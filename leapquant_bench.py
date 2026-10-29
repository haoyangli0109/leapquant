"""GDN decode cost per step on gfx950: FP32 in place, ReplaySSM, LeapQuant.

Steady state: every sequence sits at a random point of its window, and a
profile spans `CAP - 1` steps, the flush period at T == 1, so each sequence
folds exactly once inside it and the flush cost is amortised as in serving.

LeapQuant folds every layer in one launch per step (`leapquant_fold` over the
stacked pools, as ATOM runs it); `--per-layer-fold` folds inside each layer's
call instead.

The default shape is Qwen3.8-Flash-Next per rank at TP2 (36 layers, 24 v heads,
8 q/k heads); `--layers/--hv/--h` change it, e.g. `--layers 1 --hv 32 --h 16`
for the paper's kernel figure. `bf16` keeps the per-step state in BF16 and
`replayssm_bf16` gives ReplaySSM a BF16 checkpoint, the paper's per-window BF16.

    HIP_VISIBLE_DEVICES=6 python leapquant_bench.py
    HIP_VISIBLE_DEVICES=6 python leapquant_bench.py --modes leapquant --batch 512 --fold-warps 2 4 --breakdown
    HIP_VISIBLE_DEVICES=6 python leapquant_bench.py --layers 1 --hv 32 --h 16 --batch 256 --rank 8
"""

import argparse
import itertools
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule
from atom.model_ops.fla_ops.leapquant import (
    WARM_ITERS,
    leapquant_buffer_shapes,
    leapquant_fold,
    leapquant_gated_delta_rule,
    leapquant_store_state,
)
from atom.model_ops.fla_ops.replayssm import (
    replayssm_commit,
    replayssm_gated_delta_rule,
)

dev = torch.device("cuda")
# Qwen3.8-Flash-Next GDN per rank at TP2.
H, HV, K, V, LAYERS, SLOTS, RANK = 8, 24, 128, 128, 36, 513, 4
CAP = 17  # --cap; flushes every CAP - 1 tokens at T == 1
PERIOD = CAP - 1
KERNELS = ("fused_recurrent", "_replayssm", "_leapquant")


def profiled_ms(step, steps):
    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    per = defaultdict(float)
    for e in prof.key_averages():
        if any(name in e.key for name in KERNELS):
            total = getattr(e, "device_time_total", None)
            per[e.key] += (total if total is not None else e.cuda_time_total) / 1e3
    return sum(per.values()) / steps, {k: v / steps for k, v in per.items()}


def inputs(B, gen):
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
    q = r(1, B, H, K).bfloat16()
    k = r(1, B, H, K).bfloat16()
    v = r(1, B, HV, V).bfloat16()
    g = -torch.rand(1, B, HV, device=dev, generator=gen) * 0.1
    beta = torch.rand(1, B, HV, device=dev, generator=gen).bfloat16()
    return q, k, v, g, beta


def records(rec_dtype):
    """Every layer's record buffers stacked on a leading axis, as ATOM allocates them."""
    return (
        (torch.randn(LAYERS, SLOTS, HV, CAP, K, device=dev) * 0.1).to(rec_dtype),
        (torch.randn(LAYERS, SLOTS, HV, CAP, V, device=dev) * 0.1).to(rec_dtype),
        -torch.rand(LAYERS, SLOTS, HV, CAP, device=dev) * 0.1,
    )


def bench_fp32(B, gen, state_dtype=torch.float32, **_):
    states = [
        torch.randn(SLOTS, HV, K, V, device=dev).mul_(0.01).to(state_dtype)
        for _ in range(LAYERS)
    ]
    q, k, v, g, beta = inputs(B, gen)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx = torch.randperm(SLOTS - 1, device=dev, generator=gen)[:B].to(torch.int32)

    def step():
        for s in states:
            fused_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial_state=s, inplace_final_state=True,
                cu_seqlens=cu, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
            )

    return profiled_ms(step, PERIOD)


def phased_cursor(B, gen):
    idx = torch.randperm(SLOTS - 1, device=dev, generator=gen)[:B].to(torch.int32)
    write_pos = torch.zeros(SLOTS, device=dev, dtype=torch.int32)
    # commit maps h -> h + 1 (or 1 after a flush at h = CAP - 1)
    write_pos[idx.long()] = torch.randint(0, PERIOD, (B,), device=dev, generator=gen).int()
    return idx, write_pos


def bench_replayssm(B, gen, ckpt_dtype=torch.float32, **_):
    ckpts = [
        torch.randn(SLOTS, HV, K, V, device=dev).mul_(0.01).to(ckpt_dtype)
        for _ in range(LAYERS)
    ]
    bk, bu, bg = records(torch.bfloat16)
    q, k, v, g, beta = inputs(B, gen)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx, write_pos = phased_cursor(B, gen)
    acc = torch.ones(B, device=dev, dtype=torch.int32)

    def step():
        replayssm_commit(write_pos, idx, acc, 1, CAP)
        for layer in range(LAYERS):
            replayssm_gated_delta_rule(
                q, k, v, g, beta, ckpts[layer], bk[layer], bu[layer], bg[layer],
                write_pos, idx, cu, 1, use_qk_l2norm_in_kernel=True,
            )

    return profiled_ms(step, PERIOD)


def bench_leapquant(B, gen, fold_warps=2, iters=WARM_ITERS, per_layer_fold=False):
    sc_shape, comp_shape = leapquant_buffer_shapes(HV, K, V, RANK)
    z = torch.empty(LAYERS, SLOTS, HV, K, V, device=dev, dtype=torch.int8)
    sc = torch.empty(LAYERS, SLOTS, *sc_shape, device=dev)
    comp = torch.empty(LAYERS, SLOTS, *comp_shape, device=dev, dtype=torch.float16)
    all_slots = torch.arange(SLOTS, device=dev, dtype=torch.int32)
    for layer in range(LAYERS):
        state = torch.randn(SLOTS, HV, K, V, device=dev).mul_(0.01)
        leapquant_store_state(state, z[layer], sc[layer], comp[layer], all_slots)
        del state
    bk, bu, bg = records(torch.float16)
    q, k, v, g, beta = inputs(B, gen)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx, write_pos = phased_cursor(B, gen)
    acc = torch.ones(B, device=dev, dtype=torch.int32)

    def step():
        replayssm_commit(write_pos, idx, acc, 1, CAP)
        if not per_layer_fold:
            leapquant_fold(
                z, sc, comp, bk, bu, bg, write_pos, idx, iters=iters, num_warps=fold_warps
            )
        for layer in range(LAYERS):
            leapquant_gated_delta_rule(
                q, k, v, g, beta, z[layer], sc[layer], comp[layer],
                bk[layer], bu[layer], bg[layer], write_pos, idx, cu, 1,
                use_qk_l2norm_in_kernel=True, fold=per_layer_fold, iters=iters,
            )

    return profiled_ms(step, PERIOD)


BENCH = {
    "fp32": bench_fp32,
    "bf16": lambda B, gen, **kw: bench_fp32(B, gen, state_dtype=torch.bfloat16, **kw),
    "replayssm": bench_replayssm,
    "replayssm_bf16": lambda B, gen, **kw: bench_replayssm(
        B, gen, ckpt_dtype=torch.bfloat16, **kw
    ),
    "leapquant": bench_leapquant,
}


def main():
    global CAP, PERIOD, LAYERS, HV, H, RANK
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["fp32", "replayssm", "leapquant"], choices=list(BENCH))
    ap.add_argument("--batch", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    ap.add_argument("--fold-warps", nargs="+", type=int, default=[2])
    ap.add_argument("--iters", nargs="+", type=int, default=[WARM_ITERS])
    ap.add_argument("--per-layer-fold", action="store_true")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--hv", type=int, default=HV, help="v heads (one state per v head)")
    ap.add_argument("--h", type=int, default=H, help="q/k heads")
    ap.add_argument("--rank", type=int, default=RANK, help="LeapQuant compensator tokens")
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()
    CAP, PERIOD = args.cap, args.cap - 1
    LAYERS, HV, H, RANK = args.layers, args.hv, args.h, args.rank
    print(f"shape: {LAYERS} layers, {HV} v heads, {H} q/k heads, K=V={K}; "
          f"window p={PERIOD}, rank r={RANK}", flush=True)

    base = {}
    for mode in args.modes:
        knobs = [{}]
        if mode == "leapquant":
            knobs = [
                dict(fold_warps=c, iters=d, per_layer_fold=args.per_layer_fold)
                for c, d in itertools.product(args.fold_warps, args.iters)
            ]
        for kw in knobs:
            for B in args.batch:
                gen = torch.Generator(device=dev).manual_seed(0)
                ms, per = BENCH[mode](B, gen, **kw)
                torch.cuda.empty_cache()
                if mode == "fp32":
                    base[B] = ms
                tag = mode + ("" if not kw else " " + ",".join(f"{k}={v}" for k, v in kw.items()))
                vs = f"  {base[B] / ms:4.2f}x vs fp32" if B in base and mode != "fp32" else ""
                print(f"{tag:42s} B={B:4d}: {ms:7.3f} ms/step{vs}", flush=True)
                if args.breakdown:
                    for name, t in sorted(per.items(), key=lambda x: -x[1]):
                        print(f"      {t:7.3f} ms  {name}")


if __name__ == "__main__":
    main()
