"""KDA decode cost per step (Kimi-Linear-48B, TP1: 20 layers x 32 heads) on
gfx950: FP32 in place, ReplaySSM, LeapQuant.

As `leapquant_bench.py`: steady state, every sequence at a random point of its
window, and a profile spans `CAP - 1` steps so each sequence folds exactly once
inside it.  LeapQuant folds every layer in one launch per step, as ATOM runs it.

    HIP_VISIBLE_DEVICES=4 python leapquant_kda_bench.py --breakdown
"""

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from atom.model_ops.fla_ops.leapquant import (
    leapquant_buffer_shapes,
    leapquant_fold,
    leapquant_sigmoid_gating_delta_rule,
    leapquant_store_state,
)
from atom.model_ops.fla_ops.replayssm import (
    replayssm_commit,
    replayssm_sigmoid_gating_delta_rule,
)

dev = torch.device("cuda")
H = HV = 32
K = V = 128
LAYERS, RANK = 20, 4
CAP = 17
PERIOD = CAP - 1
KERNELS = ("fused_sigmoid_gating", "_replayssm", "_leapquant")


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
    a = r(1, B, HV, K).bfloat16()
    b = r(1, B, HV).bfloat16()
    A_log = torch.rand(HV, device=dev, generator=gen)
    dt_bias = r(HV * K) - 2.0
    return q, k, v, a, b, A_log, dt_bias


def records(slots, rec_dtype):
    return (
        (torch.randn(LAYERS, slots, HV, CAP, K, device=dev) * 0.1).to(rec_dtype),
        (torch.randn(LAYERS, slots, HV, CAP, V, device=dev) * 0.1).to(rec_dtype),
        -torch.rand(LAYERS, slots, HV, CAP, K, device=dev) * 0.1,
    )


def phased_cursor(B, slots, gen):
    idx = torch.randperm(slots - 1, device=dev, generator=gen)[:B].to(torch.int32)
    write_pos = torch.zeros(slots, device=dev, dtype=torch.int32)
    write_pos[idx.long()] = torch.randint(0, PERIOD, (B,), device=dev, generator=gen).int()
    return idx, write_pos


def bench_fp32(B, gen):
    slots = B + 1
    states = [torch.randn(slots, HV, V, K, device=dev).mul_(0.01) for _ in range(LAYERS)]
    q, k, v, a, b, A_log, dt_bias = inputs(B, gen)
    o = torch.empty(B, HV, V, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx = torch.randperm(slots - 1, device=dev, generator=gen)[:B].to(torch.int32)

    def step():
        for s in states:
            fused_sigmoid_gating_delta_rule_update(
                A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, o=o,
                initial_state=s, inplace_final_state=True, cu_seqlens=cu,
                ssm_state_indices=idx, use_qk_l2norm_in_kernel=True, is_kda=True,
            )  # fmt: skip

    return profiled_ms(step, PERIOD)


def bench_replayssm(B, gen):
    slots = B + 1
    ckpts = [torch.randn(slots, HV, V, K, device=dev).mul_(0.01) for _ in range(LAYERS)]
    bk, bu, bg = records(slots, torch.bfloat16)
    q, k, v, a, b, A_log, dt_bias = inputs(B, gen)
    o = torch.empty(B, HV, V, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx, write_pos = phased_cursor(B, slots, gen)
    acc = torch.ones(B, device=dev, dtype=torch.int32)

    def step():
        replayssm_commit(write_pos, idx, acc, 1, CAP)
        for layer in range(LAYERS):
            replayssm_sigmoid_gating_delta_rule(
                q, k, v, a, b, A_log, dt_bias, ckpts[layer], bk[layer], bu[layer],
                bg[layer], write_pos, idx, cu, 1, o=o, use_qk_l2norm_in_kernel=True,
            )  # fmt: skip

    return profiled_ms(step, PERIOD)


def bench_leapquant(B, gen, num_warps=1, block_v=128):
    slots = B + 1
    sc_shape, comp_shape = leapquant_buffer_shapes(HV, K, V, RANK)
    z = torch.empty(LAYERS, slots, HV, K, V, device=dev, dtype=torch.int8)
    sc = torch.empty(LAYERS, slots, *sc_shape, device=dev)
    comp = torch.empty(LAYERS, slots, *comp_shape, device=dev, dtype=torch.float16)
    all_slots = torch.arange(slots, device=dev, dtype=torch.int32)
    for layer in range(LAYERS):
        state = torch.randn(slots, HV, K, V, device=dev).mul_(0.01)
        leapquant_store_state(state, z[layer], sc[layer], comp[layer], all_slots)
        del state
    bk, bu, bg = records(slots, torch.float16)
    q, k, v, a, b, A_log, dt_bias = inputs(B, gen)
    o = torch.empty(B, HV, V, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx, write_pos = phased_cursor(B, slots, gen)
    acc = torch.ones(B, device=dev, dtype=torch.int32)

    def step():
        replayssm_commit(write_pos, idx, acc, 1, CAP)
        leapquant_fold(z, sc, comp, bk, bu, bg, write_pos, idx)
        for layer in range(LAYERS):
            leapquant_sigmoid_gating_delta_rule(
                q, k, v, a, b, A_log, dt_bias, z[layer], sc[layer], comp[layer],
                bk[layer], bu[layer], bg[layer], write_pos, idx, cu, 1, o=o,
                use_qk_l2norm_in_kernel=True, fold=False, block_v=block_v,
                num_warps=num_warps,
            )  # fmt: skip

    return profiled_ms(step, PERIOD)


BENCH = {"fp32": bench_fp32, "replayssm": bench_replayssm, "leapquant": bench_leapquant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(BENCH))
    ap.add_argument("--batch", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    ap.add_argument("--warps", nargs="+", type=int, default=[1], help="LeapQuant decode")
    ap.add_argument("--block-v", nargs="+", type=int, default=[128], help="LeapQuant decode")
    ap.add_argument("--breakdown", action="store_true")
    args = ap.parse_args()

    base = {}
    for mode in args.modes:
        knobs = [{}]
        if mode == "leapquant":
            knobs = [dict(num_warps=w, block_v=bv) for w in args.warps for bv in args.block_v]
        for kw in knobs:
            for B in args.batch:
                gen = torch.Generator(device=dev).manual_seed(0)
                ms, per = BENCH[mode](B, gen, **kw)
                torch.cuda.empty_cache()
                if mode == "fp32":
                    base[B] = ms
                tag = mode + ("" if not kw else " " + ",".join(f"{k}={v}" for k, v in kw.items()))
                vs = f"  {base[B] / ms:4.2f}x vs fp32" if B in base and mode != "fp32" else ""
                print(f"{tag:32s} B={B:4d}: {ms:6.2f} ms/step{vs}", flush=True)
                if args.breakdown:
                    for name, t in sorted(per.items(), key=lambda x: -x[1]):
                        print(f"      {t:7.3f} ms  {name}")


if __name__ == "__main__":
    main()
