"""Correctness of ATOM's LeapQuant kernels, and their error against the baselines.

  A. quantiser: Triton store/load against a torch mirror of the same algorithm,
     and one-shot reconstruction error against plain INT8 and BF16
  B. decode: commit + fold + decode over many windows. A torch shadow that
     follows the same semantics (exact fp32 steps, boundary replaced by the
     kernel's at every flush) must match the kernel to fp32 rounding; the
     error against the exact FP32 trajectory is then LeapQuant's own.
"""

import math
import os

import torch

from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule
from atom.model_ops.fla_ops.leapquant import (
    COLD_ITERS,
    WARM_ITERS,
    leapquant_buffer_shapes,
    leapquant_fold,
    leapquant_gated_delta_rule,
    leapquant_load_state,
    leapquant_store_state,
)
from atom.model_ops.fla_ops.replayssm import replayssm_commit

dev = torch.device("cuda")
# Qwen3.8-Flash-Next GDN per rank at TP2.
H, HV, K, V = 8, 24, 128, 128
# A cap of 17 flushes every 16 tokens, the paper's window.
RANK, CAP = 4, int(os.environ.get("LQ_CAP", 17))
SCALE = K**-0.5


class Source:
    """GDN-like inputs: per-channel scales and a shared direction per head give
    the state the outlier rows/columns and low-rank energy real ones have."""

    def __init__(self, n, seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.gen, self.n = gen, n
        r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
        self.k_scale = r(H, K).mul(0.8).exp()
        self.k_bias = r(H, K) * 2.0
        self.v_scale = r(HV, V).mul(0.8).exp()
        self.v_bias = r(HV, V)
        self.g_bias = r(HV) * 0.5 - 3.0

    def step(self):
        r = lambda *s: torch.randn(*s, device=dev, generator=self.gen)  # noqa: E731
        n = self.n
        q = r(n, H, K) * self.k_scale + self.k_bias
        k = r(n, H, K) * self.k_scale + self.k_bias
        v = r(n, HV, V) * self.v_scale + self.v_bias
        g = -torch.nn.functional.softplus(r(n, HV) * 0.5 + self.g_bias)
        beta = torch.sigmoid(r(n, HV))
        # bf16 values as the model produces them, carried in fp32 so the
        # kernels write an fp32 output: a bf16 output alone is 2e-3 off.
        bf = lambda x: x.bfloat16().float()  # noqa: E731
        return bf(q), bf(k), bf(v), g.float(), bf(beta)


def torch_step(S, q, k, v, g, beta):
    """One exact GDN decode step on fp32 states [N, HV, K, V]."""
    rep = HV // H
    qf = q.float().repeat_interleave(rep, 1)
    kf = k.float().repeat_interleave(rep, 1)
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * SCALE
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    S = S * g.exp()[..., None, None]
    u = beta.float()[..., None] * (v.float() - torch.einsum("nhkv,nhk->nhv", S, kf))
    S = S + kf[..., :, None] * u[..., None, :]
    return S, torch.einsum("nhkv,nhk->nhv", S, qf)


TINY = 1e-30


def unit(x):
    """Mirror of `_lq_unit`: a negligible direction becomes zeros."""
    m = x.abs().amax(-1, keepdim=True)
    x = torch.where(m > TINY, x / m.clamp_min(TINY), 0.0)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1.0)


def ref_quantize(S, rank, iters, warm=None):
    """Torch mirror of `_lq_quantize`; returns the dequantised state and keys."""
    S = S.clone()
    comp = torch.zeros(*S.shape[:-2], rank, K + V, device=dev, dtype=torch.float16)
    low = torch.zeros_like(S)
    for r in range(rank):
        if warm is None:
            kv = S.square().sum(-1).sqrt()
        else:
            # k = S u_old: one power step chained across windows
            ko = warm[..., r, :K].float()
            ko = torch.where(ko.abs().amax(-1, keepdim=True) > TINY, ko, 1.0)
            kv = torch.einsum("...kv,...v->...k", S, warm[..., r, K:].float())
            kv = torch.where(kv.abs().amax(-1, keepdim=True) > TINY, kv, ko)
        kv = unit(kv)
        for _ in range(iters):
            uv = torch.einsum("...kv,...k->...v", S, kv)
            kv = unit(torch.einsum("...kv,...v->...k", S, uv))
        uv = torch.einsum("...kv,...k->...v", S, kv)
        kh, uh = kv.half(), uv.clamp(-65504, 65504).half()
        comp[..., r, :K], comp[..., r, K:] = kh, uh
        outer = kh.float()[..., :, None] * uh.float()[..., None, :]
        S, low = S - outer, low + outer
    c = S.abs().mean(-1).sqrt().clamp_min(1e-6)
    Sr = S * (1.0 / c)[..., :, None]
    amax = Sr.abs().amax(-2)
    b = torch.where(amax > TINY, amax / 127, 1.0)
    x = Sr * (1.0 / b)[..., None, :]
    zq = x.round().clamp(-127, 127)
    return zq * c[..., :, None] * b[..., None, :] + low, comp


def int8_per_channel(S):
    """The naive baseline: symmetric INT8 per value channel, no help."""
    b = S.abs().amax(-2, keepdim=True).clamp_min(1e-30) / 127
    return (S / b).round().clamp(-127, 127) * b


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def make_states(n, steps, seed):
    """Run the exact recurrence from zero for `steps` tokens."""
    src = Source(n, seed)
    S = torch.zeros(n, HV, K, V, device=dev)
    for _ in range(steps):
        S, _ = torch_step(S, *src.step())
    return S, src


def part_a():
    print("== A. quantiser ==")
    S, _ = make_states(32, 2048, seed=1)
    slots = torch.arange(32, device=dev, dtype=torch.int32)
    z = torch.empty(64, HV, K, V, device=dev, dtype=torch.int8)
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    sc = torch.empty(64, *side_sc, device=dev)
    comp = torch.empty(64, *side_comp, device=dev, dtype=torch.float16)
    leapquant_store_state(S, z, sc, comp, slots)
    got = leapquant_load_state(z, sc, comp, slots)
    want, want_comp = ref_quantize(S, RANK, COLD_ITERS)
    print(f"kernel vs torch mirror   rel {rel(got, want):.2e}")

    # warm refit, as at a window boundary
    S2 = S * 0.97 + 0.01 * torch.randn_like(S) * S.std()
    leapquant_store_state(S2, z, sc, comp, slots, iters=COLD_ITERS)
    cold = leapquant_load_state(z, sc, comp, slots)
    warm, _ = ref_quantize(S2, RANK, WARM_ITERS, warm=want_comp)
    print(f"warm({WARM_ITERS} it) vs cold({COLD_ITERS} it) error  "
          f"{rel(warm, S2):.2e} vs {rel(cold, S2):.2e}")

    ratio = lambda X: (X.abs().amax((-2, -1)) / X.abs().median(-1).values.median(-1).values).median()  # noqa: E731
    print(f"state |max|/median {ratio(S).item():.0f}x (median over heads)")
    for name, approx in (
        ("LeapQuant INT8 (r=4)", got),
        ("INT8 per-channel", int8_per_channel(S)),
        ("BF16", S.bfloat16().float()),
    ):
        print(f"  one-shot rel error  {name:22s} {rel(approx, S):.2e}")


def part_b(n=64, prefill=512, steps=16 * CAP):
    print(f"== B. decode, {n} seqs x {HV} heads, {steps} steps (cap {CAP}) ==")
    S0, src = make_states(n, prefill, seed=2)
    slots = torch.randperm(2 * n, device=dev)[:n].to(torch.int32)
    num_slots = 2 * n

    z = torch.zeros(num_slots, HV, K, V, device=dev, dtype=torch.int8)
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    sc = torch.zeros(num_slots, *side_sc, device=dev)
    comp = torch.zeros(num_slots, *side_comp, device=dev, dtype=torch.float16)
    buf_k = torch.zeros(num_slots, HV, CAP, K, device=dev, dtype=torch.float16)
    buf_u = torch.zeros(num_slots, HV, CAP, V, device=dev, dtype=torch.float16)
    buf_g = torch.zeros(num_slots, HV, CAP, device=dev)
    write_pos = torch.zeros(num_slots, device=dev, dtype=torch.int32)
    accepted = torch.ones(n, device=dev, dtype=torch.int32)
    cu = torch.arange(n + 1, device=dev, dtype=torch.int32)

    leapquant_store_state(S0, z, sc, comp, slots)
    write_pos[slots.long()] = -1  # the prefill sentinel
    shadow = leapquant_load_state(z, sc, comp, slots)
    print(f"prefill boundary rel error {rel(shadow, S0):.2e}")

    exact = S0.clone()
    pool = torch.zeros(num_slots, HV, K, V, device=dev)
    pool[slots.long()] = S0
    bf16 = pool.bfloat16()
    int8 = S0.clone()
    int8_w = S0.clone()

    worst_shadow = 0.0
    errs = {k: [] for k in ("LeapQuant", "INT8/window", "BF16/step", "INT8/step")}
    for t in range(steps):
        q, k, v, g, beta = src.step()
        replayssm_commit(write_pos, slots, accepted, 1, CAP)
        h = write_pos[slots.long()]
        flush = h + 2 > CAP
        o = leapquant_gated_delta_rule(
            q[None], k[None], v[None], g[None], beta[None],
            z, sc, comp, buf_k, buf_u, buf_g, write_pos, slots, cu, 1,
            use_qk_l2norm_in_kernel=True,
        )[0].float()
        if flush.any():
            # the kernel's own boundary for the sequences that just folded
            folded = leapquant_load_state(z, sc, comp, slots)
            shadow = torch.where(flush[:, None, None, None], folded, shadow)
            int8_w = torch.where(flush[:, None, None, None], int8_per_channel(int8_w), int8_w)
        shadow, o_shadow = torch_step(shadow, q, k, v, g, beta)
        worst_shadow = max(worst_shadow, rel(o, o_shadow))

        exact, o_exact = torch_step(exact, q, k, v, g, beta)
        int8, o_int8 = torch_step(int8, q, k, v, g, beta)
        int8 = int8_per_channel(int8)
        int8_w, o_int8_w = torch_step(int8_w, q, k, v, g, beta)
        o_bf16, _ = fused_recurrent_gated_delta_rule(
            q[None], k[None], v[None], g[None], beta[None],
            initial_state=bf16, inplace_final_state=True, cu_seqlens=cu,
            ssm_state_indices=slots, use_qk_l2norm_in_kernel=True,
        )
        errs["LeapQuant"].append(rel(o, o_exact))
        errs["INT8/window"].append(rel(o_int8_w, o_exact))
        errs["BF16/step"].append(rel(o_bf16[0].float(), o_exact))
        errs["INT8/step"].append(rel(o_int8, o_exact))

    print(f"kernel vs shadow, worst step rel {worst_shadow:.2e}")
    print("output rel error vs exact FP32 (mean over windows):")
    w = steps // 4
    for name, e in errs.items():
        segs = [sum(e[i : i + w]) / w for i in range(0, steps, w)]
        print(f"  {name:12s} " + "  ".join(f"{x:.2e}" for x in segs))
    return worst_shadow


def part_c(layers=3, n=24, steps=3 * CAP):
    """One stacked `leapquant_fold` per step == a fold inside every layer."""
    print(f"== C. stacked fold over {layers} layers vs per-layer fold ==")
    num_slots = 2 * n
    slots = torch.randperm(num_slots, device=dev)[:n].to(torch.int32)
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    z = torch.zeros(layers, num_slots, HV, K, V, device=dev, dtype=torch.int8)
    sc = torch.zeros(layers, num_slots, *side_sc, device=dev)
    comp = torch.zeros(layers, num_slots, *side_comp, device=dev, dtype=torch.float16)
    buf_k = torch.zeros(layers, num_slots, HV, CAP, K, device=dev, dtype=torch.float16)
    buf_u = torch.zeros(layers, num_slots, HV, CAP, V, device=dev, dtype=torch.float16)
    buf_g = torch.zeros(layers, num_slots, HV, CAP, device=dev)
    srcs = []
    for l in range(layers):
        S0, src = make_states(n, 256, seed=10 + l)
        leapquant_store_state(S0, z[l], sc[l], comp[l], slots)
        srcs.append(src)
    ref = [x.clone() for x in (z, sc, comp, buf_k, buf_u, buf_g)]
    write_pos = torch.zeros(num_slots, device=dev, dtype=torch.int32)
    write_pos[slots.long()] = -1
    accepted = torch.ones(n, device=dev, dtype=torch.int32)
    cu = torch.arange(n + 1, device=dev, dtype=torch.int32)
    worst = 0.0
    for _ in range(steps):
        replayssm_commit(write_pos, slots, accepted, 1, CAP)
        leapquant_fold(z, sc, comp, buf_k, buf_u, buf_g, write_pos, slots)
        for l in range(layers):
            q, k, v, g, beta = (x[None] for x in srcs[l].step())
            args = (q, k, v, g, beta)
            o = leapquant_gated_delta_rule(
                *args, z[l], sc[l], comp[l], buf_k[l], buf_u[l], buf_g[l],
                write_pos, slots, cu, 1, use_qk_l2norm_in_kernel=True, fold=False,
            )
            o_ref = leapquant_gated_delta_rule(
                *args, *(x[l] for x in ref), write_pos, slots, cu, 1,
                use_qk_l2norm_in_kernel=True,
            )
            worst = max(worst, (o - o_ref).abs().max().item())
    same = all(torch.equal(a, b) for a, b in zip((z, sc, comp), ref[:3]))
    print(f"max |output diff| {worst:.1e}, boundary states identical: {same}")
    return worst == 0.0 and same


def part_d():
    """States real heads do produce: exactly low-rank (a decay that wiped all
    but the last token), zero, and tiny.  Each must store finite, by the cold
    prefill path and by the warm fold."""
    print("== D. degenerate states ==")
    gen = torch.Generator(device=dev).manual_seed(5)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
    unitk = lambda: torch.nn.functional.normalize(r(HV, K), dim=-1)  # noqa: E731
    outer = lambda a, b: a[..., :, None] * b[..., None, :]  # noqa: E731
    k1, u1, k2, u2 = unitk(), r(HV, V), unitk(), r(HV, V)
    cases = {
        "rank 1": outer(k1, u1),
        "rank 2": outer(k1, u1) + 0.3 * outer(k2, u2),
        "zero": torch.zeros(HV, K, V, device=dev),
        "tiny (1e-25)": r(HV, K, V) * 1e-25,
        "tiny rank 1 (1e-12)": outer(k1, u1) * 1e-12,
        "generic": make_states(1, 512, seed=3)[0][0],
    }
    n = len(cases)
    S = torch.stack(list(cases.values()))
    slots = torch.arange(n, device=dev, dtype=torch.int32)
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    z = torch.zeros(n, HV, K, V, device=dev, dtype=torch.int8)
    sc = torch.zeros(n, *side_sc, device=dev)
    comp = torch.zeros(n, *side_comp, device=dev, dtype=torch.float16)
    leapquant_store_state(S, z, sc, comp, slots)
    cold = leapquant_load_state(z, sc, comp, slots)

    # Warm: 16 records of the rank-1 token on top, then one fold.
    S0 = cold.clone()
    buf_k = torch.zeros(n, HV, CAP, K, device=dev, dtype=torch.float16)
    buf_u = torch.zeros(n, HV, CAP, V, device=dev, dtype=torch.float16)
    buf_g = torch.zeros(n, HV, CAP, device=dev)
    p = CAP - 1
    buf_k[:, :, :p] = k1.half()[None, :, None, :]
    buf_u[:, :, :p] = (0.1 * u1).half()[None, :, None, :]
    want = S0 + p * outer(buf_k[:, :, 0].float(), buf_u[:, :, 0].float())
    write_pos = torch.full((n,), p, device=dev, dtype=torch.int32)
    leapquant_fold(z, sc, comp, buf_k, buf_u, buf_g, write_pos, slots)
    warm = leapquant_load_state(z, sc, comp, slots)

    ok = True
    for i, name in enumerate(cases):
        m = S[i].abs().max()
        e_c = rel(cold[i] / m, S[i] / m) if m > 0 else cold[i].abs().max().item()
        e_w = rel(warm[i], want[i])
        fin = bool(torch.isfinite(cold[i]).all() and torch.isfinite(warm[i]).all())
        fin = fin and bool(torch.isfinite(comp[i].float()).all())
        print(f"  {name:20s} cold rel {e_c:.2e}  warm-fold rel {e_w:.2e}  finite {fin}")
        ok = ok and fin and e_c < 2e-2 and e_w < 2e-2
    return ok


def part_e(n=64, runs=200):
    """Every kernel, run on the same inputs, gives the same bytes.  Races show
    up far more often on a GPU another job keeps busy, so run it on one."""
    print(f"== E. determinism, {runs} runs per kernel ==")
    S0, src = make_states(n, 256, seed=12)
    slots = torch.arange(n, device=dev, dtype=torch.int32)
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    bufs = [
        torch.zeros(n, HV, K, V, device=dev, dtype=torch.int8),
        torch.zeros(n, *side_sc, device=dev),
        torch.zeros(n, *side_comp, device=dev, dtype=torch.float16),
        torch.zeros(n, HV, CAP, K, device=dev, dtype=torch.float16),
        torch.zeros(n, HV, CAP, V, device=dev, dtype=torch.float16),
        torch.zeros(n, HV, CAP, device=dev),
    ]
    write_pos = torch.full((n,), -1, device=dev, dtype=torch.int32)
    accepted = torch.ones(n, device=dev, dtype=torch.int32)
    cu = torch.arange(n + 1, device=dev, dtype=torch.int32)
    leapquant_store_state(S0, *bufs[:3], slots)
    # Walk one window, keeping the inputs of a step mid-window and of the flush.
    for _ in range(CAP):
        x = [t[None] for t in src.step()]
        replayssm_commit(write_pos, slots, accepted, 1, CAP)
        if int(write_pos[0]) == CAP // 2:
            mid = ([b.clone() for b in bufs], write_pos.clone(), x)
        if int(write_pos[0]) == CAP - 1:
            break
        leapquant_gated_delta_rule(
            *x, *bufs, write_pos, slots, cu, 1, use_qk_l2norm_in_kernel=True
        )
    flush = ([b.clone() for b in bufs], write_pos.clone(), x)

    def decode(state, *st):
        _, pos, xs = state
        return leapquant_gated_delta_rule(
            *xs, *st, pos, slots, cu, 1, use_qk_l2norm_in_kernel=True
        )

    cases = {
        "decode, mid-window": (lambda *st: decode(mid, *st), mid[0]),
        "decode + fold, flush step": (lambda *st: decode(flush, *st), flush[0]),
        "stacked fold": (
            lambda *st: leapquant_fold(*(b[None] for b in st), flush[1], slots),
            flush[0],
        ),
        "store (quantise)": (
            lambda *st: leapquant_store_state(S0, *st, slots),
            [torch.zeros_like(b) for b in bufs[:3]],
        ),
        "load (dequantise)": (
            lambda *st: leapquant_load_state(*st, slots), [b.clone() for b in bufs[:3]]
        ),
    }
    ok = True
    for name, (fn, state) in cases.items():
        first, differ = None, 0
        for _ in range(runs):
            st = [b.clone() for b in state]
            out = fn(*st)
            got = st + ([out] if isinstance(out, torch.Tensor) else [])
            if first is None:
                first = got
            else:
                differ += not all(torch.equal(a, b) for a, b in zip(first, got))
        print(f"{name:>26}: {differ}/{runs - 1} runs differ from the first")
        ok &= differ == 0
    return ok


if __name__ == "__main__":
    part_a()
    worst = part_b()
    assert worst < 1e-3, worst
    assert math.isfinite(worst)
    assert part_c()
    assert part_d()
    assert part_e()
