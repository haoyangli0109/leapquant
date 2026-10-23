"""Correctness of LeapQuant on KDA (Kimi Delta Attention), offline.

  A. prefill layout: `chunk_kimi_delta_attn(state_v_first=False)`, the state
     LeapQuant stores, is the V-first state the fp32 pool holds, transposed
  B. decode: commit + fold + decode over many windows.  A torch shadow (exact
     fp32 steps, boundary replaced by the kernel's at each flush) must match
     to fp32 rounding; the error against the exact trajectory is LeapQuant's
     own.  The exact trajectory is also checked against ATOM's fp32 ReplaySSM
     KDA kernel over every window, so the reference is the production one.
  C. one stacked `leapquant_fold` over layers == a fold inside every layer
  D. `replayssm_replay_records` with per-channel gates, K-first and V-first
  E. every kernel above, run 200 times on the same inputs, gives the same
     bytes: a lane-level race on ROCm shows up as an occasional wrong tile,
     far more often on a GPU another job keeps busy, so run it on one

Gate parameters (A_log, dt_bias) come from Kimi-Linear-48B's first KDA layer.
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from atom.model_ops.fla_ops.leapquant import (
    leapquant_buffer_shapes,
    leapquant_fold,
    leapquant_load_state,
    leapquant_sigmoid_gating_delta_rule,
    leapquant_store_state,
)
from atom.model_ops.fla_ops.replayssm import (
    replayssm_commit,
    replayssm_replay_records,
    replayssm_sigmoid_gating_delta_rule,
)

dev = torch.device("cuda")
MODEL = "/shareddata/haoyanli/moonshotai/Kimi-Linear-48B-A3B-Instruct"
# Kimi-Linear-48B KDA at TP1.
H = HV = 32
K = V = 128
RANK, CAP = 4, int(os.environ.get("LQ_CAP", 17))
SCALE = K**-0.5


def load_gate_params(layer=0):
    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    out = []
    for name in ("A_log", "dt_bias"):
        key = f"model.layers.{layer}.self_attn.{name}"
        with safe_open(f"{MODEL}/{index[key]}", "pt") as f:
            out.append(f.get_tensor(key).float().reshape(-1).to(dev))
    return out


A_LOG, DT_BIAS = load_gate_params()


def kda_gate(a):
    """[n, HV, K] raw gate input -> per-channel log-decay (no lower bound)."""
    x = a.float() + DT_BIAS.view(HV, K)
    return -A_LOG.exp()[:, None] * F.softplus(x)


class Source:
    """KDA-like inputs: per-channel scales/biases for the outlier structure."""

    def __init__(self, n, seed=0):
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.gen, self.n = gen, n
        r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
        self.k_scale = r(H, K).mul(0.8).exp()
        self.k_bias = r(H, K) * 2.0
        self.v_scale = r(HV, V).mul(0.8).exp()
        self.v_bias = r(HV, V)
        self.a_bias = r(HV, K)

    def step(self):
        r = lambda *s: torch.randn(*s, device=dev, generator=self.gen)  # noqa: E731
        n = self.n
        bf = lambda x: x.bfloat16()  # noqa: E731
        q = bf(r(n, H, K) * self.k_scale + self.k_bias)
        k = bf(r(n, H, K) * self.k_scale + self.k_bias)
        v = bf(r(n, HV, V) * self.v_scale + self.v_bias)
        a = bf(r(n, HV, K) + self.a_bias)
        b = bf(r(n, HV) + 1.0)
        return q, k, v, a, b


def torch_step(S, q, k, v, a, b):
    """One exact KDA decode step on fp32 K-first states [N, HV, K, V]."""
    qf = q.float()
    kf = k.float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * SCALE
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    S = S * kda_gate(a).exp()[..., :, None]
    beta = torch.sigmoid(b.float())
    u = beta[..., None] * (v.float() - torch.einsum("nhkv,nhk->nhv", S, kf))
    S = S + kf[..., :, None] * u[..., None, :]
    return S, torch.einsum("nhkv,nhk->nhv", S, qf)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def make_states(n, steps, seed):
    src = Source(n, seed)
    S = torch.zeros(n, HV, K, V, device=dev)
    for _ in range(steps):
        S, _ = torch_step(S, *src.step())
    return S, src


def side_buffers(num_slots, layers=None):
    lead = (num_slots,) if layers is None else (layers, num_slots)
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    z = torch.zeros(*lead, HV, K, V, device=dev, dtype=torch.int8)
    sc = torch.zeros(*lead, *side_sc, device=dev)
    comp = torch.zeros(*lead, *side_comp, device=dev, dtype=torch.float16)
    buf_k = torch.zeros(*lead, HV, CAP, K, device=dev, dtype=torch.float16)
    buf_u = torch.zeros(*lead, HV, CAP, V, device=dev, dtype=torch.float16)
    buf_g = torch.zeros(*lead, HV, CAP, K, device=dev)
    return z, sc, comp, buf_k, buf_u, buf_g


def part_a(n=4, t=300):
    """K-first prefill == V-first prefill transposed (outputs and states)."""
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    print("== A. prefill state layout ==")
    S0, src = make_states(n, 64, seed=7)
    gen = torch.Generator(device=dev).manual_seed(8)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
    T = n * t
    q, k = (r(1, T, H, K).bfloat16() for _ in range(2))
    v = r(1, T, HV, V).bfloat16()
    a = (r(1, T, HV, K) + src.a_bias).bfloat16()
    beta = (r(1, T, HV) + 1.0).float()
    cu = torch.arange(0, T + 1, t, device=dev, dtype=torch.int64)
    common = dict(
        A_log=A_LOG, dt_bias=DT_BIAS, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, cu_seqlens=cu,
    )  # fmt: skip
    o_v, s_v = chunk_kimi_delta_attn(
        q, k, v, a, beta, initial_state=S0.transpose(-1, -2).contiguous(),
        state_v_first=True, **common,
    )  # fmt: skip
    o_k, s_k = chunk_kimi_delta_attn(
        q, k, v, a, beta, initial_state=S0.contiguous(), state_v_first=False, **common
    )
    # The exact recurrence, token by token, as the reference for both.
    S = S0.clone()
    o_ref = []
    for i in range(t):
        idx = torch.arange(n, device=dev) * t + i
        S, o = torch_step(S, q[0, idx], k[0, idx], v[0, idx], a[0, idx], beta[0, idx])
        o_ref.append(o)
    o_ref = torch.stack(o_ref, 1).reshape(1, T, HV, V)
    print(f"final state dtype {s_k.dtype}, K-first vs V-first^T rel "
          f"{rel(s_k, s_v.transpose(-1, -2)):.2e}, outputs rel {rel(o_k, o_v):.2e}")  # fmt: skip
    print(f"K-first vs exact: state rel {rel(s_k, S):.2e}, output rel "
          f"{rel(o_k, o_ref):.2e}")  # fmt: skip
    return rel(s_k, S) < 2e-2 and rel(s_k, s_v.transpose(-1, -2)) < 1e-2


def part_b(n=64, prefill=512, steps=16 * CAP):
    print(f"== B. decode, {n} seqs x {HV} heads, {steps} steps (cap {CAP}) ==")
    S0, src = make_states(n, prefill, seed=2)
    num_slots = 2 * n
    slots = torch.randperm(num_slots, device=dev)[:n].to(torch.int32)
    z, sc, comp, buf_k, buf_u, buf_g = side_buffers(num_slots)
    write_pos = torch.zeros(num_slots, device=dev, dtype=torch.int32)
    accepted = torch.ones(n, device=dev, dtype=torch.int32)
    cu = torch.arange(n + 1, device=dev, dtype=torch.int32)

    leapquant_store_state(S0, z, sc, comp, slots)
    write_pos[slots.long()] = -1
    shadow = leapquant_load_state(z, sc, comp, slots)
    print(f"prefill boundary rel error {rel(shadow, S0):.2e}")

    # fp32 ReplaySSM, ATOM's KDA path under MTP: V-first checkpoint pool.  fp32
    # records here, so only the kernel's conventions separate it from exact.
    ckpt = torch.zeros(num_slots, HV, V, K, device=dev)
    ckpt[slots.long()] = S0.transpose(-1, -2)
    r_k, r_u, r_g = (torch.zeros_like(x, dtype=torch.float32) for x in (buf_k, buf_u, buf_g))  # noqa: E501
    r_pos = torch.zeros(num_slots, device=dev, dtype=torch.int32)
    r_pos[slots.long()] = -1

    exact = S0.clone()
    worst_shadow = worst_replay = 0.0
    errs = []
    for t in range(steps):
        q, k, v, a, b = src.step()
        replayssm_commit(write_pos, slots, accepted, 1, CAP)
        h = write_pos[slots.long()]
        flush = h + 2 > CAP
        out = torch.empty(n, HV, V, device=dev, dtype=torch.float32)
        leapquant_sigmoid_gating_delta_rule(
            q[None], k[None], v[None], a[None], b[None], A_LOG, DT_BIAS,
            z, sc, comp, buf_k, buf_u, buf_g, write_pos, slots, cu, 1,
            o=out, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        if flush.any():
            folded = leapquant_load_state(z, sc, comp, slots)
            shadow = torch.where(flush[:, None, None, None], folded, shadow)
        shadow, o_shadow = torch_step(shadow, q, k, v, a, b)
        worst_shadow = max(worst_shadow, rel(out, o_shadow))

        exact, o_exact = torch_step(exact, q, k, v, a, b)
        replayssm_commit(r_pos, slots, accepted, 1, CAP)
        o_r = torch.empty(n, HV, V, device=dev, dtype=torch.float32)
        replayssm_sigmoid_gating_delta_rule(
            q[None], k[None], v[None], a[None], b[None], A_LOG, DT_BIAS,
            ckpt, r_k, r_u, r_g, r_pos, slots, cu, 1, o=o_r,
            use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        worst_replay = max(worst_replay, rel(o_r, o_exact))
        errs.append(rel(out, o_exact))

    print(f"kernel vs shadow, worst step rel {worst_shadow:.2e}")
    print(f"fp32 ReplaySSM KDA vs torch exact, every window, worst step rel "
          f"{worst_replay:.2e}")  # fmt: skip
    w = steps // 4
    segs = [sum(errs[i : i + w]) / w for i in range(0, steps, w)]
    print("LeapQuant output rel error vs exact (mean per quarter): "
          + "  ".join(f"{x:.2e}" for x in segs))  # fmt: skip
    return worst_shadow, worst_replay


def part_c(layers=3, n=24, steps=3 * CAP):
    print(f"== C. stacked fold over {layers} layers vs per-layer fold ==")
    num_slots = 2 * n
    slots = torch.randperm(num_slots, device=dev)[:n].to(torch.int32)
    z, sc, comp, buf_k, buf_u, buf_g = side_buffers(num_slots, layers)
    srcs = []
    for l in range(layers):
        S0, src = make_states(n, 128, seed=10 + l)
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
            q, k, v, a, b = (x[None] for x in srcs[l].step())
            args = (q, k, v, a, b, A_LOG, DT_BIAS)
            o = leapquant_sigmoid_gating_delta_rule(
                *args, z[l], sc[l], comp[l], buf_k[l], buf_u[l], buf_g[l],
                write_pos, slots, cu, 1, use_qk_l2norm_in_kernel=True, fold=False,
            )  # fmt: skip
            o_ref = leapquant_sigmoid_gating_delta_rule(
                *args, *(x[l] for x in ref), write_pos, slots, cu, 1,
                use_qk_l2norm_in_kernel=True,
            )  # fmt: skip
            worst = max(worst, (o.float() - o_ref.float()).abs().max().item())
    same = all(torch.equal(a, b) for a, b in zip((z, sc, comp), ref[:3]))
    print(f"max |output diff| {worst:.1e}, boundary states identical: {same}")
    return worst == 0.0 and same


def part_d(n=8):
    print("== D. replay records (per-channel gates) ==")
    gen = torch.Generator(device=dev).manual_seed(4)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
    num_slots = 2 * n
    slots = torch.randperm(num_slots, device=dev)[:n].to(torch.int32)
    num = torch.randint(0, CAP, (n,), device=dev, generator=gen).to(torch.int32)
    num[0] = 0
    buf_k = r(num_slots, HV, CAP, K).half()
    buf_u = r(num_slots, HV, CAP, V).half()
    buf_g = kda_gate(r(num_slots * CAP, HV, K)).view(num_slots, CAP, HV, K)
    buf_g = buf_g.transpose(1, 2).contiguous()
    S0 = r(n, HV, K, V)
    want = S0.clone()
    for i in range(n):
        s = slots[i].long()
        for j in range(int(num[i])):
            g = buf_g[s, :, j]
            want[i] = want[i] * g.exp()[:, :, None] + buf_k[s, :, j].float()[
                :, :, None
            ] * buf_u[s, :, j].float()[:, None, :]
    got_k = replayssm_replay_records(S0.clone(), buf_k, buf_u, buf_g, slots, num)
    got_v = replayssm_replay_records(
        S0.transpose(-1, -2).contiguous(), buf_k, buf_u, buf_g, slots, num,
        v_first=True,
    )  # fmt: skip
    e_k, e_v = rel(got_k, want), rel(got_v.transpose(-1, -2), want)
    print(f"K-first rel {e_k:.2e}, V-first rel {e_v:.2e}")
    return e_k < 5e-5 and e_v < 5e-5


def part_e(n=64, runs=200):
    print(f"== E. determinism, {runs} runs per kernel ==")
    S0, src = make_states(n, 256, seed=12)
    slots = torch.arange(n, device=dev, dtype=torch.int32)
    bufs = side_buffers(n)
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
        leapquant_sigmoid_gating_delta_rule(
            *x, A_LOG, DT_BIAS, *bufs, write_pos, slots, cu, 1, use_qk_l2norm_in_kernel=True
        )
    flush = ([b.clone() for b in bufs], write_pos.clone(), x)
    num = torch.full((n,), CAP - 1, device=dev, dtype=torch.int32)

    def decode(state, *st):
        _, pos, xs = state
        return leapquant_sigmoid_gating_delta_rule(
            *xs, A_LOG, DT_BIAS, *st, pos, slots, cu, 1, use_qk_l2norm_in_kernel=True
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
        "replay records, K-first": (
            lambda s: replayssm_replay_records(s, *flush[0][3:], slots, num),
            [S0.clone()],
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
    print(f"A_log {A_LOG[:4].tolist()}..., dt_bias mean {DT_BIAS.mean():.3f}")
    ok_a = part_a()
    worst_shadow, worst_replay = part_b()
    ok_c = part_c()
    ok_d = part_d()
    ok_e = part_e()
    assert ok_a, "prefill layout"
    assert worst_shadow < 1e-3, worst_shadow
    assert worst_replay < 1e-4, worst_replay
    assert ok_c, "stacked fold"
    assert ok_d, "replay records"
    assert ok_e, "determinism"
    print("all KDA checks passed")
