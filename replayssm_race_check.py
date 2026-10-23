"""ReplaySSM's flush gives the same bytes on every run, as LeapQuant's does.

Each (sequence, head) runs as NV programs, one per V slice, and every one of
them reads the shared k/g records of the window it replays.  A flush restarts
the window at row 0, where one program writes this step's k/g.  While that
was always the program with i_v == 0, the write could land before a sibling
had read the old row 0, and the sibling then rebuilt its slice with the new
token in place of the oldest record (KDA's slices other than the writer's
came out up to 65% off).  A single-token GDN step now folds in a launch of
its own (`replayssm_flush`), as LeapQuant does, so its layer kernel only
appends past the rows it reads; the other kernels still fold in place, and
on a flush the k/g come from the slice that finishes reading last.  The same
inputs, run RUNS times, must give the same bytes:

  A. GDN `replayssm_gated_delta_rule`, flush step and the step before it, on
     one token (folded ahead) and on a 3-token verify window (folded in place)
  B. KDA `replayssm_sigmoid_gating_delta_rule`, flush step; every V slice must
     also match the exact recurrence
  C. LeapQuant KDA, flush step

The race needed contention to show: on an idle MI355X, GDN showed it from 64
sequences and KDA not up to 512; on a busy one both at 8, the batch
`prefix_e2e.py` decodes.  So the checks run next to two streams of matmuls,
under which the fused flush failed every A and B flush step, and A and B are
run on 8 sequences again at the end.
"""

import contextlib
import sys
import threading

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import leapquant_kda_check as C  # noqa: E402
from atom.model_ops.fla_ops.leapquant import (  # noqa: E402
    leapquant_sigmoid_gating_delta_rule,
    leapquant_store_state,
)
from atom.model_ops.fla_ops.replayssm import (  # noqa: E402
    replayssm_commit,
    replayssm_gated_delta_rule,
    replayssm_sigmoid_gating_delta_rule,
)

dev, CAP, RUNS = C.dev, C.CAP, 50
gen = torch.Generator(device=dev).manual_seed(3)


def r(*s):
    return torch.randn(*s, device=dev, generator=gen)


@contextlib.contextmanager
def busy(streams=2):
    """bf16 matmuls on side streams for as long as the block runs.

    One stream was not enough for GDN on 8 sequences to show the race."""
    stop = threading.Event()
    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    a @ a
    torch.cuda.synchronize()
    running = threading.Barrier(streams + 1)

    def loop(stream):
        with torch.cuda.stream(stream):
            for _ in range(8):
                a @ a
            running.wait()
            while not stop.is_set():
                stream.synchronize()
                for _ in range(8):
                    a @ a
            stream.synchronize()

    workers = [threading.Thread(target=loop, args=(torch.cuda.Stream(),)) for _ in range(streams)]
    for w in workers:
        w.start()
    running.wait()
    try:
        yield
    finally:
        stop.set()
        for w in workers:
            w.join()


def repeat(fn, state):
    """Bytes of fn's output and every buffer it mutates, for RUNS fresh copies."""
    outs = []
    for _ in range(RUNS):
        st = [x.clone() for x in state]
        outs.append([fn(*st).float()] + st)
    diff = max(
        max((a.float() - b.float()).abs().max().item() for a, b in zip(outs[0], o))
        for o in outs[1:]
    )
    return outs, diff


def part_a(n=128, T=1, H=16, HV=32, K=128, V=128):
    print(f"== A. GDN ReplaySSM (Qwen3.8-Flash-Next heads), bf16 records, T={T}, {n} sequences ==")
    q, k = r(1, n * T, H, K).bfloat16(), r(1, n * T, H, K).bfloat16()
    v = r(1, n * T, HV, V).bfloat16()
    g = -torch.rand(1, n * T, HV, device=dev, generator=gen) * 0.5
    beta = torch.rand(1, n * T, HV, device=dev, generator=gen).bfloat16()
    slots = torch.arange(n, device=dev, dtype=torch.int32)
    cu = torch.arange(0, n * T + 1, T, device=dev, dtype=torch.int32)
    state = [
        r(n, HV, K, V) * 0.1,
        (r(n, HV, CAP, K) * 0.1).bfloat16(),
        r(n, HV, CAP, V).bfloat16(),
        -torch.rand(n, HV, CAP, device=dev, generator=gen) * 0.5,
    ]
    ok = True
    # A flush fires once h + 2T > CAP, and the cursor stops at CAP - T.
    for h, name in ((CAP - 2 * T, "step before the flush"), (CAP - T, "flush step")):
        def fn(ckpt, bk, bu, bg, h=h):
            wp = torch.full((n,), h, device=dev, dtype=torch.int32)
            return replayssm_gated_delta_rule(
                q, k, v, g, beta, ckpt, bk, bu, bg, wp, slots, cu, T,
                use_qk_l2norm_in_kernel=True,
            )  # fmt: skip

        _, diff = repeat(fn, state)
        print(f"{name:>22}: {RUNS} runs identical {diff == 0}, max diff {diff:.2e}")
        ok &= diff == 0
    return ok


def part_b(n=64):
    print(f"== B. KDA ReplaySSM (Kimi-Linear-48B gates), fp32 records, flush step, {n} sequences ==")
    HV, K, V = C.HV, C.K, C.V
    src = C.Source(n, seed=5)
    q, k, v, a, b = src.step()
    S0 = r(n, HV, K, V) * 0.1
    h = CAP - 1
    buf_k, buf_u = r(n, HV, CAP, K), r(n, HV, CAP, V)
    buf_g = C.kda_gate(r(n * CAP, HV, K)).view(n, CAP, HV, K).transpose(1, 2).contiguous()
    want = S0.clone()
    for j in range(h):
        want = want * buf_g[:, :, j].exp()[..., None] + buf_k[:, :, j, :, None] * buf_u[:, :, j, None]
    _, o_want = C.torch_step(want, q, k, v, a, b)
    slots = torch.arange(n, device=dev, dtype=torch.int32)
    cu = torch.arange(n + 1, device=dev, dtype=torch.int32)

    def fn(ckpt, bk, bu, bg):
        wp = torch.full((n,), h, device=dev, dtype=torch.int32)
        o = torch.empty(n, HV, V, device=dev)
        replayssm_sigmoid_gating_delta_rule(
            q[None], k[None], v[None], a[None], b[None], C.A_LOG, C.DT_BIAS,
            ckpt, bk, bu, bg, wp, slots, cu, 1, o=o, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        return o

    outs, diff = repeat(fn, [S0.transpose(-1, -2).contiguous(), buf_k, buf_u, buf_g])
    print(f"{RUNS} runs identical {diff == 0}, max diff {diff:.2e}")
    nv = 4
    worst = torch.zeros(nv)
    for o in outs:
        err = (o[0] - o_want).norm(dim=0) / o_want.norm(dim=0)  # [HV, V]
        worst = torch.maximum(worst, err.view(HV, nv, V // nv).amax((0, 2)).cpu())
    print("worst rel error vs exact per V slice: " + "  ".join(f"{e:.1e}" for e in worst))
    return diff == 0 and worst.max().item() < 1e-4


def part_c(n=64):
    print("== C. LeapQuant KDA, flush step ==")
    S0, src = C.make_states(n, 256, seed=2)
    slots = torch.arange(n, device=dev, dtype=torch.int32)
    state = list(C.side_buffers(n))
    wp = torch.full((n,), -1, device=dev, dtype=torch.int32)
    acc = torch.ones(n, device=dev, dtype=torch.int32)
    cu = torch.arange(n + 1, device=dev, dtype=torch.int32)
    leapquant_store_state(S0, *state[:3], slots)
    for _ in range(CAP):
        x = [t[None] for t in src.step()]
        replayssm_commit(wp, slots, acc, 1, CAP)
        if int(wp[0]) == CAP - 1:
            break
        leapquant_sigmoid_gating_delta_rule(
            *x, C.A_LOG, C.DT_BIAS, *state, wp, slots, cu, 1, use_qk_l2norm_in_kernel=True
        )

    def fn(*st):
        return leapquant_sigmoid_gating_delta_rule(
            *x, C.A_LOG, C.DT_BIAS, *st, wp, slots, cu, 1, use_qk_l2norm_in_kernel=True
        )

    _, diff = repeat(fn, state)
    print(f"cursor {int(wp[0])}: {RUNS} runs identical {diff == 0}, max diff {diff:.2e}")
    return diff == 0


if __name__ == "__main__":
    with busy():
        ok = [
            part_a(), part_a(T=3), part_b(), part_c(),
            part_a(n=8), part_a(n=8, T=3), part_b(n=8),
        ]  # fmt: skip
    assert all(ok), ok
    print("every flush deterministic, every KDA slice exact")
