"""Prefix caching under LeapQuant: the state a checkpoint slot is filed under.

GDN checkpoints by forking: after a forward the request's slot is published
as is and the request moves to a fresh one, which its next forward writes
while reading the old.  Under ReplaySSM/LeapQuant a slot is a boundary state
plus the records its cursor covers, so both readers have to see all of it:

  A. `replayssm_replay_records` against an fp64 reference, every cursor phase
  B. decode reading a fork: the server's `_attach_replayssm` copies the slot
     before the commit.  A forked sequence must decode bit-identically to an
     unforked twin, and the checkpoint it left must not change.
  C. prefill resuming from a checkpoint published mid-decode, through
     `_attach_replayssm` and the layer's load + replay: the state must be
     the kernel's own (the shadow of `leapquant_check.py` part B), at every
     cursor phase, including an adopted slot whose cursor the prefill parks.
  D. midstep checkpoint writes: `leapquant_write_checkpoints` and the
     conv-only `write_state_checkpoints`.
"""

import os
from types import SimpleNamespace

import torch

from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata, GDNStateMixin
from atom.model_ops.fla_ops.leapquant import (
    leapquant_buffer_shapes,
    leapquant_gated_delta_rule,
    leapquant_load_state,
    leapquant_store_state,
    leapquant_write_checkpoints,
)
from atom.model_ops.fla_ops.replayssm import (
    replayssm_records_in_state,
    replayssm_replay_records,
)
from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints
from leapquant_check import CAP, HV, K, RANK, V, Source, make_states, rel, torch_step

dev = torch.device("cuda")
L = 2  # layers: the fold and the slot copy both run over a layer axis


def part_a(n=64, slots=96):
    print("== A. record replay vs fp64 ==")
    gen = torch.Generator(device=dev).manual_seed(5)
    r = lambda *s: torch.randn(*s, device=dev, generator=gen)  # noqa: E731
    S = r(n, HV, K, V)
    buf_k = r(slots, HV, CAP, K).half()
    buf_u = r(slots, HV, CAP, V).half()
    buf_g = -torch.rand(slots, HV, CAP, device=dev, generator=gen) * 0.3
    idx = torch.randperm(slots, device=dev, generator=gen)[:n].to(torch.int32)
    idx[3] = -1
    # every cursor value, the prefill sentinel and the post-flush one included
    write_pos = torch.full((slots,), -7, device=dev, dtype=torch.int32)
    write_pos[idx[idx >= 0].long()] = (
        torch.arange(n, device=dev, dtype=torch.int32)[idx >= 0] % (CAP + 1) - 1
    )
    nrec = replayssm_records_in_state(write_pos, idx, CAP)
    c = write_pos[idx.clamp(min=0).long()]
    want_n = torch.where(c + 2 > CAP, 0, c) + 1
    want_n = torch.where((c < 0) | (idx < 0), 0, want_n)
    assert torch.equal(nrec, want_n.int()), (nrec, want_n)
    assert int(nrec.max()) == CAP - 1 and int(nrec.min()) == 0

    ref = S.double().clone()
    for i in range(n):
        s, h = int(idx[i]), int(nrec[i])
        if s < 0 or h == 0:
            continue
        g = buf_g[s, :, :h].double()
        C = g.cumsum(-1)
        w = torch.exp(C[:, -1:] - C)
        ref[i] = torch.exp(C[:, -1])[:, None, None] * ref[i] + torch.einsum(
            "hj,hjk,hjv->hkv", w, buf_k[s, :, :h].double(), buf_u[s, :, :h].double()
        )
    for dtype in (torch.float32, torch.bfloat16):
        got = S.to(dtype, copy=True)
        replayssm_replay_records(got, buf_k, buf_u, buf_g, idx, nrec)
        tol = 1e-6 if dtype == torch.float32 else 1e-2
        err = rel(got.double(), ref)
        print(f"  {str(dtype):15s} rel {err:.2e}")
        assert err < tol
    return True


class Pool:
    """The runner-side tensors `_attach_replayssm` and the slot copy touch."""

    def __init__(self, num_slots, max_bs):
        side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
        z = lambda *s, dt=torch.float32: torch.zeros(L, num_slots, *s, device=dev, dtype=dt)  # noqa: E731
        self.runner = SimpleNamespace(
            mamba_k_cache=z(3, 8, dt=torch.bfloat16),
            mamba_v_cache=z(HV, K, V, dt=torch.int8),
            leapquant_scale=z(*side_sc),
            leapquant_comp=z(*side_comp, dt=torch.float16),
            replayssm_buf_k=z(HV, CAP, K, dt=torch.float16),
            replayssm_buf_u=z(HV, CAP, V, dt=torch.float16),
            replayssm_buf_g=z(HV, CAP),
            replayssm_write_pos=torch.zeros(num_slots, device=dev, dtype=torch.int32),
        )
        # what a slot's previous tenant would have left
        for t in (
            self.runner.leapquant_scale,
            self.runner.leapquant_comp,
            self.runner.replayssm_buf_k,
            self.runner.replayssm_buf_u,
            self.runner.replayssm_buf_g,
        ):
            t.normal_()
        self.runner.mamba_v_cache.random_(-127, 128)
        self.runner.replayssm_write_pos.random_(0, CAP)
        self.builder = SimpleNamespace(
            model_runner=self.runner,
            replayssm=True,
            leapquant=True,
            use_spec_decode=False,
            replayssm_cache_len=CAP,
            replayssm_route="auto",
            replayssm_max_query_len=1,
            num_accepted_tokens=torch.ones(max_bs, device=dev, dtype=torch.int32),
        )
        self.builder.relocate_state_slots = lambda pairs: GDNStateMixin.relocate_state_slots(
            self.builder, pairs
        )

    def attach(self, out_slots, in_slots, is_prefill, cached, fork_srcs):
        out_t = torch.tensor(out_slots, device=dev, dtype=torch.int32)
        in_t = torch.tensor(in_slots, device=dev, dtype=torch.int32)
        self.builder.non_spec_state_indices_tensor = SimpleNamespace(gpu=out_t)
        n = len(out_slots)
        md = GDNAttentionMetadata(
            num_prefills=n if is_prefill else 0,
            num_prefill_tokens=0,
            num_decodes=0 if is_prefill else n,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=n,
            non_spec_state_indices_tensor=out_t,
            non_spec_state_indices_in_tensor=in_t,
        )
        batch = SimpleNamespace(
            num_cached_tokens=cached,
            state_slots=[[s] for s in out_slots],
            state_fork_srcs=fork_srcs,
        )
        GDNStateMixin._attach_replayssm(self.builder, md, n, is_prefill, batch)
        return md

    def layer(self, i):
        r = self.runner
        return (
            r.mamba_v_cache[i],
            r.leapquant_scale[i],
            r.leapquant_comp[i],
            r.replayssm_buf_k[i],
            r.replayssm_buf_u[i],
            r.replayssm_buf_g[i],
        )

    def slot_bytes(self, s):
        r = self.runner
        caches = [r.mamba_k_cache, r.mamba_v_cache, r.leapquant_scale, r.leapquant_comp,
                  r.replayssm_buf_k, r.replayssm_buf_u, r.replayssm_buf_g]
        return [c[:, s].clone() for c in caches] + [r.replayssm_write_pos[s].clone()]


def resume_state(pool, layer, in_slot, out_slot, cached=64):
    """What a resuming prefill hands the chunk kernel, as the server builds it."""
    md = pool.attach([out_slot], [in_slot], True, [cached], [in_slot if in_slot != out_slot else -1])
    z, sc, comp, bk, bu, bg = pool.layer(layer)
    in_t = md.non_spec_state_indices_in_tensor
    s = leapquant_load_state(z, sc, comp, in_t)
    boundary = s.clone()
    assert md.replayssm_resume_records is not None
    replayssm_replay_records(s, bk, bu, bg, in_t, md.replayssm_resume_records)
    return s, boundary, int(md.replayssm_resume_records[0])


def part_bc(n=CAP + 3, prefill=512, lead=3):
    """2n sequences in twin pairs; sequence i forks after `lead + i` decode
    steps, which walks the fork across every cursor phase (the first one sits
    at the prefill sentinel), while its twin i + n never forks."""
    steps = lead + n + 2 * CAP
    print(f"== B/C. forks over every cursor phase, {n} twin pairs, {steps} steps ==")
    num_slots = 8 * n
    pool = Pool(num_slots, 2 * n)
    S0, srcs = [], []
    for l in range(L):
        S, src = make_states(n, prefill, seed=10 + l)
        S0.append(torch.cat([S, S]))
        srcs.append(src)
    free = torch.randperm(num_slots, device=dev).tolist()
    slots = [free.pop() for _ in range(2 * n)]
    cu = torch.arange(2 * n + 1, device=dev, dtype=torch.int32)

    # prefill: cold rows, cursors parked by the builder
    pool.attach(slots, slots, True, [0] * (2 * n), [-1] * (2 * n))
    slot_t = torch.tensor(slots, device=dev, dtype=torch.int32)
    for l in range(L):
        z, sc, comp, *_ = pool.layer(l)
        leapquant_store_state(S0[l], z, sc, comp, slot_t)
    assert bool((pool.runner.replayssm_write_pos[slot_t.long()] == -1).all())
    shadow = [leapquant_load_state(*pool.layer(l)[:3], slot_t) for l in range(L)]

    fork_at = {i: (0 if i == 0 else lead + i) for i in range(n)}
    worst_twin, worst_shadow, worst_resume = 0.0, 0.0, 0.0
    phases, margins = set(), []
    for t in range(steps):
        fork_srcs = [-1] * (2 * n)
        ckpts = {}
        for i, f in fork_at.items():
            if f == t:
                ckpts[i] = slots[i]
                fork_srcs[i] = slots[i]
                slots[i] = free.pop()
        ins = [s if s >= 0 else o for s, o in zip(fork_srcs, slots)]
        snap = {i: pool.slot_bytes(s) for i, s in ckpts.items()}
        before = [x.clone() for x in shadow]

        pool.attach(slots, ins, False, [prefill + t] * (2 * n), fork_srcs)
        slot_t = torch.tensor(slots, device=dev, dtype=torch.int32)
        h = pool.runner.replayssm_write_pos[slot_t.long()]
        flush = (h + 2 > CAP)[:, None, None, None]
        outs = []
        for l in range(L):
            q, k, v, g, beta = (torch.cat([x, x]) for x in srcs[l].step())
            z, sc, comp, bk, bu, bg = pool.layer(l)
            o = leapquant_gated_delta_rule(
                q[None], k[None], v[None], g[None], beta[None],
                z, sc, comp, bk, bu, bg, pool.runner.replayssm_write_pos, slot_t, cu, 1,
                use_qk_l2norm_in_kernel=True, fold=False,
            )[0].float()
            folded = leapquant_load_state(z, sc, comp, slot_t)
            shadow[l] = torch.where(flush, folded, shadow[l])
            shadow[l], o_shadow = torch_step(shadow[l], q, k, v, g, beta)
            worst_shadow = max(worst_shadow, rel(o, o_shadow))
            diff = (o[:n] - o[n:]).abs().amax((-2, -1))
            if worst_twin == 0.0 and bool((diff > 0).any()):
                i = int(diff.argmax())
                print(f"  first twin mismatch: step {t} layer {l} seq {i} "
                      f"(forks at {fork_at[i]}) |diff| {diff[i].item():.1e}")
            worst_twin = max(worst_twin, diff.max().item())
            outs.append(o)

        for i, src_slot in ckpts.items():
            after = pool.slot_bytes(src_slot)
            assert all(torch.equal(a, b) for a, b in zip(snap[i], after)), (
                f"checkpoint of seq {i} changed while its fork was read"
            )
            # C: a later request resumes from it, into a fresh slot ...
            for l in range(L):
                got, boundary, nrec = resume_state(pool, l, src_slot, free[-1])
                want = before[l][i]
                err = rel(got[0], want)
                worst_resume = max(worst_resume, err)
                if nrec > 0:
                    margins.append(rel(boundary[0], want) / max(err, 1e-12))
                # ... and an off-by-one count is far off
                for d in (-1, 1):
                    if 0 <= nrec + d <= CAP - 1:
                        s = boundary.clone()
                        replayssm_replay_records(
                            s, *pool.layer(l)[3:],
                            torch.tensor([src_slot], device=dev, dtype=torch.int32),
                            torch.tensor([nrec + d], device=dev, dtype=torch.int32),
                        )
                        assert rel(s[0], want) > 5 * err, (i, nrec, d)
            phases.add(nrec)
            # ... or adopts a copy of it, so the prefill parks that very cursor
            spare = free.pop()
            pool.builder.relocate_state_slots([(src_slot, spare)])
            got, _, nrec2 = resume_state(pool, 0, spare, spare)
            assert nrec2 == nrec and int(pool.runner.replayssm_write_pos[spare]) == -1
            assert rel(got[0], before[0][i]) <= 1.01 * worst_resume + 1e-7
            free.insert(0, spare)

    print(f"  twin outputs (forked vs not), max |diff| {worst_twin:.1e}")
    print(f"  kernel vs shadow, worst step rel        {worst_shadow:.2e}")
    print(f"  resumed state vs shadow, worst rel      {worst_resume:.2e}")
    print(f"  boundary-only error / resumed error     >= {min(margins):.0f}x")
    print(f"  record counts covered: {sorted(phases)}")
    assert worst_twin == 0.0
    assert phases == set(range(CAP)), phases
    assert worst_resume < 1e-2 and worst_shadow < 1e-2
    return True


def part_d(n=6, slots=32, T=300, chunk=64):
    print("== D. midstep checkpoint writes ==")
    gen = torch.Generator(device=dev).manual_seed(9)
    lens = [T, 130, 64]
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    nchunks = [(x + chunk - 1) // chunk for x in lens]
    chunk_offsets = torch.tensor([0, *torch.tensor(nchunks).cumsum(0).tolist()], device=dev, dtype=torch.int32)
    h = torch.randn(1, sum(nchunks), HV, K, V, device=dev, generator=gen).bfloat16()
    final = torch.randn(len(lens), HV, K, V, device=dev, generator=gen)
    # (row, slot, off, is_end): interior boundaries and the step ends
    targets = [(0, 3, 64, 0), (0, 7, 256, 0), (0, 9, T, 1), (1, 11, 128, 0), (1, 12, 130, 1), (2, 20, 64, 1)]
    rows, dst, offs, is_end = (
        torch.tensor(c, device=dev, dtype=torch.int32) for c in zip(*targets)
    )
    side_sc, side_comp = leapquant_buffer_shapes(HV, K, V, RANK)
    z = torch.zeros(slots, HV, K, V, device=dev, dtype=torch.int8)
    sc = torch.zeros(slots, *side_sc, device=dev)
    comp = torch.zeros(slots, *side_comp, device=dev, dtype=torch.float16)
    leapquant_write_checkpoints(h, final, z, sc, comp, rows, dst, offs, is_end, chunk_offsets, chunk)
    want = torch.stack([
        final[r] if e else h[0, int(chunk_offsets[r]) + o // chunk].float()
        for r, _, o, e in targets
    ])
    z2, sc2, comp2 = torch.zeros_like(z), torch.zeros_like(sc), torch.zeros_like(comp)
    leapquant_store_state(want, z2, sc2, comp2, dst)
    assert torch.equal(z, z2) and torch.equal(sc, sc2) and torch.equal(comp, comp2)
    print(f"  SSM half == store_state of the expected states (bit-exact), rel err "
          f"{rel(leapquant_load_state(z, sc, comp, dst), want):.2e}")

    D, state_len = 40, 3
    x = torch.randn(int(cu[-1]), D + 24, device=dev, generator=gen).bfloat16()[:, :D]
    conv = torch.zeros(slots, state_len, D, device=dev, dtype=torch.bfloat16).transpose(-1, -2)
    z_before = z.clone()
    write_state_checkpoints(
        h, z, x, conv, rows, dst, offs, is_end, dst, chunk_offsets, cu, chunk, copy_ssm=False
    )
    assert torch.equal(z, z_before), "conv-only write touched the SSM pool"
    for r, s, o, _ in targets:
        end = int(cu[r]) + o
        assert torch.equal(conv[s], x[end - state_len : end].T)
    print("  conv-only write: windows exact, SSM pool untouched")
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    ok = part_a() and part_bc() and part_d()
    print("ALL OK" if ok else "FAILED")
