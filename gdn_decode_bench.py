"""GDN decode kernel time per step vs batch size, as ATOM runs it on gfx950."""

import torch
from torch.profiler import ProfilerActivity, profile

from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule

dev = torch.device("cuda")
# Qwen3.8-Flash-Next GDN per rank at TP2: 16/2 key heads, 48/2 value heads.
H, HV, K, V, LAYERS, SLOTS = 8, 24, 128, 128, 36, 513
KERNEL = "fused_recurrent_gated_delta_rule_fwd_kernel"


def kernel_us(prof):
    for e in prof.key_averages():
        if KERNEL in e.key:
            total = getattr(e, "device_time_total", None)
            if total is None:
                total = e.cuda_time_total
            return total / e.count
    raise RuntimeError("kernel not found in profile")


def run(B, dtype, steps=5):
    torch.manual_seed(0)
    # One pool per layer, so the working set matches a real decode step.
    states = [
        torch.randn(SLOTS, HV, K, V, device=dev, dtype=dtype).mul_(0.01)
        for _ in range(LAYERS)
    ]
    q = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, B, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, B, HV, V, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(1, B, HV, device=dev, dtype=torch.float32)
    beta = torch.rand(1, B, HV, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(B + 1, device=dev, dtype=torch.int32)
    idx = torch.randperm(SLOTS - 1, device=dev)[:B].to(torch.int32)

    def step():
        for s in states:
            fused_recurrent_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=s,
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=idx,
                use_qk_l2norm_in_kernel=True,
            )

    step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
    us = kernel_us(prof)
    state_bytes = 2 * B * HV * K * V * states[0].element_size()
    del states
    torch.cuda.empty_cache()
    return us, state_bytes / (us * 1e-6) / 1e12


for dtype in (torch.float32, torch.bfloat16):
    for B in (1, 32, 64, 128, 256, 512):
        us, tbps = run(B, dtype)
        print(
            f"{str(dtype).split('.')[-1]:9s} B={B:4d}: {us:7.1f} us/layer, "
            f"{us * LAYERS / 1000:6.2f} ms/step (36 layers), state r+w {tbps:5.2f} TB/s",
            flush=True,
        )
