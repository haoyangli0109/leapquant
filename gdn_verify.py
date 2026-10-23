import torch
import torch.nn.functional as F
from safetensors import safe_open

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops.chunk import chunk_gated_delta_rule
from atom.model_ops.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from atom.model_ops.qwen4_exp.ops.gated import sigmoid_rmsnorm

dev = "cuda"
CKPT = "/shareddata/haoyanli/Qwen/Qwen3.8-Flash-Next/model-00003-of-00131.safetensors"
P = "model.language_model.layers.1.linear_attn."
with safe_open(CKPT, "pt") as f:
    W = {k[len(P):]: f.get_tensor(k).to(dev) for k in f.keys() if k.startswith(P)}

H, HV, K, V, D, KW, EPS = 16, 48, 128, 128, 2560, 4, 1e-6
KD, VD = H * K, HV * V # 16 * 128 = 2048, 48 * 128 = 6144
C = 2 * KD + VD # 2 * 2048 + 6144 = 10240
T = 150
torch.manual_seed(0)
x = torch.randn(T, D, device=dev).bfloat16() # [150, 2560]


# ---------------- reference: fp32, one token at a time ----------------
def ref_conv(mixed_qkv):
    y = F.conv1d(mixed_qkv.T[None], W["conv1d.weight"].float(), groups=C, padding=KW - 1) # padding来扩展，两边都补零
    return F.silu(y[0, :, :T].T).split([KD, KD, VD], dim=-1) # 所以这里去掉尾部没有意义的卷积结果


def ref_gating(a, b):
    g = -W["A_log"].float().exp() * F.softplus(a + W["dt_bias"].float())# [150, 48]
    return g, b.sigmoid() # [150, 48]


def ref_recurrence(q, k, v, g, beta, consecutive=True):
    rep = HV // H
    expand = (lambda t: t.repeat_interleave(rep, dim=1)) if consecutive else (lambda t: t.repeat(1, rep, 1))
    q = expand(q.view(T, H, K))# 复制q,k，以配合v，因为这里是group attn，共用q, k的  [150, 48, 128]
    k = expand(k.view(T, H, K)) #  [150, 48, 128]
    v = v.view(T, HV, V) # [150, 48, 128]
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5 # 归一化，除以自己的长度，q要额外乘以1/根号128，训练时候就i这样的
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    alpha = g.exp() # beta, g.shape = [150, 48]
    S = torch.zeros(HV, K, V, device=dev) # [48, 128, 128]
    o = torch.empty(T, HV, V, device=dev) # [150, 48, 128]
    for t in range(T):
        S = alpha[t, :, None, None] * S # [48, 1, 1] * [48, 128, 128]
        v_old = torch.einsum("hk,hkv->hv", k[t], S) # [48, 128] @ [48, 128, 128] -> [48, 128]
        # [48, 128, 128] + [48, 128]  ([48, 1] * ([48, 128] - [48, 128]))
        S = S + torch.einsum("hk,hv->hkv", k[t], beta[t, :, None] * (v[t] - v_old))
        o[t] = torch.einsum("hk,hkv->hv", q[t], S)
    return o, S


def ref_norm(o, z):
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS) * W["norm.weight"].float()
    return (o * z.view(T, HV, V).sigmoid()).reshape(T, VD)


def reference(x):
    xf = x.float()
    mixed_qkv = xf @ W["in_proj_qkv.weight"].float().T
    z = xf @ W["in_proj_z.weight"].float().T
    b = xf @ W["in_proj_b.weight"].float().T # [150, 48]
    a = xf @ W["in_proj_a.weight"].float().T # [150, 48]
    q, k, v = ref_conv(mixed_qkv)
    g, beta = ref_gating(a, b)
    o, S = ref_recurrence(q, k, v, g, beta)
    return ref_norm(o, z) @ W["out_proj.weight"].float().T, S


# ---------------- ATOM kernels ----------------
W_cat = torch.cat([W[f"in_proj_{n}.weight"] for n in ("qkv", "z", "b", "a")])
Wc = W["conv1d.weight"].view(C, KW)
idx = torch.tensor([0], device=dev, dtype=torch.int32)


def project(x):
    return F.linear(x, W_cat).split([C, VD, HV, HV], dim=-1)


def atom_prefill(x):
    mixed_qkv, z, b, a = project(x)
    conv_state = torch.zeros(1, C, KW - 1, device=dev, dtype=torch.bfloat16)
    qsl = torch.tensor([0, T], device=dev, dtype=torch.int32)
    q, k, v = causal_conv1d_fn(
        mixed_qkv.T, Wc, None, conv_states=conv_state, query_start_loc=qsl,
        k_dim_size=KD, v_dim_size=VD, cache_indices=idx,
        has_initial_state=torch.tensor([False], device=dev), activation="silu",
    )
    g, beta = fused_gdn_gating(W["A_log"], a, b, W["dt_bias"])
    o, S = chunk_gated_delta_rule(
        q=q.view(1, T, H, K), k=k.view(1, T, H, K), v=v.view(1, T, HV, V),
        g=g, beta=beta, initial_state=torch.zeros(1, HV, K, V, device=dev),
        output_final_state=True, cu_seqlens=qsl, head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    o = o.view(T, HV, V)
    out = F.linear(sigmoid_rmsnorm(o, z.view(T, HV, V), W["norm.weight"], EPS), W["out_proj.weight"])
    return dict(mixed_qkv=mixed_qkv, z=z, a=a, b=b, q=q, k=k, v=v, g=g[0], beta=beta[0], o=o, S=S[0], out=out)


def atom_decode(x):
    mixed_qkv, z, b, a = project(x)
    conv_state = torch.zeros(1, C, KW - 1, device=dev, dtype=torch.bfloat16)
    ssm_state = torch.zeros(1, HV, K, V, device=dev)
    cu = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    os = []
    for t in range(T):
        q, k, v = causal_conv1d_update(
            mixed_qkv[t : t + 1], conv_state, Wc, KD, VD, None, "silu",
            conv_state_indices=idx, validate_data=True,
        )
        g, beta = fused_gdn_gating(W["A_log"], a[t : t + 1], b[t : t + 1], W["dt_bias"])
        o, _ = fused_recurrent_gated_delta_rule(
            q=q.view(1, 1, H, K), k=k.view(1, 1, H, K), v=v.view(1, 1, HV, V),
            g=g, beta=beta, initial_state=ssm_state, inplace_final_state=True,
            cu_seqlens=cu, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
        )
        os.append(o.view(HV, V))
    o = torch.stack(os)
    out = F.linear(sigmoid_rmsnorm(o, z.view(T, HV, V), W["norm.weight"], EPS), W["out_proj.weight"])
    return dict(o=o, S=ssm_state[0], out=out)


def err(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    rel = ((a - b).norm() / b.norm()).item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    return f"rel_err={rel:.2e}  cos={cos:.6f}"


with torch.no_grad():
    out_ref, S_ref = reference(x)
    p = atom_prefill(x)
    d = atom_decode(x)

    # stage-by-stage: feed each reference stage the ATOM stage's own input
    rq, rk, rv = ref_conv(p["mixed_qkv"].float())
    rg, rbeta = ref_gating(p["a"].float(), p["b"].float())
    ro, rS = ref_recurrence(p["q"].float(), p["k"].float(), p["v"].float(), p["g"].float(), p["beta"].float())
    rn = ref_norm(p["o"].float(), p["z"].float())
    an = sigmoid_rmsnorm(p["o"], p["z"].view(T, HV, V), W["norm.weight"], EPS)

    print("[stage] conv1d+SiLU   q", err(p["q"], rq), "| k", err(p["k"], rk), "| v", err(p["v"], rv))
    print("[stage] gating  g    ", err(p["g"], rg), "| beta", err(p["beta"], rbeta))
    print("[stage] GDN prefill o", err(p["o"], ro), "| final S", err(p["S"], rS))
    print("[stage] GDN decode  o", err(d["o"], ro), "| final S", err(d["S"], rS))
    print("[stage] gated norm   ", err(an, rn))
    print("[e2e]   prefill out  ", err(p["out"], out_ref))
    print("[e2e]   decode  out  ", err(d["out"], out_ref))

    # control: v head h -> k head h % 16 (interleaved) instead of h // 3 must be far off
    wrong, _ = ref_recurrence(p["q"].float(), p["k"].float(), p["v"].float(), p["g"].float(), p["beta"].float(), consecutive=False)
    print("[control] interleaved head map", err(p["o"], wrong))
    print("[info] alpha range in this run: %.4f ~ %.4f" % (p["g"].exp().min(), p["g"].exp().max()))
