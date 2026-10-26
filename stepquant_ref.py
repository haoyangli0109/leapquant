"""STEPQuant 最小参考实现：多头门控 Delta 规则的循环状态，fp64，CPU 上直接跑（默认参数约 20 秒）。

写法和 leapquant_ref.py 一样：只留核心算法，每个函数对应论文的一个公式、官方代码的一个模块
（论文 arXiv:2609.38169；官方代码 github.com/Dreamer-Toby/STEPQuant，下文简称"官方"）。
代码是按论文公式重写的，官方实现里的 kernel 细节（分 tile、位打包、异步 stream）只保留数学部分。

  函数                    论文           作用                                        官方
  delta_step              式 (1)(4)      一步门控 Delta 更新 + 读出                   core.py::delta_step
  row_impact              式 (9)(10)     g = Aᵀq，行影响 ω = E[g²]                    core.py::row_impact
  impact_factors          脚注 1         ω → w（压缩 + 头内归一化）                   core.py::impact_factors
  lifetime_weight         式 (6)(7)      平均对数门 ℓ → 寿命权重 L                    core.py::lifetime_weight
  dual_axis_fit           式 (11)-(14)   键行感知双轴拟合：每步写回时的量化器           quantization.py::fit_state
  collect_stats           附录 A.3       标定：沿全精度轨迹收集 ω、ℓ、状态快照          calibration.py
  distortion_table        式 (17)(18)    每个单元在每个候选位宽下的失真 d_u(b)         calibration.py
  allocate_dp             式 (8)         比特分配：多选背包的精确 DP（Qwen）            allocation.py
  allocate_lagrangian     式 (8)         比特分配：拉格朗日 + 贪心修补（KDA）           allocation.py
  build_plan              附录 C.2       标定 → 选枢轴 → 分配，产出精度表               calibration.py::calibrate
  fused_readout           附录 F.1       不拼出 Ŝ，直接从码字算 δ 和 y                 kernels/readout.py
  rebuild_from_codes      附录 F.1       写回时从旧码字重算 X，不存 FP32 中间状态        kernels/rows.py、fit.py

两种架构：
  gdn（Qwen3.5/3.8 的 Gated DeltaNet）：每头一个标量门，分配单元是整个头；
  kda（Kimi Linear 的 KDA）：每个键通道一个门，分配单元是一个键行。
文件末尾有原理说明，建议先跑一遍，对着输出读。
"""
import math
import time
import unicodedata

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.set_num_threads(min(8, torch.get_num_threads()))   # 张量很小，线程多了反而慢，也免得占满整台机器
K = V = 128                    # 每个头的状态 S 是 K×V：行 = 键通道，列 = 值通道
GAMMA = 0.25                   # 行影响的压缩指数 γ：w ∝ ω^(γ/2) = ω^(1/8)
HORIZON = 2048                 # 寿命权重的视野 H（官方 configs/calibration/paper.json）
TINY, FP16_MAX = 1e-30, 65504.0


def fp16(x):                   # 存成 FP16 再读回：枢轴行就是这样存的
    return x.clamp(-FP16_MAX, FP16_MAX).half().double()


def fp16_scale(x):             # 正的缩放因子存 FP16：先钳到 FP16 能表示的正数范围 [2^-24, 65504]
    return x.clamp(2.0**-24, FP16_MAX).half().double()


# ---- 合成输入流 ----
class Stream:
    """模仿真实模型的两个特点：
    1. 各单元的门差别很大，寿命从约 2 个 token 到上千个 token（时间维度要用到）；
    2. q/k/v 各通道的尺度和偏置不均匀，状态里会长出离群的行和列（空间维度要用到）。
    通道统计（尺度、偏置、门偏置）在构造时固定，相当于"模型参数"；
    reseed 之后 token() 重新采样，相当于"换一段文本"：标定和测试用的是不同的 token。"""

    def __init__(self, arch, heads, seed=0):
        gen = torch.Generator().manual_seed(seed)
        rn = lambda *shape: torch.randn(*shape, generator=gen)
        self.arch, self.H = arch, heads
        self.ks, self.kb = (rn(heads, K) * 0.8).exp(), rn(heads, K) * 2   # 键/查询通道：尺度不均 → 离群行
        self.vs, self.vb = (rn(heads, V) * 0.8).exp(), rn(heads, V)       # 值通道：尺度不均 → 离群列
        if arch == 'gdn':                       # 门偏置越大，门越接近 1，寿命越长
            self.gb = torch.linspace(1.0, 8.0, heads)[:, None]           # GDN：每头一个，[H,1]
        else:
            self.gb = torch.rand(heads, K, generator=gen) * 7 + 1         # KDA：每个键通道一个，[H,K]
        self.reseed(seed + 1)

    def reseed(self, seed):
        self.gen = torch.Generator().manual_seed(seed)

    def token(self):
        rn = lambda *shape: torch.randn(*shape, generator=self.gen)
        H = self.H
        q = rn(H, K) * self.ks + self.kb
        k = rn(H, K) * self.ks + self.kb
        v = rn(H, V) * self.vs + self.vb
        q = q / q.norm(dim=-1, keepdim=True) * K**-0.5   # 和模型里一样：q、k 做 L2 归一化，q 再乘 1/√K
        k = k / k.norm(dim=-1, keepdim=True)             # ‖k‖ = 1，满足命题 1 的条件
        g = -F.softplus(rn(*self.gb.shape) * 0.5 - self.gb)   # 对数门 g ≤ 0，GDN [H,1]，KDA [H,K]
        decay = g.exp().expand(H, K).clone()             # 统一成 [H,K]：GDN 一个头的每行乘同一个数
        beta = torch.sigmoid(rn(H))                      # 写入强度 β ∈ (0,1)
        return q, k, v, decay, beta


# ---- 递推与两个"权重" ----
def delta_step(S, q, k, v, decay, beta):
    """式 (1)(4)：X = D S + β k (vᵀ - kᵀ D S)，y = Xᵀ q。
    S 可以是全精度参考状态，也可以是上一步的重建状态 Ŝ（量化解码就是在 Ŝ 上更新）。
    形状：S [H,K,V]，q/k/decay [H,K]，v [H,V]，beta [H]。"""
    R = decay[..., None] * S                                         # D S，[H,K,V]
    delta = beta[:, None] * (v - torch.einsum('hk,hkv->hv', k, R))   # δ = β (v - kᵀ D S)，[H,V]
    X = R + k[..., None] * delta[:, None, :]                         # 沿键方向 k 写回：D S + k δᵀ
    y = torch.einsum('hk,hkv->hv', q, X)                             # 读出 y = Xᵀ q，在量化之前读
    return X, y, delta


def row_impact(q, k, decay, beta):
    """式 (9)(10)/(20)：g = Aᵀ q = D (q - β k (kᵀq))，返回 g²，[H,K]。
    A = (I - β k kᵀ) D，D 是对角阵，所以 Aᵀ = D (I - β k kᵀ)。
    含义：上一步状态第 i 行的误差 E_i，以权重 g_i 进入这一步的读出，Δy = Σ_i g_i E_i。"""
    g = decay * (q - beta[:, None] * k * (k * q).sum(-1, keepdim=True))
    return g.square()


def impact_factors(omega):
    """脚注 1：w_i = ω_i^(γ/2) / GM_k(ω_k^(γ/2))，γ = 0.25，即 w ∝ ω^(1/8)，头内几何平均为 1。
    故意压得很狠：ω 在头内能差几十倍，直接用会让低影响行在列拟合里几乎没有发言权。"""
    lw = omega.clamp_min(1e-300).log() * (GAMMA / 2)       # log ω^(γ/2)
    return (lw - lw.mean(-1, keepdim=True)).exp()          # 减头内均值 = 除以几何平均


def lifetime_weight(ell, horizon=HORIZON):
    """式 (6)(7)：L = Σ_{j=0}^{H-1} e^{2jℓ}，ℓ = E[log 门] ≤ 0。
    只看门衰减时，一个误差 j 步后还剩 e^{jℓ}，平方误差剩 e^{2jℓ}；把视野内的读出都加起来。
    等比数列求和 = (1 - e^{2Hℓ}) / (1 - e^{2ℓ})；ℓ = 0（永不遗忘）时等于 H。"""
    x = 2 * ell
    safe = torch.where(x == 0, torch.full_like(x, -1.0), x)
    L = torch.expm1(horizon * safe) / torch.expm1(safe)
    return torch.where(x == 0, torch.full_like(x, float(horizon)), L)


# ---- 量化器 ----
def dual_axis_fit(X, bits, w, impact=True, lsq=True):
    """键行感知的双轴拟合（4.3 节、附录 C.1/C.3）：每个 token 写回时调用一次。

    存储格式：X̂_ij = r_i · c_j · n_i · code_ij
      r_i：键行缩放，FP16，每行一个          c_j：值列缩放，FP16，每列一个，整个头共享
      code_ij：b_i bit 有符号整数，|code| ≤ 2^(b_i-1) - 1（论文附录 C.1 的 q_ij）
      n_i = 2^(8 - b_i)：把各种位宽的码字都放到 int8 的刻度上
      bits = 16 的行是枢轴：不量化，直接存 FP16。
    n_i 的作用：KDA 同一个头里各行位宽不同，但只共享一组列缩放 c。乘 n_i 后，所有码字都在
    int8 刻度上，低位宽的行只是格点更稀。GDN 一个头只有一种位宽，n_i 是常数，会被 c 吸收，
    结果不变。

    impact=False, lsq=False 就是 Q-Mamba 的 DSQ，也就是 LeapQuant 里"平滑 + 按列 int8"那一步。
    X：[..., H, K, V]，可以多一个前导维度，标定时一次处理多个快照；bits、w：[H, K]。
    返回 code、r、c、重建值 X̂。"""
    integer = bits < 16                                      # 非枢轴行，[H,K]
    b = bits.clamp(max=8).to(X.dtype)
    qmax = 2.0 ** (b - 1) - 1                                # 每行的整数上限：8→127，6→31，4→7，2→1
    n = 2.0 ** (8 - b)                                       # 码字归一化因子
    wi = w if impact else torch.ones_like(w)
    # 行因子，式 (12)(13)：r = sqrt(m / w)，m 是行均值绝对值。
    #   开平方：大行和小行只拉近一半，剩下的交给列缩放（同 LeapQuant 的平滑、SmoothQuant 的 α=0.5）；
    #   除以 w：影响大的行 r 更小 → 步长更细。
    m = X.abs().mean(-1)                                     # [..., H, K]
    r = fp16_scale((m.clamp_min(1e-20) / wi).sqrt())         # 先存成 FP16 再参与拟合：标定和推理用同一个数
    rn = r * n                                               # 每行的有效行缩放
    Xn = X / rn[..., None]                                   # 行归一化之后，只剩"码字 × 列缩放"要拟合
    lim = qmax[..., None]                                    # [H,K,1]，广播到每个元素
    keep = integer[..., None]                                # 枢轴行不参与列缩放的拟合
    # 列缩放初值：逐列 absmax，让这一列最大的数恰好落在最外层格点上（不截断）
    c = fp16_scale(torch.where(keep, Xn.abs() / lim, 0.0).amax(-2))          # [..., H, V]
    code = (Xn / c[..., None, :]).round().clamp(-lim, lim)                  # 第一次取整
    if lsq:
        # 式 (14)：min_c Σ_i w_i² (X_ij - r_i n_i c_j code_ij)²。码字固定时这是 c_j 的一元二次函数，有闭式解。
        # 换到归一化坐标：X_ij - r_i n_i c_j code_ij = r_i n_i (Xn_ij - c_j code_ij)，所以权重是 w_i² (r_i n_i)²。
        W = torch.where(integer, wi.square() * rn.square(), 0.0)[..., None]
        num = (W * code * Xn).sum(-2)                        # Σ_i W_i code_ij Xn_ij
        den = (W * code * code).sum(-2)                      # Σ_i W_i code_ij²
        c = fp16_scale(torch.where(den > 0, num / den.clamp_min(TINY), c))  # 整列为 0 时保留初值
        code = (Xn / c[..., None, :]).round().clamp(-lim, lim)              # 按新缩放重新取整：一轮 Lloyd 迭代
    recon = rn[..., None] * c[..., None, :] * code
    recon = torch.where(keep, recon, fp16(X))                # 枢轴行：FP16 原值
    return code, r, c, recon


def rowwise_absmax(X, bits):
    """论文的均匀基线：对称、逐行 absmax，每 128 个值一个 FP16 缩放（官方 sym_int*_g128）。
    也用作"只用时间组件"的量化器：位宽按分配表来，每行仍是 absmax。bits = 16 的行存 FP16。"""
    keep = (bits < 16)[..., None]
    lim = (2.0 ** (bits.clamp(max=8).to(X.dtype) - 1) - 1)[..., None]
    s = fp16_scale(X.abs().amax(-1, keepdim=True) / lim)    # 每行一个缩放，由这一行最大的数决定
    recon = (X / s).round().clamp(-lim, lim) * s
    return torch.where(keep, recon, fp16(X))


CODECS = {                                                   # 统一接口：codec(X, bits, w) → X̂
    'rowwise': lambda X, bits, w: rowwise_absmax(X, bits),
    'dsq': lambda X, bits, w: dual_axis_fit(X, bits, w, impact=False, lsq=False)[3],
    'stepquant': lambda X, bits, w: dual_axis_fit(X, bits, w)[3],
}


def bits_per_value(bits, codec):
    """返回（名义 bit，实际 bit/值）。
    名义：只算码字，枢轴按 8 bit 计（论文的预算口径）。
    实际：码字（枢轴 16 bit）+ 缩放。逐行 absmax 每行一个 FP16 = 16/V；
    双轴每头一行一列 FP16 = 16(K+V)/(KV) = 0.25。和论文附录 D 一样，给枢轴也留了缩放槽。"""
    b = bits.to(torch.float64)
    scale = 16 / V if codec == 'rowwise' else 16 * (K + V) / (K * V)
    return b.clamp(max=8).mean().item(), b.mean().item() + scale


# ---- 离线标定与比特分配 ----
def collect_stats(stream, steps, every=32):
    """附录 A.3：沿全精度轨迹跑一遍（不带量化反馈），收集三样东西：
      ω：行影响 E[g²]，[H,K]
      ℓ：平均对数门 E[log 门]，[H,K]（GDN 同头各行相同）
      快照：每 every 步存一次参考状态，用来试量化、算失真表（官方每 8 步采一次，最多留 64 个）。"""
    H = stream.H
    S = torch.zeros(H, K, V)
    omega, ell, snaps = torch.zeros(H, K), torch.zeros(H, K), []
    for t in range(steps):
        q, k, v, decay, beta = stream.token()
        omega += row_impact(q, k, decay, beta)
        ell += decay.log()
        S = delta_step(S, q, k, v, decay, beta)[0]
        if (t + 1) % every == 0 and t >= steps // 4:         # 跳过开头状态还没长起来的一段
            snaps.append(S.clone())
    return omega / steps, ell / steps, torch.stack(snaps)


def distortion_table(snaps, w, cands, arch, codec, pivots=None):
    """d_u(b)：把所有非枢轴单元都量化到 b bit 时的重建失真（附录 A.3）。
      gdn：单元 = 头，按 w² 加权后在头内元素上取平均（式 17）→ [H, nb]
      kda：单元 = 键行，不加权的逐元素 MSE（式 18）            → [H, K, nb]
    注意失真是用真正的量化器（双轴拟合）算的，所以时间组件的分配已经"知道"空间组件的效果。"""
    out = []
    for b in cands:
        bits = torch.full(w.shape, b, dtype=torch.long)
        if pivots is not None:
            bits[pivots] = 16
        e = (CODECS[codec](snaps, bits, w) - snaps).square()    # [N,H,K,V]
        if arch == 'gdn':
            out.append((w[..., None].square() * e).mean((0, 2, 3)))
        else:
            out.append(e.mean((0, 3)))
    return torch.stack(out, -1)


def allocate_dp(cost, cands, budget):
    """式 (8) 的精确解：多选背包 DP（Qwen 用，2304 个单元，表的大小约 2304 × 13824）。
    cost [n, nb]：单元 u 选第 j 个候选位宽的代价。所有单元的元素数相同，
    所以约束 Σ n_u b_u ≤ b̄ Σ n_u 可以写成 Σ b_u ≤ budget。
    best[s] = 前 u 个单元恰好用掉 s 个"位"时的最小代价；每加一个单元松弛一次，记下选择用于回溯。"""
    n = cost.shape[0]
    best = torch.full((budget + 1,), math.inf)
    best[0] = 0.0
    pick = torch.zeros(n, budget + 1, dtype=torch.long)
    for u in range(n):
        nxt = torch.full_like(best, math.inf)
        for j, b in enumerate(cands):
            trial = torch.full_like(best, math.inf)
            trial[b:] = best[:budget + 1 - b] + cost[u, j]   # 第 u 个单元选 b bit
            better = trial < nxt
            nxt[better], pick[u, better] = trial[better], j
        best = nxt
    s, out = int(best.argmin()), [0] * n                     # 代价最小的总位宽，不一定用满预算
    for u in reversed(range(n)):                             # 回溯
        out[u] = cands[int(pick[u, s])]
        s -= out[u]
    return torch.tensor(out)


def allocate_lagrangian(cost, cands, budget):
    """式 (8) 的近似解：拉格朗日分配（KDA 用，81,920 个单元，DP 表要 4×10^10 格，算不动）。
    1. 给每 bit 一个价格 λ，每个单元各自选 argmin_b cost(b) + λ·b；二分 λ，找到总位宽不超预算的最低价格；
    2. 二分的解一般用不满预算，再按"每多 1 bit 降多少代价"从大到小贪心升级，直到用完。"""
    b = torch.tensor(cands, dtype=cost.dtype)
    choose = lambda lam: (cost + lam * b).argmin(-1)
    sel = choose(0.0)
    if b[sel].sum() <= budget:
        return b[sel].long()
    lo, hi = 0.0, float((cost.max(-1).values - cost.min(-1).values).max()) + TINY
    sel = choose(hi)                                         # 价格够高时，每个单元都选最低位宽
    for _ in range(80):
        mid = (lo + hi) / 2
        trial = choose(mid)
        if b[trial].sum() > budget:
            lo = mid
        else:
            hi, sel = mid, trial
    left = budget - int(b[sel].sum())
    while left > 0:                                          # 贪心修补，用完剩下的预算
        extra = b[None, :] - b[sel][:, None]                 # 升到每个候选要多花几 bit
        gain = (cost.gather(1, sel[:, None]) - cost) / extra.clamp_min(1)
        gain[(extra <= 0) | (extra > left)] = -math.inf
        u, j = divmod(int(gain.argmax()), len(cands))
        if gain[u, j] <= 0:
            break
        left -= int(extra[u, j])
        sel[u] = j
    return b[sel].long()


def build_plan(stats, arch, nominal, n_pivot, codec='stepquant'):
    """附录 C.2：标定 → 选枢轴 → 分配，返回 bits [H,K]（16 = 枢轴）和行影响因子 w [H,K]。
    预算口径：枢轴在名义预算里按 8 bit 计，整数预算 = b̄·n - 8·枢轴数（缩放另算）。"""
    omega, ell, snaps = stats
    H = omega.shape[0]
    w = impact_factors(omega)
    cands = [2, 4, 6, 8] if nominal == 4 else [4, 6, 8]
    if arch == 'gdn':                                        # 单元 = 头
        L = lifetime_weight(ell[:, 0])                       # GDN 同头各行的门一样，取一行即可
        cost = L[:, None] * distortion_table(snaps, w, cands, arch, codec)            # [H, nb]
        # 先不留枢轴做一次 DP，再按"分到位宽之后还剩多少寿命加权失真"排序，最大的改成枢轴
        first = allocate_dp(cost, cands, nominal * H)
        risk = cost[torch.arange(H), torch.tensor([cands.index(int(x)) for x in first])]
        piv = torch.zeros(H, dtype=torch.bool)
        piv[risk.argsort(descending=True)[:n_pivot]] = True
        unit = torch.full((H,), 16, dtype=torch.long)
        unit[~piv] = allocate_dp(cost[~piv], cands, nominal * H - 8 * n_pivot)       # 剩下的预算重新分
        bits = unit[:, None].expand(H, K).clone()
    else:                                                    # 单元 = 键行
        n, nb = H * K, len(cands)
        L = lifetime_weight(ell)                             # [H,K]
        cost = (L[..., None] * distortion_table(snaps, w, cands, arch, codec)).reshape(n, nb)
        fp_err = (fp16(snaps) - snaps).square().mean((0, 3))                          # FP16 存储的误差
        risk = cost[:, -1] - (L * fp_err).reshape(n)         # 从 INT8 升到 FP16 能少多少寿命加权失真
        piv = torch.zeros(n, dtype=torch.bool)
        piv[risk.argsort(descending=True)[:n_pivot]] = True
        # 枢轴行不参与共享列缩放的拟合，会改变其余行的失真，所以排除枢轴后重算一次
        cost = (L[..., None] * distortion_table(snaps, w, cands, arch, codec,
                                                pivots=piv.reshape(H, K))).reshape(n, nb)
        flat = torch.full((n,), 16, dtype=torch.long)
        flat[~piv] = allocate_lagrangian(cost[~piv], cands, nominal * n - 8 * n_pivot)
        bits = flat.reshape(H, K)
    return bits, w


# ---- 在线解码 ----
def run(tokens, S0, codec, bits, w):
    """按式 (4) 解码：每步在上一步的重建状态 Ŝ 上更新、读出，再量化写回。
    返回每步输出的相对误差，以及每行累积的状态平方误差 [H,K]。"""
    fn = CODECS[codec]
    S_ref, S_hat = S0.clone(), fn(S0, bits, w)               # prefill 结束：全精度状态量化一次
    out_err, state_err = [], torch.zeros(S0.shape[:2])
    for q, k, v, decay, beta in tokens:
        S_ref, y_ref, _ = delta_step(S_ref, q, k, v, decay, beta)
        X, y_hat, _ = delta_step(S_hat, q, k, v, decay, beta)   # 读出在量化之前
        S_hat = fn(X, bits, w)                                  # 写回：每个 token 都重新量化
        out_err.append(((y_hat - y_ref).norm() / y_ref.norm()).item())
        state_err += (S_hat - S_ref).square().sum(-1)
    return out_err, state_err


# ---- kernel 层面的两个恒等式（附录 F.1）----
def fused_readout(code, r, c, pivot_val, bits, q, k, v, decay, beta):
    """融合 kernel 的数学（官方 kernels/readout.py）：不拼出 Ŝ，直接从码字算 δ 和 y。
    整数行 Ŝ_ij = a_i c_j code_ij（a_i = r_i n_i），枢轴行 Ŝ_ij = P_ij。对任意向量 x：
      xᵀ D Ŝ = c ⊙ (codeᵀ (x ⊙ d ⊙ a)) + (x ⊙ d)_枢轴行ᵀ P
    x 取 k 得到 δ = β (v - kᵀ D Ŝ)；y = (D Ŝ + k δᵀ)ᵀ q = qᵀ D Ŝ + (qᵀk) δ。
    每个元素只读一次码字，乘一次缩放；kernel 里这一步和 Delta 更新、读出融在一起。"""
    integer = bits < 16
    a = torch.where(integer, r * 2.0 ** (8 - bits.clamp(max=8)), 0.0)    # 枢轴行置 0

    def read(x):                                             # xᵀ D Ŝ，[H,V]
        xd = x * decay
        return (c * torch.einsum('hk,hkv->hv', xd * a, code)
                + torch.einsum('hk,hkv->hv', xd * ~integer, pivot_val))

    delta = beta[:, None] * (v - read(k))
    y = read(q) + (q * k).sum(-1, keepdim=True) * delta
    return y, delta


def rebuild_from_codes(code, r, c, pivot_val, bits, k, decay, delta):
    """写回 kernel 不读稠密的 X（官方 kernels/rows.py、fit.py）：
    X = D Ŝ + k δᵀ，整数行 X_ij = (d_i a_i) c_j code_ij + k_i δ_j。
    所以读出时只要多存 δ（每头 d_v 个数）和每行 |X| 的部分和（求新的行因子要用），
    写回时就能从旧码字逐 tile 重算 X，拟合新缩放、写新码字，全程没有 FP32 的整块中间状态。"""
    integer = (bits < 16)[..., None]
    a = r * 2.0 ** (8 - bits.clamp(max=8))
    DS = torch.where(integer, (decay * a)[..., None] * c[:, None, :] * code,
                     decay[..., None] * pivot_val)
    return DS + k[..., None] * delta[:, None, :]


# ---- 演示 ----
def spearman(a, b):
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


def quarters(e):
    n = len(e) // 4
    return "  ".join(f"{sum(e[i * n:(i + 1) * n]) / n:.2e}" for i in range(4))


def ljust(s, width):           # 按显示宽度补空格：中文字符占两格
    return s + ' ' * max(0, width - sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in s))


def demo(arch, heads, n_pivot, prefill=512, calib=1024, decode=1024):
    print(f"\n==================== {arch.upper()}：{heads} 个头，每头 {K}×{V} ====================")
    t0 = time.time()
    st = Stream(arch, heads)
    st.reseed(100)                                           # 标定用一段 token（相当于论文的 WikiText-2）
    stats = collect_stats(st, calib)
    omega, ell, _ = stats
    st.reseed(200)                                           # 测试用另一段 token
    S0 = torch.zeros(heads, K, V)
    for _ in range(prefill):                                 # prefill：全精度递推，结束时才量化一次
        S0 = delta_step(S0, *st.token())[0]
    tokens = [st.token() for _ in range(decode)]
    full = lambda b: torch.full((heads, K), b, dtype=torch.long)
    results = {}

    # 观察 1：寿命越长的单元，累积误差越大（论文图 1b、附录 B.5）
    results['int6'] = run(tokens, S0, 'rowwise', full(6), None)
    tau = math.log(2) / (-ell)                               # 门半衰期（token）
    unit_tau, unit_err = (tau[:, 0], results['int6'][1].sum(-1)) if arch == 'gdn' else \
        (tau.flatten(), results['int6'][1].flatten())
    top = unit_tau.argsort(descending=True)[:len(unit_tau) // 4]
    unit = '头' if arch == 'gdn' else '键行'
    print(f"观察 1  均匀 INT6 下，{len(unit_tau)} 个{unit}的门半衰期 {unit_tau.min():.1f}–{unit_tau.max():.0f} token，"
          f"和累积状态误差的 Spearman 相关 = {spearman(unit_tau, unit_err):.2f}；"
          f"寿命最长的 1/4 贡献了 {unit_err[top].sum() / unit_err.sum():.0%} 的误差")

    # 观察 2：一行上的单位误差，下一步读出的平方误差恰好是 g_i²（附录 A.2、B.3）
    q, k, v, decay, beta = tokens[0]
    g2 = row_impact(q, k, decay, beta)
    y0 = delta_step(S0, q, k, v, decay, beta)[1]
    worst = 0.0
    for i in range(0, K, 8):
        E = torch.zeros_like(S0)
        u = torch.randn(heads, V)
        E[:, i] = u / u.norm(dim=-1, keepdim=True)           # Frobenius 范数为 1，只落在第 i 行
        dy2 = (delta_step(S0 + E, q, k, v, decay, beta)[1] - y0).square().sum(-1)
        worst = max(worst, ((dy2 - g2[:, i]).abs() / g2[:, i]).max().item())
    spread = (omega.quantile(0.9, dim=-1) / omega.quantile(0.1, dim=-1)).median().item()
    print(f"观察 2  单行单位误差的下一步读出误差 vs g_i²：最大相对差 {worst:.1e}；"
          f"头内 ω 的 P90/P10（各头中位数）= {spread:.1f}")

    # 标定出精度表
    plan6 = build_plan(stats, arch, 6, n_pivot)
    plan6_nopiv = build_plan(stats, arch, 6, 0)
    plan4 = build_plan(stats, arch, 4, n_pivot)
    plan6_time = build_plan(stats, arch, 6, n_pivot, codec='rowwise')
    if arch == 'gdn':
        print("分配结果（STEPQuant@6 / @4，16 = FP16 枢轴）：")
        print("  头  半衰期    寿命权重 L   @6   @4")
        for h in tau[:, 0].argsort().tolist():
            print(f"  {h:2d}  {tau[h, 0]:7.1f}  {lifetime_weight(ell[h, 0]).item():9.1f}"
                  f"   {plan6[0][h, 0].item():2d}   {plan4[0][h, 0].item():2d}")
    else:
        for name, (bits, _) in (('@6', plan6), ('@4', plan4)):
            cnt = {b: int((bits == b).sum()) for b in (2, 4, 6, 8, 16)}
            order = tau.flatten().argsort()
            mean_tau = {b: tau.flatten()[bits.flatten() == b].mean().item() for b in cnt if cnt[b]}
            print(f"分配结果 STEPQuant{name}：各位宽的行数 {cnt}；平均半衰期 "
                  + "，".join(f"{b} bit {t:.0f}" for b, t in mean_tau.items()))

    # 观察 3：融合读出和"从码字重算 X"与先拼出稠密 Ŝ 的结果一致
    bits, w = plan6
    code, r, c, S_hat = dual_axis_fit(S0, bits, w)
    pivot_val = torch.where((bits < 16)[..., None], 0.0, S_hat)
    X, y, delta = delta_step(S_hat, q, k, v, decay, beta)
    y_f, delta_f = fused_readout(code, r, c, pivot_val, bits, q, k, v, decay, beta)
    X_f = rebuild_from_codes(code, r, c, pivot_val, bits, k, decay, delta_f)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    print(f"观察 3  不拼出 Ŝ 的读出 vs 稠密：y 相对差 {rel(y_f, y):.1e}，δ {rel(delta_f, delta):.1e}；"
          f"从码字重算的 X {rel(X_f, X):.1e}")

    # 主表：各配置在同一段 token 上逐 token 量化，输出相对精确 fp64 的误差
    configs = [
        ('均匀 INT8（逐行 absmax）', 'rowwise', full(8), None),
        ('均匀 INT6（逐行 absmax）', 'rowwise', full(6), None),
        ('DSQ@6（双轴，无行影响、无最小二乘）', 'dsq', full(6), plan6[1]),
        ('只用空间@6（双轴 + 行影响 + 最小二乘）', 'stepquant', full(6), plan6[1]),
        ('只用时间@6（寿命分配 + 枢轴，逐行 absmax）', 'rowwise', *plan6_time),
        ('STEPQuant@6（不留枢轴）', 'stepquant', *plan6_nopiv),
        (f'STEPQuant@6（{n_pivot} 个枢轴{unit}）', 'stepquant', *plan6),
        ('均匀 INT4（逐行 absmax）', 'rowwise', full(4), None),
        (f'STEPQuant@4（{n_pivot} 个枢轴{unit}）', 'stepquant', *plan4),
    ]
    print(f"输出相对误差（{decode} 个 token 分四段取平均）：")
    print("  " + ljust('配置', 46) + "名义  实际 bit   " + "".join(ljust(f'第 {i} 段', 10) for i in range(1, 5)))
    for name, codec, bits, w in configs:
        key = 'int6' if name.startswith('均匀 INT6') else name
        if key not in results:
            results[key] = run(tokens, S0, codec, bits, w)
        nominal, actual = bits_per_value(bits, codec)
        print(f"  {ljust(name, 46)}{nominal:4.2f}{actual:10.3f}   {quarters(results[key][0])}")
    print(f"（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo('gdn', heads=16, n_pivot=1)      # 1/16 的头做枢轴，比论文的 32/2304 = 1.39% 粗得多，实际 bit 也就多一些
    demo('kda', heads=4, n_pivot=4)       # 512 个键行里 4 个做枢轴（0.78%，论文 512/81920 = 0.625%）

'''
=============================================================================
原理说明，STEPQuant 怎么降低量化误差
=============================================================================

和 LeapQuant 一样，问题出在"每个 token 都要重新量化一次状态"。LeapQuant 的思路是少量化几次
（窗口），STEPQuant 不改量化次数，而是回答两个问题：误差能活多久（时间），误差落在哪（空间），
然后把比特和缩放花在误差危害最大的地方。

一、误差怎么传（delta_step、row_impact）
  量化解码每步是 X_t = A_t Ŝ_{t-1} + B_t，Ŝ_t = 量化(X_t)。和全精度相减：
      E_t = A_t E_{t-1} + ε_t，        A_t = (I - β k kᵀ) D_t
      这一步的读出误差 Δy_t = Σ_i g_i E_{t-1,i,:}，   g = A_tᵀ q_t
  1. ‖A_t‖ ≤ 1，误差不会被放大；但每步都在注入新的 ε_t。旧误差靠两样东西消掉：
     门 D_t 整体衰减；Delta 项削掉误差沿当前键 k_t 的分量。和后续键都不太对齐的方向，
     只能靠门慢慢衰减，门接近 1 时能活成百上千步。 → 时间维度
  2. 第 i 行的误差以 g_i 为权重进入读出，同样大的误差，落在 g_i 大的行上危害更大。 → 空间维度
  所以一个误差的总危害 ≈ 寿命权重 L × 行影响 w² × 本次舍入误差²。后两个由模型决定，
  STEPQuant 能做的是：在前两个大的地方，把第三个压小。

二、时间：按寿命分比特（lifetime_weight、build_plan）
  只看门，误差 j 步后剩 e^{jℓ}，平方剩 e^{2jℓ}，ℓ 是平均对数门。把未来 H = 2048 步加起来就是
  寿命权重 L。例子，两个头：
  - 头 A 门 0.5（半衰期 1 个 token），L ≈ 1.3；头 B 门 0.999（半衰期约 700 个 token），L ≈ 490。
  - 每多 1 bit，步长减半，误差平方降为 1/4。两个都用 6 bit，总代价 ≈ (1.3 + 490)·d ≈ 493d。
  - 改成 A 用 4 bit、B 用 8 bit，平均还是 6 bit，总代价 ≈ 1.3×16d + 490/16·d ≈ 52d，只剩约 1/9。
  具体做法：
  1. 标定（collect_stats）：在别的文本上跑全精度轨迹，记下每个单元的 ℓ、每行的 ω、一批状态快照。
  2. 失真表（distortion_table）：把快照按每个候选位宽试量化一遍，得到 d_u(b)；代价 = L_u·d_u(b)。
  3. 枢轴：寿命最长、最难量化的少数单元直接存 FP16（论文 Qwen 32 个头、KDA 512 行）。
     Qwen 是先分一次，挑"分完还剩最多寿命加权失真"的头；KDA 挑"升到 FP16 能减少最多失真"的行。
  4. 分配：Σ L_u d_u(b_u) 最小，总位宽不超预算，就是多选背包。Qwen 2304 个单元用精确 DP；
     KDA 81,920 个单元 DP 表太大，用拉格朗日：给每 bit 定个价，各单元各自选最划算的位宽，
     二分这个价让总位宽刚好不超，再把剩下的预算贪心地补给"每 bit 收益最大"的单元。
  分配单元跟着门的粒度走：GDN 一个头一个门，整头寿命相同，按头分；KDA 每个键通道一个门，按行分。
  这些都在离线做完，推理时位宽表固定，没有任何开销。

三、空间：键行感知的双轴拟合（dual_axis_fit）
  1. 为什么要双轴。逐行 absmax 时一行的步长由这一行最大的数决定。如果某一列是离群列，
     它在每一行都是最大的那个，所有行的步长都被它撑大。例子，某行 = [20, 1.0, 0.6, -0.8]，
     第 1 列是离群列，6 bit（-31..31）：
     - 逐行 absmax，步长 20/31 ≈ 0.65，1.0、0.6、-0.8 只能取 0.65 的倍数，舍入误差最多半个步长 0.32，
       相对这几个数就是 30%–50%。
     - 先给第 1 列一个列缩放 20，这一行变成 [1, 1.0, 0.6, -0.8]，步长 1/31 ≈ 0.032，误差最多 0.016（2% 左右）。
     真实状态里离群值同时沿行和列出现（论文 Qwen：行 10×、列 19×），所以行、列各一组缩放。
  2. 行因子 r = sqrt(m / w)。开平方和你 leapquant_ref.py 的平滑完全一样（为什么开平方见那边的说明）；
     多除了一个 w：影响大的行 r 更小，乘回去时步长更细。w = ω^(1/8) 压得很温和：头内 ω 的
     P90/P10 约 16 倍，步长只差 16^(1/16) ≈ 1.19 倍。
  3. 列因子：先用 absmax 定初值、取整一次；再固定码字，求让 Σ w² (X - r c code)² 最小的 c（闭式解），
     按新的 c 重新取整。这就是一轮 Lloyd 迭代：absmax 只保证最大的数不截断，最小二乘把格点挪到
     让整列加权误差最小的位置。例子，一列 = [7, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4]，4 bit（-7..7）：
     - absmax 步长 1，2.4 全舍成 2，误差平方和 6×0.16 = 0.96；
     - 最小二乘步长 = (49 + 6×2×2.4) / (49 + 6×4) ≈ 1.066，2.4 → 2×1.066 = 2.13，7 → 7×1.066 = 7.46，
       误差平方和 ≈ 0.21 + 6×0.072 ≈ 0.64，少了三分之一。
     加 w² 的权重，是让影响大的行在挑格点时说了算。
  4. KDA 同一个头里各行位宽不同，码字乘 2^(8-b) 统一到 int8 刻度上，共用一组列缩放。
  对照主表：DSQ 就是 1 里的双轴，不带 2、3 的两处改动。

四、运行时怎么做到不拖慢（fused_readout、rebuild_from_codes）
  1. 融合读出：Ŝ = diag(a) code diag(c)，所以 kᵀ D Ŝ = c ⊙ (codeᵀ (k ⊙ d ⊙ a))，读出同理。
     一个 kernel 读一遍码字，算完 δ、更新和读出，y 立刻交给后面的层。和你的 readout() 一个思路：
     不拼出状态，只做"向量 × 低比特矩阵"。
  2. 不存 FP32 中间状态：X = D Ŝ + k δᵀ，D Ŝ 能由旧码字重算。所以读出时只多存 δ（d_v 个数）
     和每行 |X| 的部分和，写回 kernel 逐 tile 从旧码字重算 X、拟合、写新码字。
  3. 写回放在另一条 CUDA stream 上，和后面的层重叠，只要求在下一步读这一层之前写完。
  4. 位宽表离线固定，所以压缩布局、偏移量都是静态的。prefill 用全精度 chunk kernel，结束时量化一次。

五、名义 bit 和实际 bit（bits_per_value）
  名义 6 bit 只算码字，枢轴按 8 bit 计。实际还要加每头一行一列 FP16 缩放（0.25 bit）和枢轴
  从 8 升到 16 的增量：论文 Qwen@6 是 6 + 0.25 + 32/2304×8 = 6.36 bit/值，比均匀 INT6 的 6.125 还多。
  这里 16 个头里拿 1 个做枢轴，枢轴增量是 1/16×8 = 0.5 bit，比论文大得多，读主表时要看"实际 bit"一列。

六、和 LeapQuant 对照（leapquant_ref.py）
  - 主攻方向：LeapQuant 少舍入（16 步一次）；STEPQuant 把比特给误差活得久的单元。
  - 行缩放：你的 c = S.abs().mean(1).sqrt()  ↔  这里的 r = sqrt(m / w)，多了行影响 w。
  - 列缩放：你的 b = amax / 127              ↔  这里的 c，absmax 初值 + 一次加权最小二乘。
    注意名字正好反过来：你代码里 c 是行、b 是列；这里 r 是行、c 是列。
  - 难量化的部分：LeapQuant 拿出 4 个低秩方向存 FP16（补偿 token）；STEPQuant 拿出整头或整行存 FP16（枢轴）。
  - 位宽：LeapQuant 统一 int8；STEPQuant 2/4/6/8 混合。标定：LeapQuant 不要；STEPQuant 要 ω、ℓ 和快照。
  LeapQuant 去掉补偿 token 之后的 quantize()，就是 dual_axis_fit(impact=False, lsq=False)。

七、一次解码发生了什么
  离线：标定 → 失真表 → 枢轴 → 分配，得到每个头/每行的位宽和 w。
  每个 token、每一层：
  1. 融合 kernel：从码字（枢轴行从 FP16）读出 D Ŝ，算 δ，更新 X = D Ŝ + k δᵀ，读出 y = Xᵀ q，交给下一层。
  2. 另一条 stream：求行因子 r = sqrt(m / w)，absmax 定列缩放初值，加权最小二乘修正一次，
     按各行位宽取整、打包，覆盖旧码字。
  3. 下一个 token 读这一层之前，写回必须完成。

八、怎么读这次的输出（合成数据，只看趋势，数值不能和论文比）
  1. 观察 1：寿命和累积误差的秩相关接近 1，寿命最长的 1/4 单元占了大部分误差。这就是时间维度的出发点。
  2. 观察 2：恒等式成立，说明 ω = E[g²] 确实是"单行误差进入读出的放大倍数"。头内 ω 能差十几倍。
  3. 分配结果：短寿命的单元拿低位宽，长寿命的拿高位宽，最长寿命、最难量化的成了枢轴。
  4. 主表里两个组件的分工很清楚：
     - 只用时间：误差在四段之间基本不再增长（累积被止住了），但每步的误差水平还是逐行 absmax 的水平；
     - 只用空间：每步的误差小了（双轴 + 最小二乘比 DSQ 又好一点），但长寿命单元的误差照样随时间涨；
     - 两个都用：既不涨，水平也低。6 bit 左右就接近均匀 INT8，4 bit 时均匀 INT4 发散，STEPQuant 不发散。
  5. 看"实际 bit"一列：GDN 的枢轴是整头，16 个头里拿 1 个就多 0.5 bit；比特对齐的比较看"不留枢轴"那一行。
  6. 合成数据里枢轴几乎没用（留不留只差一点），因为最长寿命的头拿 8 bit 已经够了。论文的真实模型里
     枢轴作用很大：只用时间组件时去掉 32 个枢轴头，Qwen 6 bit 的 AIME 从 74.58 掉到 60.42。
     真实模型里少数头特别难量化，这个现象合成数据没有造出来。
'''
