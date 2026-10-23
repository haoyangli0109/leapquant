"""LeapQuant 最小参考实现,单个 v 头,fp64,四个操作对照 leapquant.py 的四个 kernel。"""
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
K = V = 128
P, R = 16, 4                                   # 窗口长度、补偿 token 个数
TINY, FP16_MAX = 1e-30, 65504.0


def fp16(x):
    return x.clamp(-FP16_MAX, FP16_MAX).half().double()


def unit(x):                                   # _lq_unit,先按最大值缩放,防止平方和下溢
    m = x.abs().max()
    return x / m / (x / m).norm() if m > TINY else torch.zeros_like(x)


# ---- 量化,S ≈ diag(c) Z diag(b) + Kcᵀ Uc  （_lq_quantize）----
def quantize(S, warm=None):
    S = S.clone()
    # 策略2，
    Kc, Uc = torch.zeros(R, K), torch.zeros(R, V) # [4,128], [4,128]
    # 找 kv：希望找到一个方向 k，使得 S 在该方向上的投影尽可能大
    # 固定 k 后，找最合适的 u，让秩 1 矩阵 k u 最接近 S。
    for r in range(R): # 4
        kv, iters = (S @ warm[r], 0) if warm is not None else (None, 8)
        if kv is None or kv.abs().max() <= TINY:
            # S.norm(dim=1) 算的是 S 每一行的 L2 范数，得到一个长度为 K 的向量，表示第i行有多大的强度
            kv, iters = S.norm(dim=1), 8       # 冷启动,行范数指向离群行

        kv = unit(kv)
        # 反复用当前的 S 去乘 kv，把 kv 转到能量最大的那个方向上
        for _ in range(iters):                 # 幂迭代 k ← unit((kS)Sᵀ)
            kv = unit(S @ (kv @ S))
        Kc[r], Uc[r] = fp16(kv), fp16(kv @ S)  # 给定方向,最优的 ũ 就是投影 k̃S
        S -= torch.outer(Kc[r], Uc[r])         # 用舍入后的值 deflate
    # 策略3
    c = S.abs().mean(1).sqrt().clamp_min(1e-6) # 平滑,每行平均幅值开方，这样就可以把小行和大行拉近，因为我们是按列量化，per col
    Rs = S / c[:, None]
    amax = Rs.abs().amax(0)
    b = torch.where(amax > TINY, amax / 127, 1.0)
    Z = (Rs / b).round().clamp(-127, 127)      # 按列对称 int8
    return Z, b, c, Kc, Uc # z是量化后的值，b是scale，c是平滑值，KC，KU是秩1矩阵的行列向量


def dequant(bnd):
    Z, b, c, Kc, Uc = bnd
    return c[:, None] * Z * b + Kc.T @ Uc


def weights(recs):                             # e^{C_h},以及每条记录的 e^{C_h - C_j}
    # recs = [(k0, u0, g0), (k1, u1, g1), (k2, u2, g2)]
    g = torch.tensor([float(r[2]) for r in recs]) # [g0, g1, g2]
    # g.sum().exp() = e^{g0 + g1 + g2}
    if len(recs) == 3:
        pass
    # g.sum() = g0 + g1 + g2
    # g.cumsum(0) = [g0, g0 + g1, g0 + g1 + g2]
    # g.sum() - g.cumsum(0) = [g1 + g2, g2, g3]
    return g.sum().exp(), (g.sum() - g.cumsum(0)).exp()


# ---- 读出,x S_h,不反量化、不拼出 S_h  （decode kernel 的前半）----
def readout(x, bnd, recs):
    # readout理解为一个s模拟器，返回计算x和s的乘积，只是我们并不算出来这个S
    Z, b, c, Kc, Uc = bnd  # z是量化后的值，b是scale，c是平滑值，KC，KU是秩1矩阵的行列向量
    d, w = weights(recs)
    out = torch.zeros(V)
    for wj, (kj, uj, _) in zip(w, recs):       # 记录,e^{C_h-C_j} (x·k̂_j) u_j
        out += wj * (x @ kj) * uj
    # x是q，所以x @ kj 是 q 和 kj 的点积，也就是 q 和 kj 的相似度
    # 以上是k和u的线性组合
    # 以下是S0部分
    out += d * b * ((c * x) @ Z)               # int8 残差,一次“向量 x int8 矩阵”
    out += d * (Kc @ x) @ Uc                   # 补偿 token,与记录同形、门为 0 的虚拟 token
    return out


# ---- decode 一个 token  （_leapquant_decode_kernel）----
def decode_step(q, k, v, g, beta, bnd, recs):
    q, k = q / q.norm() * K**-0.5, k / k.norm()
    xk, xq = g.exp() * k, g.exp() * q          # 本步的门推到向量上
    u = beta * (v - readout(xk, bnd, recs))    # u = β(v - k̂ e^g S_h)
    o = readout(xq, bnd, recs) + (q @ k) * u   # o = q̂ (e^g S_h + k̂ᵀu)
    recs.append((fp16(k), fp16(u), g))         # 追加记录 (k̂, u, g)
    return o


# ---- fold,窗口满了,在高精度里拼出 S_16 再量化一次  （_leapquant_fold_kernel）----
def rebuild(bnd, recs):                        # _lq_rebuild
    d, w = weights(recs)
    S = d * dequant(bnd)
    # (k̂, u, g) == recs
    for wj, (kj, uj, _) in zip(w, recs):
        S += wj * torch.outer(kj, uj)
    return S


def fold(bnd, recs):
    return quantize(rebuild(bnd, recs), warm=bnd[4])   # 热启动,用上个窗口的 Uc


# ---- 精确参考,gdn_verify.py 的逐 token 递推 ----
def step(S, q, k, v, g, beta):
    q, k = q / q.norm() * K**-0.5, k / k.norm()
    S = g.exp() * S
    S = S + torch.outer(k, beta * (v - k @ S))
    return S, q @ S


if __name__ == "__main__":
    torch.manual_seed(0)
    ks, kb = (torch.randn(K) * 0.8).exp(), torch.randn(K) * 2   # 通道尺度不均,造出离群行列
    vs, vb = (torch.randn(V) * 0.8).exp(), torch.randn(V)

    def token():
        return (torch.randn(K) * ks + kb, torch.randn(K) * ks + kb,
                torch.randn(V) * vs + vb,
                -F.softplus(torch.randn(()) * 0.5 - 3), torch.sigmoid(torch.randn(())))
        # q, k, v, g, beta

    S = torch.zeros(K, V)
    # 模拟prefill
    for _ in range(512):
        S, _ = step(S, *token())

    bnd, recs = quantize(S), []                # prefill 结束,冷启动量化,窗口为空
    S_tok = dequant(quantize(S))               # 对照,每个 token 都重新量化
    err_win, err_tok, worst = [], [], 0.0
    for t in range(256):
        # q, k, v, g, beta
        tok = token()
        if len(recs) == P:                     # 16 窗口满,先 fold,再 decode
            bnd, recs = fold(bnd, recs), []
        x = torch.randn(K)
        dense = x @ rebuild(bnd, recs)
        # x不是q,不把状态矩阵拼出来，readout 算出来的结果，和先拼出完整的 Sh 再做矩阵乘法，是不是同一个数。
        worst = max(worst, ((readout(x, bnd, recs) - dense).norm() / dense.norm()).item())
        # o 是 LeapQuant。状态留在 bnd 和 recs 里，窗口内不量化；满 16 步才在循环前面 fold 一次。
        o = decode_step(*tok, bnd, recs)
        # o_ref 是精确对照。S 从 prefill 起一直用 fp64 递推，从不量化
        S, o_ref = step(S, *tok)
        # o_tok 是「每个 token 都量化」的对照。它也做同一套递推，但每步写完立刻 quantize 再 dequant，舍入误差马上进状态
        S_tok, o_tok = step(S_tok, *tok)
        S_tok = dequant(quantize(S_tok))

        err_win.append(((o - o_ref).norm() / o_ref.norm()).item())
        err_tok.append(((o_tok - o_ref).norm() / o_ref.norm()).item())

    print(f"读出公式 vs 先拼出稠密 S_h 再乘,最大相对误差 {worst:.1e}")
    print("输出相对精确 fp64 的误差,256 个 token 分四段取平均,")
    for name, e in (("每窗口量化一次（LeapQuant）", err_win), ("每 token 量化一次", err_tok)):
        print(f"  {name}  " + "  ".join(f"{sum(e[i:i + 64]) / 64:.2e}" for i in range(0, 256, 64)))

'''
=============================================================================
原理说明,LeapQuant 怎么降低量化误差
=============================================================================

量化误差 ≈ 舍入的次数 x 每次舍入的大小。LeapQuant 用“逐窗口”减少舍入次数,
用“补偿 token”和“平滑”减小每次舍入的大小。

一、int8 的误差从哪来
  int8 每一列只有 -127..127 这 255 个格点。步长 = 这一列最大的数 / 127,
  每个数舍入后误差最多半个步长。所以一列的精度由这一列最大的数决定。
  极端例子,某列是 [127, 1.3, 0.4, -0.8],步长 = 1,舍入后变成 [127, 1, 0, -1],
  1.3 错了 23%,0.4 直接变成 0,-0.8 错了 25%。没有那个 127 的话,步长约 0.01,
  所有数的误差都不到 0.005,小了约 100 倍。下面三招都围绕这个例子。

二、第一招,逐窗口量化,少舍入几次
  （decode_step 从不改 bnd,只往 recs 里追加；窗口满了才 fold）
  比方,账本只能记到千位,余额 100 万,每天进账 300。
  - 每天记一次,100 万 + 300 四舍五入后还是 100 万,300 丢了；16 天丢 4800,账面不动。
  - 另开小本子精确记流水,16 天后一次入账,100 万 + 4800 = 1,004,800,记成 1,005,000,
    只差 200,而且只差这一次。
  GDN 的状态就是“大余额”,每个 token 的更新就是“300”,单个更新相对整个状态很小,
  每步都舍入就会被系统性地吃掉。LeapQuant 让 int8 状态在窗口内不动,新 token 只记进
  fp16 记录（小本子）,攒满 16 个再合并、舍入一次。状态每步还会乘衰减 α,旧误差会褪掉,
  误差最终停在一个水平上；每 16 步才引入一次新误差,这个水平就低得多。
  论文（Qwen3.5-9B,AIME）,FP32 87.9；每步存 INT8 只有 7.1,改成逐窗口 82.4；
  每步存 BF16 也掉到 72.1,改成逐窗口回到 87.8。主要问题是舍入太频繁,不是位数不够。

三、第二招,补偿 token,把“大个子”单独存（quantize 里拟合 Kc、Uc 的循环）
  状态里少数行、列特别大（最大可达中位数的 176 倍）,把步长撑大了。但状态很有规律,
  基本由几个主要图案叠加而成,前 4 个图案约占 96% 的能量,离群的大数主要来自它们。
  做法,量化前先找出最强的 4 个图案,用 fp16 单独存；从状态里减掉它们,只量化剩下的
  部分。剩下的部分小而平,步长就细了,相当于先把例子里的 127 拿走。
  - 代价小,4 个图案只占 4x256 个 fp16,int8 矩阵有 16384 个数。
  - 怎么找,拿矩阵反复乘一个向量,最强的方向每次被放大得最多,几轮后只剩它（幂迭代）。
    找到一个、减掉一个,重复 4 次。
  - 为什么叫 token,每个图案和一个 token 写进状态的内容同形（key 向量乘 value 向量）,
    decode 时当成窗口最前面多出来的 4 条记录,复用同一段代码。
  论文,离群程度从中位数的 176 倍降到 47 倍,AIME 82.4 -> 86.6。

四、第三招,平滑,把大行和小行拉近（quantize 里的 c,readout 里的 (c * x) @ Z）
  减掉图案后,仍有些行整体偏大、有些行偏小。按列量化时步长被大行决定,小行吃亏。
  做法,每行除以“这一行平均大小的平方根”,用的时候再乘回来。
  例子,某列里 A 行约 9,B 行约 1。
  - 不平滑,步长 9/127 ≈ 0.07,B 的误差最多 0.035,相对它自己 3.5%。
  - 平滑,A/√9 = 3,B/√1 = 1,步长 3/127 ≈ 0.024。乘回去后 B 的误差最多 0.012（1.2%）,
    A 的误差最多 0.012x3 ≈ 0.035,和原来一样。小行精度提高 3 倍,大行不吃亏。
  为什么开平方而不是除到一样大,小行在某列冒出尖峰时,放大后会反过来撑大这一列的步长,
  让大行吃亏。开平方是两头都不太吃亏的折中,思路同 SmoothQuant。
  系数每行只有一个数,用的时候乘到查询向量上,不用把矩阵还原出来。
  论文,离群程度从 47 倍降到 5.3 倍,AIME 86.6 -> 87.9,与 FP32 持平。

五、一个窗口里发生了什么
  1. 窗口开始,状态存成 4 个补偿 token（fp16）+ 每行一个平滑系数 + int8 残差。
  2. 窗口内 16 步,状态不动,每个 token 只追加一条 fp16 记录；读出时补偿 token 当成
     4 条额外记录,平滑系数乘到查询向量上。
  3. 攒满 16 条,高精度拼出完整状态,重新找 4 个图案、重新算平滑系数,量化一次。
  论文 AIME,7.1 -> 82.4 -> 86.6 -> 87.9,最后和 FP32 一样。
'''