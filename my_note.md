# Qwen3.8-Flash-Next 线性注意力层（Gated DeltaNet）笔记

整理于 2026-10-21。

**资料来源**

- 权重索引：[model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/model.safetensors.index.json)
- 本地 checkpoint：`/shareddata/haoyanli/Qwen/Qwen3.8-Flash-Next`
- ATOM 代码：`/home/haoyanli/o/lhy_atom/ATOM`

**模型配置**（`config.json`）

- `hidden_size = 2560`；48 层，其中 36 层 `linear_attention`（GDN），12 层 `full_attention`（QSA）
- `linear_num_key_heads = 16`，`linear_num_value_heads = 48`，`linear_key_head_dim = linear_value_head_dim = 128`
- `linear_conv_kernel_dim = 4`，`hidden_act = silu`，`rms_norm_eps = 1e-6`，`mamba_ssm_dtype = float32`

**内容结构**

1. 原理：从线性注意力一步步推到 Gated DeltaNet，说明四个 `in_proj` 分别干什么
2. 全貌：`linear_attn` 下 9 个权重的完整计算流程、形状和实际取值
3. 细节：短卷积 `conv1d.weight` 的计算和 ATOM 实现
4. 验证：PyTorch 参考实现，以及和 ATOM kernel 的数值对比

---

## 1. 原理：四个 in_proj 分别干什么

四个 `in_proj` 都是用同一个 token 向量 $x_t$（2560 维）乘一个矩阵，没有 bias。它们的区别只在于结果放进公式的哪个位置：

| 投影 | 每个 token 的输出 | 进入公式的量 | 管什么 |
|---|---|---|---|
| `in_proj_qkv` | 10240 维 | $q_t, k_t, v_t$ | 往哪写、写什么、从哪读 |
| `in_proj_a` | 48 个标量 | 遗忘门 $\alpha_t$ | 旧记忆保留多少 |
| `in_proj_b` | 48 个标量 | 写入强度 $\beta_t$ | 新内容写进去多少 |
| `in_proj_z` | 6144 维 | 输出门 $\mathrm{sigmoid}(z_t)$ | 读出的结果放出去多少 |

下面从最朴素的线性注意力出发，一步步加到 Gated DeltaNet，每一步引入一个投影。

### 1.1 朴素线性注意力：只用到 in_proj_qkv

去掉 softmax 之后，因果线性注意力可以写成递推：

$$
S_t = S_{t-1} + k_t^\top v_t,\qquad o_t = q_t S_t
$$

把递推展开：

$$
o_t = \sum_{j\le t} (q_t\cdot k_j)\, v_j
$$

这和 softmax 注意力一样，都是按 $q\cdot k$ 的相似度给 $v$ 加权求和，只是少了 exp 和归一化。所以可以把 $S$ 看成一个 key→value 记忆矩阵，每个头一个，大小 128×128。$k_t^\top v_t$ 是把内容 $v_t$ 存到地址 $k_t$，$q_t S_t$ 是拿 $q_t$ 去查。

`in_proj_qkv` 就是标准注意力里的 $W_Q, W_K, W_V$ 拼在一起：

$$
[\,q_t \mid k_t \mid v_t\,] = \mathrm{SiLU}\big(\mathrm{conv}(W_{qkv}\,x)\big)_t,\qquad
\hat q_t = \frac{q_t}{\|q_t\|\sqrt{128}},\qquad
\hat k_t = \frac{k_t}{\|k_t\|}
$$

其中卷积会用到前 3 个 token（见第 3 节）。教科书里的特征映射 $\phi$，在这里就是"短卷积 + SiLU + L2 归一化"。q、k 只有 16 个头，v 有 48 个头，每个 q/k 头被 3 个 v 头共用。

### 1.2 加遗忘门：引入 in_proj_a

朴素版本的 $S$ 只加不减，旧信息永远留着，越积越乱。于是每一步先衰减一下：

$$
S_t = \alpha_t S_{t-1} + k_t^\top v_t
\quad\Rightarrow\quad
o_t = \sum_{j\le t} \big(\alpha_{j+1}\alpha_{j+2}\cdots\alpha_t\big)\,(q_t\cdot k_j)\, v_j
$$

展开之后，token $j$ 对 token $t$ 的贡献多乘了中间所有 $\alpha$ 的乘积，离得越远衰减越多。$\alpha_t$ 由 `in_proj_a` 算出：

$$
a_t = W_a x_t,\qquad \Delta_t = \mathrm{softplus}(a_t + \mathrm{dt\_bias}),\qquad \alpha_t = \exp\!\big(-e^{A_{\log}}\,\Delta_t\big)
$$

可以把 $\Delta_t$ 看成每个 token 的"时间步长"。连乘就等于 $\exp\big(-e^{A_{\log}}\sum_{i=j+1}^{t}\Delta_i\big)$，也就是衰减按累计步长来算：

- $a_t$ 很大时，$\alpha_t\approx 0$，相当于在这个 token 处清空记忆，比如话题切换的地方。
- $a_t$ 很负时，$\alpha_t\approx 1$，记忆原样保留，这个 token 相当于不占时间。
- 三个量分工不同：`A_log` 决定这个头整体忘得多快（第 1 层各头的半衰期从不到 1 个 token 到上百万个 token，见 2.3 节），`dt_bias` 决定默认步长，`in_proj_a` 按内容逐个 token 调节。

### 1.3 把累加改成 delta rule：引入 in_proj_b

累加写法 $S \mathrel{+}= k^\top v$ 有个问题：同一个 key 写两次，读出来是两个 value 的叠加，旧值删不掉。delta rule 的做法是先读出旧值，只写差值。记 $S'=\alpha_t S_{t-1}$：

$$
S_t = S' + \beta_t\, \hat k_t^\top\big(v_t - \hat k_t S'\big),\qquad \beta_t = \mathrm{sigmoid}(W_b\, x_t)
$$

用同一个 $\hat k_t$ 去读更新后的状态。因为 $\|\hat k_t\|=1$，所以

$$
\hat k_t S_t = (1-\beta_t)\,\hat k_t S' + \beta_t\, v_t
$$

读出来的是旧值和新值按 $\beta_t$ 的插值：$\beta_t=1$ 时这个地址被完全覆盖成 $v_t$，$\beta_t=0$ 时相当于跳过这个 token。和 $\hat k_t$ 正交的地址不受影响，只跟着衰减。

还有一种理解方式：这一步正好是对损失 $\tfrac12\|\hat k_t S - v_t\|^2$ 做一步梯度下降，$\beta_t$ 就是学习率。换句话说，$S$ 一边读 token，一边在线学习"key → value"的映射，`in_proj_b` 决定每个 token 学多少。

这也解释了 k 为什么必须做 L2 归一化。把式子整理成

$$
S_t=\alpha_t\big(I-\beta_t\hat k_t^\top\hat k_t\big)S_{t-1}+\beta_t\hat k_t^\top v_t
$$

矩阵 $I-\beta_t\hat k_t^\top\hat k_t$ 沿 $\hat k_t$ 方向的特征值是 $1-\beta_t$，其余方向都是 1。只要 $\|\hat k_t\|=1$ 且 $\beta_t\in(0,1)$，$S$ 就不会越乘越大，而 sigmoid 正好把 $\beta_t$ 限制在 (0, 1) 内。

### 1.4 替换分母：引入 in_proj_z

原始线性注意力有一个分母，用来模仿 softmax 的归一化：

$$
o_t = \frac{\phi(q_t)\,S_t}{\phi(q_t)\sum_{j\le t}\phi(k_j)^\top}
$$

这个分母数值上不稳定。GDN 把它去掉，改成对每个头的输出做 RMSNorm，再乘一个输出门：

$$
\tilde o_t = \mathrm{RMSNorm}(o_t)\odot w_{\text{norm}}\odot \mathrm{sigmoid}(z_t),\qquad z_t = W_z\,x_t
$$

- RMSNorm 加 `norm.weight` 负责稳定输出的尺度，接替了原来分母的作用。
- $\mathrm{sigmoid}(z_t)$ 由当前 token 决定每个头、每一维放出多少。比如记忆里没有和当前 token 相关的内容时，可以把这个头的输出压低。
- 粒度不一样：$z_t$ 有 6144 维，按元素作用；$\alpha_t$、$\beta_t$ 每个头只有一个标量，作用在整个头上。
- 教科书里常把分母中的 $\sum_j \phi(k_j)^\top$ 记作 $z$。`in_proj_z` 的 $z_t$ 和它只是同名，不是同一个东西。

### 1.5 合起来就是 Gated DeltaNet

每个 token 的完整流程如下。第 $h$ 个 v 头对应第 $\lfloor h/3\rfloor$ 个 q/k 头：

```text
x_t (2560)
 ├─ in_proj_qkv → conv1d + SiLU → q, k (16 头), v (48 头) → L2 归一化 → q̂, k̂, v
 ├─ in_proj_a   → a_t (48)  → α_t = exp(−e^A_log · softplus(a_t + dt_bias))    遗忘门
 ├─ in_proj_b   → b_t (48)  → β_t = sigmoid(b_t)                                写入强度
 └─ in_proj_z   → z_t (48×128)                                                  输出门

S_t = α_t·S_{t−1} + β_t · k̂ᵀ (v_t − k̂ · α_t·S_{t−1})     每个 v 头一个 128×128 矩阵
o_t = q̂ · S_t
y_t = out_proj( concat_h [ RMSNorm(o_t) ⊙ norm.weight ⊙ sigmoid(z_t) ] )      → 2560
```

有个细节：3 个 v 头共用同一组 q/k，但各自的 $v$、$\alpha$、$\beta$、$z$ 是独立的。所以它们读写的地址相同，存的内容、遗忘速度和写入强度却可以各不一样。

### 1.6 实际怎么算

在 ATOM 里，四个投影拼成一次 GEMM，即 `[T, 2560] × [2560, 16480]`。每个 token 约 4200 万次乘加，占整层计算量的七成左右。剩下的卷积、门控和递推，都是逐元素运算或者 128×128 的小矩阵运算。

---

## 2. 全貌：linear_attn 全部 9 个权重的计算逻辑（以 layers.1 为例）

这 9 个权重合起来就是一个完整的 Gated DeltaNet 层：每个 token 输入一个 2560 维向量，输出也是 2560 维。计算分五步：输入投影、短卷积、门控、delta rule 递推、门控归一化加输出投影。36 个 GDN 层都是这套计算，只是权重值不同。下面的公式已经用第 1 层真实权重和 ATOM 的 kernel 对过数值，见第 4 节。

**符号约定**：q 和 k 各 16 个头，v 有 48 个头，每个头 128 维。$x_t\in\mathbb{R}^{2560}$ 是第 $t$ 个 token 的输入，来自本层 `attn_hyper_connection` 的输出。这个模型没有 `input_layernorm`，归一化在 hyper-connection 的 `hc_norm` 里做。

| 权重 | 形状 | 用在哪一步 |
|---|---|---|
| `in_proj_qkv` | [10240, 2560] | 1. 投影出 q/k/v |
| `in_proj_z` | [6144, 2560] | 1. 投影出输出门 z |
| `in_proj_b` | [48, 2560] | 1. 投影出写入强度 b |
| `in_proj_a` | [48, 2560] | 1. 投影出衰减步长 a |
| `conv1d.weight` | [10240, 1, 4] | 2. 短卷积 |
| `A_log` | [48] | 3. 每个头的衰减速率 |
| `dt_bias` | [48] | 3. 衰减步长的偏置 |
| `norm.weight` | [128] | 5. RMSNorm 的缩放系数 |
| `out_proj` | [2560, 6144] | 5. 投回 2560 维 |

模型层入口是 `Qwen4ExpLinearAttention.forward`（[qwen4_exp.py L567-591](../ATOM/atom/models/qwen4_exp.py#L567)）。

### 2.1 输入投影（in_proj_qkv、in_proj_z、in_proj_b、in_proj_a）

这四个都是不带 bias 的线性层，作用在同一个 $x_t$ 上：

- `in_proj_qkv` 得到 10240 维的 `mixed_qkv`。卷积之后再切成 q（2048）、k（2048）、v（6144）。
- `in_proj_z` 得到 $z_t$，48 头 × 128 维，第 5 步当输出门用。
- `in_proj_b` 和 `in_proj_a` 各得到 48 个标量，每个 v 头一个，第 3 步用来算门控。

ATOM 加载 BF16 权重时，把这四个矩阵按行拼成一个 `[16480, 2560]` 的 `in_proj_qkvzba`，一次 GEMM 算完再切开（`packed_modules_mapping`，[qwen4_exp.py L417-422](../ATOM/atom/models/qwen4_exp.py#L417)）。FP8 量化时拆成 qkvz 和 ba 两个 GEMM。

### 2.2 短卷积（conv1d.weight）

对 `mixed_qkv` 的 10240 个通道，各自做宽度为 4 的因果卷积，再过 SiLU。只有 q/k/v 经过卷积，z、a、b 不经过。详见第 3 节。

### 2.3 门控（A_log、dt_bias，配合 a、b）

对每个 v 头，每个 token 算出两个标量：

$$
\Delta_t = \mathrm{softplus}(a_t + \mathrm{dt\_bias}),\qquad
\alpha_t = \exp\!\big(-e^{A_{\log}}\cdot \Delta_t\big)\in(0,1),\qquad
\beta_t = \mathrm{sigmoid}(b_t)
$$

这套参数化和名字都来自 Mamba2：

- $A=-e^{A_{\log}}$ 是个负数，表示这个头每单位步长衰减多快。它是固定参数，不随输入变化。
- $\Delta_t$ 是随输入变化的步长，softplus 保证它为正。`dt_bias` 决定默认步长，`in_proj_a` 让模型能按 token 把步长调大或调小。
- $\alpha_t = e^{A\Delta_t}$ 是这一步旧记忆保留的比例。步长越大，忘得越多。
- $\beta_t$ 是这一步写入新信息的强度。

第 1 层的实际取值：$e^{A_{\log}}$ 在 0.036 到 44.6 之间，`dt_bias` 在 −11.4 到 0.12 之间。以 $a_t=0$ 为参考点，48 个头的 $\alpha$ 从 0.10 到几乎 1.0。换算成半衰期，最短不到 1 个 token，中位数约 8 个 token，最长上百万个 token。也就是说，有的头只记最近几个 token，有的头几乎不遗忘。

ATOM 里这一步由 `fused_gdn_gating` 完成（[attention_gdn.py L33-107](../ATOM/atom/model_ops/attention_gdn.py#L33)），输出 $g_t=\ln\alpha_t$（fp32）和 $\beta_t$。

### 2.4 Delta rule 递推（这一步没有可学习权重）

每个 v 头有一个 128×128 的状态矩阵 $S$。第 $h$ 个 v 头使用第 $\lfloor h/3\rfloor$ 个 q/k 头：v 头 0、1、2 共用 q/k 头 0，以此类推，类似 GQA（kernel 里是 `i_h = i_hv // (HV // H)`，[fused_recurrent.py L69](../ATOM/atom/model_ops/fla_ops/fused_recurrent.py#L69)）。q、k 先做 L2 归一化，q 再乘 $1/\sqrt{128}$。递推式是：

$$
S_t = \alpha_t S_{t-1} + \beta_t\,\hat k_t^\top\big(v_t - \hat k_t\,\alpha_t S_{t-1}\big),\qquad o_t = \hat q_t\,S_t
$$

按顺序读：

1. 把旧状态衰减成 $\alpha_t S_{t-1}$。
2. 用 $\hat k_t$ 从中读出这个 key 位置上原来存的值。
3. 用 $v_t$ 减去它得到误差，乘以 $\beta_t$，再按 $\hat k_t$ 写回。
4. 用 $\hat q_t$ 从新状态里读出输出。

$S$ 的大小是固定的。Qwen4Exp 按 `mamba_ssm_dtype=float32` 存储（[qwen4_exp_attn.py L274-283](../ATOM/atom/model_ops/attentions/qwen4_exp_attn.py#L274)），每个请求每层是 48×128×128 个 fp32，约 3 MB，36 层合计约 108 MB，和序列长度无关。

ATOM 里 prefill 走 `chunk_gated_delta_rule`（分块并行，[chunk.py L207](../ATOM/atom/model_ops/fla_ops/chunk.py#L207)），decode 走 `fused_recurrent_gated_delta_rule`（逐 token 循环，[fused_recurrent.py L129-156](../ATOM/atom/model_ops/fla_ops/fused_recurrent.py#L129)），两者算的是同一个式子。

### 2.5 门控 RMSNorm 和输出投影（norm.weight、out_proj）

对每个头的 128 维输出 $o_t$：

$$
\tilde o_t = \frac{o_t}{\sqrt{\mathrm{mean}(o_t^2)+10^{-6}}}\odot w_{\text{norm}}\odot \mathrm{sigmoid}(z_t)
$$

- `norm.weight` 只有 128 个数，48 个头共用。它是直接相乘的，不是 $1+w$。第 1 层的值都在 1 附近（0.945 到 1.109）。实现见 `_sigmoid_rmsnorm_kernel`（[gated.py L240-285](../ATOM/atom/model_ops/qwen4_exp/ops/gated.py#L240)）。
- 输出门是 sigmoid(z)，按元素控制每个头每一维放出多少。
- 48 个头拼回 6144 维，乘 `out_proj` 得到 2560 维。TP 下 `out_proj` 按行切分，之后 decoder 层做一次 all-reduce，再通过 hyper-connection 写回 4 路残差流。

### 2.6 计算量

每个 token 每层约 6100 万次乘加：

| 部分 | 乘加次数 |
|---|---|
| 四个输入投影 | 约 4200 万（16480 × 2560） |
| `out_proj` | 约 1600 万（2560 × 6144） |
| delta rule 递推 | 约 310 万（48 头 × 4 × 128 × 128） |
| 短卷积、门控、归一化 | 可以忽略 |

递推部分每个 token 的开销固定，不随上下文变长而增加，这是线性注意力相对 softmax 注意力的优势。

---

## 3. 细节：短卷积 conv1d.weight（index.json 第 16 行）

index.json 第 16 行是 `model.language_model.layers.0.linear_attn.conv1d.weight`，形状 `[10240, 1, 4]`，BF16。它是 GDN 线性注意力里的短因果卷积，位置在 `in_proj_qkv` 投影出 q/k/v 之后、进入 delta rule 递推之前。计算很简单：每个通道各自把当前 token 和前 3 个 token 的值加权求和，再过一次 SiLU。

### 3.1 形状含义

- `10240` 是通道数，等于 q、k、v 拼在一起的宽度。q 和 k 各是 16 头 × 128 = 2048，v 是 48 头 × 128 = 6144，合计 10240。
- `1` 表示 depthwise 卷积：每个通道只和自己做卷积，不跨通道混合，相当于 groups = 10240。
- `4` 是卷积核宽度，即 `linear_conv_kernel_dim = 4`。

### 3.2 公式

设 $x_t\in\mathbb{R}^{10240}$ 是第 $t$ 个 token 经过 `in_proj_qkv` 的输出。对每个通道 $c$：

$$
y_t[c] = \mathrm{SiLU}\Big(\sum_{j=0}^{3} W[c,j]\cdot x_{t-3+j}[c]\Big)
$$

- $W[c,3]$ 乘当前 token，$W[c,0]$ 乘 3 步之前的 token。卷积只看过去、不看未来，所以叫因果卷积。
- 没有 bias。序列开头凑不够 3 个历史 token 时补 0。如果这个请求前面已经算过一段（比如命中 prefix cache 或者分块 prefill），就从缓存里取前 3 个 token。
- 激活函数是 SiLU，因为 `hidden_act = "silu"`。

等价的 PyTorch 写法如下。在 CPU 上把它和逐 token 公式、decode 的滑动窗口写法都对过，最大误差约 1e-6：

```python
# x: [T, 10240] 是 in_proj_qkv 的输出，w: [10240, 1, 4]
y = F.silu(F.conv1d(x.T[None], w, groups=10240, padding=3)[0, :, :T].T)
q, k, v = y.split([2048, 2048, 6144], dim=-1)
```

卷积之后，q/k/v 再 reshape 成多头，做 L2 归一化，然后进入 Gated DeltaNet 递推。`z`、`a`、`b` 不经过这个卷积。

### 3.3 开销和作用

- 开销可以忽略：每个 token 每层只要 10240 × 4 ≈ 4 万次乘加，而 `in_proj_qkv` 要 10240 × 2560 ≈ 2600 万次，卷积只占它的 1/640。
- 作用是让每个 token 先混入前 3 个 token 的信息。GDN 层不用 RoPE（`Qwen4ExpLinearAttention.forward` 只接收 `hidden_states`，没有位置参数），局部的顺序信息就靠这个短卷积提供。

### 3.4 ATOM 里的实现

- **加载**（[qwen4_exp.py L521-541](../ATOM/atom/models/qwen4_exp.py#L521)）：先声明成 `ColumnParallelLinear(4 → 10240)`，再 `unsqueeze(1)` 变成和 checkpoint 一致的 `[10240, 1, 4]`。TP 切分用的是 `mamba_v2_sharded_weight_loader`：q、k、v 三段分别按 TP 切开再拼起来，这样每张卡拿到的通道仍是 `[q | k | v]` 的顺序。如果直接把 10240 这一维连续切，会切乱 q/k/v 的边界。
- **缓存**：每个请求每层有一个 `conv_state`，形状 `[3 + num_spec, 10240 / TP]`（[gdn_attn.py L414-418](../ATOM/atom/model_ops/attentions/gdn_attn.py#L414)），按模型 dtype（BF16）存。它存的是最近 3 个 token **卷积之前**的输入 $x$，不是卷积输出。多出的 `num_spec` 格给 MTP 投机解码用：验证完之后，下一步根据接受的 token 数决定从哪一格开始读这 3 个历史 token。
- **Prefill** 走 `causal_conv1d_fn`：整段序列并行计算。有历史就把 `conv_state` 当作左侧上下文，没有就补 0。算完把最后 3 个 token 的输入写回 `conv_state`。
- **Decode** 走 `causal_conv1d_update`（[causal_conv1d.py L1087-1173](../ATOM/atom/model_ops/mamba_ops/causal_conv1d.py#L1087)），每个 token 的核心循环就是上面的公式，再加一次窗口左移：

```python
for idx_token in tl.range(seqlen):
    acc = acc_preload
    matrix_w = w_col0
    matrix_x = col0                     # col0/col1/col2 = conv_state 里最近 3 个 token
    for j in tl.static_range(KERNEL_WIDTH):
        # KERNEL_WIDTH == 4 时：
        #   j=1: w_col1 * col1,  j=2: w_col2 * col2,  j=3: w_col3 * 当前 token 的 x
        acc += matrix_x * matrix_w  # [BLOCK_N]
    # ...
    col0 = col1                         # 窗口左移：丢掉最老的，当前 x 进窗口
    col1 = col2
    col2 = matrix_x
    # ...
    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
```

算完的 `acc` 按通道下标分别写进 `query`/`key`/`value` 三个输出：前 2048 个通道写 q，接下来 2048 个写 k，最后 6144 个写 v。

---

## 4. 验证

### 4.1 GDN 层的 PyTorch 参考实现

下面是整层的全部计算逻辑，精度 fp32，逐 token 递推。完整脚本在 `paper/gdn_verify.py`，运行方式：

```bash
cd /home/haoyanli/o/lhy_atom/ATOM && HIP_VISIBLE_DEVICES=6 python3 /home/haoyanli/o/lhy_atom/paper/gdn_verify.py
```

```python
H, HV, K, V, KW, EPS = 16, 48, 128, 128, 4, 1e-6
KD, VD = H * K, HV * V          # 2048, 6144
C = 2 * KD + VD                 # 10240
# W: 第 1 层 linear_attn 的 9 个权重（从 model-00003-of-00131.safetensors 读取）；T: token 数


def ref_conv(mixed_qkv):
    y = F.conv1d(mixed_qkv.T[None], W["conv1d.weight"].float(), groups=C, padding=KW - 1)
    return F.silu(y[0, :, :T].T).split([KD, KD, VD], dim=-1)


def ref_gating(a, b):
    g = -W["A_log"].float().exp() * F.softplus(a + W["dt_bias"].float())
    return g, b.sigmoid()


def ref_recurrence(q, k, v, g, beta):
    rep = HV // H
    q = q.view(T, H, K).repeat_interleave(rep, dim=1)   # v 头 h 用 q/k 头 h // 3
    k = k.view(T, H, K).repeat_interleave(rep, dim=1)
    v = v.view(T, HV, V)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K**-0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    alpha = g.exp()
    S = torch.zeros(HV, K, V, device=dev)
    o = torch.empty(T, HV, V, device=dev)
    for t in range(T):
        S = alpha[t, :, None, None] * S
        v_old = torch.einsum("hk,hkv->hv", k[t], S)
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
    b = xf @ W["in_proj_b.weight"].float().T
    a = xf @ W["in_proj_a.weight"].float().T
    q, k, v = ref_conv(mixed_qkv)
    g, beta = ref_gating(a, b)
    o, S = ref_recurrence(q, k, v, g, beta)
    return ref_norm(o, z) @ W["out_proj.weight"].float().T, S
```

### 4.2 和 ATOM kernel 的对比结果

测试设置：第 1 层真实权重，输入是 150 个 token 的随机向量（BF16，标准正态），跑在空闲的 GPU 6 上。两条路径都测：

- prefill：`causal_conv1d_fn` + `fused_gdn_gating` + `chunk_gated_delta_rule`，一次算完 150 个 token（跨 3 个 64-token 分块）。
- decode：`causal_conv1d_update` + `fused_gdn_gating` + `fused_recurrent_gated_delta_rule`，逐 token 跑 150 步。

逐阶段对比时，参考实现的每一步都用 ATOM 上一步的输出作为输入，避免误差累积。相对误差定义为 $\|a-b\|/\|b\|$。

| 阶段 | 相对误差 | 说明 |
|---|---|---|
| 短卷积 + SiLU（q/k/v） | 2.3e-3 | BF16 精度 |
| 门控 $g$ / $\beta$ | 3.1e-7 / 1.6e-3 | $g$ 是 fp32，$\beta$ 是 BF16 |
| 递推 prefill：输出 $o$ / 最终状态 $S$ | 4.2e-3 / 3.3e-3 | 分块算法，中间结果有 BF16 |
| 递推 decode：输出 $o$ / 最终状态 $S$ | 1.7e-3 / 1.9e-7 | 状态 $S$ 是 fp32 |
| 门控 RMSNorm | 2.8e-3 | BF16 精度 |
| 端到端 prefill / decode | 7.4e-3 / 6.4e-3 | 余弦相似度 0.99997 / 0.99998 |
| 对照：k 头映射改成交错方式 | 1.32 | 余弦相似度 0.137 |

最后一行是故意构造的错误版本：v 头 $h$ 改用 k 头 $h \bmod 16$。误差变成 132%，说明这个测试对公式错误足够敏感。

### 4.3 线性注意力几个结论的验证

第 1 节用到的代数结论，都在 fp64 下用随机数验证过，误差都在 1e-14 以下：

- 门控线性注意力的展开式 $o_t = \sum_{j\le t}(\alpha_{j+1}\cdots\alpha_t)(q_t\cdot k_j)v_j$；
- delta rule 用同一个 key 读回时是新旧值插值：$\hat k_t S_t = (1-\beta_t)\hat k_t S' + \beta_t v_t$；
- 和 $\hat k_t$ 正交的 key 只受衰减影响；
- 更新等价于对 $\tfrac12\|\hat k_t S - v_t\|^2$ 做一步学习率为 $\beta_t$ 的梯度下降；
- $I-\beta\hat k^\top\hat k$ 的特征值是 $1-\beta$（沿 $\hat k$ 方向）和 1（其余方向）。
