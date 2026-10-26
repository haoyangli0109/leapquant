# LeapQuant 在 ATOM 上的实验结论

2026-10-20，Qwen3.8-Flash-Next（BF16），2×MI355X（GPU 4、5），TP2+EP，CUDA graph，关闭前缀缓存。

- **LeapQuant 配置**：int8 残差，r=4 个 FP16 补偿 token，FP32 平滑因子，窗口 p=16，窗口内记录用 FP16。只支持每序列一个 token 的 decode。
- **对比基线**：
  - FP32 状态（默认）
  - BF16 状态（每步读写 BF16）
  - ReplaySSM（FP32 checkpoint 加记录回放）

## 性能

测试负载：输入 4096、输出 1024，每档并发只发一轮等长请求。

**decode 每步耗时（ms）**：取服务端稳态统计，flush 步的开销已摊入。

| 并发 | FP32 | BF16 | ReplaySSM | LeapQuant | 对 FP32 | 对 ReplaySSM |
|---|---|---|---|---|---|---|
| 128 | 28.17 | 27.16 | 26.68 | 26.03 | 1.08× | 1.02× |
| 256 | 38.28 | 36.51 | 36.39 | 34.49 | 1.11× | 1.06× |
| 512 | 59.33 | 55.27 | 54.76 | 51.92 | 1.14× | 1.05× |

- **不含 flush 的步时**：并发 512 时 median ITL 为 50.31 ms，相当于 FP32 的 1.18×。
- **重测**（2026-10-22，同样独占 GPU 4、5，代码含 fold 修复和前缀缓存支持）：

| 并发 | FP32 | LeapQuant | 对 FP32 | FP32 开缓存 | LeapQuant 开缓存 | 对 FP32 |
|---|---|---|---|---|---|---|
| 128 | 28.32 | 26.27 | 1.08× | 28.57 | 26.37 | 1.08× |
| 256 | 38.48 | 35.16 | 1.09× | 38.63 | 35.11 | 1.10× |
| 512 | 59.22 | 52.42 | 1.13× | 59.44 | 52.63 | 1.13× |

  - FP32 和 10-20 相差不到 0.6%。LeapQuant 慢了 0.9–1.9%，但不是新代码造成的：同一天把提交前的代码和当前代码交替各跑两次，均值相差不超过 1.0%，比同一份代码两次之间的差别（最多 2.1%）还小。
  - 不含 flush 的步时：并发 512 时 LeapQuant 50.72 ms、FP32 59.09 ms（1.17×）；开缓存时 50.92 对 59.54 ms，也是 1.17×。
  - 开前缀缓存不影响 decode 步时，差别不超过 0.9%。这个负载的 prompt 加输出不到 8192 token，decode 途中不打 checkpoint，只在 prompt 末尾打一个锚点。
  - 开缓存时，decode 扫描的 TTFT 不能拿来比。benchmark 用固定种子，并发 64 的请求包含并发 32 的全部请求，后面几档会命中前面几档留下的锚点。步时只取纯 decode 窗口，不受影响。
- **共享前缀**（`prefix-bench`，2026-10-22）：128 个请求共享 8448 token 的前缀，各自再接 512 token，输出 128 token，并发 32。

| 模式 | 平均 TTFT | 总耗时 | 请求/秒 |
|---|---|---|---|
| FP32 | 4.60 s | 41.0 s | 3.12 |
| LeapQuant | 4.77 s | 42.0 s | 3.05 |
| FP32 开缓存 | 0.27 s | 8.7 s | 14.7 |
| LeapQuant 开缓存 | 0.24 s | 8.4 s | 15.2 |

  - 开缓存后 TTFT 降到 1/17–1/20，吞吐提高到约 5 倍，LeapQuant 和 FP32 一样。
  - 命中最多到 8192 token。GDN 状态只在 8192 的阶梯和 prompt 末尾存 checkpoint，前缀再长，命中也只到最近的一级阶梯。
- **GDN kernel 单独计时**（36 层合计，每步）：
  - B=128：FP32 3.08 ms，ReplaySSM 2.14 ms，LeapQuant 1.54 ms（2.00×）。
  - B=512：FP32 12.02 ms，ReplaySSM 7.54 ms，LeapQuant 5.20 ms（2.31×）。其中 fold 占 2.25 ms，decode 占 2.95 ms。
- **显存**（每请求每 rank 的状态）：FP32 57.9 MB，BF16 29.6 MB，ReplaySSM 65.0 MB，LeapQuant 25.7 MB（比 FP32 少 56%）。
  - 留给 KV cache 的空间：FP32 46.9 GB，ReplaySSM 43.4 GB，LeapQuant 62.2 GB。
- **端到端收益小于论文的原因**（论文 B200 上 B=512：kernel 2.68×，decode 每步 1.22–1.37×）：
  - GDN 只占 FP32 每步时间的约 20%，即使把 GDN 优化到零，端到端最多 1.25×。
  - fold 没有和显存读取重叠。论文暴露的开销不超过 7%，这里占 LeapQuant GDN 时间的 43%。
- **窗口长度**：p=32 时 GDN 每步再快 7%（4.86 ms 对 5.21 ms），代价是每请求多约 7 MB 记录。默认仍用论文的 p=16。

## 精度

**AIME 2026 / GPQA-Diamond**（2026-10-21；fold 修复后的两行是 2026-10-22 测的）：thinking 模式，按模型卡采样（temperature 1.0、top_p 0.95、top_k 20），每题 3 个样本，最长输出 81,920 token，截断算错，和论文的协议一致。每个服务 128 个请求并发。

| 模式 | AIME 2026 | GPQA-Diamond | GPQA 平均输出 token | GPQA 截断数 |
|---|---|---|---|---|
| FP32 | 95.6 | 90.4 | 16,581 | 15 |
| BF16 | 94.4 | 88.7 | 18,258 | 30 |
| LeapQuant（修复前） | 94.4 | 90.7 | 16,614 | 11 |
| LeapQuant（修复后） | 95.6 | 91.6 | 16,876 | 16 |
| LeapQuant（修复后，开前缀缓存） | 96.7 | 89.9 | 16,724 | 15 |

- **LeapQuant 与 FP32 没有可测的差别**：GPQA 逐题和 FP32 比较（更差/更好的题数）：

| 模式 | 更差 | 更好 | 符号检验 p | 输出长度中位数比值 |
|---|---|---|---|---|
| LeapQuant（修复前） | 9 | 11 | 0.82 | 0.99 |
| LeapQuant（修复后） | 9 | 14 | 0.40 | 1.03 |
| LeapQuant（修复后，开前缀缓存） | 13 | 11 | 0.84 | 1.00 |

  - AIME 逐题：修复后和 FP32 完全一致；开前缀缓存 1 题更差、2 题更好。
  - 修复后的两次（开、关前缀缓存）GPQA 相差 1.7 分，逐题 9 题更差、8 题更好，是采样噪声。
- **BF16 有轻微退化**：GPQA 低 1.7 分，逐题 15 题更差、7 题更好（符号检验 p=0.13，单看不显著）。但推理变长：输出长度中位数多 4%（p=0.046），截断翻倍（p=0.036）。方向和论文一致，幅度小得多：论文里 BF16 在 Qwen3.5-9B 的 GPQA 上掉 15 分，在 35B-A3B 上掉 5 分。
- **AIME 到顶了**：错的几乎都是第 10、14 题在 81,920 token 处被截断，各模式都一样，区分不了各方法。修复后的两次，错的全是截断。唯一一次真正答错是 BF16 的第 9 题（156 答成 165）。
- FP32 的 GPQA 90.4 和模型卡报告的 91.7 接近。

**GSM8K**：全部 1319 题，5-shot chat，`max_gen_toks=4096`。

| 模式 | flexible | strict | 逐题对比 FP32（只 FP32 对 / 只本模式对） |
|---|---|---|---|
| FP32 | 97.35 | 96.97 | – |
| BF16 | 97.57 | 97.35 | 5 / 8 |
| ReplaySSM | 97.57 | 97.42 | 3 / 6 |
| LeapQuant | 97.57 | 97.50 | 3 / 6 |

各模式的标准误约 ±0.42，差异都在噪声范围内。GSM8K 的回答很短，连每步存 BF16 都不掉点，区分不了各方法。

**数值误差**：

- **合成 GDN 状态**（64 序列，272 步）：输出相对精确 FP32 的误差，按四分之一段统计如下。LeapQuant 的误差不随窗口累积。
  - LeapQuant：5.1e-4 → 5.3e-4
  - 逐窗口 INT8（不带补偿和平滑）：7.1e-4 → 9.2e-4
  - 每步 BF16：1.07e-3 → 1.16e-3
  - 每步 INT8：3.6e-3 → 4.0e-3
- **真实模型逐层检查**：decode 输出和 torch 参考的误差不超过 3.1e-3（含 bf16 输出舍入），边界状态量化误差不超过 3.3e-3。

## 已修复的问题

真实模型里有些 head 的状态恰好是低秩的。拟合完补偿 token 后残差趋近于零，幂迭代归一化时平方和在 fp32 下下溢，k 溢出为 fp16 inf，最终产生 NaN，输出变成一串 "!!!!"。

修复：`_lq_unit` 先按最大绝对值缩放再归一化，可忽略的分量直接置零。回归测试见 `leapquant_check.py` 的 D 部分。本文所有数据都是修复之后测的。

**fold 结果不确定**（2026-10-21，验证前缀缓存时发现）：同样的输入重复 fold 200 次，会得到几种到几十种不同结果，个别 head 的 int8 值差到 116 档。原因在 warm 起点退化时的回退 `tl.where(max|S·u_old| > tiny, S·u_old, k_old)`：它把从 tile 归约出来的向量和单独 load 的 `k_old` 混在一起，ROCm 上 2 个 warp 时结果不确定，加 barrier 也没用。

修复：退化时这个分量像冷启动一样，从行范数开始做 8 次幂迭代；不退化时计算不变。

- fold 耗时不变：64 序列 × 32 头 × 24 层全部 fold，2.87 ms 对 2.93 ms。
- `leapquant_check.py` 的 A–C 部分结果和修复前完全一样。D 部分 zero、tiny 两种状态的 warm-fold 误差从 3.3e-3 降到 1.1e-7。
- 修复后重测了 decode 步时和 AIME/GPQA（见性能、精度两节），和修复前没有可测的差别。

## 前缀缓存

2026-10-21，Qwen3.5-9B，单卡 eager。LeapQuant 现在可以开前缀缓存，不用再加 `--no-enable_prefix_caching`。

ATOM 的 GDN 前缀缓存靠分叉：发布 checkpoint 时，请求换到新槽，旧槽带着 hash 留下来当 checkpoint。checkpoint 打在三类位置：prompt 末尾的锚点（16 token 对齐）、`--state-checkpoint-interval-tokens` 的阶梯、decode 途中。

ReplaySSM 和 LeapQuant 的槽里除了边界状态，还有窗口内的记录和游标。原来的代码只考虑了边界状态，修了三处：

1. **decode 途中分叉**：kernel 只从一个槽重建再往同一个槽追加，分叉后的新槽是空的。修复：commit 之前把整个槽（边界、记录、游标）拷到新槽。
2. **从 decode 途中的 checkpoint 恢复**：槽里有 n 条记录（n ≤ 16），prefill 把游标停在 −1，记录被丢掉，相当于少了最近 n 个 token。修复：停游标之前记下 n，prefill 读初始状态后先回放这 n 条记录（`replayssm_replay_records`）。
3. **读写同一个槽**：n 必须在停游标之前取。

prefill 中途写 checkpoint 的路径（`leapquant_write_checkpoints`）也实现了，并做了离线测试，但 ATOM 现在的调度走不到。

**离线检查**（`leapquant_prefix_check.py`）：20 对孪生序列在游标的每个相位（0–16 条记录）分叉。

- 分叉和不分叉的输出逐位相同。
- 恢复后的状态相对 fp64 的误差为 3.05e-4。只恢复边界状态时，误差至少大 3274 倍。

**端到端**：64 个槽，checkpoint 间隔 256。

- **分叉不影响生成**：开 decode checkpoint 时，原回复（R1）在 272、528、784 处各分叉一次。它和不开 decode checkpoint（间隔 −1）的 R1 逐位相同，包括 logprob。共 3 个题目，每个约 940 token。
- **从 decode checkpoint 续写**：取原回复在 checkpoint 后第 1 或第 41 个 token 处截断的前缀，作为新请求发出。比较续写和原回复前 32 个 token 的 logprob，取平均差（3 个题目 × 3 个 checkpoint）：

| 模式 | +1 | +41 |
|---|---|---|
| 开缓存 | 1.1e-2 | 8.3e-3 |
| 开缓存，去掉回放 | 3.2e-1 | 4.9e-2 |
| 不开缓存（整段重算） | 1.0e-2 | 1.2e-2 |
| 只用锚点（间隔 −1） | 1.2e-2 | 1.4e-2 |

  - 去掉回放后，+1 处 9 次续写里有 4 次第一个 token 就和原回复不同。
  - 有回放时，续写和整段重算一样接近原回复。剩下的差别来自 prefill 和 decode 两条路径本身，例如 prefill 末尾会重新量化一次。

- **GSM8K**（250 题，5-shot，每批 8 个请求）：

| 模式 | 准确率 | 前缀命中率 |
|---|---|---|
| 开缓存 | 85.6 | 88.0% |
| 不开缓存 | 84.4 | – |
| 只用锚点 | 86.4 | 87.1% |

  开缓存和不开缓存有 13 题答案不同，在噪声范围内（标准误约 ±2.2）。

- **跨进程可复现**：同一配置跑两次，前 3 个题目的回复、全部续写和 GSM8K 答案逐位相同。
  - 第 4 个题目不同，原因是 aiter 的 GEMM：M=26、33 时，attention qkv 和 o_proj 的 GEMM 每次结果不同；M=16、112 时相同。这个题目的 prompt 会切出一个 32 token 的块。
  - 不开缓存时，短 prompt 整段都落在这个区间，默认 FP32 状态也一样，和 LeapQuant 无关。
  - Qwen3.8-Flash-Next 上也能看到：FP32、不开缓存，`smoke` 里同一个短问题贪心解码 6 次，回复长度在 132–193 token 之间。

**大模型验证**（2026-10-22）：Qwen3.8-Flash-Next，TP2+EP，CUDA graph，开前缀缓存（checkpoint 间隔用默认的 8192），跑精度一节的 AIME/GPQA。

- 精度和不开缓存一样，没有 NaN，也没有报错。
- 共打了 2136 个 checkpoint，平均每请求约 3 个：prompt 末尾的锚点，加上 decode 途中每 8192 token 一个。decode 途中分叉约 1400 次，修复 1 在 TP2 和 CUDA graph 下都走到了。
- 其中 1715 个被挤掉，给新请求腾槽（`--max-num-seqs 256`，同时跑 128 个请求）。
- 前缀命中率 15.1%，prefill 的 token 比不开缓存少约 14%。同一题的 3 个样本基本同时发出，很少能用上彼此的锚点。
- 这个评测里没有请求从 decode 途中的 checkpoint 续写，回放（修复 2、3）没有走到，只在上面 Qwen3.5-9B 的端到端测试里验证过。

## Kimi-Linear（KDA）

2026-10-22，Kimi-Linear-48B-A3B-Instruct（BF16），单卡 MI355X，TP1，CUDA graph，不开前缀缓存（前缀缓存一段除外）。27 层：20 层 KDA，7 层 MLA（NoPE）；KDA 32 个头，head_dim 128。

KDA 和 GDN 的区别是衰减门控按 K 通道（每头 128 个），而不是每头一个标量：g = −exp(A_log)·softplus(a + dt_bias)，没有下界。LeapQuant 的推导不变，只是标量衰减换成对角矩阵：diag(e^C)·S₀ = diag(e^C ⊙ c)·Z·diag(b) + (K_c·diag(e^C))ᵀ·U_c。衰减并进 K 侧平滑因子和补偿 token 的 key，int8 残差 Z 不用改。

**实现**（未提交）：

- **模型**：ATOM 原来只支持 Kimi-K3。`kimi_k3.py` 补上 Kimi-Linear 的差异：SiLU FFN，没有 q LoRA 和输出门的 MLA，低秩的 KDA 输出门（g_a_proj → g_b_proj），A_log 的形状，共享专家并入 FusedMoE。`model_runner.py` 注册了 `KimiLinearForCausalLM`。`config.py` 在没有 `max_position_embeddings` 时读 `model_max_length`，否则这个 NoPE 模型的上下文会被截到 8192。
- **kernel**（`leapquant.py`）：fold 支持每通道衰减。decode kernel 在 kernel 里算门控、β 和 q/k 的 l2 归一化，窗口记录多存一份每通道的 FP32 门控。fold 和 GDN 一样，commit 之后一次做完所有层。
- **prefill**：读状态时反量化成 K 在前的 FP32，用 aiter 的 chunk KDA（`state_v_first=False`），写回时再量化。FP32 状态池是 V 在前，回放 kernel（`replayssm_replay_records`）两种布局都支持。
- **前缀缓存**（`kimi_mla_gdn_attn.py`）：checkpoint 镜像带上平滑因子、补偿 token、记录和游标，layout 标识里写明秩和记录格式。
- **越界 token**：tokenizer 有 163842 个 id，embedding 只有 163840 行，多出的两个是保留 token。请求里出现它们时 GPU memory fault，整个服务退出。benchmark 的随机 prompt 并发 64 起就会碰到。
  - `llm_engine.py` 现在拒绝越界的 id，返回 HTTP 400。
  - `benchmark_serving.py` 的随机数据集只在模型词表内取 id。Qwen3.8 的 tokenizer 词表比 embedding 小，生成的 prompt 不变。

**离线检查**（`leapquant_kda_check.py`，门控参数取自第一层 KDA）：

- decode：64 序列 × 32 头 × 272 步，kernel 和 torch 影子的误差不超过 7.5e-5。LeapQuant 相对精确 FP32 的输出误差，按四分之一段：5.6e-4 → 6.7e-4 → 6.9e-4 → 6.9e-4，不随窗口累积。
- prefill 的 K 在前状态等于 V 在前的转置（逐位相同）；跨层 fold 和逐层 fold 逐位相同；每通道门控的记录回放误差 1.7e-7。
- 每个 kernel（decode、带 fold 的 flush 步、跨层 fold、量化、反量化、记录回放）同样输入跑 200 次，结果逐位相同。这项检查要在有别的任务在跑的 GPU 上做，空闲的卡上竞争很少暴露。GDN 的 kernel 补了同样的检查（`leapquant_check.py` 的 E 部分），也通过了。
  - 记录回放原来在 K 在前时约每 13 次出错一次：几个 lane 的结果不对，相对误差约 2e-3。原因和 fold 那次一样，是把 tile 归约出的向量和单独 load 的向量混用。改成同一种 load 逐条累加后不再出现，误差也从 8.9e-6 降到 1.7e-7。

**kernel 单独计时**（20 层 KDA 合计，每步，ms）：

| B | FP32 | ReplaySSM | LeapQuant | 对 FP32 |
|---|---|---|---|---|
| 32 | 0.63 | 0.61 | 0.41 | 1.52× |
| 128 | 2.33 | 1.97 | 1.38 | 1.69× |
| 256 | 4.53 | 3.77 | 2.56 | 1.77× |
| 512 | 8.51 | 7.43 | 4.97 | 1.71× |

B=512 时 fold 占 1.32 ms，decode 占 3.64 ms。

**decode 每步耗时**（ms，独占 GPU 4、5，输入 4096、输出 1024，每档并发一轮等长请求）：ATOM 先做完所有 prompt 的 prefill，再让全部请求一起 decode，所以只取最后一个 prompt 进来之后的窗口，算平均 ITL，flush 已摊入（`decode_window.py`）。

| 并发 | FP32 | LeapQuant | 对 FP32 | 不含 flush（中位数） |
|---|---|---|---|---|
| 32 | 8.41 | 7.91 | 1.06× | 1.07× |
| 64 | 10.86 | 10.38 | 1.05× | 1.06× |
| 128 | 14.39 | 13.17 | 1.09× | 1.13× |
| 256 | 21.83 | 20.05 | 1.09× | 1.13× |
| 512 | 33.30 | 29.65 | 1.12× | 1.17× |

- 和 kernel 计时对得上：并发 512 时 KDA kernel 省 3.54 ms，33.30 − 3.54 = 29.76 ms，实测 29.65 ms。KDA 只占 FP32 每步的 26%，就算把它优化到零，端到端最多 1.34×。
- 同一天测了两轮，同一档的中位数前后相差不超过 5.4%。
- **显存**（每请求的状态）：FP32 43.4 MB，LeapQuant 25.1 MB（少 42%）。LeapQuant 里窗口记录占 11.1 MB，其中一半是每通道的 FP32 门控；GDN 只有标量门控，所以能省 56%。512 个槽时，KV block 从 114.7 万增加到 121.7 万（+6%）。

**和 HF 参考实现对比**（`kimi_ref_check.py`）：把 ATOM 生成的 token 序列交给 HF 的 `modeling_kimi.py`（fla-core 0.5.2，sdpa）做 teacher forcing，逐位置比 logprob。

- **短回复**（8 个 prompt，贪心 256 token，1500 个位置），参考为 HF FP32：

| 对比 | argmax 一致 | 平均 \|Δlogprob\| |
|---|---|---|
| HF BF16（噪声下限） | 98.7% | 1.9e-2 |
| ATOM FP32 | 96.6% | 4.4e-2 |
| ATOM LeapQuant | 95.5% | 4.9e-2 |

  - ATOM 和 HF 的差别在各位置分布均匀，不随位置变大，和 ATOM 自己 prefill、decode 两条路径之间的差别（2.8e-2）同量级，不是移植错误。LeapQuant 比 FP32 多 0.5e-2。
- **长回复**（8 道 GPQA 题，temperature 0.6 采样到 16K token，共约 13 万个位置，LeapQuant 每题 fold 约 1000 次），参考为 HF BF16：平均 |Δlogprob| FP32 2.58e-2、LeapQuant 2.43e-2。两者都不随位置变大：[8K, 16K) 段 FP32 2.43e-2、LeapQuant 1.78e-2。LeapQuant 没有误差累积。

**精度**：

- **GSM8K**（1319 题，5-shot chat）：flexible FP32 86.28、LeapQuant 86.96；逐题只 FP32 对 26 题、只 LeapQuant 对 35 题。strict 两者都约 43.5（FP32 43.52、LeapQuant 43.82），因为 Kimi 的回答格式很少符合 strict 的正则。
- **AIME 2026 / GPQA-Diamond**：按模型卡采样（temperature 0.6），最长 32,768 token，每个服务并发 128。AIME 每题 3 个样本，GPQA 每题 6 个。

| 模式 | AIME 2026 | GPQA-Diamond | GPQA（截断算错） | GPQA 截断率 |
|---|---|---|---|---|
| FP32 | 57.8 | 54.6 | 50.7 | 36% |
| LeapQuant | 56.7 | 52.7 | 48.8 | 38% |

- **逐题对比 FP32**（更差/更好的题数）：AIME 7/6。GPQA 61/43，符号检验 p=0.095；截断算错时 61/47，p=0.21。
- **两轮 GPQA 分开看**：第一轮（样本 0–2）低 2.6 分，p=0.045；第二轮（样本 3–5）低 1.2 分，p=0.23。方向一致但都不显著，约 2 分的小幅下降不能排除。2026-10-23 重测又低 1.7 分，两次合起来 p≈0.016，见“2026-10-23 Kimi-Linear 重测”。按论文设置（10-24）低 2.2 分，这一次单独不显著。
- **截断和长度没有差别**：同一题同一样本，只有 FP32 截断的 142 对、只有 LeapQuant 截断的 159 对（p=0.36），输出长度中位数之比为 1.00。Qwen3.8 上 BF16 状态的退化表现为推理变长，这里没有出现。
- **这个评测噪声大**：Kimi-Linear-Instruct 不是推理模型，约 37% 的回复写满 32,768 token，多是在推理里打转。单个样本的准确率在 49.5–58.1 之间。按原样判分时，截断的回复里只要出现过正确答案也算对，所以另给了截断算错的分数。

**前缀缓存**（`prefix_e2e.py`，单卡 eager，64 个槽，checkpoint 间隔 256，4 个题目 × 3 个 checkpoint）：比较续写和原回复前 32 个 token 的 logprob，取平均差。+1、+41 各 12 次续写。

| 模式 | +1 | +41 | +1 首 token 不同 | +1 前 3 个 token 内分开 | GSM8K | GSM8K 命中率 |
|---|---|---|---|---|---|---|
| LeapQuant，开缓存（两次） | 2.8e-2 / 3.0e-2 | 3.0e-2 / 3.2e-2 | 0 / 0 | 0 / 0 | 91.2 / 91.6 | 86.5% |
| LeapQuant，开缓存，去掉回放 | 1.1e-1 | 4.8e-2 | 4 | 10 | – | – |
| LeapQuant，不开缓存 | 4.3e-2 | 3.9e-2 | 0 | 4 | 92.0 | – |
| FP32 ReplaySSM，开缓存 | 3.5e-2 | 2.8e-2 | 2 | 4 | 90.4 | 86.4% |
| FP32 ReplaySSM，开缓存，修复 flush 竞争后（三次） | 2.6e-2 / 2.4e-2 / 3.4e-2 | 2.8e-2 / 4.2e-2 / 3.5e-2 | 0 / 1 / 0 | 0 / 4 / 1 | 90.8 / 89.6 / 91.6 | 86.4–86.5% |
| FP32，不开缓存 | 2.8e-2 | 2.7e-2 | 0 | 3 | – | – |

- **有回放时，从 checkpoint 续写和 FP32 整段重算一样接近原回复**。+1 处比 LeapQuant 自己整段重算更近，+41 处差不多。LeapQuant 整段重算时，状态在 prefill 里一直是 FP32，只在末尾量化一次，和原回复 decode 时用的量化状态不同；从 checkpoint 恢复用的就是 decode 时那一份。
- **去掉回放**（恢复时丢掉窗口里的记录）：+1 的差大到约 4 倍，4 次第一个 token 就不同。其中一次原回复的 token 概率是 1.00，续写换成了另一个概率 0.62 的 token。
- **FP32 ReplaySSM 两行**：修复 flush 竞争（见限制）之前，那 2 次首 token 不同都在平局处，两个候选的概率都约 0.3；FP32 不开缓存在 +41 处的 1 次也是平局。
  - 修复后跑了三次：第一次 KDA 的 flush 也拆了出去，后两次是现在的代码。三个版本的 KDA kernel 在离线轨迹上逐位相同，所以三次之间的差别都来自 Kimi 本身跑两次就不同（见下一条）。
  - 三次里 +1 首 token 不同只有 1 次，不是平局：原回复里那个 token 的概率是 0.63，续写换成了另一个概率 0.58 的 token。
  - +41 在 2.8e-2–4.2e-2 之间。LeapQuant 不开缓存时完全不经过 checkpoint，也有 3.9e-2。
- **Kimi 在 ATOM 上跑两次结果不逐位相同**：开缓存的两次运行，4 个题目的原回复分别从第 91、0、28、29 个 token 开始不同（第 0 个是平局，两个候选各约 40%）。同一配置跑了三次（其中一次在修复记录回放之前），+1 在 1.9e-2–3.0e-2 之间，所以各行之间 1e-2 以内的差别不能当真。
- **GSM8K**（250 题，5-shot，每批 8 个请求）各行的差别都在噪声范围内，标准误约 ±1.7。

**数据和日志**：`results/kimi-{fp32,lq}{,-r2}/`（decode 测速，两轮）、`results_gsm8k/kimi-*`、`results_reason/kimi-*`、`results_prefix/kimi-*.json`；服务、评测和离线检查的日志在 `logs_kimi/`。

## 限制

- 不支持投机解码/MTP。
- Kimi-Linear 上 LeapQuant 的 GPQA 三次都低约 2 分（1.9、1.7、2.2）。temperature 0.6 的两次合并 p≈0.016，按论文设置的一次单独不显著；MMLU-Pro、AIME 和 FP32 没有差别。
- checkpoint 和运行中的请求共用状态池（`--max-num-seqs` 个槽）。池满时 checkpoint 会被挤掉，命中率随之下降：`--max-num-seqs 16` 同时跑 16 个请求时，GSM8K 命中率只有 6%。
- 回放 kernel 现在支持每通道门控和 V 在前的布局。以前 KDA 开 ReplaySSM 时，从 decode checkpoint 恢复会丢记录，现在不会了。
- **ATOM 原有的 ReplaySSM 在 flush 时有竞争，2026-10-23 已修复**：单 token 的 GDN 步像 LeapQuant 一样，把 flush 拆成单独一次 launch；MTP 的验证窗口和 KDA 仍在层 kernel 里 flush，改由最后一个读完记录的 program 写 k/g。
  - 原因：每个 (序列, head) 由几个 program 分 V 处理（GDN 2 个，KDA 4 个），它们都要读同一份 k/g 记录。flush 时窗口从第 0 行重新开始，`i_v == 0` 的 program 把这一步的 k/g 写进第 0 行。别的 program 如果还没读到旧的第 0 行，就会用新 token 顶替最老的那条记录，因为同一次 launch 里的 program 没有先后顺序。算错的状态会在 flush 时写回 checkpoint，这个序列之后的 token 都从它算起。
  - 修复前的现象（`replayssm_race_check.py`，同样输入跑 50 次）：
    - flush 前一步总是一致，flush 步会不会出错取决于调度。
    - GPU 忙时，GDN 和 KDA 8 个序列就出错。KDA 只有写入记录的那一块 V 是对的，其余几块相对误差到 0.65。
    - GPU 空闲时，GDN 64 个序列起出错，KDA 到 512 个序列都没出现。
    - 受影响的是 serial 路径，包括 GDN 的普通 decode、1–4 个 token 的 MTP 窗口，以及 KDA 的全部 decode。
  - 修法：
    - 单 token 的 GDN 步（`replayssm_flushes_ahead`）：每步 commit 之后、各层之前，用一次 launch（`replayssm_flush`）把所有层里满了的窗口折进 checkpoint。层 kernel 遇到满窗口时按空窗口处理：只读 checkpoint，从第 0 行开始追加。两次 launch 按 stream 顺序执行，所以层 kernel 写第 0 行时，已经没有 program 要读旧记录。
    - GDN 的 MTP 窗口和 KDA：flush 留在层 kernel 里。每个 (槽, head) 配一个计数器。flush 步每个 program 等自己发出的读全部返回（`s_waitcnt`）后取一个号，取到最后一个号的 program 写 k/g，再把计数器减回去。计数器在两次 launch 之间总是 0，batch 里有重复的槽也一样。UT kernel 每个 head 只有一个 program，不受影响。
    - 取号用 relaxed 原子加，不用 acq_rel。acq_rel 在 CDNA 上会把整个 L2 写回再失效，所有序列都 flush 的那一步慢 2.4–6.7 倍。这里只要求别的 program 的读先于写，等读返回就够了。
    - 拆出来的 flush 由 builder 在 CUDA graph 之外发起。直接调用 kernel 的地方（例如 sglang）默认每层先自己 flush，不用改。
  - 验证：
    - 修复后，上面每个用例 50 次逐位相同，KDA 每块 V 的误差在 1e-5 量级。脚本现在自带负载（两个 stream 跑矩阵乘），也测 GDN 3 个 token 的窗口。修复前的代码在这个负载下每个 flush 步都出错。
    - 新旧 kernel 在同样输入的多窗口轨迹上：MTP 窗口、UT 和 KDA 的输出、checkpoint 和记录都逐位相同；只有 GDN T=1 有差别，bf16 记录时输出相对差 2e-5，checkpoint 相对差 6e-6。新旧两版和精确递推的误差一样。
    - `leapquant_kda_check.py` 现在在所有窗口上拿 FP32 ReplaySSM KDA 和精确解比，最大相对误差 3.0e-7。
  - 端到端（Qwen3.8）：修复前的竞争没有造成可测的掉点，几组差别都在 ±0.44 的标准误内。
    - GSM8K：两次 97.35、97.50，修复前 97.57。
    - MTP（2 个草稿 token）：GSM8K 97.65，修复前 97.65；接受率 79.0%，修复前 78.9%。
  - 代价：
    - 线性注意力 kernel 合计（36 层 GDN / 20 层 KDA，批量 64–512，flush 时刻错开）：
      - GDN T=1 快 7–11%。拆出去的 flush 要多读一遍 checkpoint，但这个 kernel 去掉 flush 代码以后，本身快了 14–19%。
      - GDN T=3 慢不到 1%。KDA 的 T=1、T=3 都在 ±2% 内，和不 flush 时的测量波动一样。
    - 端到端：
      - Qwen3.8 普通 decode 不比修复前慢。含 flush 的平均步时低 1–5%，但同一份代码测两次就能差到 4.8%。
      - MTP 和修复前持平。两轮的 median ITL 差 −1.0% 到 +1.4%，纯 decode 窗口里每 token 的时间低 0.3–2.4%。吞吐两轮差 −4.4% 到 +0.7%，同一份代码两轮之间就差到 3.4%，主要看最慢的几个请求。
    - 最初的版本把 MTP 窗口和 KDA 的 flush 也拆了出去：GDN T=3 慢 9–15%，KDA 慢 5–13%，MTP 每步慢 1.2–3.6%。
- 一轮等长请求的测试里，所有请求同一步 flush，每 16 步出现一次尖峰。因此主指标用含 flush 的平均步时。
- Kimi-Linear 的窗口记录里，每通道 FP32 门控每请求占 5.6 MB（LeapQuant 状态的 22%）。改存 16 位能再省一半，没有做。

## 复现

```bash
ATOM_ENABLE_LEAPQUANT=1 bash run_atom.sh server-decode   # 基线：不设变量 / SSM_DTYPE=bfloat16 / ATOM_ENABLE_REPLAYSSM=1
bash run_atom.sh smoke                       # NaN 冒烟检查
TAG=leapquant bash run_atom.sh decode        # 测速，结果写到 results/<tag>/c*.json
TAG=leapquant bash run_atom.sh prefix-bench  # 共享前缀的 TTFT，写到 results/<tag>/prefix.json；服务加 PREFIX_CACHE=1 开缓存
TAG=leapquant bash run_atom.sh gsm8k         # 结果写到 results_gsm8k/<tag>/
ATOM_ENABLE_LEAPQUANT=1 bash run_atom.sh server-reason   # 长推理评测用的服务，模式变量同上；PREFIX_CACHE=1 开前缀缓存
TAG=leapquant bash run_atom.sh reason        # AIME/GPQA，结果写到 results_reason/<tag>/
python reason_eval.py --summary              # 汇总，并逐题对比 FP32
python steady_step.py results/*/server.log   # 含 flush 的平均步时
HIP_VISIBLE_DEVICES=6 bash run_atom.sh lq-check   # kernel 正确性（A–E）
bash run_atom.sh lq-bench                         # GDN 每步耗时，HIP_VISIBLE_DEVICES 选一张独占的卡
HIP_VISIBLE_DEVICES=6 bash run_atom.sh lq-prefix-check   # 前缀缓存离线检查
```

测性能要独占两张卡。和别的任务共用 GPU 时，FP32 并发 32 的步时从 15.65 ms 变成 19.78 ms，留给 KV 的空间也少了三分之一。正确性检查反过来，要放在有别的任务在跑的卡上（这台机器上是 GPU 6 的占卡进程）：竞争和不确定的结果在空闲的卡上很少暴露。`replayssm_race_check.py` 自带负载，不受这个限制。

前缀缓存端到端，结果写到 `results_prefix/<tag>.json`。先跑一次开缓存的，把 autotune 选择缓存下来，再跑其余的。和别的任务共用 GPU 时，`EXTRA_ARGS` 里加 `--gpu-memory-utilization 0.999`。

```bash
export ATOM_ENABLE_LEAPQUANT=1
TAG=lq-cache bash run_atom.sh prefix-e2e
TAG=lq-nocache CACHE=0 bash run_atom.sh prefix-e2e
TAG=lq-noladder EXTRA_ARGS="--state-checkpoint-interval-tokens -1" bash run_atom.sh prefix-e2e
PYTHONPATH=$PWD/ablate_replay ABLATE_REPLAY=1 TAG=lq-noreplay bash run_atom.sh prefix-e2e   # 去掉回放
TAGS="lq-cache lq-nocache lq-noreplay lq-noladder" bash run_atom.sh prefix-compare
```

Kimi-Linear，单卡。`run_atom.sh` 的 `kimi-*` 模式按论文的设置（附录 E）起服务和评测，结果写到仓库外的 `../kimi_eval`：

```bash
HIP_VISIBLE_DEVICES=0 KSTATE=fp32 bash run_atom.sh kimi-server   # KSTATE：fp32、bf16、lq、rs
HIP_VISIBLE_DEVICES=1 KSTATE=bf16 SERVER_PORT=8001 ENGINE_PORT=8016 bash run_atom.sh kimi-server
HIP_VISIBLE_DEVICES=2 KSTATE=lq SERVER_PORT=8002 ENGINE_PORT=8026 bash run_atom.sh kimi-server
TAG=fp32 bash run_atom.sh kimi-reason   # AIME、GPQA、MMLU-Pro；temperature 1.0，最长 65,536 token，3 个样本
TAG=bf16 PORT=8001 bash run_atom.sh kimi-reason
TAG=lq PORT=8002 bash run_atom.sh kimi-reason
bash run_atom.sh kimi-summary
TAG=fp32 bash run_atom.sh kimi-decode   # decode 测速，之后自动跑 decode_window.py
```

10-22 和 10-23 白天的测量（temperature 0.6，最长 32,768 token）用的是下面的命令：

```bash
M=/shareddata/haoyanli/moonshotai/Kimi-Linear-48B-A3B-Instruct
# FP32 服务；LeapQuant 在另一张卡上，加 ATOM_ENABLE_LEAPQUANT=1 和 --server-port 8001 --port 8007
HIP_VISIBLE_DEVICES=5 AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server --model $M \
  --trust-remote-code -tp 1 --no-enable_prefix_caching --max-num-seqs 512 --max-model-len 40960
# decode 测速，每个服务跑一遍（PORT、TAG 换成对应的值）
for c in 32 64 128 256 512; do
  python -m atom.benchmarks.benchmark_serving --model=$M --trust-remote-code --backend=vllm \
    --base-url=http://localhost:$PORT --dataset-name=random --random-input-len=4096 \
    --random-output-len=1024 --random-range-ratio=1.0 --num-prompts=$c --max-concurrency=$c \
    --request-rate=inf --ignore-eos --percentile-metrics=ttft,tpot,itl,e2el \
    --save-result --save-detailed --result-dir=results/$TAG --result-filename=c$c.json
done
python decode_window.py results/kimi-fp32-r2 results/kimi-lq-r2   # 含 flush 的平均步时
# AIME/GPQA；GPQA 补到 6 个样本：--tasks gpqa --samples 6，已有的样本会跳过
python reason_eval.py --tag kimi-fp32 --url http://localhost:8000 --model $M --max-tokens 32768 \
  --sampling '{"temperature": 0.6, "top_p": 1.0, "top_k": -1}'
python reason_eval.py --summary --base kimi-fp32 --tags kimi-fp32 kimi-lq
# GSM8K：和 run_atom.sh 的 gsm8k 一样，model 换成 $M
# 和 HF 参考对比，用法见 kimi_ref_check.py 开头；--long 8 是长回复
# 前缀缓存；不跑 GSM8K 时 EXTRA_ARGS="--n-gsm8k 0"
SMALL_MODEL=$M ATOM_ENABLE_LEAPQUANT=1 TAG=kimi-lq-cache bash run_atom.sh prefix-e2e
SMALL_MODEL=$M ATOM_ENABLE_LEAPQUANT=1 TAG=kimi-lq-nocache CACHE=0 bash run_atom.sh prefix-e2e
SMALL_MODEL=$M ATOM_ENABLE_LEAPQUANT=1 PYTHONPATH=$PWD/ablate_replay ABLATE_REPLAY=1 \
  TAG=kimi-lq-noreplay bash run_atom.sh prefix-e2e
SMALL_MODEL=$M ATOM_ENABLE_REPLAYSSM=1 TAG=kimi-rs-cache bash run_atom.sh prefix-e2e
SMALL_MODEL=$M TAG=kimi-fp32-nocache CACHE=0 bash run_atom.sh prefix-e2e
TAGS="kimi-lq-cache kimi-lq-noreplay kimi-lq-nocache kimi-rs-cache kimi-fp32-nocache" \
  bash run_atom.sh prefix-compare   # 末尾按 +1、+41 分开汇总
# 离线：正确性检查放在 GPU 6，测速用独占的卡
HIP_VISIBLE_DEVICES=6 python leapquant_kda_check.py    # A–E
python leapquant_kda_bench.py                          # KDA 每步耗时，--breakdown 分 kernel
python replayssm_race_check.py                         # ReplaySSM flush 的确定性，自带负载，任何一张卡都行
```

ReplaySSM 修复 flush 竞争前后的对比。`fused_flush/` 让服务换回修复前的 kernel，不改仓库：

```bash
PYTHONPATH=$PWD/fused_flush FUSED_FLUSH=1 ATOM_ENABLE_REPLAYSSM=1 bash run_atom.sh server-decode   # 修复前
ATOM_ENABLE_REPLAYSSM=1 bash run_atom.sh server-decode                                             # 修复后
# 两种都可以加 EXTRA_ARGS="--method mtp --num-speculative-tokens 2" 开 MTP
TAG=rs-split bash run_atom.sh decode   # MTP 时加 CONC="128 256 512"
TAG=rs-split bash run_atom.sh gsm8k
```

## 2026-10-22 记录

用当前代码（fold 修复 + 前缀缓存支持，未提交）在 Qwen3.8-Flash-Next 上重测。细节已并入上面各节，这里汇总。

- **精度**：LeapQuant 修复后 AIME 95.6、GPQA 91.6；开前缀缓存 96.7、89.9。FP32 是 95.6、90.4，没有可测的差别。
- **decode 步时**：并发 128/256/512 时，LeapQuant 是 FP32 的 1.08×/1.09×/1.13×，开缓存也一样。新代码没有可测的开销。
- **共享前缀**：开缓存后，LeapQuant 的平均 TTFT 从 4.77 s 降到 0.24 s，吞吐从 3.05 提到 15.2 请求/秒，和 FP32 一样。
- **前缀缓存第一次在大模型上验证**（TP2+EP，CUDA graph）：AIME/GPQA 期间打了 2136 个 checkpoint，decode 途中分叉约 1400 次，没有 NaN，也没有报错。
- **数据和日志**：
  - `results_reason/leapquant_fix/`、`results_reason/leapquant_cache/`：`eval.log`（末尾附汇总）和 `server.log`。
  - `results/` 下 `fp32_rerun`、`leapquant_fix`、`fp32_cache`、`leapquant_cache`：`bench.log`（smoke、decode 扫描、共享前缀，末尾附汇总）和 `server.log`。
  - `results_ab/`：提交前的代码和当前代码交替跑的 decode 步时。

## 2026-10-23 记录

修复 ATOM 原有 ReplaySSM 的 flush 竞争，未提交。原理、验证和代价见“限制”。

- **改动**：
  - `replayssm.py`：新增 `replayssm_flush` 和 `replayssm_flushes_ahead`，单 token 的 GDN 步不再在层 kernel 里 flush。serial 和 KDA kernel 的 flush 步由 `_last_reader` 选出写 k/g 的 program。
  - `gdn_attn.py` 的 builder 在 commit 之后，单 token 的 GDN 步对所有层做一次 flush。
  - `attention_gdn.py` 的 ReplaySSM 调用加了 `fold=False`。
  - 测试：`test_kda_checkpoint_slot_copy.py`、`test_kda_layout_id.py`、`test_qwen4_exp_mtp.py` 的 stub 补上了 LeapQuant 和 ReplaySSM 的新属性，修好之前 LeapQuant/KDA 改动带来的 33 个失败。剩下的 13 个失败在这些改动之前就有。
- **decode 步时**（Qwen3.8，GPU 4、5，修复前 → 修复后，ms）：
  - 普通 decode，含 flush 的平均步时：并发 128 为 27.63 → 26.35，256 为 36.45 → 35.95，512 为 55.71 → 55.06。修复后的代码测了两次，median ITL 最多差 4.8%。
  - MTP，每步时间（median ITL，修复后两轮）：128 为 49.68 → 49.20 / 50.24，256 为 71.74 → 71.70 / 71.99，512 为 110.05 → 111.62 / 111.30。MTP 窗口的 flush 也拆出去时是 51.49、72.63、112.14。
- **数据和日志**：
  - `results/rs-{fused,split}{,-mtp2}/`、`results/rs-split-r2/`、`results/rs-hybrid-mtp2{,-r2}/`：decode 测速和 `server.log`。`rs-split*` 的普通 decode 和现在的代码是同一套 kernel；`rs-split-mtp2` 是 MTP 窗口的 flush 也拆出去的版本。
  - `results_gsm8k/rs-*`。
  - `results_prefix/kimi-rs-cache-{split,hybrid,hybrid-r2}.json`。
  - `logs_rs/`：服务、GSM8K 和测速的日志。

## 2026-10-23 Kimi-Linear 重测

用当前代码（ReplaySSM 竞争修复之后）在 Kimi-Linear-48B 上重测性能和精度，并加了每步存 BF16 状态（`ATOM_GDN_SSM_DTYPE=bf16`）作对照。单卡 TP1，CUDA graph，不开前缀缓存，512 个槽，服务命令和“复现”一节相同。

- **结论**：
  - LeapQuant 比 FP32 快 3–12%，状态少 42%；GSM8K、AIME 没有可测的差别，GPQA 低 1.7 分。
  - GPQA 的下降和 10-22 那次（低 1.9 分）一致，两次合起来 p≈0.016，约 2 分的下降应该是真的。
  - BF16 状态几乎一样快（LeapQuant 只快约 1%），状态更小（每请求 22.4 MB 对 25.1 MB），GPQA 也不掉点。在测过的 GSM8K、AIME、GPQA 上，LeapQuant 相对 BF16 没有显出收益。论文里 BF16 在 Kimi 上掉得最多的是 MMLU-Pro（72.4 → 64.0），这里没有测。按论文设置重测后，LeapQuant 相对 BF16 有收益，见“2026-10-24 Kimi-Linear 按论文设置重测”。
- **KDA kernel**（20 层合计，每步，ms）：

| B | FP32 | BF16 | ReplaySSM | LeapQuant | LeapQuant 对 FP32 | LeapQuant 对 BF16 |
|---|---|---|---|---|---|---|
| 32 | 0.68 | 0.44 | 0.61 | 0.42 | 1.61× | 1.05× |
| 64 | 1.25 | 0.82 | 1.13 | 0.75 | 1.68× | 1.09× |
| 128 | 2.42 | 1.47 | 1.99 | 1.35 | 1.79× | 1.09× |
| 256 | 4.72 | 2.74 | 3.84 | 2.55 | 1.85× | 1.07× |
| 512 | 9.45 | 5.37 | 7.66 | 5.08 | 1.86× | 1.06× |

  FP32 比 10-22 慢（B=512 时 9.45 对 8.51 ms），LeapQuant 只慢 2%，所以倍数比上次的 1.52–1.77× 大。

- **decode 每步耗时**（ms，含 flush 的平均步时，`decode_window.py`）：四种模式在同一张卡上轮流起服务，跑两轮，第二轮倒序，取平均。括号里是对 FP32 的倍数。

| 并发 | FP32 | BF16 | ReplaySSM | LeapQuant |
|---|---|---|---|---|
| 32 | 8.28 | 8.00 (1.04×) | 8.07 (1.03×) | 8.04 (1.03×) |
| 64 | 10.81 | 10.13 (1.07×) | 10.42 (1.04×) | 9.97 (1.08×) |
| 128 | 14.90 | 13.64 (1.09×) | 13.63 (1.09×) | 13.48 (1.11×) |
| 256 | 22.28 | 20.52 (1.09×) | 21.27 (1.05×) | 20.25 (1.10×) |
| 512 | 33.16 | 30.23 (1.10×) | 32.39 (1.02×) | 29.74 (1.12×) |

  - 两轮之间相差：FP32、BF16、LeapQuant 不超过 2.2%，ReplaySSM 3.9%。
  - 不含 flush（中位数）：LeapQuant 1.04–1.16×，BF16 1.04–1.10×。
  - LeapQuant 和 10-22（1.05–1.12×）一致。

- **显存**：

| 模式 | 每请求状态 | 512 个槽的状态池 | KV block | 上下文 40960 时的并发上限 |
|---|---|---|---|---|
| FP32 | 43.4 MB | 20.7 GB | 114.7 万 | 447 |
| BF16 | 22.4 MB | 10.7 GB | 123.0 万 | 480 |
| ReplaySSM | 53.9 MB | 25.7 GB | 110.5 万 | 431 |
| LeapQuant | 25.1 MB | 12.0 GB | 121.9 万 | 476 |

  上下文更短时（FP32 不超过 28K，BF16 和 LeapQuant 不超过 36K）都是 512 个槽先用完，省下的显存不增加并发。

- **精度**：GSM8K 全部 1319 题，5-shot chat。AIME 2026 和 GPQA-Diamond 每题 6 个样本，temperature 0.6，最长 32,768 token，并发 128。

| 模式 | GSM8K | AIME 2026 | GPQA-Diamond | GPQA（截断算错） | GPQA 截断数 |
|---|---|---|---|---|---|
| FP32 | 87.26 | 49.4 | 53.4 | 49.9 | 434 |
| BF16 | 87.19 | 44.4 | 54.1 | 49.7 | 403 |
| LeapQuant | 86.50 | 48.9 | 51.7 | 47.6 | 461 |

  - 逐题对比 FP32（更差/更好的题数）：
    - GSM8K：BF16 25/24，LeapQuant 32/22。
    - AIME：BF16 11/6（p=0.33），LeapQuant 12/8（p=0.50）。
    - GPQA：BF16 52/52，LeapQuant 63/44（p=0.081）。
  - GPQA 上 LeapQuant 直接对 BF16：70/46（p=0.032）。同一题同一样本只有一方截断的，BF16 135 对、LeapQuant 193 对（p=0.002）。
  - LeapQuant 的 GPQA 分成前后各 3 个样本：这次低 1.0、2.4 分，10-22 低 2.6、1.2 分，四次都更低，平均低 1.8 分。两次运行按 Stouffer 合并，p≈0.016。
  - AIME 的绝对分数比 10-22 低（FP32 49.4 对 57.8），而 FP32 这条路径的代码没变过。30 道题里有 16 道在 6 个样本里有对有错，3 个样本的平均分标准差约 3.5 分，两次相差约 2.5 个标准差。三种模式是在同一次运行里比的，这个对比不受影响。

- **和论文对比**（表 1，Kimi-Linear-48B-A3B，3 个种子）：FP32 的 AIME、GPQA 是 67.5、70.3，BF16 是 64.3、68.1，LeapQuant 是 68.3、69.8。
  - BF16 一致：论文也说 Kimi 的输出短，对逐步存储不敏感，AIME、GPQA、LiveCodeBench 上都在 5 分以内。
  - LeapQuant 的 AIME 一致。GPQA 论文低 0.5 分，这里两次都低约 1.8 分。按这里的逐题正确率估算，3 个种子时两种方法之差的标准差约 1.9 分（AIME 约 4.9 分），论文的数据排除不了这个幅度。
  - 设置不同：论文用 temperature 1.0、最长 65,536 token，这里用 0.6、32,768 token。这里 GPQA 37%、AIME 46% 的回复写满上限，绝对分数比论文低约 17 分。LeapQuant 截断得更多，截断算错时差距是 2.3 分。按论文设置的结果见下一节。

- **改动**：`kimi_k3.py` 的 prefill 写回状态时转换成状态池的 dtype。aiter 的 chunk KDA 不管初始状态是什么 dtype 都返回 fp32，原来的代码在 BF16 池上第一次 prefill 就报错退出；FP32 池上这个转换什么都不做。
- **数据和日志**：`/home/haoyanli/o/lhy_atom/kimi_eval/`，不在这个仓库里。
  - `results/<模式>-r{1,2}/`：decode 测速。
  - `results_gsm8k/`、`results_reason/`：精度。
  - `logs/`：服务和评测的日志；`kda_bench.log` 是 kernel 计时。
  - 脚本：`serve.sh`（起服务）、`drive.sh`（起服务、跑任务、停服务）、`perf.sh`、`sweep.sh`、`accuracy.sh`、`kda_bench.py`（`leapquant_kda_bench.py` 加了 BF16）。

## 2026-10-24 Kimi-Linear 按论文设置重测

按论文附录 E 的设置重测 Kimi-Linear 的精度：temperature 1.0、top_p 1.0、不限 top_k、最长 65,536 token（模型长度 73,728），每题 3 个样本。题目是 AIME 2026 30 题、GPQA-Diamond 198 题、MMLU-Pro 全部 12,032 题。每种状态一张卡，并发 256（MMLU-Pro 后段 512），命令是 `run_atom.sh` 的 `kimi-*` 模式。LiveCodeBench 本地没有数据和判题环境，没有测。

| 模式 | AIME 2026 | GPQA-Diamond | MMLU-Pro | 论文（AIME / GPQA / MMLU-Pro） |
|---|---|---|---|---|
| FP32 | 62.2 | 69.5 | 71.7 | 67.5 / 70.3 / 72.4 |
| BF16 | 62.2 | 66.5 | 71.4 | 64.3 / 68.1 / 64.0 |
| LeapQuant | 71.1 | 67.3 | 71.9 | 68.3 / 69.8 / 72.1 |

- **结论**：
  - FP32 和论文一致。LeapQuant 和 FP32 没有可测的差别。
  - BF16 在 MMLU-Pro 上只低 0.3 分，没有复现论文的 8.4 分；但三项上都更容易写满上限。
  - 相对 BF16，LeapQuant 在 MMLU-Pro 上显著更好，写满上限的也少得多。10-23 “没有显出收益”的结论只在 temperature 0.6 的设置下成立。
- **逐题对比**（更差/更好的题数）：

| 对比 | AIME | GPQA | MMLU-Pro |
|---|---|---|---|
| BF16 对 FP32 | 7/7 | 50/35（p=0.13） | 1902/1779（p=0.044） |
| LeapQuant 对 FP32 | 4/9（p=0.27） | 47/39（p=0.45） | 1763/1855（p=0.13） |
| LeapQuant 对 BF16 | 6/10（p=0.45） | 40/44 | 1748/1928（p=0.003） |

- **写满 65,536 token**（共 90 / 594 / 36,096 条）：FP32 6 / 11 / 216，BF16 14 / 26 / 444，LeapQuant 4 / 12 / 195。
  - 同一题同一样本只有一方写满的：BF16 对 FP32 是 10 对 2、22 对 7、366 对 138（p=0.039、0.008、7e-25）；LeapQuant 对 FP32 是 3 对 5、10 对 9、155 对 176，没有差别。
  - BF16 的平均输出也更长，三项分别多 13%、10%、7%。这和论文说的 BF16 让推理变长一致，只是在 Kimi 上对分数的影响小得多。
- **LeapQuant 的 GPQA**：这次低 2.2 分，但 3 个样本时差值的标准差约 1.9 分，单独不显著。加上 temperature 0.6 的两次（低 1.9、1.7 分），三次都更低。
- **为什么这次分数比 temperature 0.6、32K 时高**：主要是温度，其次是输出上限。
  - 拆分方法：把这次超过 32K 的回复都算错，就相当于“1.0 温度、32K 上限”。下表都是截断算错的分数：

| 模式 | 任务 | 0.6、32K | 1.0、32K（模拟） | 1.0、64K |
|---|---|---|---|---|
| FP32 | GPQA | 49.9 | 63.1 | 69.4 |
| BF16 | GPQA | 49.7 | 60.8 | 65.2 |
| LeapQuant | GPQA | 47.6 | 62.1 | 67.2 |
| FP32 | AIME | 49.4 | 55.6 | 62.2 |

  - 温度贡献 GPQA 的 11–15 分。0.6 时 GPQA 有 34–39%、AIME 有 43–46% 的回复写满 32K，其中 GPQA 94%、AIME 85% 是在原地重复同一段话。判断标准是：尾部 300 个字符在全文出现 3 次以上，或尾部 2 万字符的 zlib 压缩率低于 0.15。
    - 例如 GPQA 第 1 题的一个回复，最后 160 个字符在全文出现了 95 次，第一次出现在全文 11% 处，后面近九成的 token 都在重复。
    - 温度低时采样接近贪心，一段话一旦开始重复，就很难再跳出来，这是长生成里常见的退化。1.0 的随机性能打断它：这次写满 64K 的只有 2%（GPQA）和 7%（AIME），而且大多不是重复。
  - 上限贡献 GPQA 的 4–6 分。1.0 时回复更长：正常结束的 GPQA 回复中位数是 7.5K token，0.6 时是 3.8K。15–17% 的 GPQA 回复超过 32K，FP32 的 86 条里有 37 条最后答对。
  - 0.6 时正常结束的回复其实更准（FP32 的 GPQA 78.6% 对 70.7%，AIME 90.8% 对 66.7%）。一方面打转的多是难题，留下的偏简单：0.6 时 6 个样本都没截断的 67 道 GPQA 题，1.0 时正确率 83.1%；截断 4–6 次的 57 道只有 50.9%。另一方面低温度本身更稳。但这点收益远小于打转丢掉的分数。
  - 0.6 不是 Kimi-Linear 模型目录给的（目录里没有采样参数），论文用的是 1.0。
  - 打转也放大了 0.6 时各状态之间的差别。那时 LeapQuant 的 GPQA 下降有一部分来自更多截断（只有一方截断的 154 对 127）；1.0 时两者截断一样多（10 对 9）。
- **中途中断**：10-23 16:02（UTC）`/home/haoyanli` 写满，客户端和服务写不了文件，全部退出。AIME、GPQA 已经跑完，MMLU-Pro 完成的部分也都在。输出搬到 `/shareddata/haoyanli/kimi_eval` 后，MMLU-Pro 以 512 并发续跑完成，进程改在独立会话（`setsid`）里运行。
- **数据和日志**：`/shareddata/haoyanli/kimi_eval/`，不在这个仓库里。
  - `results_reason_paper/<模式>/`：每条结果和完整回复。
  - `logs/`：服务和评测的日志。
  - `analyze_paper.py`：上面的各项统计；`why_higher.py`：和 temperature 0.6 那次的拆分、打转检测；`resume_paper.sh`：续跑。
- **改动**：
  - `reason_eval.py` 加了 MMLU-Pro（`--tasks mmlupro`；`--mmlupro-n` 取固定顺序的前 N 题）和 `--out`。
  - `run_atom.sh` 加了 `kimi-server`、`kimi-reason`、`kimi-summary`、`kimi-decode`、`kimi-gsm8k`。
