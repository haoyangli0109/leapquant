#!/bin/bash
# Qwen3.8-Flash-Next BF16 (336 GB) needs two MI355X. Pure TP2 fails MoE warmup on
# gfx950 (no BF16 ck_moe_stage1 kernel for the 320-wide per-rank intermediate);
# EP keeps every expert at its full 640 width.
#
#   bash run_atom.sh server   # OpenAI server on GPUs 4,5 (port 8000)
#   bash run_atom.sh bench    # from another terminal, once the server is up
#
# High-concurrency decode experiments, in the paper's decode-step setup (4K context):
#   bash run_atom.sh server-decode                      # FP32 state, the baseline
#   SSM_DTYPE=bfloat16 bash run_atom.sh server-decode   # half the state bytes
#   ATOM_ENABLE_REPLAYSSM=1 bash run_atom.sh server-decode
#   ATOM_ENABLE_LEAPQUANT=1 bash run_atom.sh server-decode   # int8 state, p=16, r=4
#   TAG=fp32 bash run_atom.sh decode   # sweep; CONC="256 512" for a subset
#   bash run_atom.sh summary           # decode step time of every TAG side by side
#   bash run_atom.sh smoke             # 10 greedy replies from the running server, NaN check
#   TAG=fp32 bash run_atom.sh gsm8k    # accuracy of the running server, 5-shot chat
#   python steady_step.py server.log   # mean step time incl. flushes, from the server's stats
#
# AIME 2026 / GPQA-Diamond (thinking mode, 3 samples, up to 81,920 tokens), two
# servers side by side on GPUs 4-7. Start the second only once the first is up:
# each start clears the compile cache.
#   bash run_atom.sh server-reason                                   # GPUs 4,5, :8000
#   HIP_VISIBLE_DEVICES=6,7 SERVER_PORT=8001 ENGINE_PORT=8016 \
#     ATOM_ENABLE_LEAPQUANT=1 bash run_atom.sh server-reason         # GPUs 6,7, :8001
#   TAG=fp32 bash run_atom.sh reason
#   TAG=leapquant URLS="--url http://localhost:8001" bash run_atom.sh reason
#   python reason_eval.py --summary
#   HIP_VISIBLE_DEVICES=6 bash run_atom.sh gdn        # GDN decode kernel alone
#   HIP_VISIBLE_DEVICES=6 bash run_atom.sh lq-check   # LeapQuant kernels vs torch / FP32
#   HIP_VISIBLE_DEVICES=6 bash run_atom.sh lq-bench   # GDN cost per step: FP32 / ReplaySSM / LeapQuant
#
# Prefix cache. The server modes run without it; PREFIX_CACHE=1 turns it on.
#   HIP_VISIBLE_DEVICES=6 bash run_atom.sh lq-prefix-check   # checkpoints and resumes, offline
#   HIP_VISIBLE_DEVICES=6 ATOM_ENABLE_LEAPQUANT=1 TAG=lq-cache bash run_atom.sh prefix-e2e
#   HIP_VISIBLE_DEVICES=7 ATOM_ENABLE_LEAPQUANT=1 TAG=lq-nocache CACHE=0 bash run_atom.sh prefix-e2e
#   TAGS="lq-cache lq-nocache" bash run_atom.sh prefix-compare
#   TAG=lq-pc bash run_atom.sh prefix-bench   # shared-prefix TTFT on a running server-decode
# prefix-e2e runs Qwen3.5-9B on one GPU. On a GPU other jobs share, add
# EXTRA_ARGS="--gpu-memory-utilization 0.999": ATOM counts their memory as used.
set -e
export AITER_LOG_LEVEL=WARNING
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5}
MODEL=/shareddata/haoyanli/Qwen/Qwen3.8-Flash-Next
HERE=$(cd "$(dirname "$0")" && pwd)
NO_CACHE=$([ "${PREFIX_CACHE:-0}" = 1 ] || echo --no-enable_prefix_caching)

summarize() {
  python - "$HERE/results" <<'EOF'
import json, pathlib, sys

runs = sorted(p for p in pathlib.Path(sys.argv[1]).iterdir() if p.is_dir())
concs = sorted({int(f.stem[1:]) for d in runs for f in d.glob("c*.json")})
# Median ITL is the step time of the batch-wide decode; the mean also carries
# the steps that a flush lands on, should many requests flush together.
for key in ("median_itl_ms", "mean_itl_ms"):
    print(f"decode step = {key} (ms)")
    print(f"{'conc':>6}" + "".join(f"{d.name:>12}" for d in runs))
    for c in concs:
        cells = []
        for d in runs:
            f = d / f"c{c}.json"
            ms = json.loads(f.read_text())[key] if f.exists() else None
            cells.append(f"{ms:12.2f}" if ms is not None else f"{'-':>12}")
        print(f"{c:>6}" + "".join(cells))
EOF
}

case "${1:-server}" in
server)
  rm -rf /root/.cache/atom/*
  python -m atom.entrypoints.openai_server \
    --model $MODEL --trust-remote-code \
    -tp 2 --enable-expert-parallel
  ;;
bench)
  for c in 32 128 256 512; do
    python -m atom.benchmarks.benchmark_serving \
      --model=$MODEL --backend=vllm --base-url=http://localhost:8000 \
      --dataset-name=random \
      --random-input-len=4096 --random-output-len=1024 --random-range-ratio=0.8 \
      --num-prompts=$((c * 4)) --max-concurrency=$c \
      --request-rate=inf --ignore-eos \
      --save-result --percentile-metrics="ttft,tpot,itl,e2el"
  done
  ;;
server-decode)
  rm -rf /root/.cache/atom/*
  python -m atom.entrypoints.openai_server \
    --model $MODEL --trust-remote-code \
    -tp 2 --enable-expert-parallel \
    $NO_CACHE \
    --hf-overrides "{\"mamba_ssm_dtype\": \"${SSM_DTYPE:-float32}\"}" \
    ${EXTRA_ARGS:-}
  ;;
decode)
  out=$HERE/results/${TAG:?set TAG to name this run, e.g. TAG=fp32}
  mkdir -p $out
  # One wave of equal-length requests per concurrency: after the prefill burst
  # every step decodes exactly $c requests, so the median ITL is the step time.
  for c in ${CONC:-32 64 128 256 512}; do
    python -m atom.benchmarks.benchmark_serving \
      --model=$MODEL --backend=vllm --base-url=http://localhost:8000 \
      --dataset-name=random \
      --random-input-len=4096 --random-output-len=1024 --random-range-ratio=1.0 \
      --num-prompts=$c --max-concurrency=$c \
      --request-rate=inf --ignore-eos \
      --percentile-metrics="ttft,tpot,itl,e2el" \
      --save-result --result-dir=$out --result-filename=c$c.json
  done
  summarize
  ;;
prefix-bench)
  # Prompts that share an 8448-token prefix, then 512 tokens of their own. GDN
  # state is only checkpointed on the 8192-token ladder (and at prompt ends),
  # so 8192 is the most a hit can cover; the extra 256 keep that rung clear of
  # where the shared text meets each prompt's own and its tokens could merge.
  out=$HERE/results/${TAG:?set TAG to name this run}
  mkdir -p $out
  python -m atom.benchmarks.benchmark_serving \
    --model=$MODEL --backend=vllm --base-url=http://localhost:8000 \
    --dataset-name=random --random-prefix-len=8448 \
    --random-input-len=512 --random-output-len=128 --random-range-ratio=1.0 \
    --num-prompts=128 --max-concurrency=32 \
    --request-rate=inf --ignore-eos \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --save-result --result-dir=$out --result-filename=prefix.json
  ;;
summary)
  summarize
  ;;
smoke)
  python $HERE/smoke_5050.py
  ;;
gsm8k)
  # As in ATOM/recipes/Qwen3.8-Flash-Next.md: all 1319 questions; the stop
  # string keeps GSM8K's "Question:" from cutting a reasoning response short.
  HF_HUB_CACHE=/shareddata/huggingface/hub HF_DATASETS_CACHE=/shareddata/huggingface/datasets \
  lm_eval --model local-chat-completions \
    --model_args "model=$MODEL,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=${NUM_CONCURRENT:-128},max_retries=3,timeout=1800,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k --num_fewshot 5 \
    --apply_chat_template --fewshot_as_multiturn \
    --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
    --log_samples --output_path $HERE/results_gsm8k/${TAG:?set TAG to name this run}
  ;;
server-reason)
  # As server-decode, with fewer state slots (clients keep <= 128 requests in
  # flight per server) so that more memory goes to the long replies' KV.
  rm -rf /root/.cache/atom/*
  python -m atom.entrypoints.openai_server \
    --model $MODEL --trust-remote-code \
    -tp 2 --enable-expert-parallel \
    $NO_CACHE \
    --max-num-seqs 256 \
    --server-port ${SERVER_PORT:-8000} --port ${ENGINE_PORT:-8006} \
    --hf-overrides "{\"mamba_ssm_dtype\": \"${SSM_DTYPE:-float32}\"}" \
    ${EXTRA_ARGS:-}
  ;;
reason)
  cd $HERE && python reason_eval.py --tag ${TAG:?set TAG to name this run} ${URLS:---url http://localhost:8000}
  ;;
gdn)
  python $HERE/gdn_decode_bench.py
  ;;
lq-check)
  cd $HERE && python leapquant_check.py
  ;;
lq-bench)
  cd $HERE && python leapquant_bench.py --breakdown
  ;;
lq-prefix-check)
  cd $HERE && python leapquant_prefix_check.py
  ;;
prefix-e2e)
  # A 256-token checkpoint interval, so that replies take checkpoints between
  # decode steps and follow-ups resume from them. Checkpoints share the state
  # pool (max-num-seqs entries) with running requests, so GSM8K goes in waves
  # of 8 and leaves the rest to them. Autotune choices are cached, so that two
  # runs differ only in what they configure.
  cd $HERE && TRITON_CACHE_AUTOTUNING=${TRITON_CACHE_AUTOTUNING:-1} \
    python prefix_e2e.py --tag ${TAG:?set TAG to name this run} --gsm8k-wave 8 \
    --model ${SMALL_MODEL:-/shareddata/haoyanli/Qwen/Qwen3.5-9B} --trust-remote-code \
    --enforce-eager --max-num-seqs 64 --max-num-batched-tokens 4096 --max-model-len 4096 \
    --state-checkpoint-interval-tokens 256 \
    $([ "${CACHE:-1}" = 1 ] || echo --no-enable_prefix_caching) \
    ${EXTRA_ARGS:-}
  ;;
prefix-compare)
  cd $HERE && python prefix_e2e.py --compare ${TAGS:?set TAGS, e.g. TAGS=\"lq-cache lq-nocache\"}
  ;;
*)
  echo "usage: $0 [server|bench|server-decode|decode|summary|smoke|gsm8k|server-reason|reason|gdn|lq-check|lq-bench|lq-prefix-check|prefix-e2e|prefix-compare]" >&2
  exit 1
  ;;
esac
# 为什么这样设计：
#
# 和论文的 decode 测速条件对齐：上下文 4K（输入 4096、输出 1024），最高并发 512，和论文 B200 用的 bs=512 一样。
# 每档并发只发一轮等长请求：prefill 结束后，每步都正好 decode c 个请求，所以取 median ITL 作为每步耗时。mean TPOT 里混进了 prefill 阶段的等待，会偏大，比如并发 512 时是 87 ms，而 median ITL 是 59 ms。
# 关闭前缀缓存：随机 prompt 用不上前缀缓存，开着还会多写 GDN checkpoint，干扰测量。
# TAG 必须填：防止误覆盖已有结果。结果存成 results/<TAG>/c<并发>.json。
# gdn 模式：就是上次那张表的分子，用"GDN kernel 耗时 / 每步耗时"就能算出占比。
