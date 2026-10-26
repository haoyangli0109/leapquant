#!/bin/bash
# Qwen3.8-Flash-Next 原始模型（完整 48 层 BF16，2×MI355X，TP2+EP，不开 MTP）上，
# 三种线性注意力状态的性能和精度对比：
#   native     原生推理：FP32 状态，每个 decode 步读写完整状态（基线）
#   replayssm  ReplaySSM：FP32 checkpoint + 每个 token 一条 (k, u, g) 记录，记录攒满才 flush
#   leapquant  LeapQuant：在 ReplaySSM 上把 checkpoint 换成 int8 残差 + FP16 补偿 token
#
# 参数按论文（第 4.1 节 Setups、第 4.3 节 Decode speedup、附录 E）：
#   LeapQuant  窗口 p=16，r=4 个 FP16 补偿 token，FP32 平滑，int8 残差
#   性能  decode 每步耗时，上下文 4K：输入 4096、输出 1024，并发 32–512，开 CUDA graph。
#         每档并发只发一轮等长请求，prefill 结束后每步正好 decode c 个请求。
#         同一组卡上依次起三个服务，跑两轮，第二轮倒序，抵消机器漂移。
#   精度  AIME 2026、GPQA-Diamond：每题 3 个样本，thinking 模式，采样按模型卡
#         （temperature 1.0、top_p 0.95、top_k 20、presence 0），最长输出 81,920 token，截断算错；
#         max-model-len = 81,920 + 8,192，生成不会先撞上上下文长度。
#         GSM8K：全部 1319 题，5-shot chat，max_gen_toks 4096，和以前的结论可比。
#         MMLU-Pro 可选：TASKS="aime26 gpqa mmlupro"，MMLUPRO_N=<题数> 取固定随机子集。
#         LiveCodeBench 需要代码沙箱，这里不测。
#   三种模式都关前缀缓存、状态用 FP32（mamba_ssm_dtype=float32），其余参数完全相同。
#
# 用法（结果写到 $QOUT，默认 ../qwen_eval/compare_naive，不进 paper 仓库）：
#   nohup bash compare_naive.sh all > ../qwen_eval/compare_naive.log 2>&1 &   # 性能 → 精度 → 汇总
#   bash compare_naive.sh perf       # 只测性能，用 PERF_GPUS（默认 4,5），要独占
#   bash compare_naive.sh accuracy   # 只测精度，ACC_GPUS 三组卡并行（默认 "0,1 2,3 4,5"）
#   bash compare_naive.sh summary    # 汇总表
# 中断后重跑同一个命令会接着跑：测完的性能轮次和 GSM8K 跳过，AIME/GPQA 跳过已完成的样本。
#
# 其他可调的环境变量：CONC（并发档位）、TASKS、SAMPLES、GSM8K=0（不测 GSM8K）、
# ATOM_ROOT（用哪份 ATOM 代码，默认 ../ATOM）、EXTRA_ARGS（追加给三个服务的参数）。
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
MODEL=/shareddata/haoyanli/Qwen/Qwen3.8-Flash-Next
ATOM_ROOT=${ATOM_ROOT:-$(dirname "$HERE")/ATOM}
QOUT=${QOUT:-$(dirname "$HERE")/qwen_eval/compare_naive}
MODES="native replayssm leapquant"
PERF_GPUS=${PERF_GPUS:-4,5}
ACC_GPUS=${ACC_GPUS:-"0,1 2,3 4,5"}
CONC=${CONC:-"32 64 128 256 512"}
TASKS=${TASKS:-"aime26 gpqa"}
SAMPLES=${SAMPLES:-3}
MAX_TOKENS=81920
SAMPLING='{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}'

export AITER_LOG_LEVEL=WARNING
# cwd 是 paper，里面没有 atom 包，所以 python -m 会从 PYTHONPATH 里的 $ATOM_ROOT 导入
export PYTHONPATH=$ATOM_ROOT${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_CACHE=/shareddata/huggingface/hub HF_DATASETS_CACHE=/shareddata/huggingface/datasets
LOGS=$QOUT/logs
mkdir -p "$LOGS"

log() { echo "$(date '+%F %T') $*"; }

# 三种模式只差这几个环境变量。先清掉可能从外面带进来、会改变 GDN 路径的变量
mode_env() {
  unset ATOM_ENABLE_REPLAYSSM ATOM_ENABLE_LEAPQUANT ATOM_LEAPQUANT_WINDOW ATOM_LEAPQUANT_RANK \
    ATOM_REPLAYSSM_CACHE_LEN ATOM_REPLAYSSM_ROUTE ATOM_ENABLE_GDN_DECODE_LOSSY_FAST \
    ATOM_GDN_PREFILL_BACKEND ATOM_GDN_DECODE_BACKEND ATOM_GDN_SSM_DTYPE
  case $1 in
    native) export ATOM_ENABLE_REPLAYSSM=0 ;;
    replayssm) export ATOM_ENABLE_REPLAYSSM=1 ;;
    leapquant) export ATOM_ENABLE_LEAPQUANT=1 ATOM_LEAPQUANT_WINDOW=16 ATOM_LEAPQUANT_RANK=4 ;;
    *) echo "unknown mode $1" >&2; return 1 ;;
  esac
}

# builder 在服务日志里打印的模式必须和要求一致，否则这组结果不算
mode_ok() {
  case $1 in
    native) grep -q "ReplaySSM disabled for linear attention" "$2" && ! grep -q "LeapQuant enabled" "$2" ;;
    replayssm) grep -q "ReplaySSM enabled for linear attention" "$2" && ! grep -q "LeapQuant enabled" "$2" ;;
    leapquant) grep -q "LeapQuant enabled for linear attention" "$2" ;;
  esac
}

# 等这几张卡的显存空出来，即上一个服务的进程都退干净了
wait_gpus() {
  for _ in $(seq 1 60); do
    HIP_VISIBLE_DEVICES=$1 python - <<'EOF' 2>/dev/null && return 0
import sys, torch
n = torch.cuda.device_count()
sys.exit(0 if n and all(torch.cuda.mem_get_info(i)[0] > 250 * 2**30 for i in range(n)) else 1)
EOF
    sleep 10
  done
  log "GPUs $1 still busy"
  return 1
}

ptree() { echo "$1"; local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do ptree "$c"; done; }

# 停服务：先 SIGINT 让它正常退出；3 分钟后还没退完，就把整棵进程树 SIGKILL
stop_server() {
  local pids p
  pids=$(ptree "$1")
  kill -INT "$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || break; sleep 3; done
  for p in $pids; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
  wait "$1" 2>/dev/null
  return 0
}

# start_server MODE KIND GPUS SPORT EPORT LOG：后台起服务，PID 放进 SERVER_PID
start_server() {
  local mode=$1 kind=$2 gpus=$3 sport=$4 eport=$5 slog=$6 extra
  case $kind in
    # 性能：512 个状态槽位，正好是最高并发
    perf) extra=(--max-num-seqs 512) ;;
    # 精度：每个服务最多 128 个请求在跑，少留状态槽位，把显存让给长回复的 KV；
    # 模型长度 = 输出上限 + 8192（论文附录 E）
    reason) extra=(--max-num-seqs 256 --max-model-len $((MAX_TOKENS + 8192))) ;;
  esac
  (
    mode_env "$mode" || exit 1
    export HIP_VISIBLE_DEVICES=$gpus
    exec python -m atom.entrypoints.openai_server \
      --model $MODEL --trust-remote-code \
      -tp 2 --enable-expert-parallel \
      --no-enable_prefix_caching \
      --hf-overrides '{"mamba_ssm_dtype": "float32"}' \
      --server-port "$sport" --port "$eport" \
      "${extra[@]}" ${EXTRA_ARGS:-}
  ) > "$slog" 2>&1 &
  SERVER_PID=$!
}

# 等服务就绪（日志里出现 Uvicorn running）。第一次启动要编译，最多等 40 分钟
wait_up() {
  for _ in $(seq 1 480); do
    grep -q "Uvicorn running" "$2" 2>/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# drive MODE KIND GPUS SPORT EPORT NAME CMD...：起服务 → 等就绪 → 核对模式 → 跑 CMD → 停服务。
# 引擎挂了时 HTTP 服务还活着，请求会一直挂着，所以发现日志里有 died unexpectedly 就终止 CMD
drive() {
  local mode=$1 kind=$2 gpus=$3 sport=$4 eport=$5 name=$6
  shift 6
  local slog=$LOGS/$name.server.log pid job rc
  wait_gpus "$gpus" || return 1
  start_server "$mode" "$kind" "$gpus" "$sport" "$eport" "$slog"
  pid=$SERVER_PID
  if ! wait_up "$pid" "$slog"; then
    log "$name: server did not come up, see $slog"
    stop_server "$pid"
    return 1
  fi
  if ! mode_ok "$mode" "$slog"; then
    log "$name: server log does not show mode $mode, see $slog"
    stop_server "$pid"
    return 1
  fi
  log "$name: up (pid $pid, GPUs $gpus, port $sport)"
  "$@" &
  job=$!
  while kill -0 "$job" 2>/dev/null; do
    if grep -q "died unexpectedly" "$slog"; then
      log "$name: engine died"
      kill -TERM "$job" 2>/dev/null
      break
    fi
    sleep 10
  done
  wait "$job"
  rc=$?
  log "$name: job exit $rc"
  stop_server "$pid"
  log "$name: stopped"
  return $rc
}

# ---------------------------------------------------------------- 性能
perf_done() { local c; for c in $CONC; do [ -f "$1/c$c.json" ] || return 1; done; }

# sweep PORT OUT：每档并发一轮等长请求，--save-detailed 留下逐 token 时间给 decode 窗口统计
sweep() {
  local c
  mkdir -p "$2"
  for c in $CONC; do
    python -m atom.benchmarks.benchmark_serving \
      --model=$MODEL --backend=vllm --base-url=http://localhost:$1 \
      --dataset-name=random \
      --random-input-len=4096 --random-output-len=1024 --random-range-ratio=1.0 \
      --num-prompts=$c --max-concurrency=$c \
      --request-rate=inf --ignore-eos \
      --percentile-metrics="ttft,tpot,itl,e2el" \
      --save-result --save-detailed --result-dir="$2" --result-filename=c$c.json || return 1
  done
}

perf() {
  local r m order out
  for r in r1 r2; do
    order=$MODES
    [ $r = r2 ] && order=$(echo $MODES | awk '{for (i = NF; i > 0; i--) printf "%s ", $i}')
    for m in $order; do
      out=$QOUT/perf/$m-$r
      if perf_done "$out"; then
        log "perf $m-$r: done already"
        continue
      fi
      # 稳态步时取自这次服务的日志，所以没测完的一轮整轮重测
      rm -rf -- "$QOUT/perf/$m-$r"
      log "perf $m-$r: start on GPUs $PERF_GPUS"
      drive $m perf $PERF_GPUS 8000 8006 perf-$m-$r sweep 8000 "$out" > "$LOGS/sweep-$m-$r.log" 2>&1
      log "perf $m-$r: $(grep -E 'did not come up|does not show|engine died|job exit' "$LOGS/sweep-$m-$r.log" | tail -n 1)"
    done
  done
}

# ---------------------------------------------------------------- 精度
gsm8k() {   # gsm8k PORT TAG
  lm_eval --model local-chat-completions \
    --model_args "model=$MODEL,base_url=http://localhost:$1/v1/chat/completions,num_concurrent=128,max_retries=3,timeout=1800,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn \
    --gen_kwargs 'max_gen_toks=4096,until=<|im_end|>' \
    --log_samples --output_path "$QOUT/results_gsm8k/$2"
}

reason() {   # reason PORT TAG
  python "$HERE/reason_eval.py" --tag "$2" --url "http://localhost:$1" \
    --out "$QOUT/results_reason" --tasks $TASKS --samples $SAMPLES \
    --max-tokens $MAX_TOKENS --concurrency 128 --sampling "$SAMPLING" \
    ${MMLUPRO_N:+--mmlupro-n $MMLUPRO_N}
}

gsm8k_done() { ls "$QOUT/results_gsm8k/$1"/*/results_*.json >/dev/null 2>&1; }

# reason_done TAG：TASKS 的每个 (题, 样本) 是否都已有结果
reason_done() {
  python - "$1" <<EOF
import json, pathlib, sys
sys.path.insert(0, "$HERE")
import reason_eval as R
out = pathlib.Path("$QOUT/results_reason") / sys.argv[1]
for task in "$TASKS".split():
    want = {(it["idx"], it["sample"]) for it in R.load_items(task, $SAMPLES, ${MMLUPRO_N:-None})}
    p = out / f"{task}.jsonl"
    got = {(r["idx"], r["sample"]) for r in map(json.loads, p.read_text().splitlines())} if p.exists() else set()
    if not want <= got:
        sys.exit(1)
EOF
}

acc_done() {
  if [ "${GSM8K:-1}" = 1 ]; then gsm8k_done "$1" || return 1; fi
  reason_done "$1"
}

acc_job() {   # acc_job PORT TAG：先 GSM8K，再 AIME/GPQA
  local rc=0
  if [ "${GSM8K:-1}" = 1 ] && ! gsm8k_done "$2"; then
    gsm8k "$1" "$2" || rc=1
  fi
  reason "$1" "$2" || rc=1
  return $rc
}

accuracy() {
  local gpus=($ACC_GPUS) sports=(8000 8001 8002) eports=(8006 8016 8026) i=0 m pids=()
  for m in $MODES; do
    if acc_done "$m"; then
      log "accuracy $m: done already"
      i=$((i + 1))
      continue
    fi
    rm -f "$LOGS/acc-$m.server.log"
    drive $m reason ${gpus[$i]} ${sports[$i]} ${eports[$i]} acc-$m acc_job ${sports[$i]} $m \
      > "$LOGS/acc-$m.log" 2>&1 &
    pids+=($!)
    log "accuracy $m: start on GPUs ${gpus[$i]}, port ${sports[$i]}"
    # 等这个服务起来（或失败）再起下一个，免得几个进程同时编译、写同一个缓存目录
    for _ in $(seq 1 480); do
      grep -q "Uvicorn running" "$LOGS/acc-$m.server.log" 2>/dev/null && break
      kill -0 ${pids[-1]} 2>/dev/null || break
      sleep 5
    done
    i=$((i + 1))
  done
  for i in "${!pids[@]}"; do
    wait ${pids[$i]}
  done
  for m in $MODES; do
    log "accuracy $m: $(grep -E 'did not come up|does not show|engine died|job exit' "$LOGS/acc-$m.log" 2>/dev/null | tail -n 1)"
  done
}

# ---------------------------------------------------------------- 汇总
summary() {
  python - "$QOUT" "$MODES" "$CONC" <<'EOF'
import collections, glob, json, pathlib, re, statistics, sys

q, modes, concs = pathlib.Path(sys.argv[1]), sys.argv[2].split(), [int(c) for c in sys.argv[3].split()]
logs = q / "logs"
stat = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) "
                  r"tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")


def steady(path):
    """服务端统计窗口里的稳态步时，flush 摊在里面；算法同 steady_step.py"""
    by = collections.defaultdict(list)
    for line in open(path, errors="ignore"):
        m = stat.search(line)
        if m and float(m[1]) == 0 and int(m[4]) == 0 and int(m[3]) > 0:
            by[int(m[3])].append(float(m[2]))
    return {n: 1000 * n / statistics.median(sorted(x)[len(x) // 4:] if len(x) > 2 else x)
            for n, x in by.items()}


def window(path):
    """最后一个请求拿到首 token 之后的 ITL：均值含 flush，中位数不含；算法同 decode_window.py"""
    r = json.loads(path.read_text())
    last, gaps = max(r["ttfts"]), []
    for t, itl in zip(r["ttfts"], r["itls"]):
        for g in itl:
            if t >= last - 2e-3:
                gaps.append(g)
            t += g
    return 1000 * statistics.mean(gaps), 1000 * statistics.median(gaps)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


perf = {}
for m in modes:
    rounds = sorted(p.name.rsplit("-", 1)[1] for p in (q / "perf").glob(f"{m}-r*") if p.is_dir())
    for c in concs:
        st, wm, wmed, thr = [], [], [], []
        for r in rounds:
            slog, f = logs / f"perf-{m}-{r}.server.log", q / "perf" / f"{m}-{r}" / f"c{c}.json"
            if slog.exists():
                st.append(steady(slog).get(c))
            if f.exists():
                a, b = window(f)
                wm.append(a)
                wmed.append(b)
                thr.append(json.loads(f.read_text())["output_throughput"])
        perf[m, c] = dict(steady=mean(st), wmean=mean(wm), wmed=mean(wmed), thr=mean(thr), n=len(rounds))


def table(title, key, higher_better=False):
    print(f"== {title} ==")
    print(f"{'conc':>6}" + "".join(f"{m:>12}" for m in modes) + "".join(f"{m + '/native':>20}" for m in modes[1:]))
    for c in concs:
        vals = [perf[m, c][key] for m in modes]
        row = "".join(f"{v:12.2f}" if v is not None else f"{'-':>12}" for v in vals)
        for v in vals[1:]:
            ok = vals[0] is not None and v is not None
            row += f"{(v / vals[0] if higher_better else vals[0] / v):19.3f}x" if ok else f"{'-':>20}"
        print(f"{c:>6}" + row)
    print()


print(f"轮数：" + ", ".join(f"{m} {perf[m, concs[0]]['n']}" for m in modes) + "（多轮取平均）\n")
table("decode 每步耗时，服务端稳态统计，flush 已摊入（ms；比值 = native / 本模式，>1 更快）", "steady")
table("decode 窗口 ITL 均值，含 flush（ms）", "wmean")
table("decode 窗口 ITL 中位数，不含 flush（ms）", "wmed")
table("输出吞吐，含 prefill（tok/s；比值 = 本模式 / native）。native 的 GDN prefill 走 FlyDSL，"
      "另外两种走 Triton，这一项不只反映 decode", "thr", higher_better=True)

print("== 显存（每 rank，取第一轮性能服务的日志）==")
pool = re.compile(r"sub-pool (\S+): entries=(\d+), entry_bytes=(\d+), reserved=([\d.]+)GB")
for m in modes:
    found = sorted(logs.glob(f"perf-{m}-r*.server.log"))
    if not found:
        continue
    rows = dict.fromkeys(pool.findall(found[0].read_text(errors="ignore")))
    print(f"{m:>10}  " + " | ".join(f"{n} {int(b) / 1e6:.1f} MB x {e} = {gb} GB" for n, e, b, gb in rows))
print()

print("== GSM8K（1319 题，5-shot chat）==")
for m in modes:
    files = sorted(glob.glob(str(q / "results_gsm8k" / m / "*" / "results_*.json")))
    if not files:
        print(f"{m:>10}  -")
        continue
    r = json.load(open(files[-1]))["results"]["gsm8k"]
    print(f"{m:>10}  flexible {100 * r['exact_match,flexible-extract']:.2f} ± "
          f"{100 * r['exact_match_stderr,flexible-extract']:.2f}   strict {100 * r['exact_match,strict-match']:.2f} ± "
          f"{100 * r['exact_match_stderr,strict-match']:.2f}")
EOF
  echo
  echo "== GSM8K 逐题对比 native（flexible）=="
  local have=() m
  for m in $MODES; do
    ls "$QOUT/results_gsm8k/$m"/*/samples_gsm8k_*.jsonl >/dev/null 2>&1 && have+=($m)
  done
  if [ "${have[0]}" = native ] && [ ${#have[@]} -gt 1 ]; then
    (cd "$QOUT/results_gsm8k" && python "$HERE/gsm8k_pair.py" "${have[@]}")
  else
    echo "-"
  fi
  echo
  echo "== AIME 2026 / GPQA-Diamond（thinking，每题 $SAMPLES 个样本，最长 $MAX_TOKENS token）=="
  python "$HERE/reason_eval.py" --summary --out "$QOUT/results_reason" --base native --tags $MODES || echo "-"
}

record_version() {
  {
    echo "== $(date '+%F %T') $1"
    echo "ATOM_ROOT $ATOM_ROOT"
    echo "HEAD $(git -C "$ATOM_ROOT" rev-parse HEAD) $(git -C "$ATOM_ROOT" log -1 --format=%s)"
    echo "uncommitted $(git -C "$ATOM_ROOT" diff HEAD --stat | tail -n 1), sha1 $(git -C "$ATOM_ROOT" diff HEAD | sha1sum | cut -c1-12)"
  } >> "$QOUT/atom_version.txt"
}

# 只在开头清一次编译缓存（拿到当前代码），之后的启动都复用；有别的 ATOM 服务在跑就不清
clear_cache() {
  if pgrep -f "atom.entrypoints.openai_server" >/dev/null; then
    log "another ATOM server is running, compile cache kept"
  else
    rm -rf /root/.cache/atom/*
  fi
}

case "${1:-all}" in
  perf) record_version perf; clear_cache; perf ;;
  accuracy) record_version accuracy; accuracy ;;
  summary) summary ;;
  all)
    record_version all
    clear_cache
    perf
    accuracy
    summary | tee "$QOUT/summary.txt"
    log "all done, summary in $QOUT/summary.txt"
    ;;
  *) echo "usage: $0 [all|perf|accuracy|summary]" >&2; exit 1 ;;
esac
