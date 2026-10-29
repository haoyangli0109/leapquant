#!/bin/bash
# Qwen3.5-9B（BF16 权重，单卡 MI355X，TP1，CUDA graph，不开前缀缓存）上按论文测性能，四种状态：
#   native     FP32 状态（模型 config 的 mamba_ssm_dtype），每个 decode 步读写完整状态，论文的基线
#   bf16       每步读写 BF16 状态，论文里的 BF16 基线
#   replayssm  FP32 checkpoint + 每个 token 一条 (k, u, g) 记录，攒满 16 条才 flush
#   leapquant  int8 残差 + r=4 个 FP16 补偿 token + FP32 平滑，窗口 p=16
#
# 和论文的对应（第 4.3 节、表 2/3、附录 C）：
#   kernel  单个 GDN 层、32 个 v 头、16 个 q/k 头、dk=dv=128，正好是 Qwen3.5-9B 的 GDN 层；
#           bs 32/128/512。消融都在 bs 256：BF16 每步、BF16 逐窗口（BF16 checkpoint 的
#           ReplaySSM）、窗口 p=4/8/16/32、补偿 token r=2/4/8/16。另测 Qwen3.5-9B 的实际形状：
#           24 层，ATOM 每步把所有层的 fold 合成一次启动。
#   decode  上下文 4K，bs 32/128/512；附录另有 1K/2K/8K。每档并发一轮等长请求，输入等于上下文
#           长度，输出 1024，取最后一个 prompt 进来之后的 decode 窗口的 ITL 均值（含 flush）。
#           四种模式在 GPU 4–7 上排成拉丁方：每种模式在每张卡、每个时段各测一次，
#           抵消卡间差异和时间漂移。
#
# 用法（结果写到 $QOUT，默认 ../qwen_eval/q35_perf，不进这个仓库）：
#   setsid nohup bash q35_perf.sh all > ../qwen_eval/q35_perf/run.log 2>&1 &   # kernel → decode → 汇总
#   bash q35_perf.sh kernel | decode | summary
# 中断后重跑同一个命令会接着跑：测完的 (上下文, 并发) 组合跳过。
# 可调的环境变量：GPUS（默认 "4 5 6 7"）、MODES、CTXS（默认 "4096 1024 2048 8192"）、
# CONC（默认 "32 64 128 256 512"）、OUT_LEN（默认 1024）、ATOM_ROOT（默认 ../ATOM）、EXTRA_ARGS。
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
MODEL=/shareddata/haoyanli/Qwen/Qwen3.5-9B
ATOM_ROOT=${ATOM_ROOT:-$(dirname "$HERE")/ATOM}
QOUT=${QOUT:-$(dirname "$HERE")/qwen_eval/q35_perf}
GPUS=(${GPUS:-4 5 6 7})
MODES=(${MODES:-native bf16 replayssm leapquant})
CTXS=${CTXS:-"4096 1024 2048 8192"}
CONC=${CONC:-"32 64 128 256 512"}
OUT_LEN=${OUT_LEN:-1024}
MAX_LEN=$(($(echo $CTXS | tr ' ' '\n' | sort -n | tail -n 1) + OUT_LEN + 64))

export AITER_LOG_LEVEL=WARNING
# cwd 是 leapquant，里面没有 atom 包，所以 python -m 会从 PYTHONPATH 里的 $ATOM_ROOT 导入
export PYTHONPATH=$ATOM_ROOT${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_CACHE=/shareddata/huggingface/hub
LOGS=$QOUT/logs
mkdir -p "$LOGS"

log() { echo "$(date '+%F %T') $*"; }

# 四种模式只差这几个环境变量和 mamba_ssm_dtype。先清掉可能从外面带进来、会改变 GDN 路径的变量
mode_env() {
  unset ATOM_ENABLE_REPLAYSSM ATOM_ENABLE_LEAPQUANT ATOM_LEAPQUANT_WINDOW ATOM_LEAPQUANT_RANK \
    ATOM_REPLAYSSM_CACHE_LEN ATOM_REPLAYSSM_ROUTE ATOM_ENABLE_GDN_DECODE_LOSSY_FAST \
    ATOM_GDN_PREFILL_BACKEND ATOM_GDN_DECODE_BACKEND ATOM_GDN_SSM_DTYPE
  case $1 in
    native | bf16) export ATOM_ENABLE_REPLAYSSM=0 ;;
    replayssm) export ATOM_ENABLE_REPLAYSSM=1 ;;
    leapquant) export ATOM_ENABLE_LEAPQUANT=1 ATOM_LEAPQUANT_WINDOW=16 ATOM_LEAPQUANT_RANK=4 ;;
    *) echo "unknown mode $1" >&2; return 1 ;;
  esac
}

ssm_dtype() { if [ "$1" = bf16 ]; then echo bfloat16; else echo float32; fi; }

# builder 打印的模式和状态池每槽字节数都要对：24 层合计，FP32 状态约 51.5 MB，BF16 约 26.3 MB，
# ReplaySSM 再加约 6.7 MB 记录。否则这组结果不算
mode_ok() {
  local b
  b=$(grep -oP "sub-pool state: entries=\d+, entry_bytes=\K\d+" "$2" | head -n 1)
  [ -n "$b" ] || return 1
  case $1 in
    native) grep -q "ReplaySSM disabled for linear attention" "$2" && ! grep -q "LeapQuant enabled" "$2" && [ "$b" -gt 45000000 ] ;;
    bf16) grep -q "ReplaySSM disabled for linear attention" "$2" && ! grep -q "LeapQuant enabled" "$2" && [ "$b" -lt 30000000 ] ;;
    replayssm) grep -q "ReplaySSM enabled for linear attention" "$2" && ! grep -q "LeapQuant enabled" "$2" && [ "$b" -gt 45000000 ] ;;
    leapquant) grep -q "LeapQuant enabled for linear attention" "$2" ;;
  esac
}

# 等这张卡的显存空出来，即上一个服务的进程都退干净了
wait_gpu() {
  for _ in $(seq 1 60); do
    HIP_VISIBLE_DEVICES=$1 python - <<'EOF' 2>/dev/null && return 0
import sys, torch
sys.exit(0 if torch.cuda.device_count() and torch.cuda.mem_get_info(0)[0] > 250 * 2**30 else 1)
EOF
    sleep 10
  done
  log "GPU $1 still busy"
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

# start_server MODE GPU SPORT EPORT LOG：后台起服务，PID 放进 SERVER_PID。
# 512 个状态槽位，正好是最高并发；模型长度够最长的上下文加输出
start_server() {
  local mode=$1 gpu=$2 sport=$3 eport=$4 slog=$5
  (
    mode_env "$mode" || exit 1
    export HIP_VISIBLE_DEVICES=$gpu
    exec python -m atom.entrypoints.openai_server \
      --model $MODEL --trust-remote-code -tp 1 \
      --no-enable_prefix_caching \
      --hf-overrides "{\"mamba_ssm_dtype\": \"$(ssm_dtype "$mode")\"}" \
      --server-port "$sport" --port "$eport" \
      --max-num-seqs 512 --max-model-len $MAX_LEN ${EXTRA_ARGS:-}
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

# 贪心续写一句话。状态算坏了（例如 NaN）会输出一串乱码
smoke() {
  curl -s -m 120 "http://localhost:$1/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" |
    python -c 'import json, sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'
}

perf_done() { local x c; for x in $CTXS; do for c in $CONC; do [ -f "$1/ctx$x/c$c.json" ] || return 1; done; done; }

# sweep PORT OUT：每个上下文、每档并发一轮等长请求，--save-detailed 留下逐 token 时间给 decode 窗口统计
sweep() {
  local x c
  for x in $CTXS; do
    mkdir -p "$2/ctx$x"
    for c in $CONC; do
      [ -f "$2/ctx$x/c$c.json" ] && continue
      python -m atom.benchmarks.benchmark_serving \
        --model=$MODEL --backend=vllm --base-url="http://localhost:$1" \
        --dataset-name=random \
        --random-input-len="$x" --random-output-len=$OUT_LEN --random-range-ratio=1.0 \
        --num-prompts="$c" --max-concurrency="$c" \
        --request-rate=inf --ignore-eos \
        --percentile-metrics="ttft,tpot,itl,e2el" \
        --save-result --save-detailed --result-dir="$2/ctx$x" --result-filename="c$c.json" || return 1
    done
  done
}

# drive MODE GPU SPORT EPORT NAME OUT：起服务 → 等就绪 → 核对模式 → 冒烟 → 扫描 → 停服务。
# 服务一就绪就留下 .up-g<GPU>，下一条线的第一个服务等它再起。
# 引擎挂了时 HTTP 服务还活着，请求会一直挂着，所以发现日志里有 died unexpectedly 就终止扫描
drive() {
  local mode=$1 gpu=$2 sport=$3 eport=$4 name=$5 out=$6
  local slog=$LOGS/$name.server.log pid job rc up=0
  wait_gpu "$gpu" || return 1
  start_server "$mode" "$gpu" "$sport" "$eport" "$slog"
  pid=$SERVER_PID
  wait_up "$pid" "$slog" && up=1
  touch "$LOGS/.up-g$gpu"
  if [ $up = 0 ]; then
    log "$name: server did not come up, see $slog"
    stop_server "$pid"
    return 1
  fi
  if ! mode_ok "$mode" "$slog"; then
    log "$name: server log does not show mode $mode, see $slog"
    stop_server "$pid"
    return 1
  fi
  log "$name: up (pid $pid, GPU $gpu, port $sport), smoke $(smoke "$sport")"
  sweep "$sport" "$out" &
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

# ---------------------------------------------------------------- kernel
# 四张卡各跑一组，几分钟。fold 在论文里和层 kernel 融在一起，所以论文形状另测一遍 --per-layer-fold
kernel() {
  local K=$QOUT/kernel pids=() p r
  local bench="python $HERE/leapquant_bench.py"
  local paper="--layers 1 --hv 32 --h 16" q35="--layers 24 --hv 32 --h 16"
  local all="--modes fp32 bf16 replayssm replayssm_bf16 leapquant"
  mkdir -p "$K"
  (
    export HIP_VISIBLE_DEVICES=${GPUS[0]}
    $bench $paper $all --batch 32 128 256 512 --breakdown
    $bench $paper --modes fp32 leapquant --per-layer-fold --batch 32 128 256 512 --breakdown
  ) > "$K/paper_shape.log" 2>&1 &
  pids+=($!)
  (
    export HIP_VISIBLE_DEVICES=${GPUS[1]}
    $bench $q35 $all --batch 32 64 128 256 512 --breakdown
  ) > "$K/q35_shape.log" 2>&1 &
  pids+=($!)
  # 窗口长度 p，--cap = p + 1
  (
    export HIP_VISIBLE_DEVICES=${GPUS[2]}
    for p in 4 8 16 32; do
      $bench $paper --modes fp32 leapquant --batch 256 --cap $((p + 1))
      $bench $q35 --modes fp32 leapquant --batch 256 --cap $((p + 1))
    done
  ) > "$K/ablation_p.log" 2>&1 &
  pids+=($!)
  # 补偿 token 数 r
  (
    export HIP_VISIBLE_DEVICES=${GPUS[3]}
    for r in 2 4 8 16; do
      $bench $paper --modes fp32 leapquant --batch 256 --rank $r
      $bench $q35 --modes fp32 leapquant --batch 256 --rank $r
    done
  ) > "$K/ablation_r.log" 2>&1 &
  pids+=($!)
  wait "${pids[@]}"
}

# ---------------------------------------------------------------- decode
# 第 i 条线（GPUS[i]）第 s 个时段测 MODES[(i + s) % n]。冷启动一个一个来：
# 等前一条线的第一个服务起来再起，免得几个进程同时编译、写同一个缓存目录
decode() {
  local n=${#MODES[@]} i pids=()
  rm -f "$LOGS"/.up-g*
  for i in "${!GPUS[@]}"; do
    (
      local s m g=${GPUS[$i]} out name
      if [ "$i" -gt 0 ]; then
        for _ in $(seq 1 480); do [ -f "$LOGS/.up-g${GPUS[$((i - 1))]}" ] && break; sleep 5; done
      fi
      for s in $(seq 0 $((n - 1))); do
        m=${MODES[$(((i + s) % n))]}
        out=$QOUT/perf/$m-g$g
        name=perf-$m-g$g
        if perf_done "$out"; then
          log "decode $m on GPU $g: done already"
          touch "$LOGS/.up-g$g"
          continue
        fi
        log "decode $m on GPU $g: start"
        drive "$m" "$g" $((8100 + 10 * i)) $((8106 + 10 * i)) "$name" "$out" > "$LOGS/sweep-$m-g$g.log" 2>&1
        log "decode $m on GPU $g: $(grep -E 'did not come up|does not show|engine died|job exit' "$LOGS/sweep-$m-g$g.log" | tail -n 1)"
      done
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}

# ---------------------------------------------------------------- 汇总
summary() {
  echo "== kernel（每步耗时，ms；倍数相对同一次运行里的 FP32）=="
  local f
  for f in paper_shape q35_shape ablation_p ablation_r; do
    [ -f "$QOUT/kernel/$f.log" ] || continue
    echo "-- $f"
    grep -E "^shape:|ms/step" "$QOUT/kernel/$f.log"
  done
  echo
  python - "$QOUT" "${MODES[*]}" "$CTXS" "$CONC" <<'EOF'
import json, pathlib, re, statistics, sys

q, modes = pathlib.Path(sys.argv[1]), sys.argv[2].split()
ctxs = sorted(int(x) for x in sys.argv[3].split())
concs = [int(c) for c in sys.argv[4].split()]


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


runs = {m: sorted(p for p in (q / "perf").glob(f"{m}-g*") if p.is_dir()) for m in modes}
print("各模式的测量次数（每张卡一次）：" + ", ".join(f"{m} {len(runs[m])}" for m in modes) + "\n")
for x in ctxs:
    print(f"== 上下文 {x}：decode 窗口 ITL 均值，含 flush（ms，各卡平均；括号里是 native / 本模式）==")
    print(f"{'conc':>6}" + "".join(f"{m:>18}" for m in modes) + f"{'卡间极差/均值':>16}")
    for c in concs:
        means, cells, spread = {}, [], []
        for m in modes:
            vals = [window(f)[0] for d in runs[m] if (f := d / f"ctx{x}" / f"c{c}.json").exists()]
            means[m] = statistics.mean(vals) if vals else None
            if len(vals) > 1:
                spread.append((max(vals) - min(vals)) / statistics.mean(vals))
        for m in modes:
            v, b = means[m], means[modes[0]]
            if v is None:
                cells.append(f"{'-':>18}")
            elif m == modes[0] or b is None:
                cells.append(f"{v:18.2f}")
            else:
                cells.append(f"{v:9.2f} ({b / v:5.3f}x)")
        print(f"{c:>6}" + "".join(cells) + (f"{100 * max(spread):14.1f}%" if spread else f"{'-':>16}"))
    print()

print("== 上下文 4096：decode 窗口 ITL 中位数，不含 flush（ms，各卡平均）==")
print(f"{'conc':>6}" + "".join(f"{m:>12}" for m in modes))
for c in concs:
    row = []
    for m in modes:
        vals = [window(f)[1] for d in runs[m] if (f := d / "ctx4096" / f"c{c}.json").exists()]
        row.append(f"{statistics.mean(vals):12.2f}" if vals else f"{'-':>12}")
    print(f"{c:>6}" + "".join(row))
print()

print("== 显存（取每种模式的第一次服务日志）==")
pool = re.compile(r"sub-pool (\S+): entries=(\d+), entry_bytes=(\d+), reserved=([\d.]+)GB")
cap = re.compile(r"100% \(\s*(\d+) tok\):\s+(\d+) blk/req → max_concurrent=(\d+)")
for m in modes:
    found = sorted((q / "logs").glob(f"perf-{m}-g*.server.log"))
    if not found:
        continue
    text = found[0].read_text(errors="ignore")
    rows = dict.fromkeys(pool.findall(text))
    cc = cap.search(text)
    print(f"{m:>10}  " + " | ".join(f"{n} {int(b) / 1e6:.2f} MB x {e} = {gb} GB" for n, e, b, gb in rows)
          + (f" | {cc[1]} tok 时并发上限 {cc[3]}" if cc else ""))
EOF
}

record_version() {
  {
    echo "== $(date '+%F %T') $1"
    echo "ATOM_ROOT $ATOM_ROOT"
    echo "HEAD $(git -C "$ATOM_ROOT" rev-parse HEAD) $(git -C "$ATOM_ROOT" log -1 --format=%s)"
    echo "uncommitted $(git -C "$ATOM_ROOT" diff HEAD --stat | tail -n 1), sha1 $(git -C "$ATOM_ROOT" diff HEAD | sha1sum | cut -c1-12)"
  } >> "$QOUT/atom_version.txt"
}

case "${1:-all}" in
  kernel) record_version kernel; kernel ;;
  decode) record_version decode; decode ;;
  summary) summary ;;
  all)
    record_version all
    log "kernel benches on GPUs ${GPUS[*]}"
    kernel
    log "decode sweeps on GPUs ${GPUS[*]}"
    decode
    summary | tee "$QOUT/summary.txt"
    log "all done, summary in $QOUT/summary.txt"
    ;;
  *) echo "usage: $0 [all|kernel|decode|summary]" >&2; exit 1 ;;
esac
