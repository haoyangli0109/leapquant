#!/bin/bash
# 论文图 9 的端到端吞吐（第 4.3 节）：Qwen3.5-9B，单卡 MI355X，按原模型的输入、输出长度对整个数据集做
# 离线推理，比较 FP32 状态（native）和 LeapQuant 的输出吞吐。
#   参考长度  q35_e2e_ref.py 用 vLLM 跑原模型：FP32 状态，模型卡采样（含 presence_penalty 1.5），
#            最长 81,920 token，每题一次。结果在 $QOUT/ref，按题号轮流分给 4 个分片。
#   服务     ATOM，TP1，CUDA graph，开前缀缓存，每 1024 token 一个状态 checkpoint（论文：vLLM/SGLang 默认
#            每 1K token 一个）；max-num-seqs 1024、max-num-batched-tokens 16384 是 vLLM 离线推理在
#            大显存卡上的默认值；模型长度 = 81,920 + 8,192。
#   回放     q35_e2e_replay.py：一个数据集的请求一次全部交给服务，prompt 用原 token，输出长度用原模型的
#            长度（ignore_eos），两种模式生成的 token 完全一样多。每个数据集单独起一次服务。
#
# 用法（结果写到 $QOUT，默认 ../qwen_eval/q35_e2e，不进这个仓库）：
#   REF="4:3 5:0" bash q35_e2e.sh ref       # 补齐参考长度，"GPU:分片"，已完成的题跳过
#   setsid nohup bash q35_e2e.sh replay > ../qwen_eval/q35_e2e/replay.log 2>&1 &
#   bash q35_e2e.sh summary
# LANES 每项是一条线，"候选卡:模式[:数据集]"，例如 "7:native,leapquant" 或 "4/5/6:leapquant,native:mmlupro"。
# 数据集写成 "gsm8k@3" 时整个数据集发 3 遍（相当于每题 3 个样本），结果存成 gsm8k_x3.json；
# 一条线跑多个数据集时用逗号隔开，例如 "7:native,leapquant:gsm8k@3,gpqa@3"。
# 一条线先在候选卡里等到一张完全空闲的卡（显存全空、利用率连续几次接近 0，最多等 WAIT_MIN 分钟），
# 之后一直用它：每个数据集按列出的顺序把各模式跑一遍。
# 卡号都是 HIP_VISIBLE_DEVICES 的编号；这台机器上它和 rocm-smi 的编号不同，查利用率和进程时按 PCI 总线号换算。
# 其他环境变量：TASKS（默认 "gsm8k gpqa mmlupro"）、WAIT_MIN（默认 360）、PORT_BASE（默认 8200，同时跑两份时错开）、
# ATOM_ROOT（默认 ../ATOM）、EXTRA_ARGS（追加给服务）。中断后重跑会跳过已经有结果的 (模式, 卡, 数据集)。
set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
MODEL=/shareddata/haoyanli/Qwen/Qwen3.5-9B
ATOM_ROOT=${ATOM_ROOT:-$(dirname "$HERE")/ATOM}
QOUT=${QOUT:-$(dirname "$HERE")/qwen_eval/q35_e2e}
LANES=(${LANES:-7:native,leapquant})
TASKS=${TASKS:-"gsm8k gpqa mmlupro"}
WAIT_MIN=${WAIT_MIN:-360}
PORT_BASE=${PORT_BASE:-8200}
MAX_LEN=$((81920 + 8192))

export AITER_LOG_LEVEL=WARNING
export HF_HUB_CACHE=/shareddata/huggingface/hub
# vLLM（参考长度）不能看到 ATOM；ATOM 服务在 start_server 里单独设置
unset PYTHONPATH
LOGS=$QOUT/logs
mkdir -p "$LOGS"

log() { echo "$(date '+%F %T') $*"; }

# ---------------------------------------------------------------- 参考长度
ref() {
  local p g s pids=()
  for p in ${REF:-4:0 5:1 6:2 7:3}; do
    g=${p%:*} s=${p#*:}
    log "ref shard $s on GPU $g"
    VLLM_PLUGINS="" HIP_VISIBLE_DEVICES=$g python "$HERE/q35_e2e_ref.py" --shard "$s" --shards 4 \
      --out "$QOUT/ref" >> "$LOGS/ref-$s.log" 2>&1 &
    pids+=($!)
    sleep 30
  done
  wait "${pids[@]}"
  python "$HERE/q35_e2e_ref.py" --summary --out "$QOUT/ref"
}

# ---------------------------------------------------------------- 回放
mode_env() {
  unset ATOM_ENABLE_REPLAYSSM ATOM_ENABLE_LEAPQUANT ATOM_LEAPQUANT_WINDOW ATOM_LEAPQUANT_RANK \
    ATOM_REPLAYSSM_CACHE_LEN ATOM_REPLAYSSM_ROUTE ATOM_ENABLE_GDN_DECODE_LOSSY_FAST \
    ATOM_GDN_PREFILL_BACKEND ATOM_GDN_DECODE_BACKEND ATOM_GDN_SSM_DTYPE
  case $1 in
    native) export ATOM_ENABLE_REPLAYSSM=0 ;;
    leapquant) export ATOM_ENABLE_LEAPQUANT=1 ATOM_LEAPQUANT_WINDOW=16 ATOM_LEAPQUANT_RANK=4 ;;
    *) echo "unknown mode $1" >&2; return 1 ;;
  esac
}

# builder 打印的模式要对；native 的状态池每槽约 51.5 MB（FP32 状态），否则这组结果不算
mode_ok() {
  local b
  b=$(grep -oP "sub-pool state: entries=\d+, entry_bytes=\K\d+" "$2" | head -n 1)
  [ -n "$b" ] || return 1
  case $1 in
    native) grep -q "ReplaySSM disabled for linear attention" "$2" && ! grep -q "LeapQuant enabled" "$2" && [ "$b" -gt 45000000 ] ;;
    leapquant) grep -q "LeapQuant enabled for linear attention" "$2" ;;
  esac
}

mem_free() {
  HIP_VISIBLE_DEVICES=$1 python - <<'EOF' 2>/dev/null
import sys, torch
sys.exit(0 if torch.cuda.device_count() and torch.cuda.mem_get_info(0)[0] > 280 * 2**30 else 1)
EOF
}

# smi_index HIP编号：同一张卡在 rocm-smi 里的编号（按 PCI 总线号对应）
smi_index() {
  local bus
  bus=$(HIP_VISIBLE_DEVICES=$1 python -c "import torch; print('%02X' % torch.cuda.get_device_properties(0).pci_bus_id)" 2>/dev/null)
  rocm-smi --showbus 2>/dev/null | grep -oP "GPU\[\K\d+(?=\]\s*: PCI Bus: [0-9A-Fa-f]{4}:${bus}:)"
}

# 利用率连续 6 次（30 秒）不超过 5%：显存全空的卡上也可能有别人的进程在算
compute_idle() {
  local u smi
  smi=$(smi_index "$1")
  [ -n "$smi" ] || return 1
  for _ in 1 2 3 4 5 6; do
    u=$(rocm-smi --showuse -d "$smi" 2>/dev/null | grep -oP 'GPU use \(%\): \K\d+')
    [ -n "$u" ] && [ "$u" -le 5 ] || return 1
    sleep 5
  done
}

# card_pids HIP编号：rocm-smi 看到的、正在用这张卡的进程（宿主机 PID）
card_pids() {
  local smi
  smi=$(smi_index "$1")
  rocm-smi --showpidgpus 2>/dev/null | awk -v d="$smi" '/^PID/{pid=$2} /^[0-9 ]+$/{for (i = 1; i <= NF; i++) if ($i == d) print pid}' | sort -u | tr '\n' ' '
}

# 等上一个服务退干净（这条线自己的卡，只看显存）
wait_gpu() {
  for _ in $(seq 1 60); do
    mem_free "$1" && return 0
    sleep 10
  done
  log "GPU $1 still busy"
  return 1
}

# pick_gpu CANDIDATES...：等到其中一张卡完全空闲且没被别的线占用，占上它并打印卡号
pick_gpu() {
  local g end=$((SECONDS + 60 * WAIT_MIN))
  while [ $SECONDS -lt $end ]; do
    for g in "$@"; do
      [ -e "$LOGS/.lane-g$g" ] && continue
      if mem_free "$g" && compute_idle "$g"; then
        echo $$ > "$LOGS/.lane-g$g"
        echo "$g"
        return 0
      fi
    done
    sleep 60
  done
  return 1
}

ptree() { echo "$1"; local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do ptree "$c"; done; }

stop_server() {
  local pids p
  pids=$(ptree "$1")
  kill -INT "$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || break; sleep 3; done
  for p in $pids; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done
  wait "$1" 2>/dev/null
  return 0
}

start_server() {   # start_server MODE GPU SPORT EPORT LOG
  local mode=$1 gpu=$2 sport=$3 eport=$4 slog=$5
  (
    mode_env "$mode" || exit 1
    export HIP_VISIBLE_DEVICES=$gpu PYTHONPATH=$ATOM_ROOT
    exec python -m atom.entrypoints.openai_server \
      --model $MODEL --trust-remote-code -tp 1 \
      --hf-overrides '{"mamba_ssm_dtype": "float32"}' \
      --state-checkpoint-interval-tokens 1024 \
      --max-num-seqs 1024 --max-num-batched-tokens 16384 --max-model-len $MAX_LEN \
      --server-port "$sport" --port "$eport" ${EXTRA_ARGS:-}
  ) > "$slog" 2>&1 &
  SERVER_PID=$!
}

wait_up() {
  for _ in $(seq 1 480); do
    grep -q "Uvicorn running" "$2" 2>/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# drive MODE GPU SPORT EPORT TASK OUT REPEAT：起服务 → 核对模式 → 回放一个数据集 → 存 /metrics → 停服务
drive() {
  local mode=$1 gpu=$2 sport=$3 eport=$4 task=$5 out=$6 rep=${7:-1}
  local slog=${out%.json}.server.log pid job rc
  wait_gpu "$gpu" || return 1
  start_server "$mode" "$gpu" "$sport" "$eport" "$slog"
  pid=$SERVER_PID
  if ! wait_up "$pid" "$slog"; then
    log "$mode $task GPU $gpu: server did not come up, see $slog"
    stop_server "$pid"
    return 1
  fi
  if ! mode_ok "$mode" "$slog"; then
    log "$mode $task GPU $gpu: server log does not show mode $mode, see $slog"
    stop_server "$pid"
    return 1
  fi
  log "$mode $task GPU $gpu: up (pid $pid, port $sport)"
  python "$HERE/q35_e2e_replay.py" --url "http://localhost:$sport" --ref "$QOUT/ref" \
    --task "$task" --repeat "$rep" --out "$out" &
  job=$!
  # 每 30 秒记一次这张卡上的进程：只有我们的服务时是 1 个，多出来的就是别人的任务
  (
    while kill -0 "$job" 2>/dev/null; do
      echo "$(date '+%T') $(card_pids "$gpu")"
      sleep 30
    done
  ) > "${out%.json}.card_pids.txt" 2>&1 &
  while kill -0 "$job" 2>/dev/null; do
    if grep -q "died unexpectedly" "$slog"; then
      log "$mode $task GPU $gpu: engine died"
      kill -TERM "$job" 2>/dev/null
      break
    fi
    sleep 10
  done
  wait "$job"
  rc=$?
  curl -s -m 30 "http://localhost:$sport/metrics" > "${out%.json}.metrics.txt"
  log "$mode $task GPU $gpu: replay exit $rc"
  stop_server "$pid"
  return $rc
}

replay() {
  local i pids=()
  for i in "${!LANES[@]}"; do
    (
      local cands modes tasks g t m out base rep name
      IFS=: read -r cands modes tasks <<< "${LANES[$i]}"
      tasks=${tasks:-$TASKS}
      log "lane $i: waiting for a free GPU among ${cands//\// }"
      if ! g=$(pick_gpu ${cands//\// }); then
        log "lane $i: no free GPU among ${cands//\// } within $WAIT_MIN min"
        exit 1
      fi
      log "lane $i: GPU $g"
      for t in ${tasks//,/ }; do
        base=${t%@*} rep=1 name=${t%@*}
        if [[ $t == *@* ]]; then rep=${t#*@} name=${base}_x$rep; fi
        for m in ${modes//,/ }; do
          out=$QOUT/replay/$m-g$g/$name.json
          if [ -f "$out" ]; then
            log "$m $name GPU $g: done already"
            continue
          fi
          mkdir -p "$(dirname "$out")"
          log "$m $name GPU $g: start"
          drive "$m" "$g" $((PORT_BASE + 10 * i)) $((PORT_BASE + 6 + 10 * i)) "$base" "$out" "$rep" >> "$LOGS/replay-$m-g$g.log" 2>&1
          log "$m $name GPU $g: $(grep -E 'did not come up|does not show|engine died|replay exit' "$LOGS/replay-$m-g$g.log" | tail -n 1)"
        done
      done
      rm -f "$LOGS/.lane-g$g"
    ) &
    pids+=($!)
    # 冷启动错开，免得几个进程同时编译、写同一个缓存目录
    sleep 90
  done
  wait "${pids[@]}"
}

# ---------------------------------------------------------------- 汇总
# 每个结果文件一行：整体吞吐（论文的口径：生成 token 数 / 第一个请求到最后一个回复的时间）、
# 饱和阶段吞吐（在跑请求数不低于峰值 80% 的时段，不含请求陆续结束后的长尾）、卡上出现过的进程数。
summary() {
  python - "$QOUT" <<'EOF'
import collections, json, pathlib, re, statistics, sys
from datetime import datetime

q = pathlib.Path(sys.argv[1])
# Figure 9 of the paper, Qwen3.5-9B: LeapQuant / FP32 output throughput on B200 and RTX PRO 6000.
paper = {"gsm8k": (1.23, 1.40), "mmlupro": (1.30, 1.38), "gpqa": (1.44, 1.31)}
line_re = re.compile(r"\[atom (\d\d:\d\d:\d\d)\] Engine \d+: Avg prompt throughput: [\d.]+ tokens/s, "
                     r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
                     r"GPU KV cache usage: ([\d.]+)%")


def server(path):
    """Saturated-phase throughput, median running requests while others wait, peak KV use."""
    pts, prev, kv, run_wait = [], None, [], []
    for line in path.read_text(errors="ignore").splitlines():
        m = line_re.search(line)
        if not m:
            continue
        t = datetime.strptime(m[1], "%H:%M:%S")
        kv.append(float(m[5]))
        if int(m[4]) > 0:
            run_wait.append(int(m[3]))
        if prev is not None:
            dt = (t - prev).total_seconds() % 86400
            pts.append((dt, float(m[2]) * dt, int(m[3])))
        prev = t
    peak = max((p[2] for p in pts), default=0)
    sat = [p for p in pts if peak and p[2] >= 0.8 * peak]
    sat_tput = sum(p[1] for p in sat) / sum(p[0] for p in sat) if sat else None
    return sat_tput, (statistics.median(run_wait) if run_wait else None), (max(kv) if kv else None)


def preemptions(path):
    m = re.search(r"^atom:preemptions\S*\s+([\d.e+]+)", path.read_text(), re.M) if path.exists() else None
    return int(float(m[1])) if m else None


def max_pids(path):
    if not path.exists():
        return None
    return max((len(l.split()) - 1 for l in path.read_text().splitlines()), default=0)


runs = collections.defaultdict(lambda: collections.defaultdict(list))
for p in sorted(q.glob("replay/*-g*/*.json")):
    mode, gpu = p.parent.name.rsplit("-g", 1)
    r = json.loads(p.read_text())
    sat, run_wait, kv = server(p.with_suffix(".server.log"))
    runs[p.stem][mode].append(dict(
        gpu=gpu, tput=r["output_throughput"], wall=r["wall_s"], sat=sat, run_wait=run_wait, kv=kv,
        pre=preemptions(p.parent / f"{p.stem}.metrics.txt"), pids=max_pids(p.parent / f"{p.stem}.card_pids.txt"),
        requests=r["requests"], tokens=r["output_tokens"]))

print(f"{'task':>11} {'mode':>10} {'GPU':>4} {'requests':>8} {'M tok':>6} {'wall min':>9} {'tok/s':>7} "
      f"{'saturated tok/s':>16} {'running (queue)':>16} {'peak KV%':>9} {'preempt':>8} {'max pids':>9}")
for task in sorted(runs):
    for mode in ("native", "leapquant"):
        for r in runs[task].get(mode, []):
            print(f"{task:>11} {mode:>10} {r['gpu']:>4} {r['requests']:8d} {r['tokens'] / 1e6:6.2f} {r['wall'] / 60:9.1f} "
                  f"{r['tput']:7.0f} {r['sat'] or 0:16.0f} {str(r['run_wait']):>16} {str(r['kv']):>9} "
                  f"{str(r['pre']):>8} {str(r['pids']):>9}")
    n, l = runs[task].get("native"), runs[task].get("leapquant")
    if n and l:
        base = task.split("_x")[0]
        ratio = statistics.mean(r["tput"] for r in l) / statistics.mean(r["tput"] for r in n)
        sats = [r["sat"] for r in n + l]
        sat_ratio = (statistics.mean(r["sat"] for r in l) / statistics.mean(r["sat"] for r in n)
                     if all(sats) else float("nan"))
        print(f"{task:>11} LeapQuant / FP32: overall {ratio:.3f}x, saturated {sat_ratio:.3f}x; "
              f"paper B200 {paper[base][0]:.2f}x, RTX PRO 6000 {paper[base][1]:.2f}x")
    print()
EOF
}

case "${1:-replay}" in
  ref) ref ;;
  replay) replay; summary | tee "$QOUT/summary.txt" ;;
  summary) summary ;;
  *) echo "usage: $0 [ref|replay|summary]" >&2; exit 1 ;;
esac
