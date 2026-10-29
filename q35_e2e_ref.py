"""Qwen3.5-9B's own replies to GSM8K, MMLU-Pro and GPQA-Diamond, for the
paper's end-to-end benchmark (section 4.3): it replays every question with "the
input and output lengths of the original model".

The original model runs in vLLM as in the paper, without the ATOM plugin: FP32
state (the checkpoint's `mamba_ssm_dtype`), prefix caching in its default mode,
the model card's thinking-mode sampling (temperature 1.0, top_p 0.95, top_k 20,
presence_penalty 1.5), up to 81,920 output tokens. One reply per question.

Questions are dealt round-robin to `--shards` processes; each writes one line
per finished reply to `ref-<shard>.jsonl` (replies in `ref-<shard>-text.jsonl`)
and skips the ones already there, so a restarted shard resumes.

    VLLM_PLUGINS="" HIP_VISIBLE_DEVICES=4 python q35_e2e_ref.py --shard 0 --shards 4 --out DIR
    python q35_e2e_ref.py --summary --out DIR
"""

import argparse
import json
import os
import pathlib
import re
import statistics
import sys
import time

MODEL = "/shareddata/haoyanli/Qwen/Qwen3.5-9B"
DATA = "/shareddata/huggingface/datasets"
MAX_TOKENS = 81920
# Output limit plus 8,192 (the paper's appendix E).
MAX_MODEL_LEN = MAX_TOKENS + 8192
SAMPLING = dict(
    temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
)
# GPQA first: its long replies set the tail, so they start before the rest.
TASKS = ("gpqa", "mmlupro", "gsm8k")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reason_eval as R  # noqa: E402


def load_items(task):
    if task in ("gpqa", "mmlupro"):
        return [dict(task=task, **it) for it in R.load_items(task, 1)]
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=DATA)
    return [
        dict(
            task="gsm8k", idx=i, sample=0,
            prompt=f"{row['question'].strip()}\n\n{R.AIME_SUFFIX}",
            gold=row["answer"].split("####")[-1].strip().replace(",", ""),
        )
        for i, row in enumerate(ds)
    ]


def number(s):
    """The one number in a boxed GSM8K answer ("$18", "18 dollars", "1,200")."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(nums[0]) if len(nums) == 1 else None


def correct(task, text, gold, finish_reason):
    # Truncated replies count as wrong; the rest are scored after the reasoning.
    if finish_reason == "length":
        return False
    answer = text.split("</think>")[-1]
    if task == "gsm8k":
        got = number(R.last_boxed(answer))
        return got is not None and abs(got - float(gold)) < 1e-6
    return R.extract(task, answer) == gold


def generate(args):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, text_path = out / f"ref-{args.shard}.jsonl", out / f"ref-{args.shard}-text.jsonl"
    done = set()
    if meta_path.exists():
        done = {(r["task"], r["idx"]) for r in map(json.loads, meta_path.read_text().splitlines())}
    items = [it for t in TASKS for it in load_items(t)]
    mine = [it for k, it in enumerate(items) if k % args.shards == args.shard]
    todo = [it for it in mine if (it["task"], it["idx"]) not in done]
    print(f"shard {args.shard}/{args.shards}: {len(mine)} questions, {len(todo)} to go", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9,
        seed=args.shard, limit_mm_per_prompt={"image": 0, "video": 0},
    )
    cc = llm.llm_engine.vllm_config.cache_config
    print(f"vLLM: ssm state {cc.mamba_ssm_cache_dtype}, mamba cache {cc.mamba_cache_mode} "
          f"every {cc.mamba_block_size} tokens, prefix caching {cc.enable_prefix_caching}, "
          f"max_num_seqs {llm.llm_engine.vllm_config.scheduler_config.max_num_seqs}", flush=True)
    params = SamplingParams(max_tokens=MAX_TOKENS, **SAMPLING)
    engine = llm.llm_engine
    by_id = {}
    for it in todo:
        text = tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, add_special_tokens=False).input_ids
        rid = f"{it['task']}-{it['idx']}"
        by_id[rid] = (it, ids)
        engine.add_request(rid, {"prompt_token_ids": ids}, params)

    t0, n_done, n_tok = time.time(), 0, 0
    with meta_path.open("a") as fm, text_path.open("a") as ft:
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                it, ids = by_id.pop(o.request_id)
                c = o.outputs[0]
                rec = dict(
                    task=it["task"], idx=it["idx"], prompt_len=len(ids), output_len=len(c.token_ids),
                    finish_reason=c.finish_reason, gold=it["gold"],
                    correct=correct(it["task"], c.text, it["gold"], c.finish_reason),
                    prompt_token_ids=ids,
                )
                fm.write(json.dumps(rec) + "\n")
                ft.write(json.dumps(dict(task=it["task"], idx=it["idx"], text=c.text)) + "\n")
                fm.flush()
                ft.flush()
                n_done += 1
                n_tok += len(c.token_ids)
                if n_done % 200 == 0:
                    dt = time.time() - t0
                    print(f"[{dt / 60:6.1f} min] {n_done}/{len(todo)} done, {n_tok / dt:.0f} tok/s", flush=True)
    print(f"shard {args.shard} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def summary(args):
    rows = [r for p in sorted(pathlib.Path(args.out).glob("ref-[0-9]*.jsonl")) if "-text" not in p.name
            for r in map(json.loads, p.read_text().splitlines())]
    paper = {"gsm8k": 96.1, "gpqa": 81.3, "mmlupro": 83.3}
    print(f"{'task':>8} {'n':>6} {'acc':>6} {'paper':>6} {'prompt mean':>12} {'output mean':>12} "
          f"{'median':>7} {'p90':>7} {'max':>7} {'truncated':>10} {'output tokens':>14}")
    for t in TASKS:
        rs = [r for r in rows if r["task"] == t]
        if not rs:
            continue
        lens = sorted(r["output_len"] for r in rs)
        print(f"{t:>8} {len(rs):6d} {100 * statistics.mean(r['correct'] for r in rs):6.1f} {paper[t]:6.1f} "
              f"{statistics.mean(r['prompt_len'] for r in rs):12.0f} {statistics.mean(lens):12.0f} "
              f"{statistics.median(lens):7.0f} {lens[int(0.9 * (len(lens) - 1))]:7d} {lens[-1]:7d} "
              f"{sum(r['finish_reason'] == 'length' for r in rs):10d} {sum(lens):14d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
