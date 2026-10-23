"""AIME 2026 and GPQA-Diamond accuracy of running ATOM servers, thinking mode.

As in the paper (section 4.1, appendix E): the model card's thinking-mode
sampling, three samples per question, up to 81,920 output tokens.  Each
(question, sample) becomes one line of ``results_reason/<tag>/<task>.jsonl``
as soon as it finishes, so an interrupted run resumes where it stopped; the
full replies go to ``<task>_outputs.jsonl`` next to it.

    python reason_eval.py --tag fp32 --url http://localhost:8000
    python reason_eval.py --tag bf16 --url http://localhost:8000 --url http://localhost:8001
    python reason_eval.py --summary

Kimi-Linear (an instruct model: no thinking, shorter replies):

    python reason_eval.py --tag kimi-fp32 --model <Kimi-Linear path> \
        --sampling '{"temperature": 0.6, "top_p": 1.0, "top_k": -1}' --max-tokens 32768
    python reason_eval.py --summary --base kimi-fp32 --tags kimi-fp32 kimi-lq
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import pathlib
import random
import re
import statistics
import time

import aiohttp

MODEL = "/shareddata/haoyanli/Qwen/Qwen3.8-Flash-Next"
DATA = "/shareddata/huggingface/datasets"
OUT = pathlib.Path(__file__).resolve().parent / "results_reason"
TASKS = ("aime26", "gpqa")
# min_p 0 and repetition_penalty 1 are the server's defaults already.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
AIME_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
GPQA_SUFFIX = (
    "Please reason step by step, and put the letter of the correct option "
    "(A, B, C or D) within \\boxed{}."
)


def load_items(task: str, samples: int) -> list[dict]:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset

    items = []
    if task == "aime26":
        ds = load_dataset("math-ai/aime26", split="test", cache_dir=DATA)
        for s in range(samples):
            for i, row in enumerate(ds):
                prompt = f"{row['problem'].strip()}\n\n{AIME_SUFFIX}"
                items.append(dict(idx=i, sample=s, prompt=prompt, gold=str(int(row["answer"]))))
        return items
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir=DATA)
    for s in range(samples):
        for i, row in enumerate(ds):
            options = [row["Correct Answer"]] + [row[f"Incorrect Answer {j}"] for j in (1, 2, 3)]
            order = list(range(4))
            # The same shuffle for every mode, a different one per sample.
            random.Random(f"gpqa-{i}-{s}").shuffle(order)
            lines = [f"{'ABCD'[p]}) {options[o].strip()}" for p, o in enumerate(order)]
            prompt = f"{row['Question'].strip()}\n\n" + "\n".join(lines) + f"\n\n{GPQA_SUFFIX}"
            items.append(dict(idx=i, sample=s, prompt=prompt, gold="ABCD"[order.index(0)]))
    return items


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for k in range(brace, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : k].strip()
    return None


def extract(task: str, content: str) -> str | None:
    box = last_boxed(content)
    if task == "aime26":
        if box is None:
            return None
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", box)
        s = re.sub(r"\\[,!; ]|[$\s,]|\^\{?\\circ\}?", "", s)
        nums = re.findall(r"\d+", s)
        # "277", "m+n=277": one integer.  "2^{10}" stays unparsed on purpose.
        return str(int(nums[0])) if len(nums) == 1 else None
    if box is not None:
        m = re.search(r"\b([ABCD])\b", re.sub(r"\\text(?:bf)?\{([^}]*)\}", r"\1", box))
        return m.group(1) if m else None
    found = re.findall(r"[Aa]nswer(?: is)?\s*:?\s*\**\(?([ABCD])\)?\b", content)
    return found[-1] if found else None


async def run_one(session, url, task, item, args, files):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": args.max_tokens,
        **SAMPLING,
        **args.sampling,
    }
    t0 = time.time()
    for attempt in range(3):
        try:
            async with session.post(f"{url}/v1/chat/completions", json=payload) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
                data = await r.json()
            break
        except Exception as e:  # noqa: BLE001 -- retried, then left for a resume
            print(f"[{task} {item['idx']}/{item['sample']} @ {url}] attempt {attempt}: {e}", flush=True)
            await asyncio.sleep(10 * (attempt + 1))
    else:
        return None
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    pred = extract(task, content)
    rec = dict(
        idx=item["idx"],
        sample=item["sample"],
        gold=item["gold"],
        pred=pred,
        correct=pred == item["gold"],
        finish=choice.get("finish_reason"),
        completion_tokens=data["usage"]["completion_tokens"],
        latency_s=round(time.time() - t0, 1),
        # NaN logits decode as token 0, i.e. a run of "!".
        bang="!!!!!!!!" in reasoning or "!!!!!!!!" in content,
        url=url,
    )
    files[task][0].write(json.dumps(rec) + "\n")
    files[task][0].flush()
    full = dict(idx=item["idx"], sample=item["sample"], reasoning=reasoning, content=content)
    files[task][1].write(json.dumps(full) + "\n")
    files[task][1].flush()
    return rec


async def evaluate(args):
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue = asyncio.Queue()
    files, total = {}, 0
    # AIME first: its long replies set the tail, so they start before GPQA's.
    for task in args.tasks:
        path = out / f"{task}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                done.add((r["idx"], r["sample"]))
        todo = [it for it in load_items(task, args.samples) if (it["idx"], it["sample"]) not in done]
        print(f"{task}: {len(done)} done, {len(todo)} to run", flush=True)
        for it in todo:
            queue.put_nowait((task, it))
        total += len(todo)
        files[task] = (path.open("a"), (out / f"{task}_outputs.jsonl").open("a"))

    stats = collections.Counter()
    t_start = time.time()

    async def worker(session, url):
        while True:
            try:
                task, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            rec = await run_one(session, url, task, item, args, files)
            if rec is None:
                stats["failed"] += 1
                continue
            stats["done"] += 1
            stats[f"{task}_n"] += 1
            stats[f"{task}_ok"] += rec["correct"]
            stats["tokens"] += rec["completion_tokens"]

    async def report():
        while True:
            await asyncio.sleep(120)
            el = time.time() - t_start
            accs = "  ".join(
                f"{t} {stats[f'{t}_ok']}/{stats[f'{t}_n']}" for t in args.tasks if stats[f"{t}_n"]
            )
            print(
                f"[{el / 60:6.1f} min] {stats['done']}/{total} done, {stats['failed']} failed, "
                f"{stats['tokens'] / el:.0f} tok/s  {accs}",
                flush=True,
            )

    timeout = aiohttp.ClientTimeout(total=8 * 3600)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        reporter = asyncio.create_task(report())
        await asyncio.gather(
            *(worker(session, url) for url in args.url for _ in range(args.concurrency))
        )
        reporter.cancel()
    for a, b in files.values():
        a.close()
        b.close()
    print(f"finished in {(time.time() - t_start) / 60:.1f} min, {stats['failed']} failed", flush=True)


def per_question(tag: str, task: str) -> dict[int, float] | None:
    path = OUT / tag / f"{task}.jsonl"
    if not path.exists():
        return None
    acc = collections.defaultdict(list)
    for line in path.read_text().splitlines():
        r = json.loads(line)
        acc[r["idx"]].append(r["correct"])
    return {i: sum(v) / len(v) for i, v in acc.items()}


def summary(base: str = "fp32", only: list[str] | None = None):
    tags = sorted(p.name for p in OUT.iterdir() if p.is_dir())
    if only:
        tags = [t for t in tags if t in only]
    for tag_dir in (OUT / t for t in tags):
        for task in TASKS:
            path = tag_dir / f"{task}.jsonl"
            if not path.exists():
                continue
            recs = [json.loads(l) for l in path.read_text().splitlines()]
            by_sample = collections.defaultdict(list)
            for r in recs:
                by_sample[r["sample"]].append(r["correct"])
            per = {s: 100 * sum(v) / len(v) for s, v in sorted(by_sample.items())}
            mean = statistics.mean(per.values())
            toks = statistics.mean(r["completion_tokens"] for r in recs)
            cut = sum(r["finish"] == "length" for r in recs)
            bang = sum(r["bang"] for r in recs)
            per_s = " ".join(f"{v:5.1f}" for v in per.values())
            print(
                f"{tag_dir.name:>10} {task:>7}: mean {mean:5.1f}  per-sample [{per_s}]  "
                f"n={len(recs)}  avg tokens {toks:7.0f}  truncated {cut}  '!!!!' {bang}"
            )
    # Per question, the share of its samples answered correctly, against `base`.
    for tag in tags:
        for task in TASKS:
            b, o = per_question(base, task), per_question(tag, task)
            if tag == base or not b or not o:
                continue
            common = b.keys() & o.keys()
            worse = sum(o[i] < b[i] for i in common)
            better = sum(o[i] > b[i] for i in common)
            print(f"{tag:>10} {task:>7} vs {base}: {len(common)} questions, worse on {worse}, better on {better}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="results_reason/<tag>; required unless --summary")
    ap.add_argument("--url", action="append", help="server base URL; repeat to spread the load")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=81920)
    ap.add_argument("--concurrency", type=int, default=128, help="in-flight requests per URL")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument(
        "--sampling", type=json.loads, default={},
        help='JSON overriding the thinking-mode sampling, e.g. \'{"temperature": 0.6}\'',
    )  # fmt: skip
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--base", default="fp32", help="--summary: the tag to compare against")
    ap.add_argument("--tags", nargs="+", help="--summary: only these tags")
    args = ap.parse_args()
    if args.summary:
        summary(args.base, args.tags)
    else:
        if not args.tag:
            ap.error("--tag is required")
        args.url = args.url or ["http://localhost:8000"]
        asyncio.run(evaluate(args))
