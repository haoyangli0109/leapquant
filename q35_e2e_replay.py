"""The paper's end-to-end benchmark (section 4.3, figure 6) against a running
ATOM server: offline inference over a whole dataset with the input and output
lengths of the original model.

Every question of `--task` goes out at once, in dataset order, with the
original model's prompt tokens and `max_tokens` = its reply length under
`ignore_eos`, so each mode generates exactly the same tokens' worth of work.
`--max-inflight` (default 3x the server's 1024 sequences) only bounds the open
connections; the server's queue never runs dry, as in offline inference.
`--repeat N` sends the dataset N times over, as N samples per question would be.
Sampling is the model card's; ATOM does not apply presence_penalty, which only
shaped the reference lengths. Output throughput = generated tokens / wall time
from the first request to the last reply.

    python q35_e2e_replay.py --url http://localhost:8100 --ref DIR/ref --task gsm8k --out run.json
"""

import argparse
import asyncio
import json
import pathlib
import time

import aiohttp

MODEL = "/shareddata/haoyanli/Qwen/Qwen3.5-9B"
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def load_ref(ref_dir, task, limit=None):
    rows = [r for p in sorted(pathlib.Path(ref_dir).glob("ref-[0-9]*.jsonl")) if "-text" not in p.name
            for r in map(json.loads, p.read_text().splitlines()) if r["task"] == task]
    rows.sort(key=lambda r: r["idx"])
    return rows[:limit]


async def one(session, url, row, timeline, sem):
    payload = {
        "model": MODEL,
        "prompt_token_ids": row["prompt_token_ids"],
        "max_tokens": row["output_len"],
        "ignore_eos": True,
        **SAMPLING,
    }
    async with sem:
        t0 = time.time()
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            body = await resp.json()
        t1 = time.time()
    if resp.status != 200:
        raise RuntimeError(f"{row['task']}-{row['idx']}: HTTP {resp.status}: {str(body)[:300]}")
    got = body["usage"]["completion_tokens"]
    timeline.append((row["idx"], t0, t1, row["prompt_len"], row["output_len"], got))


async def replay(args):
    rows = load_ref(args.ref, args.task, args.limit)
    if not rows:
        raise SystemExit(f"no reference replies for {args.task} in {args.ref}")
    rows = rows * args.repeat
    timeline = []
    sem = asyncio.Semaphore(args.max_inflight)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=6 * 3600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t0 = time.time()
        await asyncio.gather(*(one(session, args.url, r, timeline, sem) for r in rows))
        wall = time.time() - t0
    out_tokens = sum(t[5] for t in timeline)
    want = sum(r["output_len"] for r in rows)
    result = dict(
        task=args.task, repeat=args.repeat, requests=len(rows), output_tokens=out_tokens,
        expected_output_tokens=want,
        prompt_tokens=sum(r["prompt_len"] for r in rows), wall_s=wall,
        output_throughput=out_tokens / wall,
        short_replies=sum(t[5] != t[4] for t in timeline),
        timeline=sorted(timeline),
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result))
    print(f"{args.task}: {len(rows)} requests, {out_tokens} output tokens (expected {want}), "
          f"{wall:.1f} s, {out_tokens / wall:.0f} tok/s, short replies {result['short_replies']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--task", required=True, choices=["gsm8k", "mmlupro", "gpqa"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke tests)")
    ap.add_argument("--max-inflight", type=int, default=3072)
    ap.add_argument("--repeat", type=int, default=1)
    asyncio.run(replay(ap.parse_args()))


if __name__ == "__main__":
    main()
