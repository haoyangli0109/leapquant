"""Greedy replies from a running server, to eyeball a Kimi-Linear build.

    python kimi_smoke.py [--url http://localhost:8000] [--out replies.json]

`--out` saves the replies; `--ref` compares against a saved file and prints
how many tokens each reply shares with the reference before they diverge.
"""

import argparse
import concurrent.futures
import json
import urllib.request

PROMPTS = [
    "What is 17*23? Answer briefly.",
    "Write a haiku about the ocean.",
    "Explain in two sentences why the sky is blue.",
    "List the first ten prime numbers.",
    "Translate to French: 'The weather is nice today, let us go for a walk.'",
    "A train travels 120 km in 1.5 hours. What is its average speed? Show your work.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Who wrote 'Pride and Prejudice', and in what year was it published?",
]


def ask(url, model, prompt, max_tokens):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument(
        "--model", default="/shareddata/haoyanli/moonshotai/Kimi-Linear-48B-A3B-Instruct"
    )
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out")
    p.add_argument("--ref")
    args = p.parse_args()

    with concurrent.futures.ThreadPoolExecutor(len(PROMPTS)) as ex:
        replies = list(
            ex.map(lambda q: ask(args.url, args.model, q, args.max_tokens), PROMPTS)
        )
    ref = json.load(open(args.ref)) if args.ref else None
    for i, (q, a) in enumerate(zip(PROMPTS, replies)):
        print(f"--- [{i}] {q}")
        print(a)
        if ref is not None:
            n = next(
                (j for j, (x, y) in enumerate(zip(a, ref[i])) if x != y),
                min(len(a), len(ref[i])),
            )
            print(f"    [same as reference for the first {n} chars of {len(ref[i])}]")
    bad = [i for i, a in enumerate(replies) if not a or "!!!!" in a or "nan" in a]
    print(f"\n{len(replies)} replies, {len(bad)} suspicious: {bad}")
    if args.out:
        json.dump(replies, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
