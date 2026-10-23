"""The GDN state under the prefix cache, end to end: offline engine, one GPU.

  HIP_VISIBLE_DEVICES=5 ATOM_ENABLE_LEAPQUANT=1 python prefix_e2e.py --tag lq-cache \
      --model /shareddata/haoyanli/Qwen/Qwen3.5-9B --trust-remote-code \
      --enforce-eager --state-checkpoint-interval-tokens 256
  (the same with --no-enable_prefix_caching --tag lq-nocache)
  python prefix_e2e.py --compare lq-cache lq-nocache

repeat     One prompt twice. The second resumes from a checkpoint a prefill
           wrote and must decode the same tokens as the first.
follow-up  A prompt plus the start of its own greedy reply, cut 1 or 41 tokens
           past one of the checkpoints the reply took between decode steps.
           It resumes from that checkpoint, a boundary plus records, so its
           continuation should track the reply's: same tokens, and sampled
           logprobs closer to the reply's than a cold prefill gets. The
           prompts differ in length mod 16, the hash block, so their
           checkpoints hold different numbers of records.
gsm8k      5-shot questions behind one shared prefix, greedy; accuracy.

Checkpoints share the state pool with running requests and are evicted when
it fills, so give it room: --max-num-seqs 64, and --gsm8k-wave 8.
"""

import argparse
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results_prefix"
GSM8K = pathlib.Path(
    "/shareddata/huggingface/datasets/openai___gsm8k/main/0.0.0/"
    "740312add88f781978c0658806c59bc2815b9866"
)
TOPICS = [
    "the history of the printing press",
    "how a city should plan for a once-in-a-century flood",
    "why bridges fail, with three famous examples",
    "the life cycle of a star, from nebula to remnant",
]
HBS = 16  # prefix-cache hash block
INTERVAL = 256  # --state-checkpoint-interval-tokens of the runs with a ladder
RUNGS = (1, 2, 3)
PAST = (1, 41)
FOLLOW_TOKENS = 128
CLOSE = 32  # leading tokens whose logprobs are compared


def decode_checkpoints(p: int):
    """(position, records) of the checkpoints a reply to a p-token prompt takes.

    The prompt-end anchor sits on the hash grid strictly inside the prompt;
    generation then keeps one at the first block boundary INTERVAL past the
    last. The state at position b covers tokens [0, b); the reply's first
    decode step wrote record 0, and a fold empties the window every 16.
    """
    anchor = (p - 1) // HBS * HBS
    return [(b, (b - 1 - p) % HBS + 1) for b in (anchor + k * INTERVAL for k in RUNGS)]


def logprob_gap(cont, reply, m: int) -> float | None:
    """Mean |logprob difference| over the tokens both sides sampled alike."""
    n = min(agree(cont["token_ids"], reply["token_ids"][m:]), CLOSE)
    if n == 0:
        return None
    a, b = cont["logprobs"][:n], reply["logprobs"][m:m + n]
    return sum(abs(x - y) for x, y in zip(a, b)) / n


def number(text: str):
    found = re.findall(r"-?[\d,]*\.?\d+", text.split("Question:")[0])
    return found[-1].replace(",", "").rstrip(".") if found else None


def shots(k: int) -> str:
    from datasets import Dataset

    train = Dataset.from_file(str(GSM8K / "gsm8k-train.arrow"))
    return "".join(
        f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
        for r in train.select(range(k))
    )


def gsm8k_prompts(n: int):
    from datasets import Dataset

    test = Dataset.from_file(str(GSM8K / "gsm8k-test.arrow"))
    prefix = shots(5)
    rows = test.select(range(n))
    prompts = [prefix + f"Question: {r['question']}\nAnswer:" for r in rows]
    golds = [r["answer"].split("####")[-1].strip().replace(",", "") for r in rows]
    return prompts, golds


def agree(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def run(args, engine_args):
    from atom import SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = engine_args.create_engine()

    def greedy(n, stop=None, logprobs=None):
        return SamplingParams(temperature=0.0, max_tokens=n, stop_strings=stop,
                              logprobs=logprobs)

    def chat(text):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tok(rendered, add_special_tokens=False).input_ids

    def stats():
        s = llm.get_cache_statistics()
        return {k: s.get(k) for k in ("requests", "cached_tokens", "reusable_tokens",
                                      "checkpoints_kept", "checkpoints_evicted", "hit")}

    res = {"tag": args.tag, "stats": {}}
    t0 = time.time()

    long_doc = chat("Summarise these worked problems in five bullet points.\n\n" + shots(16))
    first = llm.generate([long_doc], greedy(128))[0]["token_ids"]
    second = llm.generate([long_doc], greedy(128))[0]["token_ids"]
    res["repeat"] = {"prompt_len": len(long_doc), "first": first, "second": second}
    res["stats"]["repeat"] = stats()
    print(f"repeat: prompt {len(long_doc)} tokens, identical={first == second}",
          flush=True)

    res["follow"] = []
    for topic in TOPICS:
        prompt = chat(f"Write a long, detailed essay on {topic}.")
        p = len(prompt)
        cuts = [(b + d - p, b, n) for b, n in decode_checkpoints(p) for d in PAST]
        out = llm.generate([prompt], greedy(cuts[-1][0] + FOLLOW_TOKENS + HBS, logprobs=True))
        reply = {k: out[0][k] for k in ("token_ids", "logprobs")}
        entry = {"prompt_len": p, "reply": reply, "cont": {}}
        for m, b, n in cuts:
            if m + FOLLOW_TOKENS > len(reply["token_ids"]):
                continue
            out = llm.generate([prompt + reply["token_ids"][:m]],
                               greedy(FOLLOW_TOKENS, logprobs=True))
            cont = {k: out[0][k] for k in ("token_ids", "logprobs")}
            entry["cont"][m] = cont
            gap = logprob_gap(cont, reply, m)
            print(f"follow-up: prompt {p}, checkpoint {b} with {n} records, "
                  f"resumed {p + m - b} past it: agrees for "
                  f"{agree(cont['token_ids'], reply['token_ids'][m:])}/{FOLLOW_TOKENS}, "
                  f"logprob gap {gap if gap is None else f'{gap:.2e}'}", flush=True)
        res["follow"].append(entry)
    res["stats"]["follow"] = stats()

    res["gsm8k"] = None
    if args.n_gsm8k > 0:
        prompts, golds = gsm8k_prompts(args.n_gsm8k)
        wave = args.gsm8k_wave or len(prompts)
        outs = []
        for i in range(0, len(prompts), wave):
            outs += llm.generate(prompts[i:i + wave], greedy(256, ["Question:"]))
        preds = [number(o["text"]) for o in outs]
        res["gsm8k"] = {"texts": [o["text"] for o in outs], "preds": preds,
                        "golds": golds,
                        "acc": sum(p == g for p, g in zip(preds, golds)) / len(golds)}
        res["stats"]["gsm8k"] = stats()
        print(f"gsm8k: {res['gsm8k']['acc']:.4f} over {len(golds)}", flush=True)

    res["seconds"] = time.time() - t0
    print(json.dumps(res["stats"], indent=1))
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(res))
    llm.close()


def compare(tags):
    """Follow-up cells read "<tokens agreeing> <logprob gap>"."""
    runs = [json.loads((OUT / f"{t}.json").read_text()) for t in tags]
    print(f"{'':30}" + "".join(f"{r['tag']:>20}" for r in runs))
    rows = [("repeat: 2nd == 1st", [str(r["repeat"]["first"] == r["repeat"]["second"])
                                     for r in runs])]
    gaps = [{d: [] for d in PAST} for _ in runs]
    spans = [{d: [] for d in PAST} for _ in runs]
    for i in range(len(TOPICS)):
        base = runs[0]["follow"][i]["reply"]
        cells = []
        for r in runs:
            reply = r["follow"][i]["reply"]
            same = agree(reply["token_ids"], base["token_ids"])
            cells.append(f"{same}/{len(base['token_ids'])}"
                         + (" bitwise" if reply == base else ""))
        rows.append((f"reply {i} vs {runs[0]['tag']}", cells))
        p = runs[0]["follow"][i]["prompt_len"]
        for b, n in decode_checkpoints(p):
            for d in PAST:
                m, cells = b + d - p, []
                for j, r in enumerate(runs):
                    e = r["follow"][i]
                    c = e["cont"].get(str(m))
                    if c is None:
                        cells.append("-")
                        continue
                    g = logprob_gap(c, e["reply"], m)
                    if g is not None:
                        gaps[j][d].append(g)
                    same = agree(c["token_ids"], e["reply"]["token_ids"][m:])
                    spans[j][d].append(same)
                    cells.append(f"{same} {'-' if g is None else f'{g:.1e}'}")
                rows.append((f"  {b} ({n:2d} records) +{d}", cells))
    for d in PAST:
        rows.append((f"mean logprob gap +{d}", [f"{sum(g[d]) / len(g[d]):.2e}" if g[d] else "-"
                                                for g in gaps]))
        rows.append((f"first token differs +{d}", [f"{s[d].count(0)}/{len(s[d])}" for s in spans]))
        rows.append((f"apart within 3 tokens +{d}", [f"{sum(x <= 3 for x in s[d])}/{len(s[d])}"
                                                     for s in spans]))
    last = [r["stats"]["gsm8k" if r["gsm8k"] else "follow"] for r in runs]
    rows.append(("gsm8k acc", [f"{r['gsm8k']['acc']:.4f}" if r["gsm8k"] else "-"
                               for r in runs]))
    rows.append(("prefix hit rate", [f"{s['hit'] or 0:.3f}" for s in last]))
    rows.append(("checkpoints kept", [str(s["checkpoints_kept"]) for s in last]))
    for name, cells in rows:
        print(f"{name:30}" + "".join(f"{c:>20}" for c in cells))
    graded = [r for r in runs if r["gsm8k"]]
    if len(graded) == 2:
        a, b = (r["gsm8k"]["preds"] for r in graded)
        print(f"gsm8k answers that differ: {sum(x != y for x, y in zip(a, b))}/{len(a)}")


def main():
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    if "--compare" in sys.argv:
        compare(sys.argv[sys.argv.index("--compare") + 1:])
        return
    parser = FlexibleArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n-gsm8k", type=int, default=250)
    # Requests in flight at once. Checkpoints share the state pool with
    # running requests, so a full pool keeps none and nothing hits.
    parser.add_argument("--gsm8k-wave", type=int, default=0)
    args = parser.parse_args()
    run(args, EngineArgs.from_cli_args(args))


if __name__ == "__main__":
    main()
