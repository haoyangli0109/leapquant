"""Kimi-Linear on ATOM against the Hugging Face reference, teacher forced.

  HIP_VISIBLE_DEVICES=0 python kimi_ref_check.py --tag fp32 --model $M \
      --trust-remote-code --enforce-eager
  (the same with ATOM_ENABLE_LEAPQUANT=1 --tag lq)
  HIP_VISIBLE_DEVICES=0 PYTHONPATH=/tmp/fla_pkg python kimi_ref_check.py --score bfloat16 fp32 lq
  HIP_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/fla_pkg python kimi_ref_check.py --score float32 fp32 lq
  python kimi_ref_check.py --report fp32 lq

An ATOM run saves greedy replies with each sampled token's logprob. --score
runs prompt plus reply through the checkpoint's modeling_kimi.py, with the bf16
weights computed in the given dtype, and reports how often ATOM's token is the
reference's argmax and the mean |logprob difference|. The bf16 reference
against the fp32 one is the noise floor bf16 arithmetic alone sets. The
reference needs fla-core (pip install --no-deps --target /tmp/fla_pkg
fla-core) and runs its MLA with sdpa: flash-attn is not installed.
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results_ref"
MODEL = pathlib.Path("/shareddata/haoyanli/moonshotai/Kimi-Linear-48B-A3B-Instruct")
REF = pathlib.Path("/tmp/kimi_hf_ref")
PROMPTS = [
    "Write a long, detailed essay on the history of the printing press.",
    "Write a long, detailed essay on why bridges fail, with three famous examples.",
    "Explain step by step how to solve x^2 - 5x + 6 = 0, then check both roots.",
    "A train travels 120 km in 1.5 hours. What is its average speed? Show your work.",
    "Write a Python function that returns the n-th Fibonacci number, and explain it.",
    "Describe the life cycle of a star, from nebula to remnant.",
    "List ten countries in Africa with their capitals, one per line.",
    "Translate to French and German: 'The weather is nice today, let us go for a walk.'",
]
MAX_TOKENS = 256
LONG_TOKENS = 16384
TIE = 0.1  # top-2 logprob margin under which a different argmax is a near tie
BUCKETS = (0, 512, 2048, 4096, 8192, LONG_TOKENS)


def long_prompts(n):
    """The n GPQA questions the FP32 server answered at greatest length
    without running out of tokens, as reason_eval.py asked them."""
    import collections

    from reason_eval import load_items

    lens = collections.defaultdict(list)
    for line in open(HERE / "results_reason/kimi-fp32/gpqa.jsonl"):
        r = json.loads(line)
        if r["finish"] == "stop":
            lens[r["idx"]].append(r["completion_tokens"])
    picked = sorted(lens, key=lambda q: -sum(lens[q]) / len(lens[q]))[:n]
    items = {it["idx"]: it["prompt"] for it in load_items("gpqa", 1)}
    return [items[q] for q in picked]


def run(args, engine_args):
    from atom import SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    texts = long_prompts(args.long) if args.long else PROMPTS
    prompts = [
        tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                    add_generation_prompt=True),
            add_special_tokens=False).input_ids
        for p in texts
    ]
    llm = engine_args.create_engine()
    # Sampled as reason_eval.py does for Kimi; the logprobs stay those of the raw logits.
    sampling = (SamplingParams(temperature=0.6, max_tokens=LONG_TOKENS, logprobs=True)
                if args.long else
                SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=True))
    outs = llm.generate(prompts, sampling)
    llm.close()
    res = {"tag": args.tag, "replies": [
        {"prompt": p, "token_ids": list(o["token_ids"]), "logprobs": list(o["logprobs"])}
        for p, o in zip(prompts, outs)
    ]}
    OUT.mkdir(exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(res))
    print(f"saved {sum(len(r['token_ids']) for r in res['replies'])} tokens to "
          f"{OUT / (args.tag + '.json')}")


# modeling_kimi.py targets transformers 4.57 and flash-attn.
PATCHES = [
    ('"flash_attention_2"', '"sdpa"'),
    ("from transformers.utils.generic import OutputRecorder, check_model_inputs",
     "from transformers.utils.generic import check_model_inputs\n"
     "from transformers.utils.output_capturing import OutputRecorder"),
    ("            input_embeds=inputs_embeds,\n            attention_mask=attention_mask,\n"
     "            cache_position=cache_position,\n",
     "            inputs_embeds=inputs_embeds,\n            attention_mask=attention_mask,\n"),
    # fla-core 0.5 changed fused_kda_gate's signature; this is its formula.
    ("        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)\n",
     "        g = -self.A_log.float().exp() * F.softplus(rearrange(\n"
     "            g.float() + self.dt_bias.float(), '... (h d) -> ... h d', d=self.head_dim))\n"),
]


def reference_dir():
    """The checkpoint, with modeling_kimi.py patched to run here."""
    REF.mkdir(exist_ok=True)
    for f in MODEL.iterdir():
        dst = REF / f.name
        if f.name == "modeling_kimi.py":
            text = f.read_text()
            for old, new in PATCHES:
                assert old in text, old
                text = text.replace(old, new)
            dst.write_text(text)
        elif not dst.exists():
            dst.symlink_to(f)
    return REF


def score(dtype, tags):
    """The reference's view of each run's replies, saved as <tag>.ref-<dtype>.json:
    per token, the logprob of ATOM's token and the reference's own argmax."""
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        reference_dir(), trust_remote_code=True, dtype=getattr(torch, dtype), device_map="cuda"
    ).eval()
    # The checkpoint stores the KDA gate parameters in fp32, as ATOM keeps them.
    where = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    for name, p in model.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            with safe_open(MODEL / where[name], "pt", device="cuda") as f:
                p.data = f.get_tensor(name)
    for tag in tags:
        scored = []
        for r in json.loads((OUT / f"{tag}.json").read_text())["replies"]:
            p, toks = r["prompt"], r["token_ids"]
            with torch.no_grad():
                logits = model(torch.tensor([p + toks], device="cuda"), use_cache=False).logits
            lp = logits[0, len(p) - 1:-1].float().log_softmax(-1)
            top = lp.max(-1)
            scored.append({
                "lp": lp.gather(1, torch.tensor(toks, device="cuda")[:, None])[:, 0].tolist(),
                "top_id": top.indices.tolist(),
                "top_lp": top.values.tolist(),
            })
        (OUT / f"{tag}.ref-{dtype}.json").write_text(json.dumps(scored))
    report(tags)


def report(tags):
    """ATOM against each saved reference, and the bf16 reference against fp32."""
    print(f"{'':36}{'tokens':>8}{'argmax':>9}{'ties':>6}{'mean |dlp|':>12}"
          f"{'p99 |dlp|':>11}{'max |dlp|':>11}  worst (reply, position)")
    for tag in tags:
        run_ = json.loads((OUT / f"{tag}.json").read_text())["replies"]
        refs = {d: json.loads(f.read_text()) for d in ("float32", "bfloat16")
                if (f := OUT / f"{tag}.ref-{d}.json").exists()}
        atom = [{"lp": r["logprobs"], "top_id": r["token_ids"], "top_lp": r["logprobs"]}
                for r in run_]
        pairs = [(f"{tag} vs ref-{d}", atom, ref) for d, ref in refs.items()]
        if len(refs) == 2:
            pairs.append((f"  ref-bfloat16 vs ref-float32", refs["bfloat16"], refs["float32"]))
        for name, a, b in pairs:
            diffs, hits, ties = [], 0, 0
            for i, (x, y) in enumerate(zip(a, b)):
                for j in range(len(x["lp"])):
                    hit = x["top_id"][j] == y["top_id"][j]
                    hits += hit
                    # A miss the reference itself nearly calls: ATOM's token is close to its top.
                    ties += not hit and y["top_lp"][j] - y["lp"][j] < TIE
                    diffs.append((abs(x["lp"][j] - y["lp"][j]), i, j))
            n = len(diffs)
            d = sorted(diffs)
            worst = ", ".join(f"({i}, {j})" for _, i, j in d[-3:][::-1])
            print(f"{name:36}{n:>8}{hits / n:>9.4f}{ties:>6}"
                  f"{sum(x for x, _, _ in d) / n:>12.2e}{d[int(0.99 * (n - 1))][0]:>11.2e}"
                  f"{d[-1][0]:>11.2e}  {worst}")
            if max(len(x["lp"]) for x in a) > BUCKETS[1]:
                for lo, hi in zip(BUCKETS, BUCKETS[1:]):
                    cell = [x for x, _, j in diffs if lo <= j < hi]
                    if cell:
                        print(f"{f'    tokens [{lo}, {hi})':36}{len(cell):>8}"
                              f"{'':15}{sum(cell) / len(cell):>12.2e}")


def main():
    if "--score" in sys.argv:
        i = sys.argv.index("--score")
        score(sys.argv[i + 1], sys.argv[i + 2:])
        return
    if "--report" in sys.argv:
        report(sys.argv[sys.argv.index("--report") + 1:])
        return
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--long", type=int, default=0,
                        help="sample this many long GPQA replies instead")
    args = parser.parse_args()
    run(args, EngineArgs.from_cli_args(args))


if __name__ == "__main__":
    main()
