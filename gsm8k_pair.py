"""Question-by-question agreement between GSM8K runs.  python gsm8k_pair.py base other..."""
import glob
import json
import sys

def load(tag):
    path = sorted(glob.glob(f"{tag}/*/samples_gsm8k_*.jsonl"))[-1]
    out = {}
    for line in open(path):
        s = json.loads(line)
        if s.get("filter", "flexible-extract") == "flexible-extract":
            out[s["doc_id"]] = bool(s["exact_match"])
    return out

base_tag, *others = sys.argv[1:]
base = load(base_tag)
for tag in others:
    o = load(tag)
    ids = base.keys() & o.keys()
    both = sum(base[i] and o[i] for i in ids)
    lost = sum(base[i] and not o[i] for i in ids)
    won = sum(o[i] and not base[i] for i in ids)
    print(f"{tag:10s} vs {base_tag}: n={len(ids)}  {base_tag} {sum(base[i] for i in ids)/len(ids):.4f}  "
          f"{tag} {sum(o[i] for i in ids)/len(ids):.4f}  both right {both}  only {base_tag} {lost}  only {tag} {won}")
