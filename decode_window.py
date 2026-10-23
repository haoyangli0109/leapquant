"""Decode step time from `benchmark_serving --save-detailed` results.

ATOM prefills every waiting prompt before it decodes again, so a wave of equal
requests decodes in lockstep once the last prompt is in: from then on each step
gives every request one token, and the mean inter-token latency over that
window is the mean step time, flushes included.  The median leaves them out.

  python decode_window.py results/kimi-fp32-r2 results/kimi-lq-r2
"""
import json
import pathlib
import statistics
import sys

import numpy as np


def window(path: pathlib.Path) -> tuple[float, float, int]:
    r = json.loads(path.read_text())
    last_first_token = max(r["ttfts"])
    gaps = []
    for ttft, itl in zip(r["ttfts"], r["itls"]):
        itl = np.asarray(itl)
        start = ttft + np.cumsum(itl) - itl
        # A gap that opens before the last first token spans prefill steps.
        gaps.extend(itl[start >= last_first_token - 2e-3])
    return 1000 * statistics.mean(gaps), 1000 * statistics.median(gaps), len(gaps)


runs = [pathlib.Path(p) for p in sys.argv[1:]]
concs = sorted({int(f.stem[1:]) for d in runs for f in d.glob("c*.json")})
print(f"{'conc':>6}" + "".join(f"{d.name + ' mean/median':>30}" for d in runs))
for c in concs:
    cells = []
    for d in runs:
        f = d / f"c{c}.json"
        if not f.exists():
            cells.append(f"{'-':>30}")
            continue
        mean, median, n = window(f)
        cells.append(f"{mean:13.2f} /{median:7.2f} ({n:6d})")
    print(f"{c:>6}" + "".join(cells))
