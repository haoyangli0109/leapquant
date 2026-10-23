"""Average decode step time per batch size from an ATOM server log.

Uses only pure-decode stats windows (no prompt tokens, nothing waiting), so a
flush that lands on some steps is averaged in, unlike the benchmark's median
ITL.  python steady_step.py server.log [more.log ...]
"""
import re
import statistics
import sys

pat = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) "
                 r"tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")
for path in sys.argv[1:]:
    by = {}
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m and float(m[1]) == 0.0 and int(m[4]) == 0 and int(m[3]) > 0:
            by.setdefault(int(m[3]), []).append(float(m[2]))
    cells = []
    for n in sorted(by):
        # drop windows that straddle the start or end of the decode phase
        xs = sorted(by[n])[len(by[n]) // 4:] if len(by[n]) > 2 else by[n]
        cells.append(f"B={n}: {1000 * n / statistics.median(xs):.2f} ms ({len(by[n])} win)")
    print(path, " | ".join(c for c, n in zip(cells, sorted(by)) if n in (32, 64, 128, 256, 512)))
