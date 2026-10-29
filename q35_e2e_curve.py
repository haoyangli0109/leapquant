"""Cumulative generated tokens over a replay, from the server's 10-second stats.

Each "Avg generation throughput" line covers the preceding stats interval, so
summing throughput x interval gives the token curve. Prints when 50/90/99/100%
of the tokens were out, the throughput up to the 90% mark, and the throughput
over the intervals with at least 80% of the peak number of running requests
(the saturated phase, before the batch drains into the long tail).

    python q35_e2e_curve.py DIR/replay/native-g7/gsm8k.server.log ...
"""
import re
import sys
from datetime import datetime

line_re = re.compile(r"\[atom (\d\d:\d\d:\d\d)\] Engine \d+: Avg prompt throughput: [\d.]+ tokens/s, "
                     r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs")


def curve(path):
    pts, prev = [], None
    for line in open(path, errors="ignore"):
        m = line_re.search(line)
        if not m:
            continue
        t = datetime.strptime(m[1], "%H:%M:%S")
        if prev is not None:
            dt = (t - prev).total_seconds() % 86400
            pts.append((dt, float(m[2]) * dt, int(m[3])))
        prev = t
    # drop the idle lines before the replay and after it
    while pts and pts[0][2] == 0:
        pts.pop(0)
    while pts and pts[-1][1] == 0:
        pts.pop()
    return pts


for path in sys.argv[1:]:
    pts = curve(path)
    total = sum(p[1] for p in pts)
    marks, acc, t, out = {}, 0.0, 0.0, []
    for dt, tok, run in pts:
        t += dt
        acc += tok
        for q in (0.5, 0.9, 0.99, 1.0):
            if q not in marks and acc >= q * total - 1:
                marks[q] = t
    t90 = marks[0.9]
    peak = max(p[2] for p in pts)
    sat = [p for p in pts if p[2] >= 0.8 * peak]
    sat_tput = sum(p[1] for p in sat) / sum(p[0] for p in sat)
    print(f"{path.split('/replay/')[-1]}: {total / 1e6:.2f} M tokens; 50/90/99/100% at "
          + "/".join(f"{marks[q]:.0f}" for q in (0.5, 0.9, 0.99, 1.0))
          + f" s; overall {total / marks[1.0]:.0f} tok/s, to 90% {0.9 * total / t90:.0f} tok/s, "
          f"saturated (>= {0.8 * peak:.0f} running, {sum(p[0] for p in sat):.0f} s) {sat_tput:.0f} tok/s")
