"""Per-episode view of a screen.py --dump file (scratch).

    python scratch/per_ep.py scratch/dump_64.json pull v2_c3
"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
refs = d["refs"]
names = sys.argv[2:]
Jn = np.array([r["J_naive_cents"] for r in refs], float)
Jo = np.array([r["J_oracle_cents"] if r["J_oracle_cents"] is not None else np.nan for r in refs], float)
den = Jn - Jo
print(f"{'ep':>3} {'stratum':>7} {'naive bn':>9} {'oracle bn':>9} " + " ".join(f"{n:>14}" for n in names) + "   (episode RSS)")
rss = {n: (Jn - np.array(d["J"][n], float)) / den for n in names}
for i, r in enumerate(refs):
    print(f"{r['episode']:3d} {str(r['stratum']):>7} {Jn[i] / 1e11:9.0f} {Jo[i] / 1e11:9.0f} " + " ".join(f"{rss[n][i]:14.3f}" for n in names))
for s in (1, 2, 3, 4):
    m = np.array([r["stratum"] == s for r in refs])
    print(f"stratum {s}: n={m.sum()}  " + "  ".join(f"{n}={np.nanmean(rss[n][m]):.3f}" for n in names) + f"   mean denominators {np.nanmean(den[m]) / 1e11:.0f} bn")
print("mean episode RSS:", {n: round(float(np.nanmean(rss[n])), 4) for n in names})
if len(names) >= 2:
    diff = rss[names[-1]] - rss[names[0]]
    order = np.argsort(diff)
    print("worst for", names[-1], "vs", names[0], [(int(refs[i]["episode"]), round(float(diff[i]), 3)) for i in order[:6]])
    print("best  for", names[-1], "vs", names[0], [(int(refs[i]["episode"]), round(float(diff[i]), 3)) for i in order[-6:]])
