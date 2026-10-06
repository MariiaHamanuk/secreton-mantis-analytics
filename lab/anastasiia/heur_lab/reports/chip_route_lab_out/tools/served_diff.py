"""Scratch: chips served per week, difference of two recorded agents: python served_diff.py ep_list tagA tagB"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
a, b = sys.argv[2], sys.argv[3]
diff = {"le": np.zeros(52), "mat": np.zeros(52)}
for ep in eps:
    ra = Rec(pickle.load(open(f"lab_scratch/recs/{a}_{ep}.pkl", "rb")))
    rb = Rec(pickle.load(open(f"lab_scratch/recs/{b}_{ep}.pkl", "rb")))
    for r, sign in ((ra, 1.0), (rb, -1.0)):
        sv = np.asarray(r.obs["last_week.sinks.served"])[1:]
        for di, (nd, k) in enumerate(r.demands):
            fam = "le" if r.com[k] == "chip_le" else "mat"
            diff[fam] += sign * sv[:, di] / len(eps)
print(f"served({a}) - served({b}), mean per episode, thousand units, by week")
for fam in ("le", "mat"):
    print(fam, " ".join(f"{v/1e3:5.0f}" for v in diff[fam]))
    print(f"   total {diff[fam].sum()/1e6:.3f}M; weeks 1-25 {diff[fam][:25].sum()/1e6:.3f}M, 26-40 {diff[fam][25:40].sum()/1e6:.3f}M, 41-52 {diff[fam][40:].sum()/1e6:.3f}M")
