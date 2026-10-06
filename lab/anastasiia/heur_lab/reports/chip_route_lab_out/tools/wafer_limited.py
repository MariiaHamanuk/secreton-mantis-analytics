"""Scratch: weeks in which a fab started everything it had on hand while having less than its capacity (wafer-limited)."""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
for tag in tags:
    print(f"--- {tag}: share of fab-weeks wafer-limited (start >= 98% of wafers on hand, and on hand < 98% of cap_eff), by week bucket")
    buckets = [(1, 5), (6, 10), (11, 20), (21, 38)]
    tab = {}
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        cap = np.asarray(r.obs["graph_now.fab.cap_eff"])  # row t-1
        for fi, fab in enumerate(r.fabs):
            nm = r.node[fab]
            st = r.wip_starts(nm)  # (T,)
            stock = r.stock(nm, "wafer")  # start of week t = index t-1; index t = end of week t
            for t in range(1, 39):
                W = stock[t] + st[t - 1]
                lim = st[t - 1] >= 0.98 * W and W < 0.98 * cap[t - 1, fi] and W > 1.0
                for bi, (a, b) in enumerate(buckets):
                    if a <= t <= b:
                        tab.setdefault((nm, bi), []).append(lim)
    for nm in sorted({k[0] for k in tab}):
        print(f"  {nm:18s} " + "  ".join(f"{a}-{b}: {100 * np.mean(tab[(nm, bi)]):3.0f}%" for bi, (a, b) in enumerate(buckets)))
