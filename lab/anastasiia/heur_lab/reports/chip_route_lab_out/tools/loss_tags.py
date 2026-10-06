"""Scratch: disposal by node/commodity (chips only) from saved records: python loss_tags.py ep_list tag1 tag2 ..."""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec
from loss import disposal_by_slot

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
res = {}
names = None
for tag in tags:
    tot = None
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        disp = disposal_by_slot(r)[0][1:].sum(axis=0)
        tot = disp if tot is None else tot + disp
        names = [(r.node[nd], r.com[k]) for nd, k in r.stock_slots]
    res[tag] = tot / len(eps)
rows = []
for i, (nd, k) in enumerate(names):
    if k.startswith("chip") and any(res[t][i] > 5e3 for t in tags):
        rows.append((nd, k, [res[t][i] for t in tags]))
print(f"{'node':18s} {'commodity':13s} " + " ".join(f"{t:>9s}" for t in tags) + "   (thousand units disposed per episode)")
for nd, k, vals in sorted(rows, key=lambda x: -max(x[2])):
    print(f"{nd:18s} {k:13s} " + " ".join(f"{v/1e3:9.0f}" for v in vals))
