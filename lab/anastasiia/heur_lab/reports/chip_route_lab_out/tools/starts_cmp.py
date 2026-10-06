"""Scratch: lots started by fab, per week bucket, for two recorded agents: python starts_cmp.py ep_list tagA tagB"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
buckets = [(1, 10), (11, 20), (21, 30), (31, 44), (45, 52)]
res = {}
for tag in tags:
    acc = {}
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        for fab in r.fabs:
            nm = r.node[fab]
            st = r.wip_starts(nm)
            acc.setdefault(nm, []).append(st)
    res[tag] = {nm: np.mean(v, axis=0) for nm, v in acc.items()}
print("lots started per week (k), mean over episodes, by fab and week bucket")
print(f"{'fab':18s} {'agent':6s} " + " ".join(f"{a}-{b:>2d}".rjust(9) for a, b in buckets))
for nm in res[tags[0]]:
    for tag in tags:
        v = res[tag][nm]
        print(f"{nm:18s} {tag:6s} " + " ".join(f"{v[a-1:b].mean()/1e3:9.1f}" for a, b in buckets))
