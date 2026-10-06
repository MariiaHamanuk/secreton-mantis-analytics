"""Scratch: tariff paid by slot (bn USD per episode), approximated from executed flows and the weekly tariff rates.

    python tariffs.py ep tag1 tag2 ...
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

ep = int(sys.argv[1])
for tag in sys.argv[2:]:
    r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
    st = r.static
    E, sl = st["edges"], st["action_slots"]
    v = np.asarray(st["commodities"]["v"])
    tar = np.asarray(r.obs["graph_now.tariff"])  # (T+1, E, K) row t-1 = week t
    exe = np.asarray(r.obs["last_week.clip.executed"])  # row t = week t
    res = []
    T = r.T
    for s in range(len(sl["edge"])):
        e, k, lane = sl["edge"][s], sl["k"][s], sl["lane"][s]
        route = [e] if lane is None else st["lanes"]["edges"][lane]
        cost = 0.0
        for t in range(1, T + 1):
            rate = sum(tar[t - 1, x, k] for x in route[:1])  # first edge only is charged at dispatch
            cost += rate * v[k] * exe[t, s]
        if cost > 5e7:
            res.append((cost, s, E["id"][e], r.com[k]))
    res.sort(reverse=True)
    print(f"--- {tag} ep{ep}: dispatch-edge tariffs (first edge of each slot), total {sum(c for c, *_ in res)/1e9:.1f}bn; top:")
    for cost, s, eid, k in res[:8]:
        print(f"    {cost/1e9:6.2f}bn  slot {s:3d} {eid} {k}")
    cc = r.costs.sum(axis=0) / 1e9
    print(f"    (tariff cost component {cc[2]:.1f}bn)")
