"""Scratch: average cargo waiting in strait queues by commodity (units), for recorded agents.

    python queues.py ep_list tag1 tag2 ...
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
names = ["wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"]
print(f"{'agent':8s} " + " ".join(f"{n:>13s}" for n in names) + "   (mean units in strait queues per week, thousands; and in transit)")
for tag in tags:
    acc = {n: 0.0 for n in names}
    transit = {n: 0.0 for n in names}
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        qty = np.asarray(r.obs["queue_lots.qty"])
        qobs = np.asarray(r.obs["queue_lots.qty.observed"])
        rows = r.layout["lot_keys"]
        for ri, key in enumerate(rows):
            k = r.com[key[1]]
            if k in acc:
                acc[k] += float((qty[:, ri, :] * (qobs[:, ri, :] == 1)).sum()) / qty.shape[0]
        pq = np.asarray(r.obs["pipeline.qty"])
        po = np.asarray(r.obs["pipeline.qty.observed"])
        pk = np.asarray(r.obs["pipeline.k"])
        for n in names:
            kk = r.com_ix[n]
            transit[n] += float((pq * (po == 1) * (pk == kk)).sum()) / pq.shape[0]
    m = len(eps)
    print(f"{tag:8s} " + " ".join(f"{acc[n]/m/1e3:13.0f}" for n in names) + "    transit: " + " ".join(f"{transit[n]/m/1e3:7.0f}" for n in names))
