"""Scratch: tariff (first-edge approximation) paid on wafer / raw chip / chip slots vs fuel, mean bn USD per episode."""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
for tag in sys.argv[2:]:
    acc = {"wafer": 0.0, "raw": 0.0, "chip": 0.0, "fuel": 0.0}
    comp = 0.0
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        st = r.static
        E, sl = st["edges"], st["action_slots"]
        v = np.asarray(st["commodities"]["v"])
        tar = np.asarray(r.obs["graph_now.tariff"])
        exe = np.asarray(r.obs["last_week.clip.executed"])
        for s in range(len(sl["edge"])):
            e, k = sl["edge"][s], sl["k"][s]
            name = r.com[k]
            grp = "wafer" if name == "wafer" else "raw" if name.endswith("_raw") else "chip" if name.startswith("chip") else "fuel"
            for t in range(1, r.T + 1):
                acc[grp] += tar[t - 1, e, k] * v[k] * exe[t, s]
        comp += r.costs[:, 2].sum()
    n = len(eps)
    print(f"{tag:6s} " + "  ".join(f"{g} {x / n / 1e9:5.2f}" for g, x in acc.items()) + f"   (tariff component {comp / n / 1e9:.2f}bn)")
