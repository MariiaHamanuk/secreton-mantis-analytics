"""Scratch: per episode, the alive capacity of each stage of the chip chain at week 1 and week 26 (from recorded pull episodes)."""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

tag = sys.argv[1]
eps = [int(x) for x in sys.argv[2].split(",")]
weeks = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 26]
for ep in eps:
    r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
    st = r.static
    E, sl = st["edges"], st["action_slots"]
    nname = r.node
    cname = r.com
    mask = np.asarray(r.obs["action_mask"])
    u = np.asarray(r.obs["graph_now.u"])
    print(f"=== episode {ep}")
    for w in weeks:
        i = w - 1
        line = []
        # route capacity per slot: first edge u when alive
        def alive_cap(s):
            e = sl["edge"][s]
            return u[i, e] if mask[i, s] == 1 else 0.0

        # fab outlets: raw chip slots from each fab
        for fab in r.fabs:
            fname = nname[fab]
            prod = r.fab_attr(fname)["product"]
            k = r.com_ix[prod]
            ss = [s for s in range(len(sl["edge"])) if sl["k"][s] == k and E["tail"][sl["edge"][s]] == fab]
            dest = {}
            for s in ss:
                e = sl["edge"][s]
                lane = sl["lane"][s]
                head = E["head"][e] if lane is None else E["head"][st["lanes"]["edges"][lane][-1]]
                dest[nname[head]] = dest.get(nname[head], 0.0) + alive_cap(s)
            line.append(f"{fname[4:12]} out: " + ", ".join(f"{d[5:]}={v/1e3:.0f}k" for d, v in dest.items()))
        print(f"  week {w:2d} " + " | ".join(line))
        line = []
        for o in r.osats:
            oname = nname[o]
            parts = []
            for c in ("chip_le", "chip_mat"):
                k = r.com_ix[c]
                ss = [s for s in range(len(sl["edge"])) if sl["k"][s] == k and E["tail"][sl["edge"][s]] == o]
                dest = {}
                for s in ss:
                    e = sl["edge"][s]
                    lane = sl["lane"][s]
                    head = E["head"][e] if lane is None else E["head"][st["lanes"]["edges"][lane][-1]]
                    dest[nname[head]] = dest.get(nname[head], 0.0) + alive_cap(s)
                parts.append(c[5:] + ":" + ",".join(f"{d[5:]}={v/1e3:.0f}" for d, v in dest.items() if v > 0))
            line.append(f"{oname[5:]} " + " ".join(parts))
        print(f"          " + " | ".join(line))
