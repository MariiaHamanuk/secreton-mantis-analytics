"""Scratch: shipped quantities by (plant -> market) and served per market, for recorded episodes.

    python ship.py ep tag1 tag2 ...
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
    exe = np.asarray(r.obs["last_week.clip.executed"])[1:]  # (T, slots)
    req = np.asarray(r.obs["last_week.clip.requested"])[1:]
    out = {}
    for s in range(len(sl["edge"])):
        e, k, lane = sl["edge"][s], sl["k"][s], sl["lane"][s]
        tail = E["tail"][e]
        head = E["head"][e] if lane is None else E["head"][st["lanes"]["edges"][lane][-1]]
        if r.node[tail].startswith("osat") and r.node[head].startswith("sink"):
            key = (r.com[k], r.node[tail][5:], r.node[head][5:])
            out[key] = out.get(key, 0.0) + exe[:, s].sum()
    print(f"--- {tag} ep{ep}: shipped plant->market (M units), served per market")
    for c in ("chip_le", "chip_mat"):
        line = []
        for (cc, o, m), v in sorted(out.items()):
            if cc == c and v > 1e4:
                line.append(f"{o}->{m} {v/1e6:.2f}")
        print(f"  {c:9s} " + ", ".join(line))
    served = np.asarray(r.obs["last_week.sinks.served"])[1:].sum(axis=0)
    print("  served: " + ", ".join(f"{r.node[n][5:]}/{r.com[k][5:]} {served[i]/1e6:.2f}" for i, (n, k) in enumerate(r.demands) if served[i] > 0))
