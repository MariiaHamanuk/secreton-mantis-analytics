"""Scratch: shipments missed at the plants: stock left unshipped while alive edges to markets had spare capacity.

    python missed.py ep_list tag1 tag2 ...
Per week and plant: missed = min(packaged stock left after dispatch, spare capacity of the plant's alive outlet edges).
Reported in millions of units per episode, split le / mat (stock-weighted).
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
for tag in tags:
    tot = {"le": 0.0, "mat": 0.0, "stuck_le": 0.0, "stuck_mat": 0.0}
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        st = r.static
        E, sl = st["edges"], st["action_slots"]
        T = r.T
        mask = np.asarray(r.obs["action_mask"])
        u = np.asarray(r.obs["graph_now.u"])
        exe = np.asarray(r.obs["last_week.clip.executed"])
        stock = np.asarray(r.obs["stock.qty"])
        # slots by plant
        for o in r.osats:
            slots_o = [s for s in range(len(sl["edge"])) if E["tail"][sl["edge"][s]] == o and r.node[E["head"][st["lanes"]["edges"][sl["lane"][s]][-1]] if sl["lane"][s] is not None else E["head"][sl["edge"][s]]].startswith("sink")]
            for t in range(1, T):  # decisions of weeks 1..T-1 (week T cannot ship usefully)
                spare_by_edge = {}
                for s in slots_o:
                    if mask[t - 1, s] != 1:
                        continue
                    e = sl["edge"][s]
                    spare_by_edge.setdefault(e, u[t - 1, e])
                used_e = {}
                for s in slots_o:
                    e = sl["edge"][s]
                    used_e[e] = used_e.get(e, 0.0) + exe[t, s]
                spare = sum(max(0.0, spare_by_edge[e] - used_e.get(e, 0.0)) for e in spare_by_edge)
                for k, fam in ((r.com_ix["chip_le"], "le"), (r.com_ix["chip_mat"], "mat")):
                    i = r.slot_of.get((o, k))
                    if i is None:
                        continue
                    left = stock[t - 1, i] - sum(exe[t, s] for s in slots_o if sl["k"][s] == k)
                    tot[fam] += min(max(0.0, left), spare)
                    tot["stuck_" + fam] += max(0.0, left)
    n = len(eps)
    print(f"{tag:6s} missed shipments le {tot['le']/n/1e6:5.2f}M mat {tot['mat']/n/1e6:5.2f}M per episode   (unshipped stock-weeks: le {tot['stuck_le']/n/1e6:5.1f}M mat {tot['stuck_mat']/n/1e6:5.1f}M)")
