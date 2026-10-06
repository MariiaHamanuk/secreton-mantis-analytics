"""Scratch: lost sales that one more week of air shipping could have avoided.

For week t+1, market m, chip k: lost_{m,k}(t+1) versus what the plants could still have sent in week t
(unshipped stock after week t's dispatch, over alive edges with spare capacity, air/lead-1 slots only).
    python avoidable.py ep_list tag1 tag2 ...
Millions of units per episode.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
for tag in tags:
    lost_tot = {"le": 0.0, "mat": 0.0}
    avoid = {"le": 0.0, "mat": 0.0}
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        st = r.static
        E, sl = st["edges"], st["action_slots"]
        T = r.T
        mask = np.asarray(r.obs["action_mask"])
        u = np.asarray(r.obs["graph_now.u"])
        exe = np.asarray(r.obs["last_week.clip.executed"])
        stock = np.asarray(r.obs["stock.qty"])
        lost = np.asarray(r.obs["last_week.sinks.lost"])  # row t = week t
        demand = np.asarray(r.obs["last_week.sinks.demand"])
        # slot table: plant -> list of (slot, market node, edge, k, lead)
        info = []
        for s in range(len(sl["edge"])):
            e = sl["edge"][s]
            lane = sl["lane"][s]
            head = E["head"][e] if lane is None else E["head"][st["lanes"]["edges"][lane][-1]]
            if r.node[E["tail"][e]].startswith("osat") and r.node[head].startswith("sink"):
                lead = sum(E["tau0"][x] for x in ([e] if lane is None else st["lanes"]["edges"][lane]))
                info.append((s, E["tail"][e], head, e, sl["k"][s], lead))
        for t in range(1, T):  # shipments of week t serve week t+1
            for k, fam in ((r.com_ix["chip_le"], "le"), (r.com_ix["chip_mat"], "mat")):
                # unshipped stock per plant after week t's dispatch
                left = {}
                for o in r.osats:
                    i = r.slot_of.get((o, k))
                    if i is None:
                        continue
                    sent = sum(exe[t, s] for s, o2, m, e, kk, lead in info if o2 == o and kk == k)
                    left[o] = max(0.0, stock[t - 1, i] - sent)
                # spare capacity per edge after week t's dispatch (all commodities)
                used_e = {}
                for s, o2, m, e, kk, lead in info:
                    used_e[e] = used_e.get(e, 0.0) + exe[t, s]
                for di, (mn, kk) in enumerate(r.demands):
                    if kk != k:
                        continue
                    l = lost[t + 1, di]
                    lost_tot[fam] += l
                    if l <= 0:
                        continue
                    reach = 0.0
                    for o in left:
                        cap = 0.0
                        seen_e = set()
                        for s, o2, m, e, kk2, lead in info:
                            if o2 == o and m == mn and kk2 == k and lead == 1 and mask[t - 1, s] == 1 and e not in seen_e:
                                seen_e.add(e)
                                cap += max(0.0, u[t - 1, e] - used_e.get(e, 0.0))
                        reach += min(left[o], cap)
                    avoid[fam] += min(l, reach)
    n = len(eps)
    print(f"{tag:6s} lost le {lost_tot['le']/n/1e6:6.2f}M avoidable {avoid['le']/n/1e6:5.2f}M | mat lost {lost_tot['mat']/n/1e6:6.2f}M avoidable {avoid['mat']/n/1e6:5.2f}M   (per episode)")
