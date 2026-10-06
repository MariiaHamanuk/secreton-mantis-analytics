"""Scratch: per-episode chip fate for several recorded agents: python cmp_eps.py ep_list tag1 tag2 ..."""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec
from loss import disposal_by_slot

eps = [int(x) for x in sys.argv[1].split(",")]
tags = sys.argv[2:]
CH = ["chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"]
for ep in eps:
    print(f"=== episode {ep}")
    for tag in tags:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        disp, stock, out, arr, prod, cons = disposal_by_slot(r)
        served = np.asarray(r.obs["last_week.sinks.served"])[1:]
        sv = {"le": 0.0, "mat": 0.0}
        for di, (nd, k) in enumerate(r.demands):
            sv["le" if r.com[k] == "chip_le" else "mat"] += served[:, di].sum()
        dl = {"le": 0.0, "mat": 0.0}
        endraw = {"le": 0.0, "mat": 0.0}
        endpk = {"le": 0.0, "mat": 0.0}
        for i, (nd, k) in enumerate(r.stock_slots):
            c = r.com[k]
            if c in ("chip_le_raw", "chip_le"):
                dl["le"] += disp[1:, i].sum()
            if c in ("chip_mat_raw", "chip_mat"):
                dl["mat"] += disp[1:, i].sum()
            if r.node[nd].startswith("fab") and c in CH:
                endraw["le" if c == "chip_le_raw" else "mat"] += stock[-1, i]
            if r.node[nd].startswith("osat") and c in ("chip_le", "chip_mat"):
                endpk["le" if c == "chip_le" else "mat"] += stock[-1, i]
        cost = r.costs.sum(axis=0) / 1e9
        st = {"le": 0.0, "mat": 0.0}
        for fab in r.fabs:
            nm = r.node[fab]
            fam = "le" if r.fab_attr(nm)["product"] == "chip_le_raw" else "mat"
            st[fam] += r.wip_starts(nm).sum()
        print(f"  {tag:8s} starts le {st['le']/1e6:5.2f} mat {st['mat']/1e6:5.2f} | served le {sv['le']/1e6:5.2f} mat {sv['mat']/1e6:5.2f} | disposed le {dl['le']/1e6:5.2f} mat {dl['mat']/1e6:5.2f} | end raw@fab le {endraw['le']/1e6:5.2f} mat {endraw['mat']/1e6:5.2f}  pk@osat le {endpk['le']/1e6:5.2f} mat {endpk['mat']/1e6:5.2f} | short {cost[5]:7.1f} disp {cost[6]:5.1f} tar {cost[2]:5.1f} shed {cost[7]:7.1f}")
