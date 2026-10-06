"""Scratch: weekly view of one commodity family along the chain for a recorded episode.

    python show_node.py tag ep family(le|mat)
Columns: fab raw stock | plant raw stock + packaged stock | market stock; plus served and requested flows by stage.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec
from loss import disposal_by_slot

tag, ep, fam = sys.argv[1], int(sys.argv[2]), sys.argv[3]
r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
disp, stock, out, arr, prod, cons = disposal_by_slot(r)
T = r.T
raw = "chip_le_raw" if fam == "le" else "chip_mat_raw"
pk = "chip_le" if fam == "le" else "chip_mat"
cols = []
for fab in r.fabs:
    nm = r.node[fab]
    if r.fab_attr(nm)["product"] == raw:
        cols.append((nm, raw))
for o in r.osats:
    nm = r.node[o]
    if (o, r.com_ix[pk]) in r.slot_of:
        cols.append((nm, pk))
for nm in ["sink_us", "sink_eu", "sink_cn", "sink_jp"]:
    if (r.node_ix[nm], r.com_ix[pk]) in r.slot_of and nm != "sink_cn" or fam == "mat":
        cols.append((nm, pk))
ix = [r.slot_of[(r.node_ix[n], r.com_ix[c])] for n, c in cols]
print("week | stock at start (k):", " ".join(f"{n[:8]}/{c[5:9]:4s}" for n, c in cols), "| out(k) from plants | disposed(k)")
sl = r.static["action_slots"]
for t in range(1, T + 1):
    row = " ".join(f"{stock[t-1, i]/1e3:13.0f}" for i in ix)
    o_ = " ".join(f"{out[t, i]/1e3:6.0f}" for i in ix[: len(cols)] if cols[ix.index(i)][0].startswith(("osat", "fab")))
    d_ = " ".join(f"{disp[t, i]/1e3:5.0f}" for i in ix if disp[t, i] > 1e3)
    print(f"{t:3d} | {row} | {o_} | {d_}")
