import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

rec = pickle.load(open(sys.argv[1], "rb"))
r = Rec(rec)
T = r.T
np.set_printoptions(linewidth=250, suppress=True)
inst = r.inst
cols = []
for node in ["fab_tw_leading_1", "fab_tw_mature_1", "fab_kr_memory_1", "fab_jp_memory_1", "fab_eu_leading_1", "fab_eu_mature_1"]:
    prod = r.fab_attr(node)["product"]
    cols.append((node, prod))
for o in ["osat_my", "osat_tw", "osat_kr"]:
    for c in ["chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"]:
        if (r.node_ix[o], r.com_ix[c]) in r.slot_of:
            cols.append((o, c))
for s in ["sink_us", "sink_eu", "sink_cn", "sink_jp"]:
    for c in ["chip_le", "chip_mat"]:
        cols.append((s, c))
# storage caps
cap = {}
for nd in inst["nodes"]:
    for c, d in nd.get("stock", {}).items():
        cap[(nd["id"], c)] = d["storage"]
S = {c: r.stock(*c) for c in cols}
print("stock at start of week (k); '*' marks within 2% of storage")
hdr = " wk " + " ".join(f"{n[:7]:>7s}/{c[5:11]:6s}" for n, c in cols)
print(hdr)
for t in range(T + 1):
    row = []
    for c in cols:
        v = S[c][t]
        flag = "*" if v >= 0.98 * cap[c] else " "
        row.append(f"{v/1e3:13.0f}{flag}")
    print(f"{t+1:3d} " + " ".join(row))
print("cap(k):", [round(cap[c] / 1e3) for c in cols])
cc = np.asarray(r.obs["last_week.cost_components"])[1:]
print("disposal cost by week (bn):", (cc[:, 6] / 1e9).round(2))
print("tariff cost by week (bn):", (cc[:, 2] / 1e9).round(2))
print("queue holding by week (bn):", (cc[:, 4] / 1e9).round(2))
