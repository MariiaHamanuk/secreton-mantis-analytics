import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

rec = pickle.load(open(sys.argv[1], "rb"))
r = Rec(rec)
T = r.T
np.set_printoptions(linewidth=250, suppress=True)
fabs = ["fab_tw_leading_1", "fab_tw_mature_1", "fab_kr_memory_1", "fab_jp_memory_1", "fab_eu_leading_1", "fab_eu_mature_1"]
print("wafer stock at start of week (k), starts in week (k), cap_eff (k)")
cap = np.asarray(r.obs["graph_now.fab.cap_eff"])
print("wk  " + "  ".join(f"{f[4:12]:>22s}" for f in fabs))
S = {f: r.stock(f, "wafer") for f in fabs}
P = {f: r.wip_starts(f) for f in fabs}
for t in range(T):
    row = []
    for i, f in enumerate(fabs):
        row.append(f"{S[f][t]/1e3:7.0f} {P[f][t]/1e3:6.1f} {cap[t, i]/1e3:6.0f}")
    print(f"{t+1:2d}  " + "  ".join(f"{x:>22s}" for x in row))
print("grid shed per week:")
print(np.asarray(r.obs["last_week.shed.qty"])[1:, :].round(0).T)
