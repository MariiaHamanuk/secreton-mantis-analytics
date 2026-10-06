import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec
from loss import disposal_by_slot

r = Rec(pickle.load(open(sys.argv[1], "rb")))
disp, stock, out, arr, prod, cons = disposal_by_slot(r)
tot = disp[1:].sum(axis=0)
for i, (nd, k) in enumerate(r.stock_slots):
    name = f"{r.node[nd]}/{r.com[k]}"
    if abs(tot[i]) > 1 and r.com[k] in ("wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"):
        print(f"{name:34s} balance residual (disposal) total {tot[i]/1e3:10,.1f}k   min week {disp[1:, i].min()/1e3:9,.1f}k max {disp[1:, i].max()/1e3:9,.1f}k")
cc = np.asarray(r.obs["last_week.cost_components"])[1:]
print("disposal cost total (USD bn):", cc[:, 6].sum() / 1e9)
