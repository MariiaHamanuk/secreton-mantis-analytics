import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

rec = pickle.load(open(sys.argv[1], "rb"))
r = Rec(rec)
inst = r.inst
ist = inst["initial_state"]
chipish = lambda k: k.startswith("chip") or k == "wafer"
print("== initial stock (chips, wafers)")
for s in ist["stock"]:
    if chipish(s["k"]):
        print(f"  {s['node']:18s} {s['k']:13s} {s['qty']:12,.0f}")
print("== fab WIP (out_week: qty)")
agg = {}
for s in ist["fab_wip"]:
    agg.setdefault((s["node"], s["k"]), []).append((s["out_week"], round(s["qty"])))
for k, v in agg.items():
    print(f"  {k}: {v}")
print("== osat WIP")
agg = {}
for s in ist["osat_wip"]:
    agg.setdefault((s["node"], s["k"]), []).append((s["out_week"], round(s["qty"])))
for k, v in agg.items():
    print(f"  {k}: {v}")
print("== pipeline (chips, wafers), arrival_week: qty")
agg = {}
for s in ist["pipeline"]:
    if chipish(s["k"]):
        agg.setdefault((s["edge"], s["k"]), []).append((s["arrival_week"], round(s["qty"])))
for k, v in agg.items():
    print(f"  {k}: {v}")
