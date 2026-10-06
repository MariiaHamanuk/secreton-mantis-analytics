import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

r = Rec(pickle.load(open("lab_scratch/recs/pull_0.pkl", "rb")))
t = 10  # index -> week 11
ob = {k: v[t] for k, v in r.obs.items()}
np.set_printoptions(linewidth=200, suppress=True)
for k in ["week", "graph_now.u", "pipeline.edge", "pipeline.lane", "pipeline.qty", "pipeline.arrival_week", "wip.node", "wip.k", "wip.qty", "wip.out_week", "demand_forecast.qty", "graph_now.fab.cap_eff", "graph_now.osat.thr_eff", "graph_now.osat.R", "graph_now.kappa.ct", "graph_now.open", "graph_now.tau"]:
    v = ob[k]
    ov = ob.get(k + ".observed")
    print(k, v.dtype, v.shape, "observed:", None if ov is None else (int(np.sum(ov)), ov.shape))
print("pipeline live entries:", int(ob["pipeline.qty.observed"].sum()))
live = ob["pipeline.qty.observed"] == 1
print(np.c_[ob["pipeline.edge"][live], ob["pipeline.k"][live], ob["pipeline.lane"][live], ob["pipeline.lane.observed"][live], ob["pipeline.qty"][live].round(0), ob["pipeline.arrival_week"][live]][:25])
print("forecast row us le:", ob["demand_forecast.qty"][0].round(0))
print("wip live:", int(ob["wip.qty.observed"].sum()))
print("action_mask:", ob["action_mask"].tolist())
print("graph_now.u first 10:", ob["graph_now.u"][:10], ob["graph_now.u.observed"][:10])
print("queue_lots shape:", ob["queue_lots.qty"].shape, ob["queue_lots.qty.observed"].sum())
print("layout lot_keys[:3]", r.layout["lot_keys"][:3])
