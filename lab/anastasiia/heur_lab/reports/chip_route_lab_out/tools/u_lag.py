"""Scratch: does the executed flow ever exceed the capacity the observation showed (graph_now.u of the same week)?"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from dec import Rec

tot_weeks = 0
viol = 0
examples = []
for tag, eps in (("pull", range(12)), ("f1", range(8))):
    for ep in eps:
        r = Rec(pickle.load(open(f"lab_scratch/recs/{tag}_{ep}.pkl", "rb")))
        st = r.static
        E, sl = st["edges"], st["action_slots"]
        u = np.asarray(r.obs["graph_now.u"])  # row t-1: the observation of week t
        exe = np.asarray(r.obs["last_week.clip.executed"])  # row t: executed in week t
        T = r.T
        for t in range(1, T + 1):
            used = {}
            for s in range(len(sl["edge"])):
                if exe[t, s] > 0:
                    used[sl["edge"][s]] = used.get(sl["edge"][s], 0.0) + exe[t, s]
            for e, q in used.items():
                tot_weeks += 1
                cap = u[t - 1, e]
                if np.isfinite(cap) and q > cap * 1.0001 + 1e-6:
                    viol += 1
                    if len(examples) < 6:
                        examples.append((tag, ep, t, E["id"][e], round(float(q)), round(float(cap))))
print("edge-weeks with flow:", tot_weeks, " executed above the observed capacity:", viol)
for x in examples:
    print("  ", x)
