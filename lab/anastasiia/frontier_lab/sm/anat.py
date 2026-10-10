"""What binds at a strait in a saved descent: container throughput, out-edges, what passes and what waits, by week."""
import pickle
import sys

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

task, n, strait = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ep = pc.Episode.of(task, 444, n)
inst, marks = ep.inst, ep.marks
Cn = [c.id for c in inst.commodities]
acts = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/{task}_444_{n}_hybrid.pkl", "rb"))["acts"]
recs, J = ep.simulate(acts)
print(f"{task} 444 ep {n}: J {J/1e11:.2f} bn")
c = inst.node_index[strait]
ci = inst.chokepoint_ordinal[c]
outs = [e for e in inst.out_edges[c] if inst.edges[e].pool == "ct"]
mu = inst.nodes[c].chokepoint.mu[1]
print("container out-edges:", [(inst.edges[e].id, inst.edges[e].u0) for e in outs], "mu", mu)
ct = [k for k, cm in enumerate(inst.commodities) if cm.pool == "ct"]
print("week | kappa/mu | u/u0 of out-edges | passed by commodity (k) | queue by commodity (k) | prohibited (edge,k)")
for t in range(1, ep.T + 1):
    rec, ti = recs[t - 1], t - 1
    kap = float(marks.kappa[ti][ci][1])
    passed = {k: 0.0 for k in ct}
    for (e, k, lane), q in rec.x.items():
        if inst.edges[e].tail == c and k in passed:
            passed[k] += q
    queue = {k: 0.0 for k in ct}
    for (cc, k, lane), q in rec.queue.items():
        if cc == c and k in queue:
            queue[k] += q
    us = " ".join(f"{float(marks.u[ti][e])/inst.edges[e].u0:.2f}" for e in outs)
    proh = [(inst.edges[e].id.split('.')[-1], Cn[k]) for e in outs for k in ct if k in inst.edges[e].K and marks.prohibited[ti][e, k]]
    tot = sum(passed.values())
    if t % 2 == 1 or t > ep.T - 6:
        print(f"{t:3d} | {kap/mu:.2f} ({tot/1e3:5.0f}k of {kap/1e3:5.0f}k) | {us} | "
              + " ".join(f"{Cn[k][:8]} {passed[k]/1e3:.0f}" for k in ct if passed[k] > 500) + " | "
              + " ".join(f"{Cn[k][:8]} {queue[k]/1e3:.0f}" for k in ct if queue[k] > 500) + f" | {proh if t in (1, ep.T) else len(proh)}")
