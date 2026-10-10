"""A kept play at one strait: container throughput, queues by commodity, what the model sent into its lanes."""
import pickle
import sys

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

tag, task, n, strait = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
step = int(sys.argv[5]) if len(sys.argv) > 5 else 2
inst, marks = pc.world(task, 444, n)
kept = pickle.load(open(f"{ROOT}/outputs/hazard_lab/play/{tag}_{task}_444.pkl", "rb"))[n]
Cn = [c.id for c in inst.commodities]
c = inst.node_index[strait]
ci = inst.chokepoint_ordinal[c]
mu = inst.nodes[c].chokepoint.mu[1]
ct = [k for k, cm in enumerate(inst.commodities) if cm.pool == "ct"]
outs = [e for e in inst.out_edges[c] if inst.edges[e].pool == "ct"]
print(f"{tag} {task} 444 ep {n} at {strait}: J {kept['J']/1e11:.2f} bn; out-edges {[inst.edges[e].id.split('.')[-1] + ':' + '/'.join(Cn[k][:8] for k in sorted(inst.edges[e].K)) for e in outs]}")
slots = {k: [s for s, (e, kk, lane) in enumerate(inst.action_slots) if kk == k and lane is not None and c in inst.lanes[lane].chokepoints] for k in ct}
T = inst.T
print("week | kappa/mu | u/u0 out-edges | queue, thousand | sent into the strait's lanes this week, thousand | prohibited out (edge,k)")
for t in range(1, T + 1):
    ti = t - 1
    if not (t % step == 1 or step == 1 or t > T - 3):
        continue
    q = {k: kept["stock"][ti, inst.slot_index[(c, k)]] for k in ct if (c, k) in inst.slot_index}
    snt = {k: float(kept["sent"][ti, slots[k]].sum()) for k in ct}
    us = " ".join(f"{float(marks.u[ti][e])/inst.edges[e].u0:.2f}" for e in outs)
    proh = sum(bool(marks.prohibited[ti][e, k]) for e in outs for k in ct if k in inst.edges[e].K)
    print(f"{t:3d} | {float(marks.kappa[ti][ci][1])/mu:.2f} | {us} | " + " ".join(f"{Cn[k][:8]} {v/1e3:.0f}" for k, v in q.items() if v > 500)
          + " | " + " ".join(f"{Cn[k][:8]} {v/1e3:.0f}" for k, v in snt.items() if v > 500) + f" | {proh}")
tot = {Cn[k]: round(float(kept["sent"][:, slots[k]].sum()) / 1e3) for k in ct}
print("sent into the strait's lanes over the episode, thousand:", tot)
lost = kept["lost"].sum(axis=0)
print("lost sales, thousand:", {(inst.nodes[d.node].id, Cn[d.k]): round(float(v) / 1e3) for d, v in zip(inst.demands, lost) if v > 1e3})
