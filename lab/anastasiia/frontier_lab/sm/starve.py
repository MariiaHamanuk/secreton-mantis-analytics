"""A legal emulation of 'free release' at a bound strait: throttle what is sent into it, by cargo and block of weeks.

Start: a saved full-future descent. Move: the dispatches of one group of action slots (a commodity on lanes through
the strait) in one block of weeks times a factor. Judge: the simulator. Coordinate search, the cheapest kept.
"""
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

task, n = sys.argv[1], int(sys.argv[2])
straits = sys.argv[3].split(",")
block = int(sys.argv[4]) if len(sys.argv) > 4 else 4
sweeps = int(sys.argv[5]) if len(sys.argv) > 5 else 2
budget = float(sys.argv[6]) if len(sys.argv) > 6 else 420.0
t_start = time.time()
ep = pc.Episode.of(task, 444, n)
inst = ep.inst
Cn = [c.id for c in inst.commodities]
acts = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/{task}_444_{n}_hybrid.pkl", "rb"))["acts"]
recs0, J0 = ep.simulate(acts)
comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def items(recs):
    lost = np.sum([r.lost for r in recs], axis=0)
    by = {}
    for d, q in zip(inst.demands, lost):
        by[Cn[d.k]] = by.get(Cn[d.k], 0.0) + q * d.pi
    c = {name: sum(getattr(r.costs, name) for r in recs) for name in comp}
    return by, c


chk = [inst.node_index[s] for s in straits]
groups = {}
for s, (e, k, lane) in enumerate(inst.action_slots):
    if lane is None or inst.commodities[k].pool != "ct":
        continue
    for c in chk:
        if c in inst.lanes[lane].chokepoints:
            groups.setdefault((inst.nodes[c].id, Cn[k]), []).append(s)
order = {"chip_mat": 0, "chip_mat_raw": 1, "wafer": 2, "chip_le_raw": 3}
keys = sorted(groups, key=lambda g: (order[g[1]], g[0]))
print(f"{task} 444 ep {n}: start {J0/1e11:.2f} bn; groups:", {k: len(v) for k, v in groups.items()})
T = ep.T
blocks = [(a, min(T, a + block - 1)) for a in range(1, T + 1, block)]
mult = {(g, b): 1.0 for g in keys for b in range(len(blocks))}


def build(mult):
    out = []
    week_block = {t: b for b, (a, z) in enumerate(blocks) for t in range(a, z + 1)}
    slot_groups = {}
    for g in keys:
        for s in groups[g]:
            slot_groups.setdefault(s, []).append(g)
    for t, (fl, ov, ho) in enumerate(acts, start=1):
        b = week_block[t]
        new = {}
        for s, q in fl.items():
            f = 1.0
            for g in slot_groups.get(s, ()):
                f = min(f, mult[(g, b)])
            if q * f > 0:
                new[s] = q * f
        out.append((new, ov, ho))
    return out


best, sims = J0, 0
for sweep in range(sweeps):
    improved = False
    for g in keys:
        if g[1] == "chip_le_raw":
            continue
        for b in range(len(blocks)):
            if time.time() - t_start > budget:
                break
            cur = mult[(g, b)]
            for f in (0.0, 0.5, 1.0):
                if f == cur:
                    continue
                mult[(g, b)] = f
                _r, J = ep.simulate(build(mult))
                sims += 1
                if J < best - 1e6:
                    best, cur, improved = J, f, True
            mult[(g, b)] = cur
    print(f"   sweep {sweep + 1}: {best/1e11:.2f} bn ({(J0 - best)/1e11:+.2f}), {sims} plays, {time.time()-t_start:.0f}s", flush=True)
    if not improved:
        break
final = build(mult)
recs1, J1 = ep.simulate(final)
b0, c0 = items(recs0)
b1, c1 = items(recs1)
print(f"   result {J1/1e11:.2f} bn, gain {(J0 - J1)/1e11:.2f} bn")
print("   lost sales, bn, start -> result:", {k: (round(b0[k] / 1e9, 2), round(b1[k] / 1e9, 2)) for k in b0})
print("   cost items, bn, result - start:", {k: round((c1[k] - c0[k]) / 1e9, 2) for k in comp if abs(c1[k] - c0[k]) > 5e6})
for g in keys:
    row = [mult[(g, b)] for b in range(len(blocks))]
    if any(f != 1.0 for f in row):
        print("   ", g, " ".join("0" if f == 0 else "h" if f == 0.5 else "1" for f in row))
sent0 = {g: sum(acts[t][0].get(s, 0.0) for t in range(T) for s in groups[g]) for g in keys}
sent1 = {g: sum(final[t][0].get(s, 0.0) for t in range(T) for s in groups[g]) for g in keys}
print("   sent into the lanes, thousand, start -> result:", {g: (round(sent0[g] / 1e3), round(sent1[g] / 1e3)) for g in keys})
pickle.dump({"acts": final, "J": J1, "J0": J0, "mult": mult, "blocks": blocks},
            open(f"outputs/frontier_lab/scratch/sm/starve_{task}_{n}.pkl", "wb"))
if len(sys.argv) > 7 and sys.argv[7] == "descend":
    d = pc.descend(ep, final, iters=6)
    print(f"   the cell's own descent from the result: {d['J']/1e11:.2f} bn ({(J1 - d['J'])/1e11:+.2f}), passes {len(d['hist'])}")
    d0 = pc.descend(ep, acts, iters=6)
    print(f"   control, the same descent from the start: {d0['J']/1e11:.2f} bn ({(J0 - d0['J'])/1e11:+.2f})")
print(f"   total {time.time()-t_start:.0f}s")
