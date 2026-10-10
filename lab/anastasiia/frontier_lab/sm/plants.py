"""Packaging: raw chips withheld from a plant by product and block of weeks (the legal handle on the plant's split
and on what waits unpackaged), from the chain's result; judged by the simulator, then the cell's descent."""
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
SCR = "outputs/frontier_lab/scratch/sm"
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

start = pickle.load(open(f"{SCR}/chain_0_1_2_3.pkl", "rb")) | pickle.load(open(f"{SCR}/chain_4_5_6_7.pkl", "rb"))
eps = [int(a) for a in sys.argv[1].split(",")]
tot = []
for n in eps:
    t0 = time.time()
    ep = pc.Episode.of("small", 444, n)
    inst, T = ep.inst, ep.T
    Cn = [c.id for c in inst.commodities]
    acts = start[n]["acts"]
    recs, J = ep.simulate(acts)
    # weeks a plant is at its throughput with two products waiting
    bound = {}
    for t, r in enumerate(recs, start=1):
        for oi, o in enumerate(inst.osats):
            pk = sum(r.packaged.get((oi, kp), 0.0) for kp in inst.nodes[o].osat.packages.values())
            if pk >= ep.osat_thr(t, oi) * (1 - 1e-6) and ep.osat_thr(t, oi) > 0:
                bound[inst.nodes[o].id] = bound.get(inst.nodes[o].id, 0) + 1
    groups = {}
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if Cn[k] not in ("chip_mat_raw", "chip_le_raw"):
            continue
        dest = inst.lane_destination(lane) if lane is not None else inst.edges[e].head
        if dest in inst.osat_ordinal:
            groups.setdefault((inst.nodes[dest].id, Cn[k], inst.nodes[inst.edges[e].tail].id), []).append(s)
    blocks = [(a, min(T, a + 3)) for a in range(1, T + 1, 4)]
    week_block = {t: b for b, (a, z) in enumerate(blocks) for t in range(a, z + 1)}
    slot_group = {s: g for g, ss in groups.items() for s in ss}

    def build(mult):
        out = []
        for t, (fl, ov, ho) in enumerate(acts, start=1):
            new = {}
            for s, q in fl.items():
                f = mult.get((slot_group.get(s), week_block[t]), 1.0)
                if q * f > 0:
                    new[s] = q * f
            out.append((new, ov, ho))
        return out

    mult, best = {}, J
    for g, ss in groups.items():
        for b, (a, z) in enumerate(blocks):
            if not any(acts[t - 1][0].get(s, 0.0) > 0 for t in range(a, z + 1) for s in ss):
                continue
            keep = 1.0
            for f in (0.0, 0.5):
                mult[(g, b)] = f
                _r, Jn = ep.simulate(build(mult))
                if Jn < best - 1e6:
                    best, keep = Jn, f
            mult[(g, b)] = keep
    d = pc.descend(ep, build(mult), iters=6)
    cut = {f"{g[2][4:]}>{g[0][5:]}/{g[1][5:]}": "".join("0" if mult.get((g, b), 1.0) == 0 else "h" if mult.get((g, b), 1.0) == 0.5 else "." for b in range(len(blocks))) for g in groups}
    cut = {k: v for k, v in cut.items() if set(v) != {"."}}
    print(f"ep {n}: start {J/1e11:.2f}; plant-weeks at throughput {bound}; withheld {(J - best)/1e11:+.2f} bn, descent after {(best - d['J'])/1e11:+.2f}; {cut}; {time.time()-t0:.0f}s", flush=True)
    tot.append((J - min(best, d["J"])) / 1e11)
print("mean", round(float(np.mean(tot)), 2))
