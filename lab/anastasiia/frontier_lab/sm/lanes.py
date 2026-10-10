"""Lane family of the saved full-future descents, split by side: hold back (release less than the reference share)
against jump (release more), by commodity, and what the simulator pays for the freed solutions played blind."""
import copy
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

INF = float("inf")
eps = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [7]
out = {}
for n in eps:
    t0 = time.time()
    ep = pc.Episode.of("small", 444, n)
    inst = ep.inst
    Cn = [c.id for c in inst.commodities]
    d = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/small_444_{n}_hybrid.pkl", "rb"))
    acts = d["acts"]
    recs, J = ep.simulate(acts)
    mode, ref = ep.regimes(recs)
    C = ep.cell(mode, ref)
    s0 = ep.solve(C)
    if s0["status"] == "Optimal":  # read the ties as the duals say, once
        hint = ep.hints(C, s0, mode)
        mode, ref = ep.regimes(recs, hint)
        C = ep.cell(mode, ref)
        s0 = ep.solve(C)
    print(f"ep {n}: played {J/1e11:.2f} bn (kept {d['J']/1e11:.2f}), cell {s0['status']} {s0['J']/1e9:.2f} bn, build+solve {time.time()-t0:.1f}s", flush=True)
    # which lanes are empty in the reference
    empty = set()
    for t in range(1, ep.T + 1):
        rec = recs[t - 1]
        for c, k, lane in ep.lanes:
            e = inst.lane_through[(lane, c)][1]
            if float(rec.x.get((e, k, lane), 0.0)) + float(rec.queue.get((c, k, lane), 0.0)) <= 1e-9:
                empty.add((t, c, k, lane))
    lane_tags = [tg for tg in C.tags if tg[0] == "lane"]
    roles = {}
    for kind, key, role, where, idx in lane_tags:
        roles[role] = roles.get(role, 0) + 1
    print("   lane constraints by role:", roles, "empty lane-weeks:", len(empty))

    def variant(side, pick=lambda key: True):
        D = copy.copy(C)
        D.lb, D.ub, D.lo, D.hi = C.lb.copy(), C.ub.copy(), list(C.lo), list(C.hi)
        for kind, key, role, where, idx in lane_tags:
            if not pick(key):
                continue
            if role == "q1":  # Q <= 0: everything passes
                if side in ("free", "hold"):
                    D.ub[idx] = ep.m.ub[idx]
            elif role == "q0":  # x <= 0: nothing passes
                if side in ("free", "jump"):
                    D.ub[idx] = ep.m.ub[idx]
            else:  # (1 - phi) x - phi Q = 0
                if side == "free":
                    D.lo[idx], D.hi[idx] = -INF, INF
                elif side == "hold":
                    D.lo[idx] = -INF
                else:
                    D.hi[idx] = INF
        return D

    res = {"J": J / 1e11, "cell": s0["J"] / 1e9}
    base = s0["J"]
    cases = [("free", "all", lambda key: True), ("hold", "all", lambda key: True), ("jump", "all", lambda key: True),
             ("free", "empty", lambda key: key in empty), ("free", "nonempty", lambda key: key not in empty),
             ("jump", "empty", lambda key: key in empty), ("jump", "nonempty", lambda key: key not in empty)]
    for k in sorted({key[2] for _kd, key, *_ in lane_tags}):
        cases.append(("free", Cn[k], lambda key, k=k: key[2] == k))
    for c in sorted({key[1] for _kd, key, *_ in lane_tags}):
        cases.append(("free", inst.nodes[c].id, lambda key, c=c: key[1] == c))
    for side, name, pick in cases:
        s = ep.solve(variant(side, pick))
        gain = (base - s["J"]) / 1e9 if s["status"] == "Optimal" else float("nan")
        line = f"   {side:5s} {name:14s} LP gain {gain:7.2f} bn"
        if s["status"] == "Optimal" and name == "all":
            a2 = ep.actions(s["x"])
            _r2, J2 = ep.simulate(a2)
            line += f" | played blind {J2/1e11:9.2f} bn ({(J - J2)/1e11:+.2f} to the descent)"
            res[(side, name, "played")] = (J - J2) / 1e11
            if side == "free":  # what the freed solution does against the reference shares, by commodity
                x = s["x"]
                dev = {}
                for kind, key, role, where, idx in lane_tags:
                    t, c, k, lane = key
                    e = inst.lane_through[(lane, c)][1]
                    jx, jQ = ep.col("x", t, e, k, lane), ep.col("Q", t, c, k, lane)
                    phi = mode["lane"][key]
                    v = x[jx] - phi * (x[jx] + x[jQ])
                    a = dev.setdefault(Cn[k], [0.0, 0.0])
                    a[0 if v > 0 else 1] += v
                line += " | jump/hold units by commodity: " + ", ".join(f"{k} {v[0]/1e3:+.0f}k/{v[1]/1e3:+.0f}k" for k, v in dev.items())
        res[(side, name)] = gain
        print(line, flush=True)
    # the reference's container queues, unit-weeks by strait and commodity
    q = {}
    for rec in recs:
        for (c, k, lane), v in rec.queue.items():
            if inst.commodities[k].pool == "ct" and v > 0:
                q[(inst.nodes[c].id, Cn[k])] = q.get((inst.nodes[c].id, Cn[k]), 0.0) + v
    print("   reference queues, thousand unit-weeks:", {key: round(v / 1e3) for key, v in sorted(q.items()) if v > 5e3})
    out[n] = res
    print(f"   ep {n} done in {time.time()-t0:.1f}s", flush=True)
pickle.dump(out, open(f"outputs/frontier_lab/scratch/sm/lanes_{'_'.join(map(str, eps))}.pkl", "wb"))
