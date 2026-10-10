"""The disposal and lift regimes of the saved descents, split by role and by what the stock is."""
import copy
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

INF = float("inf")
eps = [int(a) for a in sys.argv[1].split(",")]
passes = int(sys.argv[2]) if len(sys.argv) > 2 else 6
tot = {}
for n in eps:
    t0 = time.time()
    ep = pc.Episode.of("small", 444, n)
    inst = ep.inst
    Cn = [c.id for c in inst.commodities]
    acts = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/small_444_{n}_hybrid.pkl", "rb"))["acts"]
    recs, J = ep.simulate(acts)
    mode, ref = ep.regimes(recs)
    C = ep.cell(mode, ref)
    s0 = ep.solve(C)
    hint = ep.hints(C, s0, mode)
    mode, ref = ep.regimes(recs, hint)
    C = ep.cell(mode, ref)
    s0 = ep.solve(C)
    print(f"ep {n}: played {J/1e11:.2f} bn")

    def freed(pick):
        D = copy.copy(C)
        D.lb, D.ub, D.lo, D.hi = C.lb.copy(), C.ub.copy(), list(C.lo), list(C.hi)
        k = 0
        for kind, key, role, where, idx in C.tags:
            if kind in ("disp", "lift") and pick(kind, key, role):
                D.lb[idx], D.ub[idx] = ep.m.lb[idx], ep.m.ub[idx]
                k += 1
        return D, k

    def what(key):
        sl = inst.stock_slots[key[1]]
        return inst.nodes[sl.node].type, Cn[sl.k]

    cases = [("all", lambda kind, key, role: True)]
    for role in ("d0", "d1", "LA", "LS"):
        cases.append((role, lambda kind, key, r, role=role: r == role))
    kinds = sorted({(role, *what(key)) for kind, key, role, where, idx in C.tags if kind in ("disp", "lift")})
    for role, ntype, cname in kinds:
        cases.append((f"{role} {ntype} {cname}", lambda kind, key, r, role=role, ntype=ntype, cname=cname: r == role and what(key) == (ntype, cname)))
    for name, pick in cases:
        D, k = freed(pick)
        s = ep.solve(D)
        if s["status"] != "Optimal":
            continue
        gain = (s0["J"] - s["J"]) / 1e9
        if gain < 0.05 and name != "all":
            continue
        a2 = ep.actions(s["x"])
        _r, J2 = ep.simulate(a2)
        line = f"   {name:28s} rows {k:5d} LP gain {gain:6.2f} | played blind {(J - J2)/1e11:+7.2f}"
        if name == "all" or gain > 0.3:
            d = pc.descend(ep, a2, iters=passes)
            line += f" | descent from that play {(J - d['J'])/1e11:+6.2f}"
            tot.setdefault(name, []).append((J - d["J"]) / 1e11)
            if name == "all":
                recs3, _ = ep.simulate(d["acts"])
                lost0, lost1 = np.sum([r.lost for r in recs], axis=0), np.sum([r.lost for r in recs3], axis=0)
                sold = {}
                for dm, a, b in zip(inst.demands, lost0, lost1):
                    sold[Cn[dm.k]] = sold.get(Cn[dm.k], 0.0) + (a - b) * dm.pi / 1e9
                shed = (sum(r.costs.shed for r in recs) - sum(r.costs.shed for r in recs3)) / 1e9
                lots = (np.sum([r.lots_started for r in recs3], axis=0) - np.sum([r.lots_started for r in recs], axis=0)) / 1e3
                lift = (np.sum([r.lift for r in recs3], axis=0) - np.sum([r.lift for r in recs], axis=0))
                disp = (np.sum([r.disposal for r in recs3], axis=0) - np.sum([r.disposal for r in recs], axis=0))
                top = np.argsort(-np.abs(lift))[:5]
                line += (f"\n        sales gained bn {({k_: round(v, 2) for k_, v in sold.items()})}, shed saved {shed:+.2f} bn, lots k {np.round(lots).tolist()}"
                         f"\n        lift change: {[(inst.nodes[inst.stock_slots[s].node].id, Cn[inst.stock_slots[s].k], round(float(lift[s]))) for s in top if abs(lift[s]) > 1]}"
                         f"\n        disposal change: {[(inst.nodes[inst.stock_slots[s].node].id, Cn[inst.stock_slots[s].k], round(float(disp[s]))) for s in np.argsort(-np.abs(disp))[:5] if abs(disp[s]) > 1]}")
        print(line, flush=True)
    print(f"   {time.time()-t0:.0f}s")
print({k: (round(float(np.mean(v)), 2), len(v)) for k, v in tot.items()})
