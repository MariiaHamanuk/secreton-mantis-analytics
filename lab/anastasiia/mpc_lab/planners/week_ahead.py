"""Closed loop with the base-load rule: what each week's plan expects of the weeks ahead against what then happens."""
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
LAGS = (1, 2, 4, 8)


def one(n, known, every):
    import planner_B as B

    h = B.make("small", 111, n, 52, known)
    env, pol, obs = B.start(h)
    inst, L = h.inst, h.L
    F, G, D = len(inst.fabs), len(inst.grids), len(inst.demands)
    K = [k.id for k in inst.commodities]
    wafer = K.index("wafer")
    wslot = [inst.slot_index[(node, wafer)] for node in inst.fabs]
    packs = sorted((oo, kp) for oo, node in enumerate(inst.osats) for kp in inst.nodes[node].osat.packages.values())
    names = ("lots started", "fab energy", "fab wafer stock", "packaged", "base load shed", "generation", "demand served")
    plans, stats, done = {}, {"trials": 0, "accepted": 0}, False
    while not done:
        week = int(obs["week"])
        out = B.zfix_plan(h, pol, obs, every, stats)
        model, x = out[0], out[1]
        ix = model.index
        weeks = B.model_weeks(model)

        def col(*key):
            j = ix.get(key)
            return 0.0 if j is None else float(x[j])

        for lag in LAGS:
            r = lag + 1
            if r > weeks:
                continue
            gk = {g: [k[3] for k in ix if k[0] == "G" and k[1] == r and k[2] == g] for g in range(G)}
            plans[(week + lag, lag)] = (
                np.array([col("p", r, f) for f in range(F)]),
                np.array([col("E", r, f) for f in range(F)]),
                np.array([col("I", r, s) for s in wslot]),
                np.array([col("xi", r, oo, kp) for oo, kp in packs]),
                np.array([col("ysh", r, g) for g in range(G)]),
                np.array([sum(col("G", r, g, k) for k in gk[g]) for g in range(G)]),
                np.array([col("D", r, d) for d in range(D)]),
            )
        obs, _, done, _, _ = env.step(L.week1_action(inst, model, x, obs, L.prohibited_now(pol._memory, week)))
    R = env.trajectory.records
    real = {
        r.week: (
            r.lots_started, r.energy, r.stock[wslot], np.array([r.packaged.get(k, 0.0) for k in packs]), r.shed,
            np.array([sum(v for (g2, _), v in r.segment.items() if g2 == g) for g in range(G)]), r.served,
        )
        for r in R
    }
    out = {}
    for lag in LAGS:
        for i, name in enumerate(names):
            p = np.array([plans[(w, lag)][i] for w in sorted(real) if (w, lag) in plans])
            a = np.array([real[w][i] for w in sorted(real) if (w, lag) in plans])
            out[(lag, name)] = (p.sum(), a.sum(), np.abs(p - a).sum())
    return out, names


if __name__ == "__main__":
    N, known, every = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    res = Parallel(n_jobs=4)(delayed(one)(n, known, every) for n in range(N))
    names = res[0][1]
    print(f"{N} episodes, known: {known}; the plan's value for a week, made `lag` weeks before, against the simulator")
    print(f"{'quantity':18s} " + " ".join(f"{'lag ' + str(l) + ': sim/plan |diff|/plan':>30s}" for l in LAGS))
    for name in names:
        cells = []
        for lag in LAGS:
            p = sum(r[0][(lag, name)][0] for r in res); a = sum(r[0][(lag, name)][1] for r in res); d = sum(r[0][(lag, name)][2] for r in res)
            cells.append(f"{a / max(p, 1e-9):14.3f} {d / max(p, 1e-9):15.3f}")
        print(f"{name:18s} " + " ".join(cells))
