"""Plan (full horizon, everything known, base-load-first in every week) against what the simulator does with it."""
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
COMP = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def one(n, flags, tl):
    import planner_D as P
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.oracle.lp import lp_costs
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    out, sol = P.solve(n, flags, tl)
    m, x = sol
    inst, omega, marks, fb = P.world(n)
    nc, T, tm = m.meta["nc"], m.T, m.meta["template"]
    env = Env(fallback=fb)
    obs, info = env.reset(inst, "standard", omega, _policy_seed(P.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="d")
    done, t = False, 0
    while not done:
        xt = np.zeros_like(m.lb)
        xt[:nc] = x[t * nc : (t + 1) * nc]
        obs, r, done, tr, inf = env.step(L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t])))
        t += 1
    R = env.trajectory.records

    def c(tag, t, *rest):
        j = tm.get((tag, *rest))
        return 0.0 if j is None else float(x[(t - 1) * nc + j])

    q = {}
    # every flow the plan has, by (edge, commodity): dispatches and strait releases
    xkeys = sorted({(k[1], k[2]) for k in tm if k[0] == "x"})
    lanes = {}
    for k in tm:
        if k[0] == "x":
            lanes.setdefault((k[1], k[2]), []).append(k[3])
    chk = set(inst.chokepoints)
    disp = [ek for ek in xkeys if inst.edges[ek[0]].tail not in chk]
    rel = [ek for ek in xkeys if inst.edges[ek[0]].tail in chk]

    def flows(keys):
        plan = np.array([[sum(c("x", t, e, k, ln) for ln in lanes[(e, k)]) for (e, k) in keys] for t in range(1, T + 1)])
        real = np.array([[sum(v for (ee, kk, _), v in R[t - 1].x.items() if (ee, kk) == (e, k)) for (e, k) in keys] for t in range(1, T + 1)])
        return plan, real

    q["dispatch flows (routes we choose)"] = flows(disp)
    q["strait releases"] = flows(rel)
    G = len(inst.grids)
    gk = {go: [k[2] for k in tm if k[0] == "G" and k[1] == go] for go in range(G)}
    q["grid generation"] = (
        np.array([[sum(c("G", t, go, k) for k in gk[go]) for go in range(G)] for t in range(1, T + 1)]),
        np.array([[sum(v for (g2, _), v in R[t - 1].segment.items() if g2 == go) for go in range(G)] for t in range(1, T + 1)]),
    )
    q["base load shed"] = (np.array([[c("ysh", t, go) for go in range(G)] for t in range(1, T + 1)]), np.array([R[t - 1].shed for t in range(1, T + 1)]))
    F = len(inst.fabs)
    q["fab energy"] = (np.array([[c("E", t, f) for f in range(F)] for t in range(1, T + 1)]), np.array([R[t - 1].energy for t in range(1, T + 1)]))
    q["lots started"] = (np.array([[c("p", t, f) for f in range(F)] for t in range(1, T + 1)]), np.array([R[t - 1].lots_started for t in range(1, T + 1)]))
    pk = sorted(k[1:] for k in tm if k[0] == "xi")
    q["packaged"] = (np.array([[c("xi", t, *k) for k in pk] for t in range(1, T + 1)]), np.array([[R[t - 1].packaged.get(k, 0.0) for k in pk] for t in range(1, T + 1)]))
    D = len(inst.demands)
    q["demand served"] = (np.array([[c("D", t, d) for d in range(D)] for t in range(1, T + 1)]), np.array([R[t - 1].served for t in range(1, T + 1)]))
    slots = sorted(k[1] for k in tm if k[0] == "I")
    q["stocks"] = (np.array([[c("I", t, s) for s in slots] for t in range(1, T + 1)]), np.array([R[t - 1].stock[slots] for t in range(1, T + 1)]))
    weekly, credit = lp_costs(m, x)
    plan_c = np.array([[w.as_dict()[k] for k in COMP] for w in weekly])
    real_c = np.array([[r.costs.as_dict()[k] for k in COMP] for r in R])
    return q, plan_c, real_c, out.get("gap")


if __name__ == "__main__":
    N, tl = int(sys.argv[1]), int(sys.argv[2])
    flags = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("base",)
    res = Parallel(n_jobs=4)(delayed(one)(n, flags, tl) for n in range(N))
    print(f"{N} episodes, rules {flags}, MIP gaps {[None if r[3] is None else round(r[3], 3) for r in res]}")
    print(f"{'quantity':34s} {'plan':>12s} {'simulator':>12s} {'sim/plan':>8s} {'|diff|/plan':>11s} {'first week off (median)':>24s}")
    for name in res[0][0]:
        plan = [r[0][name][0] for r in res]; real = [r[0][name][1] for r in res]
        P_, R_ = sum(p.sum() for p in plan), sum(r.sum() for r in real)
        ad = sum(np.abs(p - r).sum() for p, r in zip(plan, real))
        firsts = []
        for p, r in zip(plan, real):
            scale = max(np.abs(p).sum(axis=1).mean(), 1e-9)
            off = np.flatnonzero(np.abs(p - r).sum(axis=1) > 0.02 * scale)
            firsts.append(off[0] + 1 if off.size else 99)
        print(f"{name:34s} {P_:12.4g} {R_:12.4g} {R_ / max(P_, 1e-9):8.3f} {ad / max(P_, 1e-9):11.3f} {int(np.median(firsts)):>10d}  (per episode: {firsts})")
    pc = np.mean([r[1].sum(axis=0) for r in res], axis=0) / 1e9; rc = np.mean([r[2].sum(axis=0) for r in res], axis=0) / 1e9
    print("cost by component, $B per episode: plan / simulator / difference")
    for k, a, b in zip(COMP, pc, rc):
        print(f"  {k:14s} {a:9.1f} {b:9.1f} {b - a:+9.1f}")
    T = res[0][1].shape[0]; qs = np.linspace(0, T, 5).astype(int)
    d = np.mean([(r[2] - r[1]).sum(axis=1) for r in res], axis=0) / 1e9
    print("simulator minus plan by weeks:", [f"{a + 1}-{b}: {d[a:b].sum():+.1f}" for a, b in zip(qs[:-1], qs[1:])])
