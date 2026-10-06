"""What is the best score with everything known, and which simulator rule costs what (worker D).

    uv run python lab/anastasiia/mpc_lab/planners/planner_D.py --episodes=4 --variants=relaxed,base,lots,pack,all --play
    uv run python lab/anastasiia/mpc_lab/planners/planner_D.py --episodes=4 --mpc    # the closed-loop bench on the same episodes

Part 1: the full-horizon LP from the true marks, with one rule at a time enforced in ALL weeks as a MILP (scipy.milp, HiGHS):
  base : base load first at every (base_first) grid and week. A binary s: shed base load (ysh > 0) only if s = 1, and then
         the grid's fabs draw nothing (sum E <= M (1 - s)) and the grid runs at full load.
  lots : a fab starts every lot it can: p = min(capacity, wafers on hand) unless the grid's energy binds. Binary b per
         fab-week (p = capacity, or no wafer left over) and binary u per grid-week (u = 1: energy not binding, all fabs start
         every feasible lot; u = 0: all the grid's output is used, y + sum E = G at full load). The pro rata split of a
         short grid between its two fabs is NOT enforced (relaxed).
  pack : an OSAT packages min(throughput, raw chips): binary a (a = 1: nothing is left raw; a = 0: throughput is full).
         The pro rata split across the two packaged commodities is NOT enforced in the plain variant.
Every variant keeps the package's pro rata segment loading (lam, short; every week in planning_rules) and drops the
week-1 lot_start / osat_start rows and every other price (they are replaced by the rules above). "relaxed" is that model
with no rule: it measures what the loading rows alone cost.
Part 2 (--play): the solution's flows are played open loop through the real Env, week by week.
"""

import sys
import time

import fire
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
from package_baselines import rss  # noqa: E402

ENTROPY, TASK = 111, "small"


def world(n):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(TASK, ENTROPY, n, REPLICATIONS, str(default_cache_dir()))


class Extra:
    """Rows and columns appended to a model."""

    def __init__(self, ncol):
        self.ncol, self.lb, self.ub, self.integer = ncol, [], [], []
        self.r, self.c, self.v, self.rlo, self.rhi = [], [], [], [], []
        self.nrow = 0

    def binary(self):
        self.lb.append(0.0), self.ub.append(1.0), self.integer.append(1)
        self.ncol += 1
        return self.ncol - 1

    def row(self, coefs, lo=-np.inf, hi=np.inf):
        for j, a in coefs:
            self.r.append(self.nrow), self.c.append(j), self.v.append(a)
        self.rlo.append(lo), self.rhi.append(hi)
        self.nrow += 1


def build(n, flags, pr_prev=None):
    """(inst, model, objective, extra, A_all pieces) for episode n with the rules in ``flags``."""
    from shockbench_flow.marks import osat_throughput
    from shockbench_flow.oracle.lp import build_lp

    inst, omega, marks, fb = world(n)
    m = build_lp(inst, marks, planning_rules=True)
    nc, T = m.meta["nc"], m.T
    tm = m.meta["template"]
    keep_eq = np.array([nm[0] not in ("lot_start", "osat_start") for nm in m.eq_names])
    A_eq, b_eq = m.A_eq[keep_eq], m.b_eq[keep_eq]
    # objective: the true cost, plus the package's price on a fuel segment's shortfall only
    pri = np.zeros_like(m.cost)
    for j, k in enumerate(m.keys):
        if k[0] == "short":
            pri[j] = m.meta["priority"][j]
    obj = m.cost - m.salvage + pri
    lb, ub = m.lb.copy(), m.ub.copy()
    ex = Extra(len(obj))

    def col(tag, t, *rest):
        return (t - 1) * nc + tm[(tag, *rest)]

    thr_R = osat_throughput(inst, marks.R_osat)
    rows_eq_extra = []
    nb = {"s": 0, "u": 0, "b": 0, "a": 0}
    for t in range(1, T + 1):
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = inst.grid_fabs[go]
            Gub = sum(ga.shares[k] for k in ga.shares) * marks.G_bar[t - 1, go]
            Eall = [col("E", t, fo) for fo in fabs]
            Emax = sum(inst.nodes[inst.fabs[fo]].fab.e * m.ub[tm[("p", fo)]] for fo in fabs) / max(1e-12, 1) if fabs else 0
            # E = e p / R exactly when the energy rule or lots rule is on (no free energy)
            if ("lots" in flags or "base" in flags) and fabs:
                for fo in fabs:
                    fa = inst.nodes[inst.fabs[fo]].fab
                    R = marks.R[t - 1, fo]
                    pj, Ej = col("p", t, fo), col("E", t, fo)
                    if R > 0:
                        ex.row([(pj, fa.e), (Ej, -R)], 0.0, 0.0)
                    else:
                        ub[Ej] = 0.0
            if "base" in flags and ga.priority == "base_first":
                s = ex.binary()
                nb["s"] += 1
                ybar = marks.y_bar[t - 1, go]
                ex.row([(col("ysh", t, go), 1.0), (s, -ybar)], -np.inf, 0.0)
                Em = sum(inst.nodes[inst.fabs[fo]].fab.e * m.ub[tm[("p", fo)]] / max(marks.R[t - 1, fo], 1e-9) * marks.R[t - 1, fo] / 1.0 for fo in fabs)
                # sum E <= Em (1 - s); E = e p / R <= e alpha cap0
                Em = sum(inst.nodes[inst.fabs[fo]].fab.e * marks.alpha_bar[t - 1, fo] * inst.nodes[inst.fabs[fo]].fab.cap0 for fo in fabs)
                if fabs:
                    ex.row([(j, 1.0) for j in Eall] + [(s, Em)], -np.inf, Em)
                ex.row([(col("lam", t, go), 1.0), (s, -1.0)], 0.0, np.inf)
            if "lots" in flags and fabs:
                u = ex.binary()
                nb["u"] += 1
                # u = 0: all output used at full load
                Gj = [col("G", t, go, k) for k in (ga.fuels + ((None,) if None in ga.shares else ()))]
                ex.row([(j, 1.0) for j in Gj] + [(col("y", t, go), -1.0)] + [(j, -1.0) for j in Eall] + [(u, -Gub)], -np.inf, 0.0)
                ex.row([(col("lam", t, go), 1.0), (u, 1.0)], 1.0, np.inf)
                for fo in fabs:
                    fa = inst.nodes[inst.fabs[fo]].fab
                    s_in = inst.slot_index[(inst.fabs[fo], fa.input)]
                    cap = m.ub[col("p", t, fo)]
                    if cap <= 0:
                        continue
                    b = ex.binary()
                    nb["b"] += 1
                    pj = col("p", t, fo)
                    ex.row([(pj, 1.0), (b, -cap), (u, -cap)], -cap, np.inf)  # p >= cap (b + u - 1)
                    st = inst.stock_slots[s_in].storage
                    Mw = st if np.isfinite(st) else 1e7
                    left = [(col("I", t, s_in), 1.0)] + ([(col("O", t, s_in), 1.0)] if ("O", s_in) in tm else [])
                    ex.row(left + [(b, -Mw), (u, Mw)], -np.inf, 0.0 + 0.0)  # I + O <= Mw (b + 1 - u)  -> shift
                    ex.rhi[-1] = Mw  # I + O - Mw b + Mw u <= Mw
        if "pack" in flags:
            for oo, o in enumerate(inst.osats):
                oa = inst.nodes[o].osat
                thr = float(thr_R[t - 1, oo])
                a = ex.binary()
                nb["a"] += 1
                pairs = sorted(oa.packages.items(), key=lambda it: it[1])
                xis = []
                for kr, kp in pairs:
                    s_raw = inst.slot_index[(o, kr)]
                    st = inst.stock_slots[s_raw].storage
                    Mw = st if np.isfinite(st) else 1e7
                    left = [(col("I", t, s_raw), 1.0)] + ([(col("O", t, s_raw), 1.0)] if ("O", s_raw) in tm else [])
                    ex.row(left + [(a, Mw)], -np.inf, Mw)  # a = 1 -> nothing left raw
                    xis.append((col("xi", t, oo, kp), 1.0))
                ex.row(xis + [(a, thr)], thr, np.inf)  # a = 0 -> throughput full
                if pr_prev is not None and len(pairs) == 2 and thr > 0:
                    # pro rata with the raw stock of the previous solution, in the weeks it was binding
                    raw = pr_prev[(t, oo)]
                    if raw is not None:
                        (k1, k2) = [col("xi", t, oo, kp) for _, kp in pairs]
                        ex.row([(k1, raw[1]), (k2, -raw[0])], -np.inf, np.inf)  # activated below by a
    return inst, marks, m, A_eq, b_eq, obj, lb, ub, ex, nb


def solve(n, flags, tl=600, gap=1e-4, threads=2):
    from scipy.optimize import LinearConstraint, Bounds, milp, linprog
    from shockbench_flow.oracle.lp import lp_cents

    t0 = time.time()
    inst, marks, m, A_eq, b_eq, obj, lb, ub, ex, nb = build(n, flags)
    n0 = len(obj)
    N = ex.ncol
    pad = lambda A: sparse.hstack([A, sparse.csr_matrix((A.shape[0], N - n0))]).tocsr()
    cons = [LinearConstraint(pad(A_eq), b_eq, b_eq), LinearConstraint(pad(m.A_ub), -np.inf, m.b_ub)]
    if ex.nrow:
        Ex = sparse.coo_matrix((ex.v, (ex.r, ex.c)), shape=(ex.nrow, N)).tocsr()
        cons.append(LinearConstraint(Ex, np.array(ex.rlo), np.array(ex.rhi)))
    c = np.concatenate([obj, np.zeros(N - n0)])
    integ = np.concatenate([np.zeros(n0), ex.integer])
    bnd = Bounds(np.concatenate([lb, ex.lb]), np.concatenate([ub, ex.ub]))
    if not ex.integer:
        res = linprog(c[:n0], A_ub=m.A_ub, b_ub=m.b_ub, A_eq=A_eq, b_eq=b_eq, bounds=np.column_stack([lb, ub]), method="highs")
        x, status, gapv, dual = res.x, res.status, 0.0, res.fun
    else:
        res = milp(c, constraints=cons, integrality=integ, bounds=bnd,
                   options=dict(time_limit=tl, mip_rel_gap=gap, presolve=True))
        x, status, gapv = res.x, res.status, getattr(res, "mip_gap", None)
        dual = getattr(res, "mip_dual_bound", None)
    out = dict(n=n, flags=flags, status=int(status), gap=gapv, secs=time.time() - t0, nb=nb)
    if x is None:
        out["J"] = None
        return out, None
    x = np.asarray(x)
    out["J"] = lp_cents(m, x[:n0])
    if dual is not None:  # J of the true cost lower bound: objective differs by the (tiny) short price; report both
        out["dual_obj"] = float(dual)
        out["obj"] = float(c @ x)
    return out, (m, x[:n0])


def play(n, m, x):
    """Open-loop: the plan's week-t action in the real simulator. Returns J cents and the clipped-away share."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = world(n)
    nc = m.meta["nc"]
    env = Env(fallback=fb)
    seed = _policy_seed(ENTROPY, n, NO_ZIP_SHA256)
    obs, info = env.reset(inst, "standard", omega, seed, marks=marks, policy_name="planD")
    done, t = False, 0
    while not done:
        xt = np.zeros_like(m.lb)
        xt[:nc] = x[t * nc : (t + 1) * nc]
        act = L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t]))
        obs, r, done, tr, inf = env.step(act)
        t += 1
    return int(env.trajectory.J_cents)


def one(n, variants, tl, do_play):
    res = {}
    for v in variants:
        flags = {"relaxed": (), "base": ("base",), "lots": ("lots",), "pack": ("pack",), "all": ("base", "lots", "pack"),
                 "base+lots": ("base", "lots")}[v]
        out, sol = solve(n, flags, tl)
        if do_play and sol is not None and v in do_play:
            out["played"] = play(n, *sol)
        res[v] = out
        print(n, v, {k: (round(val, 5) if isinstance(val, float) else val) for k, val in out.items()}, flush=True)
    return res


def main(episodes=4, variants="relaxed,base,lots,pack,all", play_variants="all", tl=600, n_jobs=2):
    from sbf_starter import scoring

    refs = list(scoring.episode_set(TASK, episodes, entropy=ENTROPY, verbose=False).references)
    variants = variants.split(",") if isinstance(variants, str) else list(variants)
    pv = set(play_variants.split(",")) if isinstance(play_variants, str) else set(play_variants or ())
    res = Parallel(n_jobs=n_jobs)(delayed(one)(n, variants, tl, pv) for n in range(episodes))
    print(f"\n{episodes} episodes of root {ENTROPY}, network {TASK}")
    for v in variants:
        for key, label in (("J", "optimum"), ("played", "open-loop play")):
            if not all(key in r[v] and r[v][key] is not None for r in res):
                continue
            costs = [r[v][key] for r in res]
            sc, lv = rss(refs, costs)
            gaps = [r[v].get("gap") for r in res]
            print(f"{v:10s} {label:15s} score {sc:.4f}  by level {[round(a,3) for a in lv.values()]}  gaps {[None if g is None else round(g,4) for g in gaps]}")
    print("per-episode J cents", {v: [r[v]["J"] for r in res] for v in variants})


if __name__ == "__main__":
    fire.Fire(main)
