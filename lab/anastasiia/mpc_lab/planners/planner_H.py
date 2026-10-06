"""Worker H: the simulator's automatic steps (grid fuel, lot starts, packaging) written exactly into the plan.

    # open loop: full-horizon plan with the rules, played in the simulator; agreement table
    uv run python lab/anastasiia/mpc_lab/planners/planner_H.py validate --episodes=4 --rules=base,fuel,lots,pack --tl=300
    # closed loop (window MILP re-planned every week)
    uv run python lab/anastasiia/mpc_lab/planners/planner_H.py closed --episodes=16 --rules=... --hb=16 --tl=3 --n_jobs=4

Rules (``add_rules``), per grid g and week t (all grids here are base_first):
  s_gt  base load is shed (ysh > 0 only if s = 1); then the fabs get no energy.            (D's "base")
  u_gt  1 = the grid is not short: every fab starts p = min(capacity, wafers on hand);
        0 = the grid is short: lam = 1, y + sum E = G (all output used), s may be 1.
  fuel  (u = 0) every fuel segment is really exhausted: G_k = its cap (zeta G-bar, times the ration if gas is rationed;
        binary r = "last week's stock reached psi I-bar") or the fuel stock after burning is 0.  Without it the plan
        calls a grid short by reporting spurious ``short`` (priced like a fuel) and holds wafers.
  lots  b_ft  1 = wafers on hand >= capacity.  With u = 1: b = 1 -> p = capacity, b = 0 -> no wafer left (no holding back).
        Two fabs on a grid with u = 0: p_1 / cap_1 = p_2 / cap_2 when both have wafers >= capacity (shared ratio rho,
        exact there; relaxed when a fab has fewer wafers than capacity).
  pack  a_ot  1 = every raw chip is packaged, 0 = the throughput is full (exact for one packaged commodity; with two
        the pro rata split is NOT enforced unless ``prorata``: then it is linearised on the raw shares of a previous plan).
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
COMP = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


class Extra:
    """Columns and rows appended to a model (columns after the model's)."""

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


def add_rules(inst, m, ex, ub, rules, t_from=1, t_to=None, i0=None, prev=None, only_gw=None, only_ow=None, phat_ref=None):
    """Append the rules to the model ``m`` (full horizon or a window). ``ub`` is a copy of m.ub that is modified in place.

    i0: initial stock (needed for the ration at t_from = 1). prev: {(t, o): share of product 1 in the raw stock} for ``prorata``.
    Returns the count of binaries by kind.
    """
    nc = m.meta["nc"]
    tm = m.meta["template"]
    T = len(m.lb) // nc
    t_to = T if t_to is None else min(t_to, T)
    neq, nub = len(m.eq_rows), len(m.ub_rows)
    eqi = {r: i for i, r in enumerate(m.eq_rows)}
    ubi = {r: i for i, r in enumerate(m.ub_rows)}
    col = lambda tag, t, *r: (t - 1) * nc + tm[(tag, *r)]
    has = lambda tag, *r: (tag, *r) in tm
    cnt = {"s": 0, "u": 0, "b": 0, "w": 0, "r": 0, "a": 0}
    psi = inst.params.psi
    use_fuel, use_lots, use_pack = "fuel" in rules, "lots" in rules, "pack" in rules
    BIG = 1e7

    def storage(slot):
        st = inst.stock_slots[slot].storage
        return float(st) if st is not None and np.isfinite(st) else BIG

    for t in range(t_from, t_to + 1):
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            if not fabs:
                continue
            base_first = ga.priority == "base_first"
            ybar = float(m.b_eq[(t - 1) * neq + eqi[("baseload", go)]])
            segs = [k for k in ga.fuels] + ([None] if None in ga.shares else [])
            Gj = [col("G", t, go, k) for k in segs]
            Eall = [col("E", t, fo) for fo in fabs]
            caps = {fo: float(ub[col("p", t, fo)]) for fo in fabs}
            Gub = float(sum(ub[j] for j in Gj))
            # E = e p / R exactly
            for fo in fabs:
                fa = inst.nodes[inst.fabs[fo]].fab
                R = -m.A_ub[(t - 1) * nub + ubi[("fab_energy", fo)], col("E", t, fo)]
                if R > 0:
                    ex.row([(col("p", t, fo), fa.e), (col("E", t, fo), -R)], 0.0, 0.0)
                else:
                    ub[col("E", t, fo)] = 0.0
            Em = sum(inst.nodes[inst.fabs[fo]].fab.e * caps[fo] / max(1e-9, 1.0) for fo in fabs)
            Em = max(Em, 1.0)
            if only_gw is not None and (t, go) not in only_gw:
                continue
            s = None
            if "base" in rules and base_first:
                s = ex.binary(); cnt["s"] += 1
                ex.row([(col("ysh", t, go), 1.0), (s, -ybar)], -np.inf, 0.0)
                ex.row([(j, 1.0) for j in Eall] + [(s, Em)], -np.inf, Em)
                ex.row([(col("lam", t, go), 1.0), (s, -1.0)], 0.0, np.inf)
            u = None
            if use_lots or use_fuel:
                u = ex.binary(); cnt["u"] += 1
                # u = 0: all of the grid's output is used, at full load
                ex.row([(j, 1.0) for j in Gj] + [(col("y", t, go), -1.0)] + [(j, -1.0) for j in Eall] + [(u, -Gub)], -np.inf, 0.0)
                ex.row([(col("lam", t, go), 1.0), (u, 1.0)], 1.0, np.inf)
                if s is not None:
                    ex.row([(u, 1.0), (s, 1.0)], -np.inf, 1.0)
            if use_fuel and u is not None:
                for k in ga.fuels:
                    sl = inst.slot_index[(g, k)]
                    ak = float(ub[col("G", t, go, k)])
                    if ak <= 0:
                        continue
                    Iend = [(col("I", t, sl), 1.0)] + ([(col("O", t, sl), 1.0)] if has("O", sl) else [])
                    Mi = storage(sl)
                    w = ex.binary(); cnt["w"] += 1
                    # leftover fuel only if the segment is at its cap:  I + O <= Mi (w + u)
                    ex.row(Iend + [(w, -Mi), (u, -Mi)], -np.inf, 0.0)
                    gk = col("G", t, go, k)
                    thr = psi * ga.ibar[k] if k == ga.rationed else 0.0
                    if thr > 0:
                        r = ex.binary(); cnt["r"] += 1
                        if t == 1:
                            Iprev = [(None, float(i0[sl]))]
                        else:
                            Iprev = [(col("I", t - 1, sl), 1.0)]
                        cprev = sum(a for j, a in Iprev if j is None)
                        var = [(j, a) for j, a in Iprev if j is not None]
                        # r = 1 <-> last week's stock >= thr
                        ex.row(var + [(r, -thr)], -cprev, np.inf)  # Iprev >= thr r
                        ex.row(var + [(r, -Mi)], -np.inf, thr - cprev)  # Iprev <= thr + Mi r
                        # w = 1, u = 0: G >= min(ak, ak Iprev / thr)
                        ex.row([(gk, 1.0), (w, -ak), (u, ak), (r, -ak)], -ak, np.inf)  # r = 1 : G >= ak
                        # G >= ak Iprev/thr - ak(1-w) - ak u - ak r
                        ex.row([(gk, 1.0)] + [(j, -ak / thr * a) for j, a in var] + [(w, -ak), (u, ak), (r, ak)],
                               ak * cprev / thr - ak, np.inf)
                    else:
                        ex.row([(gk, 1.0), (w, -ak), (u, ak)], 0.0, np.inf)  # G >= ak (w - u)
            if use_lots:
                for fo in fabs:
                    fa = inst.nodes[inst.fabs[fo]].fab
                    sl = inst.slot_index[(inst.fabs[fo], fa.input)]
                    cap = caps[fo]
                    if cap <= 0:
                        continue
                    Mw = storage(sl)
                    pj = col("p", t, fo)
                    Iend = [(col("I", t, sl), 1.0)] + ([(col("O", t, sl), 1.0)] if has("O", sl) else [])
                    b = ex.binary(); cnt["b"] += 1
                    # b = 1 <-> wafers on hand W = I + O + p >= cap
                    ex.row(Iend + [(pj, 1.0), (b, -cap)], 0.0, np.inf)  # W >= cap b
                    ex.row(Iend + [(pj, 1.0), (b, -Mw)], -np.inf, cap)  # W <= cap + Mw b
                    ex.row([(pj, 1.0), (b, -cap), (u, -cap)], -cap, np.inf)  # u = 1, b = 1: p = cap
                    ex.row(Iend + [(b, -Mw), (u, Mw)], -np.inf, Mw)  # u = 1, b = 0: nothing left
                    caps[fo] = (cap, b)
                if len(fabs) == 2 and all(isinstance(caps[fo], tuple) for fo in fabs):
                    (c1, b1), (c2, b2) = caps[fabs[0]], caps[fabs[1]]
                    p1, p2 = col("p", t, fabs[0]), col("p", t, fabs[1])
                    ref = None if phat_ref is None else (phat_ref.get((t, fabs[0])), phat_ref.get((t, fabs[1])))
                    if ref is not None and ref[0] is not None and ref[0] > 1e-6 and ref[1] > 1e-6:
                        # u = 0: p1 / phat1 = p2 / phat2 with phat taken from the previous plan (fixed-point iteration)
                        r1, r2 = ref
                        Mr = r2 * c1 + r1 * c2
                        for sgn in (1.0, -1.0):
                            ex.row([(p1, sgn * r2), (p2, -sgn * r1), (u, Mr)], -np.inf, Mr)
                    else:
                        Mr = c1 * c2
                        # u = 0 and b1 = b2 = 1: c2 p1 = c1 p2
                        for sgn in (1.0, -1.0):
                            ex.row([(p1, sgn * c2), (p2, -sgn * c1), (b1, Mr), (b2, Mr), (u, -Mr)], -np.inf, 2 * Mr)
        if use_pack:
            for oo, o in enumerate(inst.osats):
                if only_ow is not None and (t, oo) not in only_ow:
                    continue
                oa = inst.nodes[o].osat
                pairs = sorted(oa.packages.items(), key=lambda it: it[1])
                if len(pairs) > 1:
                    thr = float(m.b_ub[(t - 1) * nub + ubi[("osat", oo)]])
                else:
                    thr = float(ub[col("xi", t, oo, pairs[0][1])])
                if thr <= 0:
                    continue
                a = ex.binary(); cnt["a"] += 1
                xis = []
                for kr, kp in pairs:
                    sl = inst.slot_index[(o, kr)]
                    Mw = storage(sl)
                    Iend = [(col("I", t, sl), 1.0)] + ([(col("O", t, sl), 1.0)] if has("O", sl) else [])
                    ex.row(Iend + [(a, Mw)], -np.inf, Mw)  # a = 1: nothing left raw
                    xis.append((col("xi", t, oo, kp), 1.0))
                ex.row(xis + [(a, thr)], thr, np.inf)  # a = 0: throughput full
                if prev is not None and len(pairs) == 2 and (t, oo) in prev:
                    th = prev[(t, oo)]  # share of product 1 in the raw stock of the previous plan
                    if th is not None:
                        (x1, _), (x2, _) = xis
                        # a = 0: xi_1 = th (xi_1 + xi_2); relaxed when a = 1
                        Mx = thr
                        for sgn in (1.0, -1.0):
                            ex.row([(x1, sgn * (1 - th)), (x2, -sgn * th), (a, -Mx)], -np.inf, 0.0)
    return cnt


def world(n):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(TASK, ENTROPY, n, REPLICATIONS, str(default_cache_dir()))


def parse(rules):
    if isinstance(rules, (tuple, list)):
        return tuple(rules)
    return tuple(r for r in str(rules).split(",") if r)


def solve_full(n, rules, tl=300, gap=1e-4, prev=None, only_gw=None, only_ow=None, world_=None, phat_ref=None):
    """The full-horizon plan (everything known) with ``rules``; returns (info, (model, x))."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from shockbench_flow.dynamics.sim import initial_stock
    from shockbench_flow.oracle.lp import build_lp, lp_cents

    t0 = time.time()
    inst, omega, marks, fb = world_ or world(n)
    m = build_lp(inst, marks, planning_rules=True)
    drop = set()
    if "lots" in rules:
        drop.add("lot_start")
    if "pack" in rules:
        drop.add("osat_start")
    keep = np.array([nm[0] not in drop for nm in m.eq_names])
    A_eq, b_eq = m.A_eq[keep], m.b_eq[keep]
    pri = np.zeros_like(m.cost)
    for j, k in enumerate(m.keys):
        if k[0] == "short":
            pri[j] = m.meta["priority"][j]
    obj = m.cost - m.salvage + pri
    ub = m.ub.copy()
    ex = Extra(len(obj))
    cnt = add_rules(inst, m, ex, ub, rules, t_from=1, i0=initial_stock(inst), prev=prev, only_gw=only_gw, only_ow=only_ow, phat_ref=phat_ref)
    n0, N = len(obj), ex.ncol
    out = dict(n=n, rules=rules, nb=cnt)
    pad = lambda A: sparse.hstack([A, sparse.csr_matrix((A.shape[0], N - n0))]).tocsr()
    cons = [LinearConstraint(pad(A_eq), b_eq, b_eq), LinearConstraint(pad(m.A_ub), -np.inf, m.b_ub)]
    if ex.nrow:
        Ex = sparse.coo_matrix((ex.v, (ex.r, ex.c)), shape=(ex.nrow, N)).tocsr()
        cons.append(LinearConstraint(Ex, np.array(ex.rlo), np.array(ex.rhi)))
    c = np.concatenate([obj, np.zeros(N - n0)])
    res = milp(c, constraints=cons, integrality=np.concatenate([np.zeros(n0), ex.integer]),
               bounds=Bounds(np.concatenate([m.lb, ex.lb]), np.concatenate([ub, ex.ub])),
               options=dict(time_limit=tl, mip_rel_gap=gap, presolve=True))
    x, out["gap"] = res.x, (getattr(res, "mip_gap", None) if ex.integer else 0.0)
    out["secs"] = time.time() - t0
    if x is None:
        out["J"] = None
        return out, None
    x = np.asarray(x)
    out["J"] = lp_cents(m, x[:n0])
    return out, (m, x[:n0])


def violations(inst, m, x, i0, tol=1e-3, t_from=1, t_to=None):
    """Grid-weeks and OSAT-weeks where the plan's production differs from the simulator's step 7 on the plan's own stocks."""
    from shockbench_flow.dynamics.production import package

    nc, tm = m.meta["nc"], m.meta["template"]
    T = len(m.lb) // nc if t_to is None else min(t_to, len(m.lb) // nc)
    neq, nub = len(m.eq_rows), len(m.ub_rows)
    eqi = {r: i for i, r in enumerate(m.eq_rows)}
    ubi = {r: i for i, r in enumerate(m.ub_rows)}
    c = lambda tag, t, *r: float(x[(t - 1) * nc + tm[(tag, *r)]]) if (tag, *r) in tm else 0.0
    psi = inst.params.psi
    gw, ow = set(), set()
    kinds = {"lots": 0, "shed": 0, "pack": 0}
    phat_all = {}
    for t in range(t_from, T + 1):
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            if not fabs:
                continue
            ybar = float(m.b_eq[(t - 1) * neq + eqi[("baseload", go)]])
            gav = m.ub[(t - 1) * nc + tm[("G", go, None)]] if None in ga.shares else 0.0
            for k in ga.fuels:
                sl = inst.slot_index[(g, k)]
                ak = float(m.ub[(t - 1) * nc + tm[("G", go, k)]])
                if k == ga.rationed and psi * ga.ibar[k] > 0:
                    ip = float(i0[sl]) if t == 1 else c("I", t - 1, sl)
                    ak *= min(1.0, ip / (psi * ga.ibar[k]))
                F = c("I", t, sl) + c("O", t, sl) + c("G", t, go, k)
                gav += min(ak, F)
            y = min(ybar, gav)
            phat, ehat = {}, {}
            for fo in fabs:
                fa = inst.nodes[inst.fabs[fo]].fab
                sl = inst.slot_index[(inst.fabs[fo], fa.input)]
                cap = float(m.ub[(t - 1) * nc + tm[("p", fo)]])
                R = -m.A_ub[(t - 1) * nub + ubi[("fab_energy", fo)], (t - 1) * nc + tm[("E", fo)]]
                phat[fo] = min(cap, c("I", t, sl) + c("O", t, sl) + c("p", t, fo))
                ehat[fo] = fa.e * phat[fo] / R if R > 0 else 0.0
                phat_all[(t, fo)] = phat[fo]
            tot = sum(ehat.values())
            ratio = min(1.0, (gav - y) / tot) if tot > 0 else 0.0
            bad = False
            for fo in fabs:
                cap = float(m.ub[(t - 1) * nc + tm[("p", fo)]])
                if abs(c("p", t, fo) - phat[fo] * ratio) > tol * max(cap, 1.0):
                    bad = True
                    kinds["lots"] += 1
            if abs(c("ysh", t, go) - (ybar - y)) > tol * max(ybar, 1.0):
                bad = True
                kinds["shed"] += 1
            if bad:
                gw.add((t, go))
        for oo, o in enumerate(inst.osats):
            oa = inst.nodes[o].osat
            pairs = sorted(oa.packages.items(), key=lambda it: it[1])
            thr = float(m.b_ub[(t - 1) * nub + ubi[("osat", oo)]]) if len(pairs) > 1 else float(m.ub[(t - 1) * nc + tm[("xi", oo, pairs[0][1])]])
            raw = [c("I", t, inst.slot_index[(o, kr)]) + c("O", t, inst.slot_index[(o, kr)]) + c("xi", t, oo, kp) for kr, kp in pairs]
            xs = package(raw, thr)
            if any(abs(c("xi", t, oo, kp) - q) > tol * max(thr, 1.0) for (kr, kp), q in zip(pairs, xs)):
                ow.add((t, oo))
                kinds["pack"] += 1
    return gw, ow, kinds, phat_all


def solve_lazy(n, rules, tl=60, gap=1e-3, rounds=8, log=True):
    """Constraint generation: binaries only on grid-weeks / OSAT-weeks where the plan breaks the simulator's rules."""
    from shockbench_flow.dynamics.sim import initial_stock

    w = world(n)
    inst = w[0]
    i0 = initial_stock(inst)
    gw, ow = set(), set()
    hist = []
    pref = None
    for it in range(rounds + 1):
        out, sol = solve_full(n, rules, tl, gap, only_gw=gw, only_ow=ow, world_=w, phat_ref=pref)
        if sol is None:
            out["hist"] = hist
            return out, None
        m, x = sol
        g2, o2, kinds, pref = violations(inst, m, x, i0)
        hist.append((it, len(gw), len(ow), len(g2 - gw), len(o2 - ow), round(out["secs"]), out["gap"], out["J"]))
        if log:
            print(f"  ep {n} round {it}: binaries on {len(gw)} grid-weeks {len(ow)} osat-weeks; violations {len(g2)} / {len(o2)} (new {len(g2 - gw)}/{len(o2 - ow)}) {kinds} gap {out['gap']} J {out['J']} {out['secs']:.0f}s", file=sys.stderr, flush=True)
        if not (g2 - gw) and not (o2 - ow):
            break
        gw |= g2
        ow |= o2
    out["hist"] = hist
    out["rounds"] = it
    out["flagged"] = (len(gw), len(ow))
    return out, (m, x)


def play(n, m, x):
    """Open loop in the real Env. Returns (J cents, records)."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = world(n)
    nc = m.meta["nc"]
    env = Env(fallback=fb)
    obs, info = env.reset(inst, "standard", omega, _policy_seed(ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="planH")
    done, t = False, 0
    while not done:
        xt = np.zeros_like(m.lb)
        xt[:nc] = x[t * nc : (t + 1) * nc]
        obs, r, done, tr, inf = env.step(L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t])))
        t += 1
    return int(env.trajectory.J_cents), env.trajectory.records


def agreement(inst, m, x, R):
    """Plan vs simulator totals: lots, wafer stock at fabs, packaging, demand served."""
    nc, T, tm = m.meta["nc"], m.T, m.meta["template"]
    c = lambda tag, t, *r: float(x[(t - 1) * nc + tm[(tag, *r)]]) if (tag, *r) in tm else 0.0
    F, D = len(inst.fabs), len(inst.demands)
    wslots = [inst.slot_index[(f, inst.nodes[f].fab.input)] for f in inst.fabs]
    pk = sorted(k[1:] for k in tm if k[0] == "xi")
    pairs = {
        "lots started": (np.array([[c("p", t, f) for f in range(F)] for t in range(1, T + 1)]), np.array([R[t - 1].lots_started for t in range(1, T + 1)])),
        "wafer stock at fabs": (np.array([[c("I", t, s) for s in wslots] for t in range(1, T + 1)]), np.array([R[t - 1].stock[wslots] for t in range(1, T + 1)])),
        "packaged": (np.array([[c("xi", t, *k) for k in pk] for t in range(1, T + 1)]), np.array([[R[t - 1].packaged.get(k, 0.0) for k in pk] for t in range(1, T + 1)])),
        "demand served": (np.array([[c("D", t, d) for d in range(D)] for t in range(1, T + 1)]), np.array([R[t - 1].served for t in range(1, T + 1)])),
    }
    return pairs


def one(n, rules, tl, gap, prorata, lazy=True):
    prev = None
    out, sol = solve_lazy(n, rules, tl, gap) if lazy else solve_full(n, rules, tl, gap)
    if prorata and sol is not None and "pack" in rules:
        inst, _, _, _ = world(n)
        m, x = sol
        nc, tm, T = m.meta["nc"], m.meta["template"], m.T
        prev = {}
        for t in range(1, T + 1):
            for oo, o in enumerate(inst.osats):
                pairs = sorted(inst.nodes[o].osat.packages.items(), key=lambda it: it[1])
                if len(pairs) == 2:
                    raw = []
                    for kr, kp in pairs:
                        sl = inst.slot_index[(o, kr)]
                        raw.append(x[(t - 1) * nc + tm[("I", sl)]] + (x[(t - 1) * nc + tm[("O", sl)]] if ("O", sl) in tm else 0.0)
                                   + x[(t - 1) * nc + tm[("xi", oo, kp)]])
                    prev[(t, oo)] = raw[0] / (raw[0] + raw[1]) if raw[0] + raw[1] > 1e-6 else None
        out, sol = solve_full(n, rules, tl, gap, prev=prev)
    if sol is None:
        return out, None
    m, x = sol
    J, R = play(n, m, x)
    inst, _, _, _ = world(n)
    ag = agreement(inst, m, x, R)
    stat = {k: (float(p.sum()), float(r.sum()), float(np.abs(p - r).sum())) for k, (p, r) in ag.items()}
    out["played"] = J
    return out, stat


def validate(episodes=4, rules="base", tl=60, gap=1e-3, n_jobs=4, prorata=False, lazy=True):
    from sbf_starter import scoring

    refs = list(scoring.episode_set(TASK, episodes, entropy=ENTROPY, verbose=False).references)
    if isinstance(rules, (tuple, list)):
        rules = ",".join(rules)
    variants = [parse(v) for v in rules.split(";")]
    for rl in variants:
        res = Parallel(n_jobs=n_jobs)(delayed(one)(n, rl, tl, gap, prorata, lazy) for n in range(episodes))
        claim = [r[0]["J"] for r in res]
        played = [r[0].get("played") for r in res]
        if any(v is None for v in claim + played):
            print(rl, "failed", [r[0] for r in res])
            continue
        sc, lv = rss(refs, claim)
        sp, lp_ = rss(refs, played)
        for r in res:
            print("   ep", r[0]["n"], "claimed J", r[0]["J"], "played J", r[0]["played"], "flagged", r[0].get("flagged"), "rounds", r[0].get("rounds"), "hist", r[0].get("hist"))
        print(f"\nrules {rl} prorata={prorata}: {episodes} episodes, claimed {sc:.4f}, played {sp:.4f}  gaps "
              f"{[None if r[0].get('gap') is None else round(r[0]['gap'], 4) for r in res]}  secs {[round(r[0]['secs']) for r in res]} nb {res[0][0]['nb']}")
        for k in res[0][1]:
            P = sum(r[1][k][0] for r in res); S = sum(r[1][k][1] for r in res); A = sum(r[1][k][2] for r in res)
            print(f"   {k:22s} plan {P:12.4g} sim {S:12.4g} sim/plan {S / max(P, 1e-9):6.3f} |diff|/plan {A / max(P, 1e-9):6.3f}", flush=True)


def window_plan(inst, model, rules, hb, tl, rounds, gap, stats):
    """Lazy MILP of the window: rules in weeks 2..hb on the grid-weeks / OSAT-weeks the plan breaks. Returns x (or None)."""
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp

    n0 = len(model.lb)
    nc = model.meta["nc"]
    weeks = n0 // nc
    gw, ow, pref, best = set(), set(), None, None
    t_end = time.process_time() + tl
    obj = model.objective().copy()
    if "noprice" in rules:  # the exact rules replace the package's price on kept wafers / raw chips in weeks 2..hb
        pri = model.meta["priority"]
        tm = model.meta["template"]
        slots = [inst.slot_index[(f, inst.nodes[f].fab.input)] for f in inst.fabs]
        for o in inst.osats:
            slots += [inst.slot_index[(o, kr)] for kr in inst.nodes[o].osat.packages]
        for t in range(2, min(hb, weeks) + 1):
            for sl in slots:
                for tag in ("I", "O"):
                    if (tag, sl) in tm:
                        j = (t - 1) * nc + tm[(tag, sl)]
                        obj[j] -= pri[j]
    for it in range(rounds + 1):
        ex = Extra(n0)
        ub = model.ub.copy()
        add_rules(inst, model, ex, ub, rules, t_from=2, t_to=hb, only_gw=gw, only_ow=ow, phat_ref=pref)
        N = ex.ncol
        pad = lambda A: sparse.hstack([A, sparse.csr_matrix((A.shape[0], N - n0))]).tocsr()
        cons = [LinearConstraint(pad(model.A_ub), -np.inf, model.b_ub)]
        if model.A_eq.shape[0]:
            cons.append(LinearConstraint(pad(model.A_eq), model.b_eq, model.b_eq))
        if ex.nrow:
            cons.append(LinearConstraint(sparse.coo_matrix((ex.v, (ex.r, ex.c)), shape=(ex.nrow, N)).tocsr(), np.array(ex.rlo), np.array(ex.rhi)))
        left = max(0.2, t_end - time.process_time()) if it else max(tl, 1.0)
        res = milp(np.concatenate([obj, np.zeros(N - n0)]), constraints=cons,
                   integrality=np.concatenate([np.zeros(n0), ex.integer]),
                   bounds=Bounds(np.concatenate([model.lb, ex.lb]), np.concatenate([ub, ex.ub])),
                   options=dict(time_limit=left, mip_rel_gap=gap))
        if res.x is None:
            break
        x = np.asarray(res.x)[:n0]
        best = x
        g2, o2, kinds, pref = violations(inst, model, x, None, t_from=2, t_to=hb)
        stats["rounds"] += 1
        if not (g2 - gw) and not (o2 - ow):
            break
        if time.process_time() > t_end:
            break
        gw |= g2
        ow |= o2
    stats["flag"].append(len(gw) + len(ow))
    return best


def closed_episode(known_name, task, entropy, n, horizon, rules, hb, tl, rounds, gap):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world
    import mpc_foresight as mf

    known = mf.GROUPS[known_name]
    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    q = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    context = PolicyContext(fq_quantile=q, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {w: i for i, w in L.NOW_FIELDS}
    stats = {"cpu": [], "rounds": 0, "flag": [], "fallback": 0}
    use = bool(rules)

    class Planner(MpcDet):
        def _horizon(self, inst, plan):
            return horizon

        def act(self, obs):
            t0 = time.process_time()
            week = int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            arrays = {k: a.copy() for k, a in L.persistence_arrays(inst, obs, self._memory, weeks).items()}
            f = week - 1
            for name in known:
                arrays[name] = np.array(getattr(marks, name)[f : f + weeks])
                if name in now:
                    arrays[now[name]] = np.array(getattr(marks, now[name])[f : f + weeks])
            model = L.rolled_lp(inst, obs, L.read_only(arrays), weeks, planning_rules=True)
            x = window_plan(inst, model, rules, hb, tl, rounds, gap, stats) if use and weeks >= 2 else None
            if x is None:
                from scipy.optimize import linprog
                rows = {}
                if model.A_ub.shape[0]:
                    rows.update(A_ub=model.A_ub, b_ub=model.b_ub)
                if model.A_eq.shape[0]:
                    rows.update(A_eq=model.A_eq, b_eq=model.b_eq)
                plan = linprog(model.objective(), bounds=np.column_stack([model.lb, model.ub]), method="highs-ds", **rows)
                if plan.status != 0 or plan.x is None:
                    stats["cpu"].append(time.process_time() - t0)
                    return self._fallback.act(obs)
                x = np.asarray(plan.x)
                stats["fallback"] += 1
            a = L.week1_action(inst, model, x, obs, L.prohibited_now(self._memory, week))
            stats["cpu"].append(time.process_time() - t0)
            return a

    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    J = int(rollout(inst, Planner(None, context), omega, "standard", seed, marks=marks, fallback=fallback).J_cents)
    cpu = np.array(stats["cpu"])
    return J, float(cpu.mean()), float(cpu.max()), stats["rounds"], float(np.mean(stats["flag"])) if stats["flag"] else 0.0, stats["fallback"]


def closed(episodes=16, rules="base,fuel,lots", hb=16, tl=3.0, rounds=3, gap=1e-3, known="everything", horizon=52, n_jobs=4, start=0):
    import mpc_foresight  # noqa: F401
    from sbf_starter import scoring

    if isinstance(rules, (tuple, list)):
        rules = ",".join(rules)
    rl = parse(rules)
    refs = list(scoring.episode_set(TASK, start + episodes, entropy=ENTROPY, verbose=False).references)[start:]
    name = "nothing (the MPC as it is)" if known == "nothing" else known
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(closed_episode)(name, TASK, ENTROPY, n, horizon, rl, hb, tl, rounds, gap) for n in range(start, start + episodes))
    costs = [o[0] for o in out]
    sc, lv = rss(refs, costs)
    print(f"known={known} eps={episodes} rules={rl} hb={hb} tl={tl} rounds={rounds}: score {sc:.4f} levels {[round(v, 3) for v in lv.values()]} "
          f"cpu/week mean {np.mean([o[1] for o in out]):.2f} max {np.max([o[2] for o in out]):.2f} lazy rounds/ep {np.mean([o[3] for o in out]):.0f} "
          f"flagged {np.mean([o[4] for o in out]):.0f} fallbacks {sum(o[5] for o in out)} ({time.time()-t0:.0f}s) costs {costs}", flush=True)


if __name__ == "__main__":
    fire.Fire({"validate": validate, "closed": closed})
