"""Replan-frequency curve of the plain LP under persistence: plan at week 1, re-plan at the weeks of a schedule.

    uv run python lab/anastasiia/mpc_lab/planners/replan_curve.py run --episodes=8 --entropy=444 --n_jobs=2

The plain LP is persist_plan's round 0: ``build_lp(..., planning_rules=True)`` with the lot-start rows dropped, the
exact E = e p / R rows of planner_H.add_rules and no binaries; objective cost - salvage + the package's price on
``short`` only (planner_H.solve_full with empty flag sets).

Closed loop, one Env per (episode, schedule). At each week t of the schedule the window t..T is rebuilt from what an
agent observes at week t: the rolled start state from the observation (``lp_common.rolled_window``: stock, pipeline,
queues, WIP, backlog — the state the previous plan's execution reached in the TRUE simulator) and the persistence
forecast ``lp_common.persistence_arrays`` (week t's ``graph_now`` repeated, announced pending prohibitions switched on
at their week, sigma_scr 0, demand forecast h <= 7 then d-bar m_sea). The ObservedGraph memory is updated every week.
Between replans the week's slice of the last plan is executed via ``lp_common.week1_action`` with the TRUE week's
prohibitions masked (as persist_plan.play_on). Anchors: the plain LP on the TRUE marks played blind, and the hybrid
(planner_LSF cache, replayed).

Schedule {1} must reproduce persist_plan's plain-LP persistence number (attribution table, column ``persist``).
"""

import os
import pickle
import sys
import time

import fire
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import persist_plan as PP  # noqa: E402  (sets SBF_CACHE_DIR)

H = PP.H
LOG = os.path.join(HERE, "replan_curve.log")
BN = PP.BN
SCHEDULES = {
    "{1}": [1],
    "{1,27}": [1, 27],
    "{1,14,27,40}": [1, 14, 27, 40],
    "step4": list(range(1, 53, 4)),
    "step2": list(range(1, 53, 2)),
    "weekly": list(range(1, 53)),
}


def logline(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def plain_lp(inst, m, rules=PP.RULES):
    """planner_H.solve_full's round 0 on a given model (no binaries). Returns (x or None, claimed J cents)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from shockbench_flow.oracle.lp import lp_cents

    drop = {"lot_start"} if "lots" in rules else set()
    keep = np.array([nm[0] not in drop for nm in m.eq_names])
    A_eq, b_eq = m.A_eq[keep], m.b_eq[keep]
    pri = np.zeros_like(m.cost)
    for j, k in enumerate(m.keys):
        if k[0] == "short":
            pri[j] = m.meta["priority"][j]
    obj = m.cost - m.salvage + pri
    ub = m.ub.copy()
    ex = H.Extra(len(obj))
    H.add_rules(inst, m, ex, ub, rules, t_from=1, i0=None, only_gw=set(), only_ow=set())
    assert ex.ncol == len(obj), "plain LP must add no binaries"
    cons = []
    if A_eq.shape[0]:
        cons.append(LinearConstraint(A_eq, b_eq, b_eq))
    if m.A_ub.shape[0]:
        cons.append(LinearConstraint(m.A_ub, -np.inf, m.b_ub))
    if ex.nrow:
        cons.append(LinearConstraint(sparse.coo_matrix((ex.v, (ex.r, ex.c)), shape=(ex.nrow, len(obj))).tocsr(), np.array(ex.rlo), np.array(ex.rhi)))
    res = milp(obj, constraints=cons, integrality=np.zeros(len(obj)), bounds=Bounds(m.lb, ub), options=dict(presolve=True))
    if res.x is None:
        return None, None
    x = np.asarray(res.x)
    return x, lp_cents(m, x)


def closed(n, w, weeks):
    """Play episode n closed loop with replans at ``weeks``; returns (J cents, stats)."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = w
    T = inst.T
    env = Env(fallback=fb)
    obs, _ = env.reset(inst, "standard", omega, _policy_seed(H.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="replan")
    mem = L.ObservedGraph.nominal(inst)
    plan = None  # (model, x, start week)
    st = dict(solves=0, fails=0, secs=0.0, claims=[], churn=[])
    sched = set(weeks)
    done, t = False, 1
    while not done:
        mem.update(inst, obs)
        if t in sched or plan is None:
            t0 = time.process_time()
            Hh = T - t + 1
            arrays = L.persistence_arrays(inst, obs, mem, Hh)
            inst_r, backlog = L.rolled_window(inst, obs, Hh)
            m = L.rolled_lp(inst, obs, arrays, Hh, window=(inst_r, backlog), planning_rules=True)
            x, claim = plain_lp(inst_r, m)
            st["solves"] += 1
            st["secs"] += time.process_time() - t0
            if x is not None:
                if plan is not None:  # churn: the week-t action of the new plan vs the old plan's slice for week t
                    pm_, px, ps = plan
                    nc = pm_.meta["nc"]
                    j = t - ps
                    if j * nc < len(px):
                        cols = L.action_columns(inst, m)
                        old, new = px[j * nc + cols], x[cols]
                        st["churn"].append(float(np.abs(new - old).sum() / max(1e-9, np.abs(old).sum() + np.abs(new).sum())))
                plan = (m, x, t)
                st["claims"].append((t, claim))
            else:
                st["fails"] += 1
        m, x, ps = plan
        nc = m.meta["nc"]
        xt = np.zeros_like(m.lb)
        j = t - ps
        xt[:nc] = x[j * nc : (j + 1) * nc]
        obs, _r, done, _tr, _inf = env.step(L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t - 1])))
        t += 1
    return int(env.trajectory.J_cents), st, PP.summarize(env.trajectory)


def episode(n, entropy, schedules):
    PP.configure(entropy)
    path = PP.cache_path(entropy, n, "replan")
    res = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            res = pickle.load(f)
    w = H.world(n)
    inst, omega, marks, fb = w
    if "true_lp" not in res:
        m = __import__("shockbench_flow.oracle.lp", fromlist=["build_lp"]).build_lp(inst, marks, planning_rules=True)
        x, claim = plain_lp(inst, m)
        J, S = PP.play_on(n, w, m, x)
        res["true_lp"] = dict(J=J, claim=claim, S=S)
    if "hybrid" not in res:
        J, S = PP.hybrid(n, w)
        res["hybrid"] = dict(J=J, S=S)
    for name in schedules:
        if name in res:
            continue
        t0 = time.time()
        J, st, S = closed(n, w, SCHEDULES[name])
        res[name] = dict(J=J, st=st, S=S, secs=time.time() - t0)
        logline(f"ep {n} {name:14s} J {J / BN:8.1f}  (true LP blind {res['true_lp']['J'] / BN:8.1f}, hybrid {res['hybrid']['J'] / BN:8.1f})  "
                f"solves {st['solves']} fails {st['fails']} cpu {st['secs']:.0f}s churn {np.mean(st['churn']) if st['churn'] else 0:.3f}  {time.time() - t0:.0f}s")
        with open(path, "wb") as f:
            pickle.dump(res, f)
    with open(path, "wb") as f:
        pickle.dump(res, f)
    return n, res


def run(episodes=8, start=0, entropy=444, n_jobs=2, schedules=",".join(SCHEDULES)):
    PP.configure(entropy)
    names = [s for s in (schedules.split(",") if isinstance(schedules, str) else schedules) if s]
    # schedule names contain commas: rebuild from the known keys
    names = [k for k in SCHEDULES if k in (schedules if isinstance(schedules, str) else ",".join(schedules))]
    logline(f"\n=== replan_curve {time.strftime('%Y-%m-%d %H:%M:%S')}: small root {entropy} eps {start}..{start + episodes - 1} schedules {names}")
    out = Parallel(n_jobs=n_jobs)(delayed(episode)(n, entropy, names) for n in range(start, start + episodes))
    table(out, entropy, names)


def table(out, entropy, names):
    from package_baselines import rss

    res = {n: r for n, r in out}
    ns = sorted(res)
    refs = PP.references(entropy, max(ns) + 1)
    rr = [refs[i] for i in ns]
    cols = ["hybrid", "true_lp"] + names
    logline("\n| ep | " + " | ".join(cols) + " |")
    logline("| --- " * (len(cols) + 1) + "|")
    for n in ns:
        logline(f"| {n} | " + " | ".join(f"{res[n][c]['J'] / BN:.1f}" for c in cols) + " |")
    J = {c: np.array([res[n][c]["J"] for n in ns], float) for c in cols}
    logline("| mean | " + " | ".join(f"{J[c].mean() / BN:.1f}" for c in cols) + " |")
    logline("| RSS | " + " | ".join(f"{rss(rr, [int(v) for v in J[c]])[0]:.4f}" for c in cols) + " |")
    se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
    for c in names:
        d1, d2 = (J[c] - J["true_lp"]) / BN, (J[c] - J["hybrid"]) / BN
        st = [res[n][c]["st"] for n in ns]
        logline(f"  {c:14s} - true LP blind {d1.mean():+7.1f} +- {se(d1):5.1f} (median {np.median(d1):+7.1f});  - hybrid {d2.mean():+7.1f} +- {se(d2):5.1f};  "
                f"solves/ep {np.mean([s['solves'] for s in st]):.0f}, cpu/solve {np.sum([s['secs'] for s in st]) / max(1, np.sum([s['solves'] for s in st])):.2f}s, "
                f"churn {np.mean([np.mean(s['churn']) for s in st if s['churn']]) if any(s['churn'] for s in st) else 0:.3f}")


def show(episodes=8, start=0, entropy=444):
    PP.configure(entropy)
    out = []
    for n in range(start, start + episodes):
        with open(PP.cache_path(entropy, n, "replan"), "rb") as f:
            out.append((n, pickle.load(f)))
    names = [k for k in SCHEDULES if all(k in r for _, r in out)]
    table(out, entropy, names)


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
