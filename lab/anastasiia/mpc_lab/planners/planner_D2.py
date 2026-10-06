"""Closed-loop MPC with base-load-first binaries (worker D2): mpc_foresight's Foresight, plus, in weeks 2..Hb of each window,
a binary s per base_first grid and week (shed base load only if s = 1, and then the grid's fabs draw nothing and it runs at
full load), solved with scipy.optimize.milp under a time limit; the plain LP solution is the fallback.

    uv run python lab/anastasiia/mpc_lab/planners/planner_D2.py --episodes=16 --hb=16 --tl=3 --n_jobs=3 [--known=everything|nothing]

Grid-weeks where the answer is obvious get no binary (--fix=1): generation capacity (sum of zeta Gbar) well above
base load + the fabs' largest draw -> no shedding is ever needed (nothing added); capacity below base load -> shed is
certain and the fabs get nothing (their energy is fixed to 0). Fuel stock limits are not looked at.
"""

import sys
import time

import fire
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
import mpc_foresight as mf  # noqa: E402
from package_baselines import rss  # noqa: E402


def episode(known_name, task, entropy, n, horizon, hb, tl, gap, fix):
    from scipy.optimize import LinearConstraint, Bounds, linprog, milp
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    known = mf.GROUPS[known_name]
    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    q = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    context = PolicyContext(fq_quantile=q, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {w: i for i, w in L.NOW_FIELDS}
    stats = {"cpu": [], "milp_ok": 0, "fallback": 0, "bin": []}

    class Planner(MpcDet):
        def _horizon(self, inst, plan):
            return horizon

        def extra(self, inst, model, weeks):
            """(Aext, lo, hi, integrality, lb_ext, ub, ub_changes) of base-load-first for weeks 2..Hb."""
            nc, neq, nub = model.meta["nc"], len(model.eq_rows), len(model.ub_rows)
            tm = model.meta["template"]
            eqi = {r: i for i, r in enumerate(model.eq_rows)}
            ubi = {r: i for i, r in enumerate(model.ub_rows)}
            n0 = len(model.lb)
            ub = model.ub.copy()
            r_, c_, v_, lo, hi, integ = [], [], [], [], [], []
            ncol = n0
            nrow = 0

            def row(coefs, l, h):
                nonlocal nrow
                for j, a in coefs:
                    r_.append(nrow), c_.append(j), v_.append(a)
                lo.append(l), hi.append(h)
                nrow += 1

            col = lambda tag, t, *r: (t - 1) * nc + tm[(tag, *r)]
            for t in range(2, min(hb, weeks) + 1):
                for go, g in enumerate(inst.grids):
                    ga = inst.nodes[g].grid
                    fabs = inst.grid_fabs[go]
                    if ga.priority != "base_first" or not fabs:
                        continue
                    if not any(inst.nodes[inst.fabs[fo]].fab.e > 0 for fo in fabs):
                        continue
                    Gj = [col("G", t, go, k) for k in (ga.fuels + ((None,) if None in ga.shares else ()))]
                    Gub = float(sum(model.ub[j] for j in Gj))
                    ybar = float(model.b_eq[(t - 1) * neq + eqi[("baseload", go)]])
                    Em = sum(inst.nodes[inst.fabs[fo]].fab.e * model.ub[col("p", t, fo)] for fo in fabs)
                    if fix and Gub >= 1.3 * (ybar + Em):
                        continue  # capacity far above need: no shedding in any optimum
                    Ej = [col("E", t, fo) for fo in fabs]
                    for fo in fabs:  # E = e p / R exactly
                        fa = inst.nodes[inst.fabs[fo]].fab
                        pj, Ej1 = col("p", t, fo), col("E", t, fo)
                        R = -model.A_ub[(t - 1) * nub + ubi[("fab_energy", fo)], Ej1]
                        if R > 0:
                            row([(pj, fa.e), (Ej1, -R)], 0.0, 0.0)
                        else:
                            ub[Ej1] = 0.0
                    if fix and Gub <= 0.7 * ybar:
                        for j in Ej:
                            ub[j] = 0.0  # shedding certain: the fabs get nothing
                        continue
                    s = ncol
                    ncol += 1
                    integ.append(1)
                    row([(col("ysh", t, go), 1.0), (s, -ybar)], -np.inf, 0.0)
                    emax = sum(inst.nodes[inst.fabs[fo]].fab.e * model.ub[col("p", t, fo)] for fo in fabs)
                    row([(j, 1.0) for j in Ej] + [(s, emax)], -np.inf, emax)
                    row([(col("lam", t, go), 1.0), (s, -1.0)], 0.0, np.inf)
            return r_, c_, v_, lo, hi, integ, ncol, nrow, ub

        def act(self, obs):
            t0 = time.process_time()
            inst, week = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            arrays = {k: a.copy() for k, a in L.persistence_arrays(inst, obs, self._memory, weeks).items()}
            f = week - 1
            for name in known:
                arrays[name] = np.array(getattr(marks, name)[f : f + weeks])
                if name in now:
                    arrays[now[name]] = np.array(getattr(marks, now[name])[f : f + weeks])
            model = L.rolled_lp(inst, obs, L.read_only(arrays), weeks, planning_rules=True)
            x = None
            r_, c_, v_, lo, hi, integ, ncol, nrow, ub = self.extra(inst, model, weeks)
            n0 = len(model.lb)
            stats["bin"].append(len(integ))
            if integ or nrow:
                pad = lambda A: sparse.hstack([A, sparse.csr_matrix((A.shape[0], ncol - n0))]).tocsr()
                cons = [LinearConstraint(pad(model.A_ub), -np.inf, model.b_ub)]
                if model.A_eq.shape[0]:
                    cons.append(LinearConstraint(pad(model.A_eq), model.b_eq, model.b_eq))
                if nrow:
                    cons.append(LinearConstraint(sparse.coo_matrix((v_, (r_, c_)), shape=(nrow, ncol)).tocsr(), lo, hi))
                c = np.concatenate([model.objective(), np.zeros(ncol - n0)])
                bnd = Bounds(np.concatenate([model.lb, np.zeros(ncol - n0)]), np.concatenate([ub, np.ones(ncol - n0)]))
                res = milp(c, constraints=cons, integrality=np.concatenate([np.zeros(n0), integ]), bounds=bnd,
                           options=dict(time_limit=tl, mip_rel_gap=gap))
                if res.x is not None:
                    x = np.asarray(res.x)[:n0]
                    stats["milp_ok"] += 1
            if x is None:
                stats["fallback"] += 1
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
            a = L.week1_action(inst, model, x, obs, L.prohibited_now(self._memory, week))
            stats["cpu"].append(time.process_time() - t0)
            return a

    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    J = int(rollout(inst, Planner(None, context), omega, "standard", seed, marks=marks, fallback=fallback).J_cents)
    cpu = np.array(stats["cpu"])
    return J, float(cpu.mean()), float(cpu.max()), stats["milp_ok"], stats["fallback"], float(np.mean(stats["bin"]))


def main(task="small", episodes=16, entropy=111, horizon=52, hb=16, tl=3.0, gap=1e-3, fix=1, known="everything",
         n_jobs=3, start=0):
    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, start + episodes, entropy=entropy, verbose=False).references)[start:]
    name = "nothing (the MPC as it is)" if known == "nothing" else known
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(
        delayed(episode)(name, task, entropy, n, horizon, hb, tl, gap, fix) for n in range(start, start + episodes)
    )
    costs = [o[0] for o in out]
    sc, lv = rss(refs, costs)
    print(
        f"known={known} eps={episodes} hb={hb} tl={tl} gap={gap} fix={fix}: score {sc:.4f} "
        f"levels {[round(v, 3) for v in lv.values()]}  cpu/week mean {np.mean([o[1] for o in out]):.2f} "
        f"max {np.max([o[2] for o in out]):.2f}  milp ok {sum(o[3] for o in out)} fallback {sum(o[4] for o in out)} "
        f"mean binaries {np.mean([o[5] for o in out]):.0f}  ({time.time()-t0:.0f}s)  costs {costs}",
        flush=True,
    )


if __name__ == "__main__":
    fire.Fire(main)
