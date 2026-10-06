"""Planner C: the window LP of the MPC, steered by cheap changes of its objective and bounds.

    uv run python lab/anastasiia/mpc_lab/planners/planner_C.py --variants="base,prio:f=0.5" --episodes=16 --horizon=52 --known=everything

A variant is ``name:key=value,key=value``. Variants (all act on weeks 2.. of the window, the first week keeps the
package's planning rules):
  base                      the unchanged planner
  prio:f=F                  planning-rule prices (priority) of weeks >= 2 times F
  shed:f=F                  price on shed base load of weeks >= 2: F times the package's week-1 price
  fuel:pen=P,mult=M,tmax=W  soft lower bound I_fuel(t) >= M psi ibar for t < W, shortfall priced P x fuel value
  viol:rounds=R,mode=M     iterate: where week >= 2 of the plan sheds base load and powers a fab of the grid, bound the shed to 0
                            (mode 0, E to 0 if infeasible) or the fabs energy to 0 (mode 1)
  cap:L=L                   two passes: the first week's grid ratio rho_g scales the lots bound of weeks >= 2 at the grid's
                            fabs, recovering as 1 - (1 - rho) exp(-(t-1)/L)
"""

import dataclasses
import time

import fire
import numpy as np
from joblib import Parallel, delayed
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))  # the bench is one folder up
from package_baselines import rss
from scipy import sparse

import mpc_foresight as MF


def parse(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    cfg = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        cfg[k] = float(v)
    return name, cfg


def cols_of(model, tag):
    """{key[2:] or ...: [(week, column index)]} for columns with ``tag`` (keys are (tag, week, ...))."""
    out = {}
    nc = model.meta["nc"]
    for j, key in enumerate(model.columns):
        if key[0] == tag:
            for t in range(1, model.T + 1):
                out.setdefault(key[1:], {})[t] = (t - 1) * nc + j
    return out


def solve(model, c, lb, ub, A_ub, b_ub):
    from scipy.optimize import linprog

    rows = {}
    if A_ub.shape[0]:
        rows.update(A_ub=A_ub, b_ub=b_ub)
    if model.A_eq.shape[0]:
        n = c.shape[0] - model.A_eq.shape[1]
        A_eq = model.A_eq if n == 0 else sparse.hstack([model.A_eq, sparse.csr_matrix((model.A_eq.shape[0], n))]).tocsr()
        rows.update(A_eq=A_eq, b_eq=model.b_eq)
    return linprog(c, bounds=np.column_stack([lb, ub]), method="highs-ds", **rows)


def plan(model, inst, name, cfg):
    """Return the solution x (length of the unmodified model) of the steered LP, or None."""
    nc, T = model.meta["nc"], model.T
    N = model.cost.shape[0]
    c = model.objective().copy()
    lb, ub = model.lb.copy(), model.ub.copy()
    A_ub, b_ub = model.A_ub, model.b_ub
    later = np.arange(N) >= nc  # weeks >= 2
    if name == "prio":
        pr = model.meta["priority"]
        c = c - pr + np.where(later, pr * cfg["f"], pr)
    elif name == "shed":
        pr = model.meta["priority"]
        ysh = cols_of(model, "ysh")
        for go, d in ysh.items():
            p1 = pr[d[1]]
            for t in range(2, T + 1):
                c[d[t]] += cfg["f"] * p1
    elif name == "fuel":
        extra_c, rows_i, rows_j, rows_v, rhs = [], [], [], [], []
        slots = model.meta["slots"]
        I = cols_of(model, "I")
        n0 = N
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            k = ga.rationed
            s = slots.index((g, k))
            thr = cfg["mult"] * inst.params.psi * ga.ibar[k]
            v = inst.commodities[k].v
            for t in range(1, min(T, int(cfg["tmax"])) + 1):
                col = I[(s,)][t]
                r = len(rhs)
                rows_i += [r, r]
                rows_j += [col, n0 + len(extra_c)]
                rows_v += [-1.0, -1.0]
                rhs.append(-thr)
                extra_c.append(cfg["pen"] * v)
        m = len(extra_c)
        add = sparse.csr_matrix((rows_v, (rows_i, rows_j)), shape=(m, N + m))
        A_ub = sparse.vstack([sparse.hstack([A_ub, sparse.csr_matrix((A_ub.shape[0], m))]), add]).tocsr()
        b_ub = np.concatenate([b_ub, rhs])
        c = np.concatenate([c, extra_c])
        lb = np.concatenate([lb, np.zeros(m)])
        ub = np.concatenate([ub, np.full(m, np.inf)])
    elif name == "viol":
        # base load is served first: where the plan sheds base load and powers a fab in the same grid-week, forbid the shed
        ysh, E = cols_of(model, "ysh"), cols_of(model, "E")
        eps = cfg.get("eps", 1e-6)
        mode = int(cfg.get("mode", 0))  # 0: ysh -> 0 (else E -> 0 if infeasible); 1: E -> 0
        tmax = int(cfg.get("tmax", 10_000))
        for _ in range(int(cfg.get("rounds", 3))):
            res = solve(model, c, lb, ub, A_ub, b_ub)
            if res.status != 0 or res.x is None:
                return None
            x = np.asarray(res.x)
            bad = []
            for go in range(len(inst.grids)):
                for t in range(2, min(model.T, tmax) + 1):
                    if x[ysh[(go,)][t]] > eps and sum(x[E[(fo,)][t]] for fo in inst.grid_fabs[go]) > eps:
                        bad.append((go, t))
            if not bad:
                return x[:N]
            lb2, ub2 = lb.copy(), ub.copy()
            for go, t in bad:
                if mode == 0:
                    ub2[ysh[(go,)][t]] = 0.0
                else:
                    for fo in inst.grid_fabs[go]:
                        ub2[E[(fo,)][t]] = 0.0
            res = solve(model, c, lb2, ub2, A_ub, b_ub)
            if res.status != 0 and mode == 0:
                for go, t in bad:
                    ub2[ysh[(go,)][t]] = ub[ysh[(go,)][t]]
                    for fo in inst.grid_fabs[go]:
                        ub2[E[(fo,)][t]] = 0.0
                res = solve(model, c, lb2, ub2, A_ub, b_ub)
            if res.status != 0:
                return None
            ub = ub2
    elif name == "cap":
        first = solve(model, c, lb, ub, A_ub, b_ub)
        if first.status != 0:
            return None
        rho = cols_of(model, "rho")
        p = cols_of(model, "p")
        for fo, f in enumerate(inst.fabs):
            fa = inst.nodes[f].fab
            if fa.grid is None or fa.e <= 0:
                continue
            go = inst.grid_ordinal[fa.grid]
            r1 = float(first.x[rho[(go,)][1]])
            for t in range(2, T + 1):
                ub[p[(fo,)][t]] *= 1.0 - (1.0 - r1) * np.exp(-(t - 1) / cfg["L"])
    res = solve(model, c, lb, ub, A_ub, b_ub)
    if res.status != 0 or res.x is None:
        return None
    return np.asarray(res.x)[:N]


def episode(known, task, entropy, n, horizon, spec):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    name, cfg = parse(spec)
    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    quantiles = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    context = PolicyContext(fq_quantile=quantiles, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {week: instant for instant, week in L.NOW_FIELDS}
    times = []

    class Steered(MpcDet):
        def _horizon(self, inst, plan):
            return horizon

        def _window_arrays(self, inst, obs, weeks):
            arrays = {k: a.copy() for k, a in super()._window_arrays(inst, obs, weeks).items()}
            first = int(obs["week"]) - 1
            for fld in known:
                arrays[fld] = np.array(getattr(marks, fld)[first : first + weeks])
                if fld in now:
                    arrays[now[fld]] = np.array(getattr(marks, now[fld])[first : first + weeks])
            return L.read_only(arrays)

        def act(self, obs):
            inst, week = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            start = time.process_time()
            model = L.rolled_lp(inst, obs, self._window_arrays(inst, obs, weeks), weeks, planning_rules=True)
            x = plan(model, inst, name, cfg)
            times.append(time.process_time() - start)
            if x is None:
                return self._fallback.act(obs)
            return L.week1_action(inst, model, x, obs, L.prohibited_now(self._memory, week))

    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    J = int(rollout(inst, Steered(None, context), omega, "standard", seed, marks=marks, fallback=fallback).J_cents)
    return J, float(np.mean(times)), float(np.max(times))


def main(task="small", episodes=16, entropy=111, horizon=52, known="everything", variants="base", n_jobs=2):
    """Score each variant; known: a key of mpc_foresight.GROUPS."""
    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    kn = MF.GROUPS[known if known != "nothing" else "nothing (the MPC as it is)"]
    for spec in variants.split(";"):
        t0 = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(delayed(episode)(kn, task, entropy, n, horizon, spec) for n in range(episodes))
        score, by_level = rss(refs, [o[0] for o in out])
        lv = " ".join(f"{v:.3f}" for v in by_level.values())
        print(
            f"{spec:36s} known={known:10s} n={episodes} h={horizon} score={score:.4f}  {lv}  "
            f"cpu/wk mean={np.mean([o[1] for o in out]):.2f} max={np.max([o[2] for o in out]):.2f}  "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
