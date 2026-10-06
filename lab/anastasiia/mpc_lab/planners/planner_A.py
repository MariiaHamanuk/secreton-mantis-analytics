"""Planner A: window LP made honest about base-first electricity by iterated re-solving (caps on fab energy).

    uv run python lab/anastasiia/mpc_lab/planners/planner_A.py --task=small --episodes=16 --horizon=52 --variant=E1
Variants: E<n> = n extra solves capping fab energy per grid (weeks 2+); suffix _H<k> caps only weeks 2..k+1;
"L<n>" same but caps are applied as bounds on lot starts p (cap/e_f scaled), mode 'G' uses G_bar instead of plan G.
"""

import sys
import time

import fire
import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, vstack

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
from package_baselines import rss  # noqa: E402

CPU: list[float] = []


def parse(variant):
    # e.g. "E1", "E3_H8", "E2_Gbar"
    parts = variant.split("_")
    n = int(parts[0][1:]) if len(parts[0]) > 1 else 0
    H = None
    mode = "plan"
    for p in parts[1:]:
        if p.startswith("H"):
            H = int(p[1:])
        elif p == "Gbar":
            mode = "gbar"
    return n, H, mode


def solve(model, extra=None):
    from scipy.optimize import linprog

    A_ub, b_ub = model.A_ub, model.b_ub
    if extra is not None:
        A_ub = vstack([A_ub, extra[0]], format="csr")
        b_ub = np.concatenate([b_ub, extra[1]])
    rows = {}
    if A_ub.shape[0]:
        rows.update(A_ub=A_ub, b_ub=b_ub)
    if model.A_eq.shape[0]:
        rows.update(A_eq=model.A_eq, b_eq=model.b_eq)
    return linprog(model.objective(), bounds=np.column_stack([model.lb, model.ub]), method="highs-ds", **rows)


def cap_rows(inst, model, x, H, mode):
    """Rows sum_f E(t,f) <= max(0, G_t - ybar_t) for weeks 2..(H+1) of the window, per grid."""
    idx = model.index
    T = model.T
    last = T if H is None else min(T, H + 1)
    ri, ci, vals, rhs = [], [], [], []
    r = 0
    for go in range(len(inst.grids)):
        fabs = list(inst.grid_fabs[go])
        ga = inst.nodes[inst.grids[go]].grid
        fuels = tuple(ga.fuels) + ((None,) if None in ga.shares else ())
        for t in range(2, last + 1):
            ecols = [idx[("E", t, fo)] for fo in fabs if ("E", t, fo) in idx]
            if not ecols:
                continue
            ybar = x[idx[("y", t, go)]] + x[idx[("ysh", t, go)]]
            if mode == "gbar":
                g = sum(model.ub[idx[("G", t, go, k)]] for k in fuels)
            else:
                g = sum(x[idx[("G", t, go, k)]] for k in fuels)
            cap = max(0.0, g - ybar)
            for c in ecols:
                ri.append(r)
                ci.append(c)
                vals.append(1.0)
            rhs.append(cap)
            r += 1
    if r == 0:
        return None
    A = csr_matrix((vals, (ri, ci)), shape=(r, len(model.lb)))
    return A, np.array(rhs)


def episode(variant: str, known: tuple, task: str, entropy: int, n: int, horizon: int) -> tuple[int, list[float]]:
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    quantiles = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    context = PolicyContext(fq_quantile=quantiles, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {week: instant for instant, week in L.NOW_FIELDS}
    n_extra, capH, mode = parse(variant)
    times: list[float] = []

    class Foresight(MpcDet):
        def _horizon(self, inst, plan):
            return horizon

        def _window_arrays(self, inst, obs, weeks):
            arrays = {name: a.copy() for name, a in super()._window_arrays(inst, obs, weeks).items()}
            first = int(obs["week"]) - 1
            for name in known:
                arrays[name] = np.array(getattr(marks, name)[first : first + weeks])
                if name in now:
                    arrays[now[name]] = np.array(getattr(marks, now[name])[first : first + weeks])
            return L.read_only(arrays)

        def act(self, obs):
            inst, week = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            model = L.rolled_lp(inst, obs, self._window_arrays(inst, obs, weeks), weeks, planning_rules=True)
            t0 = time.process_time()
            plan = solve(model)
            if plan.status != 0 or plan.x is None:
                return self._fallback.act(obs)
            if weeks > 1:
                for _ in range(n_extra):
                    extra = cap_rows(inst, model, np.asarray(plan.x), capH, mode)
                    if extra is None:
                        break
                    p2 = solve(model, extra)
                    if p2.status != 0 or p2.x is None:
                        break
                    plan = p2
            times.append(time.process_time() - t0)
            return L.week1_action(inst, model, np.asarray(plan.x), obs, L.prohibited_now(self._memory, week))

    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    J = int(rollout(inst, Foresight(None, context), omega, "standard", seed, marks=marks, fallback=fallback).J_cents)
    return J, times


def main(task="small", episodes=16, entropy=111, horizon=52, variants="E0", known="everything", n_jobs=2):
    from sbf_starter import scoring
    import mpc_foresight as mf

    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    print(f"{task}, {episodes} episodes of root {entropy}, horizon {horizon}, known={known}")
    for v in [variants] if isinstance(variants, str) else list(variants):
        start = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(
            delayed(episode)(v, mf.GROUPS[known], task, entropy, n, horizon) for n in range(episodes)
        )
        score, by_level = rss(refs, [o[0] for o in out])
        ts = np.concatenate([o[1] for o in out])
        levels = " ".join(f"{x:.3f}" for x in by_level.values())
        print(
            f"{v:12s} {score:7.4f}  {levels}  cpu/week mean {ts.mean():.2f} max {ts.max():.2f}"
            f"  ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
