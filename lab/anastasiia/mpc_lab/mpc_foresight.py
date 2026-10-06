"""What perfect foresight of each part of the future would be worth to the MPC: a diagnostic, not an agent.

    uv run python lab/anastasiia/mpc_lab/mpc_foresight.py
    uv run python lab/anastasiia/mpc_lab/mpc_foresight.py --task=small --episodes=64

Each week the MPC plans on a forecast of the next weeks: the network as observed today, and the shown demand forecast.
Here a group of that forecast's fields is replaced by what really happens in the episode (read from the scenario, which
no agent sees), one group at a time. The score a group adds is the most a perfect predictor of it could add.

The MPC is shockbench-flow's ``mpc_det``, solved with SciPy as agents/mpc does, played in this process.
"""

import time

import fire
import numpy as np
from joblib import Parallel, delayed
from package_baselines import rss


GROUPS = {
    "nothing (the MPC as it is)": (),
    "demand": ("demand",),
    "prohibitions": ("prohibited",),
    "edge capacity": ("u",),
    "tariffs": ("tariff",),
    "straits": ("o", "kappa", "wr_class", "h_queue", "c_wr", "c"),
    "grids": ("G_bar", "y_bar"),
    "plants and supply": ("R", "alpha_bar", "R_osat", "sigma_scr", "supply"),
    "everything but demand": (
        *("prohibited", "u", "tariff", "o", "kappa", "wr_class", "h_queue", "c_wr", "c", "G_bar", "y_bar"),
        *("R", "alpha_bar", "R_osat", "sigma_scr", "supply"),
    ),
}
GROUPS["everything"] = (*GROUPS["everything but demand"], "demand")


def episode(known: tuple[str, ...], task: str, entropy: int, n: int, horizon: int) -> int:
    """J in cents of the MPC on episode n when the fields ``known`` of its forecast are the scenario's own."""
    from scipy.optimize import linprog
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
            rows = {}
            if model.A_ub.shape[0]:
                rows.update(A_ub=model.A_ub, b_ub=model.b_ub)
            if model.A_eq.shape[0]:
                rows.update(A_eq=model.A_eq, b_eq=model.b_eq)
            plan = linprog(model.objective(), bounds=np.column_stack([model.lb, model.ub]), method="highs-ds", **rows)
            if plan.status != 0 or plan.x is None:
                return self._fallback.act(obs)
            return L.week1_action(inst, model, np.asarray(plan.x), obs, L.prohibited_now(self._memory, week))

    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    return int(rollout(inst, Foresight(None, context), omega, "standard", seed, marks=marks, fallback=fallback).J_cents)


def main(
    task: str = "tiny",
    episodes: int = 16,
    entropy: int = 111,
    horizon: int = 24,
    only: str | tuple | None = None,
    n_jobs: int = -1,
) -> None:
    """Print the MPC's score with each group of ``GROUPS`` known in advance.

    Args:
        task: tiny, small or full.
        episodes: episodes 0 .. episodes - 1 of the root.
        entropy: the scenarios' root (111, the team's tuning root).
        horizon: weeks the MPC plans ahead.
        only: the groups to play, comma-separated (default: all of ``GROUPS``).
        n_jobs: workers (-1: all cores).

    """
    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    print(f"{task}, {episodes} episodes of root {entropy}, horizon {horizon}")
    print(f"{'known in advance':28s} {'score':>7} {'gain':>7}   by harm level")
    base = None
    names = list(GROUPS) if only is None else only.split(",") if isinstance(only, str) else list(only)
    for name in names:
        known = GROUPS[name]
        start = time.perf_counter()
        costs = Parallel(n_jobs=n_jobs)(delayed(episode)(known, task, entropy, n, horizon) for n in range(episodes))
        score, by_level = rss(refs, costs)
        base = score if base is None else base
        levels = " ".join(f"{v:.3f}" for v in by_level.values())
        print(
            f"{name:28s} {score:7.4f} {score - base:+7.4f}   {levels}   ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
