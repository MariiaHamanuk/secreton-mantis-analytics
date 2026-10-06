"""Score the baselines that ship inside shockbench-flow (greedy_lp, mpc_det, ...) on a root of ours: what a known
method reaches, and how many seconds a week it takes.

    uv run python lab/anastasiia/mpc_lab/package_baselines.py
    uv run python lab/anastasiia/mpc_lab/package_baselines.py --task=small --episodes=64 --names=greedy_lp,mpc_det

They are the organisers' reference policies, played in this process with the whole package at hand (HiGHS through
highspy, naive's plan): not submissions. A score here is a target, and the seconds are this machine's wall clock.
"""

import time

import fire
import numpy as np
from joblib import Parallel, delayed


WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}  # the board's weight of each harm level


def episode(name: str, task: str, entropy: int, n: int) -> tuple[int, float, float]:
    """(J in cents, the mean and the largest wall seconds of a week) of baseline ``name`` on episode n."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext, make_policy
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    quantiles = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    context = PolicyContext(fq_quantile=quantiles, generator=GeneratorRef(task, TASKS[task].gamma))
    policy = make_policy(name, context)
    seconds: list[float] = []
    act = policy.act

    def timed(obs):
        start = time.perf_counter()
        action = act(obs)
        seconds.append(time.perf_counter() - start)
        return action

    policy.act = timed
    traj = rollout(
        inst, policy, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback
    )
    return int(traj.J_cents), float(np.mean(seconds)), float(np.max(seconds))


def rss(rows: list[dict], costs: list[int]) -> tuple[float, dict[int, float]]:
    """The board's score of per-episode costs on the references ``rows``, and the score of each harm level."""
    saved, room = {}, {}
    for r, j in zip(rows, costs):
        if r["J_oracle_cents"] is None:
            continue
        saved.setdefault(r["stratum"], []).append(r["J_naive_cents"] - j)
        room.setdefault(r["stratum"], []).append(r["J_naive_cents"] - r["J_oracle_cents"])
    by_level = {s: float(np.sum(saved[s]) / np.sum(room[s])) for s in sorted(saved)}
    weights = {s: WEIGHTS[s] for s in saved} if set(saved) == set(WEIGHTS) else dict.fromkeys(saved, 1.0)
    total = sum(w * np.mean(saved[s]) for s, w in weights.items()) / sum(
        w * np.mean(room[s]) for s, w in weights.items()
    )
    return float(total), by_level


def main(
    names: str | tuple = ("naive", "greedy_lp", "mpc_det"),
    task: str = "tiny",
    episodes: int = 16,
    entropy: int = 111,
    n_jobs: int = -1,
) -> None:
    """Print each baseline's score on the cached references of the root.

    Args:
        names: baselines of ``shockbench_flow.policies.registry.NAMES``, comma-separated.
        task: tiny, small or full.
        episodes: episodes 0 .. episodes - 1 of the root.
        entropy: the scenarios' root (111, the team's tuning root).
        n_jobs: workers (-1: all cores); the seconds are honest only when the machine is not oversubscribed.

    """
    from sbf_starter import scoring

    names = names.split(",") if isinstance(names, str) else list(names)
    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    print(f"{task}, {episodes} episodes of root {entropy}")
    for name in names:
        start = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(delayed(episode)(name, task, entropy, n) for n in range(episodes))
        score, by_level = rss(refs, [o[0] for o in out])
        levels = ", ".join(f"{s}: {v:.3f}" for s, v in by_level.items())
        print(
            f"{name:22s} score {score:.4f}  by level {levels}  "
            f"seconds per week: mean {np.mean([o[1] for o in out]):.2f}, max {np.max([o[2] for o in out]):.2f}  "
            f"({time.perf_counter() - start:.0f} s)"
        )


if __name__ == "__main__":
    fire.Fire(main)
