"""An agent folder played under the scorer's CPU meter, beside the same agent's record without a meter.

    uv run python lab/anastasiia/regime_lab/budget.py run outputs/regime_lab/agents/hull_a --tag=hull_a --base=L_hre1 \
        --task=full --entropy=111 --episodes=32 --budget=4.0 --n_jobs=6
    uv run python lab/anastasiia/regime_lab/budget.py show hull_a --base=L_hre1 --task=full --budget=4.0

``run`` plays the folder as ``sbf evaluate --cpu_budget`` does (``shockbench_flow_agent.scoring``: a week whose CPU
time is over ``budget`` seconds is played by the naive rule, and what it took over the budget is charged to the next
week), in ``n_jobs`` processes, and keeps every episode's cost and its weeks handed to naive in
``outputs/regime_lab/budget/<tag>_<task>_<entropy>_<budget>.json``. ``show`` prints the score with the meter, the
score of ``base`` (a ``play.py`` record of the same agent without a meter) on the same episodes, their paired
difference with a 90% interval (episodes drawn again inside the harm levels, as ``versions.py``) and what one week
handed to naive cost. The meter counts this machine's CPU seconds under whatever else runs on it: a budget other
than the task's stands for a server that much faster or slower than this machine at that moment.
"""

import json
import pickle
import sys
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import plan  # noqa: E402

OUT = plan.OUT / "budget"


def _path(tag: str, task: str, entropy: int, budget: float) -> Path:
    return OUT / f"{tag}_{task}_{entropy}_{budget:g}.json"


def run(agent: str, tag: str, base: str = "", task: str = "full", entropy: int = 111, episodes: int = 32,
        budget: float = 4.0, n_jobs: int = 4) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    rows = es.play(str(resolve(agent).resolve()), cpu_budget=float(budget), n_jobs=n_jobs)
    OUT.mkdir(parents=True, exist_ok=True)
    _path(tag, task, entropy, budget).write_text(json.dumps(rows))
    show(tag, base=base, task=task, entropy=entropy, budget=budget)


def show(tag: str, base: str = "", task: str = "full", entropy: int = 111, budget: float = 4.0, draws: int = 2000) -> None:
    rows = json.loads(_path(tag, task, entropy, budget).read_text())
    ns = [r["episode"] for r in rows]
    refs_all = plan.references(task, entropy, max(ns) + 1)
    weeks = sum(r["weeks"] for r in rows)
    handed = sum(r["cpu_weeks"] for r in rows)
    per = sorted((r["cpu_weeks"] for r in rows), reverse=True)
    refs = [refs_all[n] for n in ns]
    costs = [r["J_policy_cents"] for r in rows]
    print(f"{tag}: {task}, root {entropy}, {len(ns)} episodes, a budget of {budget:g} CPU s a week")
    print(f"  weeks handed to naive: {handed} of {weeks} ({100 * handed / weeks:.2f} %); in {sum(1 for p in per if p)} episodes; most in one episode: {per[:8]}")
    other = sum(r["fallback_weeks"] - r["cpu_weeks"] for r in rows)
    if other:
        print(f"  weeks naive played for another cause: {other}; the first: {next(r['first_error'] for r in rows if r['first_error'])}")
    print(f"  score with the meter: {plan.score(refs, costs):.4f}")
    if not base:
        return
    kept = pickle.loads((plan.OUT / "play" / f"{base}_{task}_{entropy}.pkl").read_bytes())
    free = [kept[n]["J"] for n in ns]
    level = np.array([r["stratum"] for r in refs])
    groups = [np.flatnonzero(level == s) for s in sorted(set(level))]
    rng = np.random.default_rng(0)
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]
    diff = plan.score(refs, costs) - plan.score(refs, free)
    drawn = [plan.score([refs[i] for i in p], [costs[i] for i in p]) - plan.score([refs[i] for i in p], [free[i] for i in p]) for p in picks]
    lo, hi = np.percentile(drawn, [5, 95])
    same = sum(1 for a, b in zip(costs, free) if a == b)
    print(f"  score without a meter ({base}): {plan.score(refs, free):.4f}; with minus without: {diff:+.4f} ({lo:+.4f} to {hi:+.4f}); the same cost to the cent in {same} of {len(ns)} episodes")
    if handed:
        usd = (sum(costs) - sum(free)) / 100 / handed
        print(f"  a week handed to naive cost {usd / 1e9:+.2f} bn USD on average, {diff / (100 * handed / weeks):+.4f} of the score per 1 % of the weeks")


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
