"""Score variants of an agent that reads a ``params.json``: one line per variant, each with its paired difference from
a baseline agent on the same episodes.

    uv run python lab/anastasiia/mpc_lab/variants.py --against=lab/anastasiia/agents/pull \\
        --task=small --episodes=64 --grid="[{'horizon': 24}, {'power_weeks': 0}]"

Each variant is a copy of the agent's folder with the grid entry written as its ``params.json``. Nothing is saved: put
the winner's numbers in the agent's own ``params.json`` and record it with ``hub/eval/formal_eval.py``.
"""

import json
import shutil
import tempfile
from pathlib import Path

import fire


def main(
    agent: str = "agents/anastasiia_mpc_baseload",
    grid: list | str = ({},),
    against: str = "template",
    task: str = "tiny",
    episodes: int = 16,
    entropy: int = 111,
    cpu_budget: bool = False,
    n_jobs: int = -1,
) -> None:
    """Print the score of each ``params.json`` of ``grid`` and its difference from ``against``.

    Args:
        agent: the agent whose folder is copied (a name of agents/ or a folder).
        grid: a list of dicts, each one variant's ``params.json``, in Python syntax (None, True, False).
        against: the agent every variant is compared with.
        task: tiny, small or full.
        episodes: episodes 0 .. episodes - 1 of the root.
        entropy: the scenarios' root (111, the team's tuning root).
        cpu_budget: a week over the task's CPU budget is played by the naive rule, as on the server.
        n_jobs: workers (-1: all cores).

    """
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    grid = list(grid)
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs)
    base = es.score(str(resolve(against)), n_jobs=n_jobs, cpu_budget=cpu_budget)
    print(f"{task}, {episodes} episodes of root {entropy}; {against}: {base.rss:.4f}")
    print(f"{'score':>7} {'vs ' + against:>12} {'90% interval':>20} {'by harm level':>32} {'naive weeks':>11}  params")
    source = resolve(agent)
    with tempfile.TemporaryDirectory(prefix="sbf-variants-") as tmp:
        for i, params in enumerate(grid):
            folder = Path(tmp) / f"v{i}"
            shutil.copytree(source, folder, ignore=shutil.ignore_patterns("__pycache__"))
            (folder / "params.json").write_text(json.dumps(params) + "\n")
            cmp = es.compare(str(folder), str(resolve(against)), n_jobs=n_jobs, cpu_budget=cpu_budget)
            score = cmp.a
            lo, hi = cmp.interval or (float("nan"), float("nan"))
            levels = " ".join("-" if v is None else f"{v:.3f}" for v in score.rss_by_stratum.values())
            print(
                f"{score.rss:7.4f} {cmp.diff:+12.4f} {f'{lo:+.4f} to {hi:+.4f}':>20} {levels:>32} "
                f"{score.fallback_weeks:>11}  {json.dumps(params)}"
            )


if __name__ == "__main__":
    fire.Fire(main)
