"""Two runs against each other and against the plain rules, on the very same scenarios.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/compare_runs.py \
        --runs=outputs/rl_lab/q1_slot_control,outputs/rl_lab/q2_graph_control --task=small --entropy=666

Prints each run's paired gain over the rules and, for the first two runs, the paired difference between them with
its 90 % interval. The numbers are cost shares, not RSS: for a decision use ``hub/eval/compare.py``.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import rollout  # noqa: E402


def _jobs(run, tag, task, entropy, episodes):
    s = json.loads((run / "settings.json").read_text())
    return [
        {
            "task": task,
            "pool_index": i,
            "entropy": entropy,
            "n_scenarios": max(episodes, 1),
            "version": tag,
            "policy_path": str(run / "policy.pt"),
            "kind": s["kind"],
            "sizes": s["sizes"],
            "width": s["width"],
            "rounds": s["rounds"],
            "groups": s["groups"],
            "lo": s["lo"],
            "hi": s["hi"],
            "sample": False,
            "record": False,
            "seed": 0,
            "families": s.get("families"),
        }
        for i in range(episodes)
    ]


def main(runs, task="small", entropy=666, episodes=32, workers=2, rules=None):
    paths = [
        Path(r) if Path(r).is_absolute() else ROOT / r
        for r in (runs if isinstance(runs, (list, tuple)) else str(runs).split(","))
    ]
    first = json.loads((paths[0] / "settings.json").read_text())
    rules = rules or first["rules"]
    if not Path(rules).is_absolute():
        rules = str(ROOT / rules)

    costs = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=rollout.setup, initargs=(rules,)) as pool:
        for p in paths:
            played = list(pool.map(rollout.play, _jobs(p, p.name, task, entropy, episodes)))
            costs[p.name] = np.array([e["costs"].sum() for e in played])
            if any(e["fallbacks"] for e in played):
                print(f"  note: {p.name} fell back to the rules in {sum(e['fallbacks'] for e in played)} week(s)")
        base = list(
            pool.map(
                rollout.play, [dict(j, rules_only=True) for j in _jobs(paths[0], "rules", task, entropy, episodes)]
            )
        )
    rules_cost = np.array([e["costs"].sum() for e in base])

    print(f"{task}, root {entropy}, {episodes} episode(s), mean correction, paired with the plain rules")
    for name, c in costs.items():
        rel = (rules_cost - c) / rules_cost
        se = rel.std(ddof=1) / np.sqrt(len(rel))
        print(f"  {name:28s} gain {rel.mean():+.5f} ± {se:.5f}   better in {int((rel > 0).sum())}/{len(rel)}")
    if len(paths) >= 2:
        a, b = paths[0].name, paths[1].name
        d = (costs[b] - costs[a]) / rules_cost  # positive: the second costs more
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n  paired difference, {b} minus {a} (share of the rules' cost):")
        print(
            f"    {d.mean():+.5f} ± {se:.5f}, 90 % interval {d.mean() - 1.645 * se:+.5f} to "
            f"{d.mean() + 1.645 * se:+.5f}"
        )
        print(f"    {b} cheaper in {int((d < 0).sum())} of {len(d)} episodes")
        print("    an interval that holds 0 means these episodes cannot tell the two networks apart")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
