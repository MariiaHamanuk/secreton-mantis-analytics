"""The mean correction against the plain rules on the same scenarios, on any root.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/paired_eval.py \
        --run=outputs/rl_lab/<run> --task=small --entropy=555 --episodes=32

Point it at the training root (555) and at the held-out root (666) with the same weights: a policy that is better
on the first and worse on the second has fitted its scenarios, which is a different failure from one that is worse
on both. The number here is a cost ratio, not RSS — for the decision use ``hub/eval/compare.py``.
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


def main(run, task="small", entropy=666, episodes=32, policy="policy.pt", workers=4, sample=False):
    """``sample=True`` plays the noisy policy PPO actually optimised, not the mean the submission would play.

    The two differ: PPO maximises the average return over the spread, and ``tanh`` is not linear, so the mean
    action is not the average action. A policy whose samples match the rules while its mean loses is a different
    story from one that loses either way, and the rules of the contest force the mean.
    """
    run = Path(run) if Path(run).is_absolute() else ROOT / run
    s = json.loads((run / "settings.json").read_text())
    rules = s["rules"] if Path(s["rules"]).is_absolute() else str(ROOT / s["rules"])

    def job(path, version, sample=False):
        return [
            {
                "task": task,
                "pool_index": i,
                "entropy": entropy,
                "n_scenarios": max(episodes, 1),
                "version": version,
                "policy_path": str(path),
                "kind": s["kind"],
                "sizes": s["sizes"],
                "width": s["width"],
                "rounds": s["rounds"],
                "groups": s["groups"],
                "lo": s["lo"],
                "hi": s["hi"],
                "sample": sample,
                "record": False,
                "seed": 1000 + i,
                "families": s.get("families"),
            }
            for i in range(episodes)
        ]

    with ProcessPoolExecutor(max_workers=workers, initializer=rollout.setup, initargs=(rules,)) as pool:
        mine = list(pool.map(rollout.play, job(run / policy, "mine", sample)))
        base = list(pool.map(rollout.play, [dict(j, rules_only=True) for j in job(run / policy, "rules")]))

    a = np.array([e["costs"].sum() for e in mine])
    b = np.array([e["costs"].sum() for e in base])
    rel = (b - a) / b
    how = "sampled" if sample else "mean"
    print(f"{run.name} on {task}, root {entropy}, {episodes} episode(s), weights {policy}, {how} correction")
    print(f"  cost: network {a.mean():.4f}, rules {b.mean():.4f} (naive weeks per episode)")
    print(f"  paired gain {rel.mean():+.5f} ± {rel.std(ddof=1) / np.sqrt(len(rel)):.5f} (one standard error)")
    print(
        f"  better in {int((rel > 0).sum())} of {len(rel)} episodes; "
        f"weeks played by the rules instead of the network: {sum(e['fallbacks'] for e in mine)}"
    )
    return float(rel.mean())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
