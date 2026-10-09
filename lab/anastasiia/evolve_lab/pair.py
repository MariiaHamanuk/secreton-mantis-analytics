"""Two records of ``regime_lab/budget.py`` (plays under the scorer's meter) on the same episodes, paired.

    uv run python lab/anastasiia/evolve_lab/pair.py t_v1:2 t_v0:2 --task=small --entropy=111

Each name is ``<tag>:<budget>``. Prints both scores, the weeks the meter handed to the naive rule, and the first
minus the second with an interval (episodes drawn again inside the harm levels).
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "regime_lab"), str(HERE.parent / "mpc_lab")]
import plan  # noqa: E402


def main(new: str, old: str, task: str = "small", entropy: int = 111, draws: int = 2000, level: float = 0.90) -> None:
    def rows(spec):
        tag, budget = str(spec).rsplit(":", 1)
        return json.loads((plan.OUT / "budget" / f"{tag}_{task}_{entropy}_{float(budget):g}.json").read_text())

    a = {r["episode"]: r for r in rows(new)}
    b = {r["episode"]: r for r in rows(old)}
    ns = sorted(set(a) & set(b))
    refs_all = plan.references(task, entropy, max(ns) + 1)
    refs = [refs_all[n] for n in ns]
    ca = [a[n]["J_policy_cents"] for n in ns]
    cb = [b[n]["J_policy_cents"] for n in ns]
    stats = []
    for d in (a, b):
        weeks = sum(d[n]["weeks"] for n in ns)
        naive = sum(d[n]["fallback_weeks"] for n in ns)
        stats.append(f"{plan.score(refs, [d[n]['J_policy_cents'] for n in ns]):.4f} ({naive} of {weeks} weeks naive)")
    strata = np.array([r["stratum"] for r in refs])
    groups = [np.flatnonzero(strata == s) for s in sorted(set(strata.tolist()))]
    rng = np.random.default_rng(0)
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]
    diff = plan.score(refs, ca) - plan.score(refs, cb)
    drawn = [
        plan.score([refs[i] for i in p], [ca[i] for i in p]) - plan.score([refs[i] for i in p], [cb[i] for i in p])
        for p in picks
    ]
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(drawn, [tail, 100 - tail])
    print(
        f"{new} {stats[0]} against {old} {stats[1]} on {len(ns)} episodes of {task} {entropy}: {diff:+.4f} "
        f"({lo:+.4f} to {hi:+.4f}); cheaper in {sum(1 for x, y in zip(ca, cb) if x < y)}, the same in "
        f"{sum(1 for x, y in zip(ca, cb) if x == y)}"
    )


if __name__ == "__main__":
    fire.Fire(main)
