"""Is the exact advantage a function of the state at all, or only of the future?

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/constant_policy.py --dataset=outputs/rl_lab/cf_dest600

A model fitted on the exact advantages memorised its training states and transferred nothing. That has two very
different explanations, and this tells them apart with arithmetic alone.

If the advantage carries structure that simply is not visible in the observation, then a **state-independent**
choice — always this coordinate at this factor — should still earn something on average, and a learner that cannot
see the future would be right to fall back on it. If even the best constant choice earns nothing, the signal is
not merely hard to read from the state: there is no stable preference to read, and the search's edge comes from
knowing which week this is in that particular scenario.

Reported per candidate: its mean advantage over the states where it was offered, with a standard error, and what
the best constant choice would have captured against the per-state best.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


def main(dataset, top=12):
    folder = Path(dataset) if Path(dataset).is_absolute() else ROOT / dataset
    rows = json.loads((folder / "dataset.json").read_text())
    settings = json.loads((folder / "settings.json").read_text())

    by_pair = defaultdict(list)
    by_factor = defaultdict(list)
    best_per_state, states = [], 0
    for r in rows:
        if not r["candidates"]:
            continue
        plain = max(float(r["plain_cost_to_go"]), 1.0)
        adv = {}
        for c in r["candidates"]:
            a = (plain - float(c["cost_to_go"])) / plain
            key = (int(c["coordinate"]), float(c["factor"]))
            by_pair[key].append(a)
            by_factor[float(c["factor"])].append(a)
            adv[key] = a
        best_per_state.append(max(adv.values()))
        states += 1
    best_mean = float(np.mean(best_per_state))

    print(
        f"{states} state(s), grouping {settings.get('grouping')}, "
        f"the per-state best is worth {best_mean:+.4%} of the cost to go on average\n"
    )

    print(f"{'factor':>8s} {'offered':>8s} {'mean advantage':>16s} {'standard error':>16s}")
    for f in sorted(by_factor):
        v = np.array(by_factor[f])
        print(f"{f:8.2f} {v.size:8d} {v.mean():+16.5%} {v.std(ddof=1) / np.sqrt(v.size):16.5%}")

    ranked = sorted(
        ((np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v)), len(v), k) for k, v in by_pair.items() if len(v) >= 20),
        reverse=True,
    )
    print(f"\nbest constant (coordinate, factor) choices, of {len(by_pair)} offered at least once:")
    print(f"{'coord':>6s} {'factor':>7s} {'offered':>8s} {'mean advantage':>16s} {'standard error':>16s}")
    for m, se, n, (j, f) in ranked[:top]:
        print(f"{j:6d} {f:7.2f} {n:8d} {m:+16.5%} {se:16.5%}")

    if ranked:
        m, se, n, (j, f) = ranked[0]
        print(
            f"\nthe best constant choice (coordinate {j}, factor {f}) captures "
            f"{m / best_mean:+.2%} of the per-state best, and its own mean is {m:+.5%} ± {se:.5%}"
        )
        print(
            "a mean within a standard error of zero means there is no stable preference to learn: the search's "
            "edge is in knowing the scenario's future, not in reading the state"
        )
    return None


if __name__ == "__main__":
    import fire

    fire.Fire(main)
