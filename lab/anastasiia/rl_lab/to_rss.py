"""Per-episode costs in dollars to the board's number.

A search minimises cost; the team decides in RSS. This turns one into the other with the package's own formula and
the episode set's cached references, so a ceiling measured in a search can be read beside `hub/FORMAL_RESULTS.md`.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/to_rss.py \
        --costs=outputs/rl_lab/knobs_hindsight_small/result.json --task=small --entropy=111

``costs`` is either that file (a list of rows with ``episode``, ``base_cost`` and ``best_cost``) or a JSON list of
dollar costs in episode order. Both the shipped numbers and the searched ones are reported, so the gain is paired.
"""

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from shockbench_flow.scoring.rss import rss_pooled  # noqa: E402

from sbf_starter import scoring  # noqa: E402


def references(task, entropy, episodes, n_jobs=1):
    """``(naive cents, clairvoyant cents, harm level)`` per episode, from the cache in ``hub/refcache``."""
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    rows = {int(r["episode"]): r for r in es.references}
    naive, oracle, strata, kept = [], [], [], []
    for i in range(episodes):
        r = rows.get(i)
        if r is None or r.get("excluded") or r["J_oracle_cents"] is None or r["stratum"] is None:
            continue
        naive.append(int(r["J_naive_cents"]))
        oracle.append(int(r["J_oracle_cents"]))
        strata.append(int(r["stratum"]))
        kept.append(i)
    return np.array(naive), np.array(oracle), np.array(strata), kept


def score(costs_usd, naive, oracle, strata):
    cents = [int(round(c * 100)) for c in costs_usd]
    present = set(int(s) for s in strata)
    missing = sorted({1, 2, 3, 4} - present)
    if missing:  # the board's formula weighs every harm level and needs an episode in each
        raise SystemExit(
            f"harm level(s) {missing} have no episode in this set: the pooled RSS is undefined. "
            f"Use more episodes (the first 32 of a root usually cover all four)."
        )
    return float(rss_pooled(cents, list(naive), list(oracle), list(strata)))


def main(costs, task="small", entropy=111, episodes=None, n_jobs=1):
    path = Path(costs) if Path(costs).is_absolute() else ROOT / costs
    data = json.loads(path.read_text())
    if isinstance(data, dict):  # a static run
        rows = None
        series = {"searched": data["per_episode"]} if "per_episode" in data else None
        if series is None:
            raise SystemExit("a static result has no per-episode costs; rerun with --record_episodes")
    else:
        rows = {int(r["episode"]): r for r in data}
        series = {
            "shipped numbers": [rows[i]["base_cost"] for i in sorted(rows)],
            "searched per episode": [rows[i]["best_cost"] for i in sorted(rows)],
        }
    episodes = episodes or (max(rows) + 1 if rows else len(next(iter(series.values()))))

    naive, oracle, strata, kept = references(task, entropy, episodes, n_jobs=n_jobs)
    print(
        f"{task}, root {entropy}, {len(kept)} of {episodes} episode(s) scored "
        f"(levels {np.bincount(strata, minlength=5)[1:]})"
    )
    out = {}
    for name, costs_all in series.items():
        picked = [costs_all[i] for i in kept]
        out[name] = score(picked, naive, oracle, strata)
        print(f"  {name:24s} RSS {out[name]:.4f}")
    if len(out) == 2:
        a, b = list(out.values())
        print(f"\n  the search is worth {b - a:+.4f} RSS on this set")
    return out


if __name__ == "__main__":
    import fire

    fire.Fire(main)
