"""Cost per episode of several agent folders on the same episodes, and what choosing the best of them per episode gives.

    uv run python lab/anastasiia/heur_lab3/tools/costs.py agents/anastasiia_rules_v2 path/to/variantA path/to/variantB \
        --task=small --entropy=111 --episodes=64 --out=outputs/heur3/data/variants_small.json

The first folder is the base. Per agent: the board's score, the paired difference with the base and its 90% interval
(episodes resampled inside their harm level, as ``hub/eval/formal_eval.py`` does). Last lines: the score of the best
agent per episode chosen with hindsight, over all the folders and over each pair (base, variant). That is an upper bound
on any rule that picks one of these variants per episode: no selector without the future can beat it.

``--out`` keeps the rows (cost of the agent, of the naive rule and of the clairvoyant plan per episode, in cents), and a
later call with the same ``--out`` plays only the folders that are not in the file yet.
"""

import json
from pathlib import Path

import fire
import numpy as np


LEVEL_WEIGHTS = (0.50, 0.30, 0.15, 0.05)
KEYS = ("episode", "stratum", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents", "excluded")


def pooled(level: np.ndarray, saved: np.ndarray, room: np.ndarray) -> float:
    """The board's score: the levels' mean savings over their mean attainable savings, weighted by level."""
    present = [s for s in (1, 2, 3, 4) if (level == s).any()]
    w = {s: LEVEL_WEIGHTS[s - 1] for s in present}
    return float(
        sum(w[s] * saved[level == s].mean() for s in present) / sum(w[s] * room[level == s].mean() for s in present)
    )


def interval(level: np.ndarray, saved_a: np.ndarray, saved_b: np.ndarray, room: np.ndarray, draws: int = 2000):
    """a's score minus b's and its 90% interval (episodes resampled inside their harm level)."""
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(level == s) for s in (1, 2, 3, 4) if (level == s).any()]
    diffs = np.empty(draws)
    for i in range(draws):
        pick = np.concatenate([rng.choice(g, len(g)) for g in groups])
        diffs[i] = pooled(level[pick], saved_a[pick], room[pick]) - pooled(level[pick], saved_b[pick], room[pick])
    diff = pooled(level, saved_a, room) - pooled(level, saved_b, room)
    return diff, float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))


def main(
    *agents: str, task: str = "small", entropy: int = 111, episodes: int = 64, n_jobs: int = 2, out: str = ""
) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    kept = json.loads(Path(out).read_text()) if out and Path(out).is_file() else {}
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    for a in agents:
        if a not in kept:
            s = es.score(str(resolve(a)), n_jobs=n_jobs, cpu_budget=False)
            kept[a] = [{k: r.get(k) for k in KEYS} for r in s.rows]
            print(f"played {a}: {s.rss:.4f}, naive weeks {s.fallback_weeks}", flush=True)
            if out:
                Path(out).write_text(json.dumps(kept))
    ok = np.array([r["excluded"] is None for r in kept[agents[0]]])
    level = np.array([r["stratum"] for r in kept[agents[0]]])[ok]
    naive = np.array([r["J_naive_cents"] for r in kept[agents[0]]], dtype=float)[ok]
    room = naive - np.array([r["J_clairvoyant_cents"] for r in kept[agents[0]]], dtype=float)[ok]
    saved = {a: naive - np.array([r["J_policy_cents"] for r in kept[a]], dtype=float)[ok] for a in agents}
    base = saved[agents[0]]
    print(f"\n{task}, {episodes} episodes of root {entropy}; levels {[int((level == s).sum()) for s in (1, 2, 3, 4)]}")
    print(f"{'score':>7} {'vs base':>9} {'90% interval':>22} {'best of (base, it)':>19} {'wins':>5}  agent")
    for a in agents:
        d, lo, hi = interval(level, saved[a], base, room)
        both = pooled(level, np.maximum(saved[a], base), room)
        wins = int((saved[a] > base + 1).sum())
        print(
            f"{pooled(level, saved[a], room):7.4f} {d:+9.4f} {f'{lo:+.4f} to {hi:+.4f}':>22} {both:19.4f} "
            f"{wins:5d}  {a}"
        )
    best = np.max(np.stack([saved[a] for a in agents]), axis=0)
    d, lo, hi = interval(level, best, base, room)
    print(f"\nbest of all {len(agents)} per episode, with hindsight: {pooled(level, best, room):.4f}")
    print(f"  over the base: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")
    which = np.argmax(np.stack([saved[a] for a in agents]), axis=0)
    print("  episodes won by each:", {a: int((which == i).sum()) for i, a in enumerate(agents)})


if __name__ == "__main__":
    fire.Fire(main)
