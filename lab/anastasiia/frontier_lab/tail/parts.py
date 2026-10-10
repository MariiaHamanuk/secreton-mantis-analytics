"""A tag's loss to a base by part of the episode, each part with its own paired 90 % interval.

    uv run python lab/anastasiia/frontier_lab/tail/parts.py h3c_f h3w20c_f h3w16c_f --episodes=48

The week's costs of the kept plays (``hazard_lab/play.py``), tag minus base, as a share of the room between the naive
rule and the clairvoyant plan with the board's weights of the harm levels; the episodes are drawn again inside each
level. Plus is a loss of the tag. The parts sum to the whole but for the terminal salvage (under 0.0001).
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "hazard_lab"))
import play  # noqa: E402 - hazard_lab's harness: the kept plays and the references


# the quarters, then the weeks before the window of 26 weeks sees the episode's end, the twelve weeks from there, the rest
PARTS = {"full": ((1, 26), (27, 52), (53, 78), (79, 104), (1, 78), (79, 90), (91, 104)),
         "small": ((1, 13), (14, 26), (27, 39), (40, 52), (1, 26), (27, 38), (39, 52))}  # fmt: skip


def interval(d: np.ndarray, room: np.ndarray, level: np.ndarray, rng, draws: int = 4000) -> tuple:
    levels = sorted(set(level))
    groups = [np.flatnonzero(level == s) for s in levels]
    weights = [play.WEIGHTS[s] for s in levels]

    def stat(gs: list) -> float:
        return sum(w * d[g].mean() for w, g in zip(weights, gs)) / sum(w * room[g].mean() for w, g in zip(weights, gs))

    xs = [stat([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]
    return stat(groups), float(np.percentile(xs, 5)), float(np.percentile(xs, 95))


def main(base: str, *tags: str, task: str = "full", entropy: int = 444, episodes: int = 48, first: int = 0) -> None:
    refs = play.references(task, entropy, first + episodes)
    kept = {tag: pickle.loads((play.OUT / f"{tag}_{task}_{entropy}.pkl").read_bytes()) for tag in (base, *tags)}
    rng = np.random.default_rng(0)
    for tag in tags:
        ns = [n for n in range(first, first + episodes) if n in kept[base] and n in kept[tag]]
        room = np.array([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns]) / 100.0 / 1e9
        level = np.array([refs[n]["stratum"] for n in ns])
        dc = np.array([kept[tag][n]["costs"].sum(1) - kept[base][n]["costs"].sum(1) for n in ns]) / 1e9
        dJ = np.array([kept[tag][n]["J"] - kept[base][n]["J"] for n in ns]) / 100.0 / 1e9
        m, lo, hi = interval(dJ, room, level, rng)
        print(f"{tag} minus {base}, {len(ns)} episodes: {dJ.mean():+.2f} bn an episode, {m:+.4f} ({lo:+.4f} to {hi:+.4f}) of the room")
        for a, b in PARTS[task]:
            part = dc[:, a - 1 : b].sum(1)
            m, lo, hi = interval(part, room, level, rng)
            print(f"   weeks {a:3d}-{b:3d}  {part.mean():+7.2f} bn   {m:+.4f} ({lo:+.4f} to {hi:+.4f})   dearer in {(part > 0).sum()} of {len(ns)}")


if __name__ == "__main__":
    fire.Fire(main)
