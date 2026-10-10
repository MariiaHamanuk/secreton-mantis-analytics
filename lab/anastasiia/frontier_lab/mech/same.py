"""Do two variants gain in the same episodes? Per-episode savings of each against its own base, side by side.

    uv run python lab/anastasiia/frontier_lab/mech/same.py h3_s:two1pl h3_s:h3p3_s h3_s:two8b --episodes=0-63

Every argument is ``base:variant`` (kept plays of ``scen/play.py`` or ``hazard_lab/play.py``). For every pair of
arguments, on the episodes all of them have: the mean saving of each (bn USD an episode), the rank correlation of
the savings, the episodes where both save or both lose more than ``big`` bn, where only one does, and the paired
difference of the two variants themselves with its 90 % interval (a resampling of the episodes). Two variants with
the same base share that base's own play, which correlates their savings by itself: the line says so.
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scen"), str(HERE.parents[1] / "hazard_lab")]


def main(*pairs: str, task: str = "small", entropy: int = 444, episodes: str = "0-63", big: float = 3.0,
         by_episode: bool = False) -> None:
    import play as P
    from scipy.stats import spearmanr

    kept = {}
    for pair in pairs:
        for tag in pair.split(":"):
            if tag not in kept:
                kept[tag] = P.kept(tag, task, entropy)
    ns = [n for n in P._numbers(episodes) if all(n in kept[tag] for tag in kept)]
    if not ns:
        print("no episode is kept for every tag")
        return
    save = {}
    for pair in pairs:
        base, tag = pair.split(":")
        save[pair] = np.array([(kept[base][n]["J"] - kept[tag][n]["J"]) / 1e11 for n in ns])
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(ns), size=(2000, len(ns)))
    print(f"{task} {entropy}, {len(ns)} episodes ({ns[0]}..{ns[-1]}): savings against the own base, bn USD an episode")
    for pair in pairs:
        x = save[pair]
        lo, hi = np.percentile(x[picks].mean(axis=1), [5, 95])
        print(f"  {pair:22s} {x.mean():+6.2f} ({lo:+.2f} to {hi:+.2f}), saves in {int((x > 0).sum())}, more than {big:g} bn in "
              f"{int((x > big).sum())}, loses more than {big:g} bn in {int((x < -big).sum())}")
    for i, a in enumerate(pairs):
        for b in pairs[i + 1 :]:
            x, y = save[a], save[b]
            shared = a.split(":")[0] == b.split(":")[0]
            d = y - x
            lo, hi = np.percentile(d[picks].mean(axis=1), [5, 95])
            both = int(((x > big) & (y > big)).sum())
            only_a, only_b = int(((x > big) & (y <= big)).sum()), int(((y > big) & (x <= big)).sum())
            lose = int(((x < -big) & (y < -big)).sum())
            print(f"  {a} and {b}: rank correlation {spearmanr(x, y)[0]:+.2f}{' (one base play in both)' if shared else ''}; "
                  f"both save over {big:g} bn in {both}, only the first in {only_a}, only the second in {only_b}, both lose "
                  f"in {lose}; second minus first {d.mean():+.2f} bn ({lo:+.2f} to {hi:+.2f})")
    if by_episode:
        print("  episode: " + "  ".join(pairs))
        for j, n in enumerate(ns):
            print(f"  {n:4d}: " + "  ".join(f"{save[pair][j]:+7.2f}" for pair in pairs))


if __name__ == "__main__":
    fire.Fire(main)
