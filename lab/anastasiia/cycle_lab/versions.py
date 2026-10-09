"""The score of each recorded version of a model on one set, and its difference with a reference version.

    uv run python lab/anastasiia/regime_lab/versions.py L_hre1 hub F_t2 F_t2c --task=small --entropy=111 --episodes=64

The first tag is the reference. For every tag: the score as ``sbf evaluate`` gives it (the board's weights of the
harm levels when the set holds all four; otherwise all that was saved over all that could be), its difference with
the reference's score, and the 90% interval of that difference: the episodes are drawn again with replacement inside
each harm level, 2,000 times, both versions on the same draw, and the interval runs from the 5th to the 95th
percentile of the differences. Reads ``play.py``'s records; plays nothing.
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import plan  # noqa: E402


def main(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 64, first: int = 0, draws: int = 2000) -> None:
    ns = list(range(first, first + episodes))
    refs_all = plan.references(task, entropy, first + episodes)
    data = {t: pickle.loads((plan.OUT / "play" / f"{t}_{task}_{entropy}.pkl").read_bytes()) for t in tags}
    ns = [n for n in ns if all(n in d for d in data.values()) and refs_all[n]["J_oracle_cents"] is not None]
    refs = [refs_all[n] for n in ns]
    costs = {t: [data[t][n]["J"] for n in ns] for t in tags}
    level = np.array([r["stratum"] for r in refs])
    groups = [np.flatnonzero(level == s) for s in sorted(set(level))]
    rng = np.random.default_rng(0)
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]
    base = tags[0]
    s0 = plan.score(refs, costs[base])
    drawn0 = np.array([plan.score([refs[i] for i in p], [costs[base][i] for i in p]) for p in picks])
    print(f"{task}, root {entropy}, episodes {ns[0]}..{ns[-1]} ({len(ns)}); by harm level {[int((level == s).sum()) for s in (1, 2, 3, 4)]}")
    print(f"  {base:10s} {s0:.4f}   (the reference)")
    for t in tags[1:]:
        s = plan.score(refs, costs[t])
        drawn = np.array([plan.score([refs[i] for i in p], [costs[t][i] for i in p]) for p in picks])
        lo, hi = np.percentile(drawn - drawn0, [5, 95])
        better = sum(1 for a, b in zip(costs[t], costs[base]) if a < b)
        print(f"  {t:10s} {s:.4f}   {s - s0:+.4f} ({lo:+.4f} to {hi:+.4f}); cheaper than the reference in {better} of {len(ns)}")


if __name__ == "__main__":
    fire.Fire(main)
