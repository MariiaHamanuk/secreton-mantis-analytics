"""Where a scenario variant's saving over its base sits: by cost item, by part of the episode, by harm level.

    uv run python lab/anastasiia/frontier_lab/mech/items.py h3c_s two1l --task=small --episodes=64-127

Reads the kept plays (``scen/play.py kept``). Per item of the week's cost (``comp``), the base's cost minus the
variant's, bn USD an episode, with the 90 % interval of a resampling of the episodes; the same by quarter of the
episode; lots started, shed load and lost sales (units an episode).
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "scen"), str(HERE.parents[1] / "hazard_lab")]
COMP = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def _ci(x: np.ndarray, draws: int = 2000) -> tuple:
    rng = np.random.default_rng(0)
    means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    return float(np.percentile(means, 5)), float(np.percentile(means, 95))


def main(base: str, tag: str, task: str = "small", entropy: int = 444, episodes: str = "0-999", parts: int = 4) -> None:
    import play as P

    a, b = P.kept(base, task, entropy), P.kept(tag, task, entropy)
    ns = [n for n in P._numbers(episodes) if n in a and n in b]
    if not ns:
        print("no common episode")
        return
    refs = None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("hazard_play", HERE.parents[1] / "hazard_lab" / "play.py")
        hz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hz)
        refs = hz.references(task, entropy, max(ns) + 1)
    except Exception as error:  # the levels are an extra
        print("no references:", repr(error)[:100])
    d = np.array([(a[n]["costs"] - b[n]["costs"]) / 1e9 for n in ns])  # episode, week, item: bn USD saved (the items are in USD)
    total = np.array([(a[n]["J"] - b[n]["J"]) / 1e11 for n in ns])
    T = d.shape[1]
    print(f"{tag} against {base}, {task} {entropy}, {len(ns)} episodes ({ns[0]}..{ns[-1]}): bn USD an episode saved")
    lo, hi = _ci(total)
    print(f"  {'all':14s} {total.mean():+7.2f} ({lo:+.2f} to {hi:+.2f})   cheaper in {int((total > 0).sum())}, the same in {int((total == 0).sum())}")
    for i, name in enumerate(COMP):
        x = d[:, :, i].sum(axis=1)
        lo, hi = _ci(x)
        cuts = np.array_split(np.arange(T), parts)
        by = "  ".join(f"{d[:, c, i].sum(axis=1).mean():+6.2f}" for c in cuts)
        print(f"  {name:14s} {x.mean():+7.2f} ({lo:+.2f} to {hi:+.2f})   by part of the episode: {by}")
    cuts = np.array_split(np.arange(T), parts)
    print(f"  {'all, by part':14s} " + "  ".join(f"{d[:, c, :].sum(axis=(1, 2)).mean():+6.2f}" for c in cuts))
    for name in ("lots", "shed", "lost"):
        x = np.array([b[n][name].sum() - a[n][name].sum() for n in ns])
        base_mean = np.mean([a[n][name].sum() for n in ns])
        lo, hi = _ci(x)
        print(f"  {name:6s} variant minus base: {x.mean():+.4g} ({lo:+.4g} to {hi:+.4g}) of {base_mean:.4g}")
    if refs is not None:
        level = np.array([refs[n]["stratum"] for n in ns])
        room = np.array([(refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 1e11 for n in ns])
        for s in sorted(set(level)):
            m = level == s
            items = "  ".join(f"{COMP[i][:5]} {d[m][:, :, i].sum(axis=1).mean():+.2f}" for i in range(len(COMP)))
            print(f"  level {s} ({int(m.sum()):2d} ep, room {room[m].mean():6.0f} bn): {total[m].mean():+6.2f} bn = {total[m].sum() / room[m].sum():+.4f} of room;  {items}")
    # the picks of the variant
    rows = [w for n in ns for w in (b[n].get("notes") or {}).get("rows", [])]
    if rows:
        joint = [w for w in rows if w["pick"] == "joint"]
        print(f"  weeks {len(rows)}, joint played in {len(joint) / len(rows):.2f}; mean distance there {np.mean([w['dist'] for w in joint]):.3f}; "
              f"claimed {np.mean([w['joint_gain'] for w in joint]):.2f} bn a week (median {np.median([w['joint_gain'] for w in joint]):.2f})")


if __name__ == "__main__":
    fire.Fire(main)
