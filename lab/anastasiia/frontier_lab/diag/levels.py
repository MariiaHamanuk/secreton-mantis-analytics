"""The score by harm level on Full: the formal records, root 444, and what the level's episodes have in common.

    uv run python lab/anastasiia/frontier_lab/diag/levels.py

1. The kept formal costs (root 222: read only) of ``anastasiia_plan_hazard`` and of the later models' progress
   folders, by harm level, with the interval of each level (its episodes drawn again 4,000 times) and what the level
   scores without its worst episodes.
2. The model's settings without the clock on Full 444 episodes 0-111 (``h3c_f``), the same table, and both sets pooled.
3. Per episode of root 444: the grids that shed base load in over half of the weeks, and the worth of the lots their
   fabs did not start in those weeks (capacity times the chip's penalty): what the board's reference, which may
   power a fab while its grid sheds, has and no play can have.
"""

import glob
import json

import fire
import numpy as np
from common import BN, ROOT, WEIGHTS, Names, kept, refs


def table(lvl, naive, best, cost, rng, draws: int = 4000) -> None:
    print(
        "  lvl    n  naive  clairv   room    gap   score  90 % interval   median   p10   under 0.85 | without its worst 1, 2"
    )
    for s in (1, 2, 3, 4):
        idx = np.flatnonzero(lvl == s)
        if not len(idx):
            continue
        room, gap = naive[idx] - best[idx], cost[idx] - best[idx]
        ep = 1 - gap / room
        boot = [1 - gap[i].sum() / room[i].sum() for i in (rng.integers(0, len(idx), len(idx)) for _ in range(draws))]
        lo, hi = np.percentile(boot, [5, 95])
        order = np.argsort(-gap)
        less = [1 - gap[order[k:]].sum() / room[order[k:]].sum() for k in (1, 2) if len(idx) > k]
        print(
            f"   {s}  {len(idx):4d} {naive[idx].mean():6.0f} {best[idx].mean():7.0f} {room.mean():6.0f} {gap.mean():6.0f}  {1 - gap.sum() / room.sum():.4f}  "
            f"{lo:.3f} to {hi:.3f}   {np.median(ep):.3f}  {np.percentile(ep, 10):.3f}   {100 * np.mean(ep < 0.85):3.0f} %     | "
            + ", ".join(f"{v:.4f}" for v in less)
        )
    w = np.array([WEIGHTS[s] for s in (1, 2, 3, 4)])
    saved = np.array([(naive - cost)[lvl == s].mean() for s in (1, 2, 3, 4)])
    room = np.array([(naive - best)[lvl == s].mean() for s in (1, 2, 3, 4)])
    print(
        f"  board's weighting: {np.sum(w * saved) / np.sum(w * room):.4f}; shares of the weighted gap by level: "
        + " / ".join(f"{100 * x:.0f} %" for x in w * (room - saved) / np.sum(w * (room - saved)))
    )


def main(tag: str = "h3c_f", episodes: int = 112, tail: float = 600.0) -> None:
    rng = np.random.default_rng(0)
    record = json.loads((ROOT / "hub/eval/records/anastasiia_plan_hazard.json").read_text())
    rows = record["sets"]["full-222"]["rows"]
    lvl2 = np.array([r["stratum"] for r in rows])
    naive2 = np.array([r["J_naive_cents"] for r in rows]) / 1e11
    best2 = np.array([r["J_clairvoyant_cents"] for r in rows]) / 1e11
    models = {"anastasiia_plan_hazard": np.array([r["J_policy_cents"] for r in rows]) / 1e11}
    for name, folder in (
        ("anastasiia_plan_hazard4", "outputs/formal_eval/anastasiia_plan_hazard4/progress_*/full-222"),
    ):
        cost = np.full(len(rows), np.nan)
        for path in glob.glob(str(ROOT / folder / "*.json")):
            row = json.loads(open(path).read())
            cost[row["episode"]] = row["J_policy_cents"] / 1e11
        if np.isfinite(cost).all():
            models[name] = cost
    print("1. Formal set, Full root 222, 128 episodes (kept costs; bn USD an episode)")
    for name, cost in models.items():
        print(f" {name}")
        table(lvl2, naive2, best2, cost, rng)

    R, D = refs(), kept(tag)
    E = list(range(episodes))
    lvl = np.array([R[n]["stratum"] for n in E])
    naive = np.array([R[n]["J_naive_cents"] for n in E]) / 1e11
    best = np.array([R[n]["J_oracle_cents"] for n in E]) / 1e11
    cost = np.array([D[n]["J"] for n in E]) / 1e11
    print(f"\n2. Full root 444, episodes 0-{episodes - 1}, {tag} (the model's settings, no clock)")
    table(lvl, naive, best, cost, rng)
    print(" both sets together (240 episodes; two builds of one model)")
    table(
        np.concatenate([lvl2, lvl]),
        np.concatenate([naive2, naive]),
        np.concatenate([best2, best]),
        np.concatenate([models["anastasiia_plan_hazard"], cost]),
        rng,
    )
    both_lvl, both_gap = (
        np.concatenate([lvl2, lvl]),
        np.concatenate([models["anastasiia_plan_hazard"] - best2, cost - best]),
    )
    both_room = np.concatenate([naive2 - best2, naive - best])
    lo_hi = []
    for _ in range(4000):
        pick = [rng.choice(np.flatnonzero(both_lvl == s), (both_lvl == s).sum()) for s in (1, 2, 3, 4)]
        low = np.concatenate(pick[:2])
        high = np.concatenate(pick[2:])
        lo_hi.append(
            (1 - both_gap[high].sum() / both_room[high].sum()) - (1 - both_gap[low].sum() / both_room[low].sum())
        )
    low, high = np.isin(both_lvl, (1, 2)), np.isin(both_lvl, (3, 4))
    d = (1 - both_gap[high].sum() / both_room[high].sum()) - (1 - both_gap[low].sum() / both_room[low].sum())
    print(
        f" levels 3 and 4 less levels 1 and 2, all saved over all room: {d:+.4f} ({np.percentile(lo_hi, 5):+.4f} to {np.percentile(lo_hi, 95):+.4f})"
    )

    nm = Names("full")
    inst, N = nm.inst, nm.inst.nodes
    pi = {}
    for d_ in inst.demands:
        pi[d_.k] = max(pi.get(d_.k, 0.0), d_.pi)
    for o in inst.osats:
        for raw, packed in N[o].osat.packages.items():
            pi[raw] = max(pi.get(raw, 0.0), pi[packed])
    worth = np.array([pi[N[f].fab.product] * N[f].fab.cap0 for f in inst.fabs])
    base = np.array([N[g].grid.base_load for g in inst.grids])
    with_fabs = sorted(set(nm.fab_grid.tolist()))
    short = np.zeros((episodes, len(inst.grids)))  # share of weeks the grid sheds
    idle = np.zeros(episodes)
    for n in E:
        sheds = D[n]["shed"] > 1e-6 * base
        short[n] = sheds.mean(axis=0)
        idle[n] = (sheds[:96][:, nm.fab_grid] * worth).sum() / BN
    gap, room = cost - best, naive - best
    is_tail = gap > tail
    print(f"\n3. Root 444, {tag}: grids short of fuel and the fabs behind them")
    print(
        "  lvl    n   gap  grids with fabs that shed in over half of the weeks   worth of their idle fab weeks, bn   share of weeks shedding: "
        + " ".join(nm.grids[g][5:] for g in with_fabs)
    )
    for s in (1, 2, 3, 4):
        m = lvl == s
        print(
            f"   {s}  {m.sum():4d} {gap[m].mean():5.0f}  {(short[m][:, with_fabs] > 0.5).sum(axis=1).mean():24.2f} {idle[m].mean():38.0f}"
            f"{'':29s}" + " ".join(f"{100 * short[m][:, g].mean():3.0f}" for g in with_fabs)
        )
    print(
        f"  correlation over the {episodes} episodes: gap and idle worth {np.corrcoef(gap, idle)[0, 1]:.2f}; gap and harm "
        f"{np.corrcoef(gap, [R[n]['harm_usd'] for n in E])[0, 1]:.2f}; gap and room {np.corrcoef(gap, room)[0, 1]:.2f}"
    )
    kr = nm.grids.index("grid_kr")
    print(
        f"  episodes with a gap over {tail:.0f} bn: {is_tail.sum()} ({[int((is_tail & (lvl == s)).sum()) for s in (1, 2, 3, 4)]} by level, "
        f"{[round(float((is_tail & (lvl == s)).sum() / (lvl == s).sum()), 2) for s in (1, 2, 3, 4)]} of each level); they hold "
        f"{100 * gap[is_tail].sum() / gap.sum():.0f} % of all the gap"
    )
    print(
        f"  grid_kr sheds in over half of the weeks in {100 * (short[is_tail, kr] > 0.5).mean():.0f} % of them and in "
        f"{100 * (short[~is_tail, kr] > 0.5).mean():.0f} % of the others; by level "
        f"{[round(float((short[lvl == s, kr] > 0.5).mean()), 2) for s in (1, 2, 3, 4)]}"
    )
    print(
        "  score by level without those episodes: "
        + " / ".join(
            f"{1 - gap[(lvl == s) & ~is_tail].sum() / room[(lvl == s) & ~is_tail].sum():.4f}" for s in (1, 2, 3, 4)
        )
    )
    print("  episodes of the tail: " + ", ".join(f"{n} (L{lvl[n]}, {gap[n]:.0f})" for n in np.flatnonzero(is_tail)))


if __name__ == "__main__":
    fire.Fire(main)
