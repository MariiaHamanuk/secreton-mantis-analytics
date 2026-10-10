"""Where a shorter planning window loses: kept plays of two tags, the differences by item, quarter, grid and chip.

    uv run python lab/anastasiia/frontier_lab/tail/where.py h3c_f h3w20c_f h3w16c_f --episodes=48

Reads ``outputs/hazard_lab/play/<tag>_full_444.pkl`` (``hazard_lab/play.py``'s layout). Every number is the mean over
the episodes of (tag minus base), bn USD an episode: plus is a loss of the tag. Shed load is priced at the grid's
value of lost load, a lost sale at its market's penalty.
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


ITEMS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
BN = 1e9


def kept(tag: str, task: str, entropy: int) -> dict:
    return pickle.loads((play.OUT / f"{tag}_{task}_{entropy}.pkl").read_bytes())


def main(base: str, *tags: str, task: str = "full", entropy: int = 444, episodes: int = 48, first: int = 0,
         by_episode: bool = False) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    T = inst.T
    refs = play.references(task, entropy, first + episodes)
    data = {tag: kept(tag, task, entropy) for tag in (base, *tags)}
    ns = [n for n in range(first, first + episodes) if all(n in d for d in data.values())]
    level = np.array([refs[n]["stratum"] for n in ns])
    room = np.array([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns]) / 100.0 / BN
    voll = np.array([inst.nodes[g].grid.voll for g in inst.grids])
    pi = np.array([d.pi for d in inst.demands])
    dk = np.array([d.k for d in inst.demands])
    names = [c.id for c in inst.commodities]
    quarters = [(q * T // 4, (q + 1) * T // 4) for q in range(4)]
    slot_k = np.array([k for _e, k, _lane in inst.action_slots])
    first_edge = np.array([e for e, _k, _lane in inst.action_slots])
    supply = set(inst.supply_nodes)
    order = np.array([inst.edges[e].tail in supply for e in first_edge])
    prod = np.array([inst.nodes[f].fab.product for f in inst.fabs])
    stock_k = np.array([sl.k for sl in inst.stock_slots])
    stock_grid = np.array([inst.nodes[sl.node].grid is not None for sl in inst.stock_slots])
    stock_osat = np.array([inst.nodes[sl.node].osat is not None for sl in inst.stock_slots])

    def stack(tag: str, field: str) -> np.ndarray:
        return np.array([data[tag][n][field] for n in ns], dtype=float)

    print(f"{task} {entropy}, {len(ns)} episodes ({ns[0]}..{ns[-1]}), by harm level {[int((level == s).sum()) for s in (1, 2, 3, 4)]}")
    J0 = np.array([data[base][n]["J"] for n in ns]) / 100.0 / BN
    for tag in tags:
        J = np.array([data[tag][n]["J"] for n in ns]) / 100.0 / BN
        dJ = J - J0
        print(f"\n== {tag} minus {base}: {dJ.mean():+.2f} bn an episode ({(dJ > 0).sum()} of {len(ns)} dearer); "
              f"all over all room {dJ.sum() / room.sum():+.4f}")
        print("   by harm level: " + "  ".join(
            f"L{s} {dJ[level == s].mean():+.2f} bn ({dJ[level == s].sum() / room[level == s].sum():+.4f} of room, n={int((level == s).sum())})"
            for s in (1, 2, 3, 4) if (level == s).any()))
        dc = (stack(tag, "costs") - stack(base, "costs")) / BN  # episode, week, item
        print("   by item:    " + "  ".join(f"{name} {dc[:, :, i].sum(1).mean():+.2f}" for i, name in enumerate(ITEMS)))
        print("   by quarter: " + "  ".join(f"w{a + 1}-{b} {dc[:, a:b].sum((1, 2)).mean():+.2f}" for a, b in quarters))
        for i in (5, 7, 3):
            print(f"     {ITEMS[i]:9s} by quarter: " + "  ".join(f"{dc[:, a:b, i].sum(1).mean():+.2f}" for a, b in quarters))
        dshed = (stack(tag, "shed") - stack(base, "shed")) * voll / BN  # episode, week, grid
        print("   shed by grid (bn):  " + "  ".join(f"g{gi} {dshed[:, :, gi].sum(1).mean():+.2f}" for gi in range(len(voll))))
        dlost = (stack(tag, "lost") - stack(base, "lost")) * pi / BN
        for k in sorted(set(dk)):
            sel = dk == k
            print(f"   lost {names[k]:9s}: {dlost[:, :, sel].sum((1, 2)).mean():+.2f}  by quarter " + "  ".join(
                f"{dlost[:, a:b][:, :, sel].sum((1, 2)).mean():+.2f}" for a, b in quarters))
        lots = stack(tag, "lots") - stack(base, "lots")
        lots0 = stack(base, "lots")
        for k in sorted(set(prod)):
            sel = prod == k
            print(f"   lots {names[k]:12s}: {lots[:, :, sel].sum((1, 2)).mean():+.0f} of {lots0[:, :, sel].sum((1, 2)).mean():.0f}  by quarter " + "  ".join(
                f"{lots[:, a:b][:, :, sel].sum((1, 2)).mean():+.0f}" for a, b in quarters))
        sent, sent0 = stack(tag, "sent"), stack(base, "sent")
        for k in (0, 1, 2):
            sel = (slot_k == k) & order
            d = sent[:, :, sel].sum(2) - sent0[:, :, sel].sum(2)
            print(f"   ordered {names[k]:8s}: {d.sum(1).mean():+.0f} of {sent0[:, :, sel].sum((1, 2)).mean():.0f}  by quarter " + "  ".join(
                f"{d[:, a:b].sum(1).mean():+.0f}" for a, b in quarters))
        stock, stock0 = stack(tag, "stock"), stack(base, "stock")
        for k in (0, 1, 2):
            sel = (stock_k == k) & stock_grid
            d = stock[:, :, sel].sum(2) - stock0[:, :, sel].sum(2)
            print(f"   grid stock {names[k]:8s} (mean over weeks): {d.mean():+.0f} of {stock0[:, :, sel].sum(2).mean():.0f}  by quarter " + "  ".join(
                f"{d[:, a:b].mean():+.0f}" for a, b in quarters))
        for k in (6, 7):
            sel = (stock_k == k) & stock_osat
            d = stock[:, :, sel].sum(2) - stock0[:, :, sel].sum(2)
            print(f"   plant stock {names[k]:8s} (mean over weeks): {d.mean():+.0f} of {stock0[:, :, sel].sum(2).mean():.0f}; at the last week {d[:, -1].mean():+.0f} of {stock0[:, -1, sel].sum(1).mean():.0f}")
        if by_episode:
            for i, n in enumerate(ns):
                top = np.argsort(-np.abs(dshed[i].sum(0)))[:2]
                print(f"     ep {n:3d} L{level[i]} dJ {dJ[i]:+7.1f} ({dJ[i] / room[i]:+.4f})  shed {dc[i, :, 7].sum():+7.1f}  shortage {dc[i, :, 5].sum():+7.1f}  "
                      f"holding {dc[i, :, 3].sum():+5.1f} tariff {dc[i, :, 2].sum():+5.1f} freight {dc[i, :, 0].sum():+5.1f}  "
                      + " ".join(f"g{gi}:{dshed[i, :, gi].sum():+.1f}" for gi in top)
                      + "  q " + " ".join(f"{dc[i, a:b].sum():+.0f}" for a, b in quarters))


if __name__ == "__main__":
    fire.Fire(main)
