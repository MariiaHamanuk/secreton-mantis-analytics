"""The last weeks of the episode in kept plays: tags against a base, four weeks at a time, by item.

    uv run python lab/anastasiia/frontier_lab/tail/endzone.py h3_f tw_w20 tw_e20 --entropy=111 --episodes=32

Tag minus base, the mean over the episodes kept for both (``hazard_lab/play.py``): the weeks' cost and its shed-load
and lost-sales parts in bn USD, the lots started by chip, the packaged and the raw chips in stock and the wafers at
the fabs in thousands, the fuel burned in GWh. Plus is more in the tag.
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "hazard_lab"))
import play  # noqa: E402 - hazard_lab's harness: the kept plays


def main(base: str, *tags: str, task: str = "full", entropy: int = 444, episodes: int = 48, first: int = 0,
         since: int = 69, step: int = 4) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    T = inst.T
    kept = {tag: pickle.loads((play.OUT / f"{tag}_{task}_{entropy}.pkl").read_bytes()) for tag in (base, *tags)}
    prod = np.array([inst.nodes[f].fab.product for f in inst.fabs])
    stock_k = np.array([sl.k for sl in inst.stock_slots])
    at_plant = np.array([inst.nodes[sl.node].osat is not None for sl in inst.stock_slots])
    at_fab = np.array([inst.nodes[sl.node].fab is not None for sl in inst.stock_slots])
    pi = np.array([d.pi for d in inst.demands])
    dk = np.array([d.k for d in inst.demands])
    names = [c.id for c in inst.commodities]
    chips = sorted(set(prod))  # the raw chips, leading first
    packed = sorted(set(dk))
    for tag in tags:
        ns = [n for n in range(first, first + episodes) if n in kept[base] and n in kept[tag]]

        def d(field: str) -> np.ndarray:
            return np.array([kept[tag][n][field] for n in ns], dtype=float) - np.array([kept[base][n][field] for n in ns], dtype=float)

        costs, lots, stock, lost, seg = d("costs"), d("lots"), d("stock"), d("lost") * pi, d("segment")
        print(f"{tag} minus {base}, {task} {entropy}, {len(ns)} episodes")
        print(f"  {'weeks':9s} {'cost':>7s} {'shed':>7s} {'sales':>7s} " + " ".join(f"{'lost ' + names[k][5:]:>9s}" for k in packed)
              + " " + " ".join(f"{'lots ' + names[k][5:8]:>9s}" for k in chips)
              + " " + " ".join(f"{'plant ' + names[k][5:]:>9s}" for k in packed) + f" {'raw chips':>9s} {'wafers':>9s} {'gas':>8s} {'dearer':>7s}")
        for a in range(since - 1, T, step):
            b = min(a + step, T)
            row = f"  {a + 1:3d}-{b:3d}   {costs[:, a:b].sum((1, 2)).mean() / 1e9:+7.2f} {costs[:, a:b, 7].sum(1).mean() / 1e9:+7.2f} {costs[:, a:b, 5].sum(1).mean() / 1e9:+7.2f} "
            row += " ".join(f"{lost[:, a:b][:, :, dk == k].sum((1, 2)).mean() / 1e9:+9.2f}" for k in packed)
            row += " " + " ".join(f"{lots[:, a:b][:, :, prod == k].sum((1, 2)).mean() / 1e3:+9.1f}" for k in chips)
            row += " " + " ".join(f"{stock[:, b - 1][:, (stock_k == k) & at_plant].sum(1).mean() / 1e3:+9.1f}" for k in packed)
            row += f" {stock[:, b - 1][:, np.isin(stock_k, chips)].sum(1).mean() / 1e3:+9.1f}"
            row += f" {stock[:, b - 1][:, (stock_k == 3) & at_fab].sum(1).mean() / 1e3:+9.1f}"
            row += f" {seg[:, a:b, :, 0].sum((1, 2)).mean():+8.0f}"
            row += f" {(costs[:, a:b].sum((1, 2)) > 0).sum():3d}/{len(ns)}"
            print(row)


if __name__ == "__main__":
    fire.Fire(main)
