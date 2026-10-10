"""Where the holding cost of one commodity sits: mean stock by node and what it costs, in the kept plays of the model
and of the told planner and in the descent's plan (notes/u_fullceil.md). Reads kept files only.

    uv run python lab/anastasiia/frontier_lab/fullceil/hold.py --k=nucfuel
"""

import pickle

import common as K
import fire
import numpy as np
from descend import run_path


def main(k: str = "nucfuel", only: str | tuple | int = "", label: str = "tah0_f", task: str = "full", entropy: int = 444,
         first: int = 0, episodes: int = 16) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst = task_generator(task)[0]
    N = inst.nodes
    ki = [c.id for c in inst.commodities].index(k)
    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else list(range(first, first + episodes))
    ns = [n for n in ns if run_path(label, task, entropy, n).is_file() and pickle.loads(run_path(label, task, entropy, n).read_bytes()).get("done")]
    slots = [s for s, sl in enumerate(inst.stock_slots) if sl.k == ki and N[sl.node].type != "chokepoint"]
    plays = {tag: K.kept(tag, task, entropy) for tag in ("h3c_f", "truthallc_f", "tah0c_f")}
    stock = {tag: np.mean([d[n]["stock"] for n in ns], axis=0) for tag, d in plays.items()}  # (T, S), mean over episodes
    runs = [pickle.loads(run_path(label, task, entropy, n).read_bytes()) for n in ns]
    stock["told all, here"] = np.mean([r["weeks0"]["stock"] for r in runs], axis=0)
    stock["descent's plan"] = np.mean([r["weeks"]["stock"] for r in runs], axis=0)
    burn = {}
    for g in inst.grids:
        ga = N[g].grid
        if ga.shares.get(ki, 0.0) > 0:
            burn[g] = ga.shares[ki] * ga.deliverable
    print(f"{k}: {len(ns)} episodes {ns}; mean stock over the weeks, thousand units (weeks of the grid's full burn), and its holding, bn an episode")
    print(f"{'node':22s}{'USD/unit/week':>14s}{'storage':>10s}" + "".join(f"{tag:>26s}" for tag in stock))
    tot = dict.fromkeys(stock, 0.0)
    for s in slots:
        sl = inst.stock_slots[s]
        line = f"{N[sl.node].id:22s}{sl.holding:14.1f}{(sl.storage if sl.storage is not None else float('nan')) / 1e3:10.1f}"
        for tag, a in stock.items():
            cost = a[:, s].sum() * sl.holding / 1e9
            tot[tag] += cost
            wk = f" ({a[:, s].mean() / burn[sl.node]:.1f} wk)" if sl.node in burn else ""
            line += f"{a[:, s].mean() / 1e3:10.1f}{wk:>9s}{cost:7.1f}"
        print(line)
    print(f"{'all':46s}" + "".join(f"{'':19s}{v:7.1f}" for v in tot.values()))
    print("by 13 weeks, holding of this commodity, bn: " + "; ".join(
        f"{tag}: " + " ".join(f"{sum(a[i:i + 13, s].sum() * inst.stock_slots[s].holding for s in slots) / 1e9:.1f}" for i in range(0, a.shape[0], 13))
        for tag, a in stock.items()))


if __name__ == "__main__":
    fire.Fire(main)
