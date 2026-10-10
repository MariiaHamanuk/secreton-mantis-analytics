"""The Full ladder by cost item (notes/u_fullceil.md): the kept plays and the best executed plan of each episode, mean
bn USD an episode, and each step of the ladder by item. Reads kept files only; nothing is played.

    uv run python lab/anastasiia/frontier_lab/fullceil/items.py
"""

import pickle

import common as K
import fire
import numpy as np
from descend import run_path
from record import start_path


ITEMS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
GROUPS = {"unmet demand": ("shortage",), "shed load": ("shed",), "holding": ("holding", "queue_holding"),
          "disposal": ("disposal",), "freight, war risk, tariff": ("freight", "war_risk", "tariff")}  # fmt: skip


def main(only: str | tuple | int = "", label: str = "tah0_f", task: str = "full", entropy: int = 444, first: int = 0,
         episodes: int = 16) -> None:
    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else list(range(first, first + episodes))
    ns = [n for n in ns if run_path(label, task, entropy, n).is_file() and pickle.loads(run_path(label, task, entropy, n).read_bytes()).get("done")]
    if not ns:
        return
    plays = {tag: K.kept(tag, task, entropy) for tag in ("h3c_f", "truthallc_f", "tah0c_f")}
    rows = {tag: np.array([d[n]["costs"].sum(axis=0) for n in ns]) / 1e9 for tag, d in plays.items()}
    runs = [pickle.loads(run_path(label, task, entropy, n).read_bytes()) for n in ns]
    rows["told all, played here"] = np.array([r["weeks0"]["costs"].sum(axis=0) for r in runs]) / 1e9
    rows["descent's plan"] = np.array([r["weeks"]["costs"].sum(axis=0) for r in runs]) / 1e9
    # the weekly costs leave out the terminal salvage: the totals are checked against the kept played costs
    J = {"told all, played here": np.array([pickle.loads(start_path(label, task, entropy, n).read_bytes())["J"] for n in ns]) / K.BN,
         "descent's plan": np.array([r["J"] for r in runs]) / K.BN}  # fmt: skip
    print(f"{task} root {entropy}, {len(ns)} episodes {ns}; mean bn USD an episode")
    print(f"{'':28s}" + "".join(f"{g:>28s}" for g in GROUPS) + f"{'all items':>12s}")
    for name, a in rows.items():
        line = f"{name:28s}" + "".join(f"{a[:, [ITEMS.index(i) for i in items]].sum(axis=1).mean():28.1f}" for items in GROUPS.values())
        line += f"{a.sum(axis=1).mean():12.1f}"
        if name in J:
            line += f"   (played cost {J[name].mean():.1f}: the salvage at the end is {(a.sum(axis=1) - J[name]).mean():.1f})"
        print(line)
    steps = [("model -> told the window", "h3c_f", "truthallc_f"), ("told the window -> told all", "truthallc_f", "tah0c_f"),
             ("told all (kept) -> descent's plan", "tah0c_f", "descent's plan"),
             ("told all (here) -> descent's plan", "told all, played here", "descent's plan")]  # fmt: skip
    print("steps, what the later rung saves, by item:")
    for name, a, b in steps:
        d = rows[a] - rows[b]
        print(f"{name:34s}" + "".join(f"{g}: {d[:, [ITEMS.index(i) for i in items]].sum(axis=1).mean():+6.1f}   " for g, items in GROUPS.items())
              + f"all {d.sum(axis=1).mean():+6.1f}")
    d = rows["told all, played here"] - rows["descent's plan"]
    print("told all (here) -> descent's plan, by episode: " + "; ".join(
        f"ep {n}: unmet {d[i, 5]:+.0f}, shed {d[i, 7]:+.0f}, holding {d[i, 3] + d[i, 4]:+.0f}, disposal {d[i, 6]:+.0f}, freight etc. {d[i, :3].sum():+.0f}"
        for i, n in enumerate(ns)))


if __name__ == "__main__":
    fire.Fire(main)
