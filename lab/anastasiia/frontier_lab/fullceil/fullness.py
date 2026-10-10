"""Whole weeks and what the fabs make of them, by grid: the kept plays of the model and of the told planners and the
descent's plan (notes/u_fullceil.md). Reads kept files; builds each episode's marks for the base load.

    uv run python lab/anastasiia/frontier_lab/fullceil/fullness.py

By grid with fabs, mean over the episodes: whole weeks (no base load shed), the lots started in bn USD at the chip's
penalty, and the lots as a share of what the grid's fabs can start in its whole weeks ("fill").
"""

import pickle

import common as K
import fire
import numpy as np
from descend import run_path


def main(only: str | tuple | int = "", label: str = "tah0_f", task: str = "full", entropy: int = 444, first: int = 0,
         episodes: int = 16, by_episode: str = "") -> None:
    """``by_episode``: a grid's name; its numbers are printed for every episode too."""
    import core  # regime_lab's

    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else list(range(first, first + episodes))
    ns = [n for n in ns if run_path(label, task, entropy, n).is_file() and pickle.loads(run_path(label, task, entropy, n).read_bytes()).get("done")]
    plays = {tag: K.kept(tag, task, entropy) for tag in ("h3c_f", "truthallc_f", "tah0c_f")}
    tags = list(plays) + ["told all, here", "descent's plan"]
    inst = None
    acc = {}
    for n in ns:
        inst, marks = core.world(task, entropy, n)
        N = inst.nodes
        ybar = np.asarray(marks.y_bar, dtype=float)
        alpha = np.asarray(marks.alpha_bar, dtype=float)
        packed, pi = {}, {}
        for o in inst.osats:
            packed.update(N[o].osat.packages)
        for d in inst.demands:
            pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
        worth = np.array([pi.get(packed.get(N[f].fab.product, -1), 0.0) for f in inst.fabs])
        cap = np.array([N[f].fab.cap0 for f in inst.fabs])
        run = pickle.loads(run_path(label, task, entropy, n).read_bytes())
        data = {tag: (d[n]["shed"], d[n]["lots"]) for tag, d in plays.items()}
        data["told all, here"] = (run["weeks0"]["shed"], run["weeks0"]["lots"])
        data["descent's plan"] = (run["weeks"]["shed"], run["weeks"]["lots"])
        for gi, fabs in enumerate(inst.grid_fabs):
            if not fabs:
                continue
            fabs = list(fabs)
            for tag, (shed, lots) in data.items():
                whole = shed[:, gi] <= 1e-6 * np.maximum(1.0, ybar[: len(shed), gi])
                value = float((lots[:, fabs] * worth[fabs]).sum()) / 1e9
                room = float((whole[:, None] * alpha[: len(shed)][:, fabs] * cap[fabs] * worth[fabs]).sum()) / 1e9
                in_whole = float((whole[:, None] * lots[:, fabs] * worth[fabs]).sum()) / 1e9
                acc.setdefault((gi, tag), []).append((int(whole.sum()), value, room, in_whole))
    names = [inst.nodes[g].id for g in inst.grids]
    print(f"{task} root {entropy}, {len(ns)} episodes {ns}; by grid: whole weeks / lots, bn at the chip's penalty / fill of the whole weeks")
    print(f"{'grid':10s}" + "".join(f"{tag:>26s}" for tag in tags))
    for gi, fabs in enumerate(inst.grid_fabs):
        if not fabs:
            continue
        line = f"{names[gi]:10s}"
        for tag in tags:
            a = np.array(acc[(gi, tag)])
            line += f"{a[:, 0].mean():9.1f} {a[:, 1].mean():7.0f} {a[:, 3].sum() / max(1e-9, a[:, 2].sum()):7.2f} "
        print(line)
    line = f"{'all':10s}"
    for tag in tags:
        a = np.array([np.sum([acc[(gi, tag)][i] for gi, fabs in enumerate(inst.grid_fabs) if fabs], axis=0) for i in range(len(ns))])
        line += f"{a[:, 0].mean():9.1f} {a[:, 1].mean():7.0f} {a[:, 3].sum() / max(1e-9, a[:, 2].sum()):7.2f} "
    print(line)
    if by_episode:
        gi = names.index(by_episode)
        print(f"{by_episode} by episode: whole weeks / lots bn / fill")
        for i, n in enumerate(ns):
            print(f"  ep {n:2d} " + "".join(f"{acc[(gi, tag)][i][0]:9d} {acc[(gi, tag)][i][1]:7.0f} "
                                          f"{acc[(gi, tag)][i][3] / max(1e-9, acc[(gi, tag)][i][2]):7.2f} " for tag in tags))


if __name__ == "__main__":
    fire.Fire(main)
