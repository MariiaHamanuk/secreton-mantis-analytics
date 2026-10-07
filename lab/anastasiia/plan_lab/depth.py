"""Fab energy and shed load by how short a grid is.

    uv run python lab/anastasiia/plan_lab/depth.py full 8 hub=<arrays.npz> truth=<arrays.npz> plan=<arrays.npz>

Grid-weeks are classed by the first agent's shed share of the base load, averaged over the 13 weeks around: under
2 %, 2 to 10 %, over 10 %. For every class and agent: the energy its fabs took and the load shed, TWh per episode.
"""

import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator


task, n = sys.argv[1], int(sys.argv[2])
agents = dict(a.split("=", 1) for a in sys.argv[3:])
inst, _ = task_generator(task)
N = inst.nodes
K = [c.id for c in inst.commodities]
T = inst.T
A = {k: dict(np.load(v)) for k, v in agents.items()}
names = list(A)
H = A[names[0]]
voll = np.array([N[g].grid.voll for g in inst.grids])
edges_ = [0.0, 0.02, 0.10, 1.01]
labels = ["< 2 %", "2-10 %", "> 10 %"]
print(
    f"{task}: {n} episodes; grid-weeks classed by the first agent's shed share of base load averaged over 13 weeks around"
)
print("per episode: fab energy TWh, shed TWh, and the worth of both (shed at VOLL; fab energy at pi per lot), bn USD")
tot = {}
for gi, g in enumerate(inst.grids):
    fabs = list(inst.grid_fabs[gi])
    if not fabs:
        continue
    frac = H["shed"][:n][:, :, gi] / H["y_bar"][:n][:, :, gi]
    k = np.ones(13) / 13
    smooth = np.array([np.convolve(np.pad(r, 6, mode="edge"), k, mode="valid") for r in frac])
    for c in range(3):
        m = (smooth >= edges_[c]) & (smooth < edges_[c + 1])
        row = [m.sum() / n]
        for a in names:
            e = (A[a]["energy"][:n][:, :, fabs].sum(axis=2) * m).sum() / n / 1e3
            s = (A[a]["shed"][:n][:, :, gi] * m).sum() / n / 1e3
            row += [e, s]
        tot.setdefault(c, np.zeros(len(row)))
        tot[c] += np.array(row)
        print(
            f"{N[g].id:9} {labels[c]:7} weeks {row[0]:5.1f} | "
            + " | ".join(f"{a} fabE {row[1 + 2 * i]:6.2f} shed {row[2 + 2 * i]:7.2f}" for i, a in enumerate(names))
        )
for c in range(3):
    row = tot[c]
    print(
        f"{'all':9} {labels[c]:7} weeks {row[0]:5.1f} | "
        + " | ".join(f"{a} fabE {row[1 + 2 * i]:6.2f} shed {row[2 + 2 * i]:7.2f}" for i, a in enumerate(names))
    )
