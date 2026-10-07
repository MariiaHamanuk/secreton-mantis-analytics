"""Grid-episodes where a plan closes fewer weeks than the true-future plan, and what they hold.

    uv run python lab/anastasiia/plan_lab/closed.py full 8 hub=<arrays.npz> truth=<arrays.npz> plan=<arrays.npz>

Three agents in this order: the base, the true-future plan, the plan without foresight. A week of a grid is complete
when nothing is shed (only then its fabs get power). For the grid-episodes where the third agent closes at least 5
weeks fewer than the second, and for the rest: complete weeks, shed cost and the lots started priced at their chip's
penalty (an upper bound on what the lots sell for), per episode.
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
names = list(A)  # hub, truth, causal
pi = np.zeros(len(K))
for d in inst.demands:
    pi[d.k] = max(pi[d.k], d.pi)
packed_of = {raw: pk for o in inst.osats for raw, pk in N[o].osat.packages.items()}
rows = []
for ep in range(n):
    for gi, g in enumerate(inst.grids):
        fabs = list(inst.grid_fabs[gi])
        if not fabs:
            continue
        comp = [int((A[a]["shed"][ep, :, gi] < 1e-6).sum()) for a in names]
        shed = [A[a]["shed"][ep, :, gi].sum() * N[g].grid.voll / 1e9 for a in names]
        worth = [
            sum(A[a]["lots"][ep, :, f].sum() * pi[packed_of[N[inst.fabs[f]].fab.product]] for f in fabs) / 1e9
            for a in names
        ]
        rows.append((ep, N[g].id, comp, shed, worth))
lost = [r for r in rows if r[2][2] <= r[2][1] - 5]
kept = [r for r in rows if r not in lost]
print(f"{task}: {n} episodes, {len(rows)} grid-episodes; order: " + " / ".join(names))
for label, group in (("causal closes 5+ weeks fewer than the true-future plan", lost), ("the rest", kept)):
    c = np.array([r[2] for r in group]).sum(axis=0) / n
    s = np.array([r[3] for r in group]).sum(axis=0) / n
    w = np.array([r[4] for r in group]).sum(axis=0) / n
    print(
        f"{label}: {len(group)} grid-episodes; per episode: complete weeks "
        + "/".join(f"{v:.1f}" for v in c)
        + "; shed bn "
        + "/".join(f"{v:.0f}" for v in s)
        + "; lots at their chip's penalty, bn "
        + "/".join(f"{v:.0f}" for v in w)
    )
    print(
        f"   causal minus hub: shed {s[2] - s[0]:+.1f} bn, lots' worth {w[2] - w[0]:+.1f} bn | truth minus hub: shed {s[1] - s[0]:+.1f}, lots' worth {w[1] - w[0]:+.1f}"
    )
for r in lost:
    print(
        f"   ep {r[0]:2d} {r[1]:9} complete "
        + "/".join(f"{v:3d}" for v in r[2])
        + "  shed bn "
        + "/".join(f"{v:6.0f}" for v in r[3])
        + "  lots' worth bn "
        + "/".join(f"{v:6.0f}" for v in r[4])
    )
