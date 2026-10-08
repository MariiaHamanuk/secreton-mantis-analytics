"""Per-episode detail from ldiag's npz: shortage a-bf by market, lots by fab (agent / bf / clairv), shed a-bf by grid."""

import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator

inst, _ = task_generator("small")
N, K = inst.nodes, [c.id for c in inst.commodities]
for n in [int(x) for x in sys.argv[1].split(",")]:
    z = np.load(f"outputs/level1/diag/ep{n}.npz")
    print(f"ep {n}: a-c {(z['a_J'] - z['c_J']) / 1e9:.1f}  a-bf {(z['a_J'] - z['b_J']) / 1e9:.1f}  gap {float(z['b_gap']):.4f}")
    s = " ".join(f"{N[d.node].id[5:]}/{K[d.k][5:]}:{(z['a_short_usd'][:, i].sum() - z['b_short_usd'][:, i].sum()) / 1e9:.0f}"
                 for i, d in enumerate(inst.demands) if z["a_short_usd"][:, i].sum() > 0)
    print("   shortage a-bf by market:", s)
    print("   lots wk1-40 (k/wk) a/bf/c:", " ".join(
        f"{N[f].id[4:]}:{z['a_lots'][:40, i].mean() / 1e3:.0f}/{z['b_lots'][:40, i].mean() / 1e3:.0f}/{z['c_lots'][:40, i].mean() / 1e3:.0f}"
        for i, f in enumerate(inst.fabs)))
    print("   shed a-bf by grid:", " ".join(
        f"{N[g].id[5:]}:{(z['a_shed_usd'][:, i].sum() - z['b_shed_usd'][:, i].sum()) / 1e9:.0f}" for i, g in enumerate(inst.grids)))
