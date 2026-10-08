"""Lots by fab and week band, agent against the base-first plan; level gaps for the 20 L2-3 episodes (ldiag npz)."""

import glob
import json
import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator

inst, _ = task_generator("small")
N = inst.nodes
rows = {r["episode"]: r for r in json.load(open(sys.argv[1]))}
eps = sorted(int(p.split("ep")[-1][:-4]) for p in glob.glob("outputs/level1/diag/ep*.npz"))
Z = {n: np.load(f"outputs/level1/diag/ep{n}.npz") for n in eps}
B = ((0, 13), (13, 26), (26, 39))
for name, sel in (("L1", [n for n in eps if rows[n]["stratum"] == 1]),
                  ("L2-3", [n for n in eps if rows[n]["stratum"] in (2, 3)])):
    gap = np.mean([(rows[n]["J_naive_cents"] - rows[n]["J_clairvoyant_cents"]) / 1e11 for n in sel])
    print(f"{name}: {len(sel)} ep, mean naive-clairv gap {gap:.1f} bn")
    print("   lots a - bf, thousand per week, weeks 1-13 / 14-26 / 27-39 (agent level in brackets)")
    for i, f in enumerate(inst.fabs):
        d = [np.mean([Z[n]["a_lots"][lo:hi, i].mean() - Z[n]["b_lots"][lo:hi, i].mean() for n in sel]) / 1e3
             for lo, hi in B]
        a = [np.mean([Z[n]["a_lots"][lo:hi, i].mean() for n in sel]) / 1e3 for lo, hi in B]
        print(f"   {N[f].id:18s} " + " / ".join(f"{x:+6.1f} ({y:5.1f})" for x, y in zip(d, a)))
