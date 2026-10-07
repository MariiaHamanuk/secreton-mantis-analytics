"""Scores of saved weekly arrays on the same episodes of root 444, paired to the base arrays.

    uv run python lab/anastasiia/plan_lab/scores.py full 8 <base arrays.npz> plan=<arrays.npz> other=<arrays.npz>

The scale and the 90% interval are the harness's (``harness.pooled``, ``harness.interval``); nothing is played.
"""

import sys

import numpy as np


sys.path.insert(0, "lab/anastasiia/plan_lab")
import harness


task, n, hubf = sys.argv[1], int(sys.argv[2]), sys.argv[3]
files = dict(a.split("=", 1) for a in sys.argv[4:])
refs = harness.references(task, 444, n)
level = np.array([r["stratum"] for r in refs])
naive = np.array([r["J_naive_cents"] for r in refs], dtype=float) / 100
room = naive - np.array([r["J_oracle_cents"] for r in refs], dtype=float) / 100
hub = np.load(hubf)["J"][:n]
print(
    f"{task} 444 eps 0-{n - 1}: hub {harness.pooled(level, naive - hub, room):.4f}; 0.01 = {room.mean() / 1e11:.1f} bn an episode"
)
for label, f in files.items():
    J = np.load(f)["J"][:n]
    d, lo, hi = harness.interval(level, naive - J, naive - hub, room)
    print(
        f"   {label:22} {harness.pooled(level, naive - J, room):.4f}  {d:+.4f} ({lo:+.4f} .. {hi:+.4f})  {np.mean(J - hub) / 1e9:+7.1f} bn/ep  better {int((J < hub).sum())}/{n}"
    )
