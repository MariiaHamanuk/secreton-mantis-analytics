"""Packaged chips at the plants: what the exits could carry and what each agent shipped.

    uv run python lab/anastasiia/plan_lab/exits.py full 8 chip_le hub=<arrays.npz> truth=<arrays.npz> plan=<arrays.npz>

The arrays are weekly records of the same episodes (``harness.py --save``, ``regime.py run --save``). What a lane
could carry in a week is taken as the largest shipment of any of the agents on that lane and week (disruptions are the
same for all of them), so it is a lower bound. A week counts as starved for an agent when its plant shipped all it
held; the rest of the shortfall is idle (stock was there). M chips per episode.
"""

import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator


task, n, chip = sys.argv[1], int(sys.argv[2]), sys.argv[3]
agents = dict(a.split("=", 1) for a in sys.argv[4:])
inst, _ = task_generator(task)
N = inst.nodes
K = [c.id for c in inst.commodities]
T = inst.T
A = {k: dict(np.load(v)) for k, v in agents.items()}
names = list(A)
pk = K.index(chip)
print(f"{task}: {n} episodes, {chip}, M chips per episode")
print(f"{'plant':10}{'':28}" + "".join(f"{x:>9}" for x in names))
tot = {key: np.zeros(len(names)) for key in ("could", "shipped", "starved", "idle", "thrown", "buffer")}
for o in inst.osats:
    if pk not in N[o].osat.packages.values():
        continue
    sl = [s for s, x in enumerate(inst.stock_slots) if x.k == pk and x.node == o][0]
    lanes = [ai for ai, (ei, k, lane) in enumerate(inst.action_slots) if inst.edges[ei].tail == o and k == pk]
    cap = np.max([A[a]["sent"][:n][:, :, lanes] for a in names], axis=0)  # (n, T, lanes)
    res = {}
    for i, a in enumerate(names):
        sent = A[a]["sent"][:n][:, :, lanes]
        stock = A[a]["stock"][:n][:, :, sl]
        before = np.concatenate([np.full((n, 1), np.inf), stock[:, :-1]], axis=1)  # stock at the start of the week
        short = (cap - sent).sum(axis=2)
        starved = before - sent.sum(axis=2) < 0.02 * np.maximum(cap.sum(axis=2), 1.0)
        res[a] = (
            cap.sum() / n / 1e6,
            sent.sum() / n / 1e6,
            (short * starved).sum() / n / 1e6,
            (short * ~starved).sum() / n / 1e6,
            A[a]["disposal"][:n][:, :, sl].sum() / n / 1e6,
            stock.mean() / 1e6,
        )
        for key, v in zip(tot, res[a]):
            tot[key][i] += v
    for j, key in enumerate(tot):
        print(f"{N[o].id if j == 0 else '':10}{key:28}" + "".join(f"{res[a][j]:>9.2f}" for a in names))
for key in tot:
    print(f"{'all':10}{key:28}" + "".join(f"{v:>9.2f}" for v in tot[key]))
