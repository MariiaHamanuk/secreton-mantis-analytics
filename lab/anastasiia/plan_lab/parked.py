"""Fuel standing in the straits' queues: its mean stock, what is left there at the episode's end, what was lifted.

    uv run python lab/anastasiia/plan_lab/parked.py full 8 hub=<arrays.npz> plan=<arrays.npz>

Thousand units per episode, by fuel: lifted from the sources (sent on their lanes), standing at the straits on an
average week and at the end, burned in the grids. Fuel left at a strait at the end was lifted for nothing.
"""

import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator


def main(task: str, n: int, *agents: str) -> None:
    inst, _ = task_generator(task)
    N, E, K = inst.nodes, inst.edges, [c.id for c in inst.commodities]
    A = {a.split("=", 1)[0]: dict(np.load(a.split("=", 1)[1])) for a in agents}
    supply, chk = set(inst.supply_nodes), set(inst.chokepoints)
    fuels = sorted({k for g in inst.grids for k in N[g].grid.fuels})
    print(f"{task}: {n} episodes, thousand units per episode; order " + " / ".join(A))
    for k in fuels:
        lanes = [s for s, (e, kk, _lane) in enumerate(inst.action_slots) if kk == k and E[e].tail in supply]
        slots = [s for s, sl in enumerate(inst.stock_slots) if sl.k == k and sl.node in chk]
        rows = {"lifted": [], "at straits, mean": [], "at straits, end": [], "burned": []}
        for a in A.values():
            rows["lifted"].append(a["sent"][:n][:, :, lanes].sum() / n / 1e3)
            rows["at straits, mean"].append(a["stock"][:n][:, :, slots].sum(axis=2).mean() / 1e3)
            rows["at straits, end"].append(a["stock"][:n][:, -1, slots].sum() / n / 1e3)
            rows["burned"].append(a["segment"][:n][:, :, :, k].sum() / n / 1e3)
        for label, vals in rows.items():
            print(f"   {K[k]:8} {label:18}" + " / ".join(f"{v:7.1f}" for v in vals))
    queue = [a["cost"][:n][:, :, 4].sum() / n / 1e9 for a in A.values()]
    print(f"   queue holding, bn USD       " + " / ".join(f"{v:7.1f}" for v in queue))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), *sys.argv[3:])
