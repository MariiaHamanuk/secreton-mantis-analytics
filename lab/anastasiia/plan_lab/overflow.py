"""Lots by whether the fab's plants were overflowing when the lot started.

    uv run python lab/anastasiia/plan_lab/overflow.py full 8 hub=<arrays.npz> truth=<arrays.npz> plan=<arrays.npz>

The mark is read off the first agent's records: a fab-week is "on" when a plant that takes at least 20 % of the fab's
raw chips threw packaged chips of its product away in the four weeks up to it. For every agent: the lots it started
in the fab-weeks on and off, M per episode. Shows where a plan cuts lots and where the true-future plan does.
"""

import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator


task, n = sys.argv[1], int(sys.argv[2])
agents = dict(a.split("=", 1) for a in sys.argv[3:])
inst, _ = task_generator(task)
N, K, T = inst.nodes, [c.id for c in inst.commodities], inst.T
A = {k: dict(np.load(v)) for k, v in agents.items()}
names = list(A)
H = A[names[0]]
pi = np.zeros(len(K))
for d in inst.demands:
    pi[d.k] = max(pi[d.k], d.pi)
packed_of = {raw: pk for o in inst.osats for raw, pk in N[o].osat.packages.items()}
tot = {"on": np.zeros(len(names)), "off": np.zeros(len(names))}
cells = {"on": 0, "off": 0}
by = {}
for fi, f in enumerate(inst.fabs):
    a = N[f].fab
    raw, pk = a.product, packed_of[a.product]
    to = {}  # plant -> slots carrying this fab's raw chips to it
    for ai, (ei, k, lane) in enumerate(inst.action_slots):
        if k == raw and inst.edges[ei].tail == f:
            edges = (ei,) if lane is None else inst.lanes[lane].edges
            to.setdefault(inst.edges[edges[-1]].head, []).append(ai)
    on = np.zeros((n, T), bool)
    for ep in range(n):
        sent = {o: H["sent"][ep][:, sl].sum() for o, sl in to.items()}
        total = sum(sent.values()) or 1.0
        for o, q in sent.items():
            if q / total < 0.2:
                continue
            slot = [s for s, x in enumerate(inst.stock_slots) if x.node == o and x.k == pk][0]
            over = H["disposal"][ep, :, slot] > 0.02 * N[o].osat.thr
            now = over.copy()
            for back in range(1, 4):
                now[back:] |= over[: T - back]
            on[ep] |= now
    for key, m in (("on", on), ("off", ~on)):
        row = np.array([(A[x]["lots"][:n][:, :, fi] * m).sum() / n / 1e6 for x in names])
        tot[key] += row
        cells[key] += m.sum() / n
        by.setdefault(K[pk], {"on": np.zeros(len(names)), "off": np.zeros(len(names))})[key] += row
print(
    f"{task}: {n} episodes; lots, M per episode, by whether a plant taking 20 %+ of the fab's chips overflowed in the 4 weeks before (in {names[0]}'s records)"
)
print(f"{'':28}" + "".join(f"{x:>10}" for x in names))
for chip, d in by.items():
    for key in ("on", "off"):
        print(
            f"{chip + ', plants ' + ('overflowing' if key == 'on' else 'not overflowing'):38}"
            + "".join(f"{v:>10.2f}" for v in d[key])
        )
for key in ("on", "off"):
    print(f"{'all, ' + key + f' ({cells[key]:.0f} fab-weeks)':38}" + "".join(f"{v:>10.2f}" for v in tot[key]))
