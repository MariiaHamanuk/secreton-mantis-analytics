"""How much of the chips an agent throws away could be seen coming from its own records.

    uv run python lab/anastasiia/plan_lab/waste.py full 8 outputs/plan_lab/20261007_step0/hub_full_444.npz

A chip thrown away in week u was a lot started about ``lead`` weeks before (the fab's weeks, a week of transport and
the plant's weeks for a packaged chip). It counts as seen if the same store (a fab's raw chips, a plant's packaged
chips) was already throwing chips away in the four weeks up to the lot's start. Per chip and kind of store: M chips
thrown away per episode, the share seen, and how often a store that overflows now still overflows ``lead`` weeks on.
"""

import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator


def main(task: str, n: int, path: str, floor: float = 0.02) -> None:
    inst, _ = task_generator(task)
    N, K, T = inst.nodes, [c.id for c in inst.commodities], inst.T
    A = dict(np.load(path))
    stores = []  # (label, slot, lead, weekly size that makes an overflow count)
    fab_lead = {}
    for f in inst.fabs:
        a = N[f].fab
        fab_lead[a.product] = max(fab_lead.get(a.product, 0), a.tau)
        slot = [s for s, x in enumerate(inst.stock_slots) if x.node == f and x.k == a.product][0]
        stores.append((f"fab {K[a.product]}", slot, a.tau, a.cap0))
    for o in inst.osats:
        for raw, packed in N[o].osat.packages.items():
            slot = [s for s, x in enumerate(inst.stock_slots) if x.node == o and x.k == packed][0]
            stores.append((f"plant {K[packed]}", slot, fab_lead[raw] + 1 + N[o].osat.tau, N[o].osat.thr))
    print(f"{task}: {n} episodes of {path}")
    print(
        f"{'store':20}{'thrown, M':>10}{'seen':>7}{'still overflows':>17}{'overflows unseen':>18}{'weeks overflowing':>19}"
    )
    groups = {}
    for label, slot, lead, size in stores:
        d = A["disposal"][:n][:, :, slot]  # (episodes, weeks)
        over = d > floor * size
        now = np.zeros_like(over)  # an overflow in the four weeks up to t
        for back in range(4):
            now[:, back:] |= over[:, : T - back]
        seen = np.zeros_like(over)  # week u: the store overflowed in the four weeks up to u - lead
        seen[:, lead:] = now[:, : T - lead]
        later = np.zeros_like(over)  # week t: the store overflows in the four weeks from t + lead
        for ahead in range(lead, lead + 4):
            if ahead < T:
                later[:, : T - ahead] |= over[:, ahead:]
        valid = np.zeros_like(over)
        valid[:, : T - lead - 3] = True
        g = groups.setdefault(label, np.zeros(7))
        g += [
            d.sum(),
            (d * seen).sum(),
            (now & later & valid).sum(),
            (now & valid).sum(),
            (~now & later & valid).sum(),
            (~now & valid).sum(),
            over.sum(),
        ]
    for label, g in groups.items():
        print(
            f"{label:20}{g[0] / n / 1e6:>10.2f}{g[1] / max(g[0], 1):>7.0%}{g[2] / max(g[3], 1):>17.0%}"
            f"{g[4] / max(g[5], 1):>18.0%}{g[6] / n:>19.1f}"
        )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], *(float(x) for x in sys.argv[4:5]))
