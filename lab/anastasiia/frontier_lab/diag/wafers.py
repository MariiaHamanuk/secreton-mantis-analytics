"""Lots a fab did not start for want of wafers in weeks its grid served the whole base load, and the sales beside them.

    uv run python lab/anastasiia/frontier_lab/diag/wafers.py

A week counts when the fab's grid sheds nothing, the fab ends it with no wafers and starts under 0.95 of its capacity,
and a lot started then could still be sold (the fab's own lead time and six weeks of packing and travel before the
episode ends). The lots missing to the capacity are valued at the chip's penalty: the worth of the fab's idle top, not
of sales lost, for a chip also needs a plant and an exit. Printed: the worth per fab and play, and per episode the
model's and the told plays' worth at the fab named beside their unmet demand for the fab's chip, with the slope of
one difference on the other across the episodes (how much of the idle worth turns into sales when a play removes it).
"""

import fire
import numpy as np
from common import BN, Names, kept, level_weights, refs


def idle(nm: Names, e: dict, worth: np.ndarray) -> np.ndarray:
    """(F,) bn USD: lots short of capacity in wafer-starved weeks of a grid that sheds nothing."""
    inst, N = nm.inst, nm.inst.nodes
    out = np.zeros(len(inst.fabs))
    for fo, f in enumerate(inst.fabs):
        fab, go = N[f].fab, nm.fab_grid[fo]
        last = e["lots"].shape[0] - (int(fab.tau) + 6)
        wafers = e["stock"][:last, inst.slot_index[(f, fab.input)]]
        full = e["shed"][:last, go] < 1e-6 * N[inst.grids[go]].grid.base_load
        starved = full & (wafers < 0.01 * fab.cap0) & (e["lots"][:last, fo] < 0.95 * fab.cap0)
        out[fo] = ((fab.cap0 - e["lots"][:last, fo]) * starved).sum() * worth[fo] / BN
    return out


def main(tags: str = "h3c_f,truthallc_f,tah0c_f", fab: str = "fab_kr_memory_1", episodes: int = 16) -> None:
    nm = Names("full")
    inst, N = nm.inst, nm.inst.nodes
    names = tags.split(",")
    R = refs()
    lvl = np.array([R[n]["stratum"] for n in range(episodes)])
    w = level_weights(lvl)
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
    for o in inst.osats:
        for raw, packed in N[o].osat.packages.items():
            pi[raw] = max(pi.get(raw, 0.0), pi[packed])
    worth = np.array([pi[N[f].fab.product] for f in inst.fabs])
    V = {t: np.array([idle(nm, kept(t)[n], worth) for n in range(episodes)]) for t in names}  # (episodes, F)
    print(
        f"1. Idle worth for want of wafers, bn USD an episode, Full 444 episodes 0-{episodes - 1} (plain mean; all fabs weighted)"
    )
    for fo in np.argsort(-V[names[0]].mean(axis=0))[:8]:
        print(f"   {nm.fabs[fo]:18s} " + "  ".join(f"{t} {V[t][:, fo].mean():6.1f}" for t in names))
    print(
        f"   {'all fabs':18s} "
        + "  ".join(f"{t} {V[t].sum(axis=1).mean():6.1f} ({np.sum(w * V[t].sum(axis=1)):.1f})" for t in names)
    )

    fo = nm.fabs.index(fab)
    raw = N[inst.fabs[fo]].fab.product
    packed = next(p for o in inst.osats for r, p in N[o].osat.packages.items() if r == raw)
    sinks = nm.demand_k == packed
    lost = {
        t: np.array([(kept(t)[n]["lost"][:, sinks] * nm.penalty[sinks]).sum() for n in range(episodes)]) / BN
        for t in names
    }
    base = names[0]
    print(f"\n2. {fab}: idle worth and unmet demand for {nm.K[packed]}, bn USD, per episode ({' / '.join(names)})")
    for n in range(episodes):
        print(
            f"   ep {n:2d} L{lvl[n]}: idle "
            + " / ".join(f"{V[t][n, fo]:6.1f}" for t in names)
            + "   unmet "
            + " / ".join(f"{lost[t][n]:7.1f}" for t in names)
        )
    for t in names[1:]:
        dx, dy = V[base][:, fo] - V[t][:, fo], lost[base] - lost[t]
        slope = float(np.sum((dx - dx.mean()) * (dy - dy.mean())) / np.sum((dx - dx.mean()) ** 2))
        print(
            f"   {base} less {t}: idle worth {dx.mean():+.1f}, unmet demand {dy.mean():+.1f} bn an episode; over the episodes "
            f"correlation {np.corrcoef(dx, dy)[0, 1]:+.2f}, slope {slope:+.2f} of unmet demand a unit of idle worth"
        )


if __name__ == "__main__":
    fire.Fire(main)
