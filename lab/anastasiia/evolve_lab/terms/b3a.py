"""Refinement of b3: price fuel burnt in an OFF week only for the fuel that the next closing week will lack.

Parent b3 charged every fuel segment of an OFF fab-grid week. Here a segment is charged only when (1) its burn is a
choice in the OFF week (fuel regime R or S; under F the burn is forced and the price is a constant) and (2) the same
fuel is in regime R or S in the next closing week (ON/SOFT/HULL), i.e. it is the scarce fuel that week needs. The
weight falls with the distance to that week (1.0 within 2 weeks, 0.7 within 4, 0.4 beyond). Expected: fuel is saved
for the closable week exactly where the fuel is the gate, without distorting other fuels.
"""

import numpy as np


SHARE = 0.06


def add(ep, mode, ref):
    inst = ep.inst
    fab_grid = getattr(ep, "_b3a", None)
    if fab_grid is None:
        fab_grid = ep._b3a = [any(inst.nodes[inst.fabs[fi]].fab.e > 0 for fi in inst.grid_fabs[gi])
                              for gi in range(len(inst.grids))]
    out = np.zeros(ep.N)
    grid, fuel = mode["grid"], mode["fuel"]
    hit = False
    for gi, g in enumerate(inst.grids):
        if not fab_grid[gi]:
            continue
        ga = inst.nodes[g].grid
        nxt = None
        dist = [0] * (ep.T + 2)
        for t in range(ep.T, 0, -1):
            dist[t] = nxt if nxt is not None else 0
            if grid.get((t, gi)) in ("ON", "SOFT", "HULL"):
                nxt = 1
            elif nxt is not None:
                nxt += 1
        for t in range(1, ep.T + 1):
            if grid.get((t, gi)) != "OFF" or dist[t] <= 0:
                continue
            d = dist[t]
            w = 1.0 if d <= 2 else 0.7 if d <= 4 else 0.4
            for k in ga.fuels:
                if not ep.has("G", gi, k):
                    continue
                if fuel.get((t, gi, k)) not in ("R", "S"):
                    continue
                if fuel.get((t + d, gi, k)) not in ("R", "S"):
                    continue
                out[ep.col("G", t, gi, k)] += SHARE * ga.voll * w
                hit = True
    return out if hit else None
