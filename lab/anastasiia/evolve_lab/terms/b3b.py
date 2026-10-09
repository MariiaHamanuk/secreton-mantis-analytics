"""Refinement of b3: price OFF-week fuel burn only while the grid's stock of that fuel is short of the closing week.

Parent b3 charged all fuel burnt in an OFF fab-grid week. Here, for the next closing week (ON/SOFT/HULL, within 6
weeks), the need of fuel k is its share of the forecast generation cap; the opening stock of the window is the
proxy for what is on hand. The price (6 % of voll per GWh) is scaled by 0.25 + 0.75 * deficit, deficit = share of the
need not covered by stock, and is zero when stock covers the need. Expected: saving only where fuel is truly short.
"""

import numpy as np


SHARE = 0.06


def add(ep, mode, ref):
    inst = ep.inst
    fab_grid = getattr(ep, "_b3b", None)
    if fab_grid is None:
        fab_grid = ep._b3b = [any(inst.nodes[inst.fabs[fi]].fab.e > 0 for fi in inst.grid_fabs[gi])
                              for gi in range(len(inst.grids))]
    out = np.zeros(ep.N)
    grid = mode["grid"]
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
            if grid.get((t, gi)) != "OFF" or dist[t] <= 0 or dist[t] > 6:
                continue
            t2 = t + dist[t]
            gbar = float(ep.marks.G_bar[t2 - 1][gi])
            for k in ga.fuels:
                if not ep.has("G", gi, k) or (g, k) not in inst.slot_index:
                    continue
                need = float(ga.shares[k]) * gbar
                if need <= 0:
                    continue
                stock = float(ep.i0[ep.slot(g, k)])
                deficit = max(0.0, need - stock) / need
                if deficit <= 0:
                    continue
                out[ep.col("G", t, gi, k)] += SHARE * ga.voll * (0.25 + 0.75 * deficit)
                hit = True
    return out if hit else None
