"""Price on useless fuel burn: fuel burnt in an OFF week of a fab-grid that will have a closing week later.

In an OFF week the fabs get nothing, so fuel burnt only serves base load at voll; if the window holds a later
week asked to be whole (ON, SOFT, HULL) that fuel is worth more kept. A charge of 6 % of the grid's voll per GWh
burnt by each fuel segment (full weight if the next such week is within 4 weeks, half beyond) nudges the program to
save fuel for it. Gated by state (regimes of later weeks), not by date; grids without fabs are untouched.
"""

import numpy as np


SHARE = 0.06


def add(ep, mode, ref):
    inst = ep.inst
    fab_grid = getattr(ep, "_b3", None)
    if fab_grid is None:
        fab_grid = ep._b3 = [any(inst.nodes[inst.fabs[fi]].fab.e > 0 for fi in inst.grid_fabs[gi])
                             for gi in range(len(inst.grids))]
    out = np.zeros(ep.N)
    grid = mode["grid"]
    hit = False
    for gi, g in enumerate(inst.grids):
        if not fab_grid[gi]:
            continue
        ga = inst.nodes[g].grid
        nxt = None  # distance to the next closing week, walking back from the end
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
            w = 1.0 if dist[t] <= 4 else 0.5
            for k in ga.fuels:
                if ep.has("G", gi, k):
                    out[ep.col("G", t, gi, k)] += SHARE * ga.voll * w
                    hit = True
    return out if hit else None
