"""Runs of closed weeks: a small reward on lots started in a fab-grid's week that follows a closed (ON) week.

Targets scattered closed weeks: a grid whose week t-1 is already "ON" has its fuel and stocks warm, so a week that
is still free (MID, SOFT, HULL) is worth finishing as a whole week; the reward makes the program lean that way, so
closed weeks come in runs instead of isolated spikes. Depends on the regime of the cell, not on the date.
Magnitude: 3 % of the customs value v of the fab's raw chip per lot (instance's own commodity value), well below
the 25-40 M USD per GWh the fabs' energy is worth, so it only breaks near-ties.
"""

import numpy as np


SHARE = 0.03


def add(ep, mode, ref):
    inst = ep.inst
    cache = getattr(ep, "_b1", None)
    if cache is None:
        value = {}
        for fi, f in enumerate(inst.fabs):
            fa = inst.nodes[f].fab
            if fa.e > 0 and fa.grid is not None:
                value[fi] = SHARE * float(inst.commodities[fa.product].v)
        cache = ep._b1 = value
    if not cache:
        return None
    out = np.zeros(ep.N)
    grid = mode["grid"]
    hit = False
    for gi in range(len(inst.grids)):
        for t in range(2, ep.T + 1):
            if grid.get((t - 1, gi)) != "ON" or grid.get((t, gi)) not in ("MID", "SOFT", "HULL"):
                continue
            for fi in inst.grid_fabs[gi]:
                if fi in cache and ep.has("p", fi):
                    out[ep.col("p", t, fi)] -= cache[fi]
                    hit = True
    return out if hit else None
