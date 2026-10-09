"""State-dependent price on shed base load in weeks asked off or short (OFF, HULL), never on a date alone.

Grids without a fab that draws energy: shed is plain linear loss, so a premium (5 % of the grid's voll) on near
weeks, decaying with the week, makes the program serve what is certain before what is forecast. Grids with fabs:
the premium is 5 % of voll times the squared forecast closeability min(1, G_bar / (y_bar + full fab draw)), so a
grid the forecast says could nearly close does not shed lightly in its short weeks, while a hopeless one is left alone.
"""

import numpy as np


A = 0.05


def _cache(ep):
    inst = ep.inst
    near = np.zeros((ep.T + 1, len(inst.grids)))
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fi for fi in inst.grid_fabs[gi] if inst.nodes[inst.fabs[fi]].fab.e > 0]
        for t in range(1, ep.T + 1):
            ti = t - 1
            if not fabs:
                near[t, gi] = A * ga.voll * float(np.exp(-(t - 1) / 3.0))
                continue
            R = ep.marks.R[ti]
            tot = 0.0
            for fi in fabs:
                if R[fi] > 0 and ep.has("p", fi):
                    tot += inst.nodes[inst.fabs[fi]].fab.e * float(ep.m.ub[ep.col("p", t, fi)]) / float(R[fi])
            need = float(ep.marks.y_bar[ti][gi]) + tot
            h = min(1.0, max(0.0, float(ep.marks.G_bar[ti][gi]) / need)) if need > 0 else 0.0
            near[t, gi] = A * ga.voll * h * h
    return near


def add(ep, mode, ref):
    near = getattr(ep, "_b2", None)
    if near is None:
        near = ep._b2 = _cache(ep)
    out = np.zeros(ep.N)
    grid = mode["grid"]
    hit = False
    for gi in range(len(ep.inst.grids)):
        if not ep.has("ysh", gi):
            continue
        for t in range(1, ep.T + 1):
            if grid.get((t, gi)) in ("OFF", "HULL") and near[t, gi] > 0:
                out[ep.col("ysh", t, gi)] += near[t, gi]
                hit = True
    return out if hit else None
