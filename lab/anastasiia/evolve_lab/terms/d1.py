"""Fragile-edge price: flow dispatched on an edge that is open now but loses capacity later in the window.

Mechanism: the point forecast keeps a cut as it is, so cargo committed to an edge whose forecast capacity u falls
after week t looks as safe as cargo on a steady edge. Price per unit = 4 % of (edge freight + 1 % of the
commodity's value), times (tau + 1) weeks committed, times the share of capacity lost later (0..1). Expected
effect: tie-breaks toward steady edges and shorter commitments when routes are otherwise equal; tiny against real prices.
"""

import numpy as np


RATE = 0.04
VALUE_SHARE = 0.01


def _build(ep):
    inst, mk = ep.inst, ep.marks
    u = np.asarray(mk.u, dtype=float)
    out = np.zeros(ep.N)
    nweeks = min(ep.T, u.shape[0])
    for key in sorted(ep.tm, key=repr):
        if key[0] != "x":
            continue
        e, k = key[1], key[2]
        if e >= u.shape[1]:
            continue
        edge = inst.edges[e]
        unit = float(edge.c0) + VALUE_SHARE * float(inst.commodities[k].v)
        j = ep.tm[key]
        for t in range(1, nweeks):
            now = u[t - 1, e]
            if not np.isfinite(now) or now <= 0.0:
                continue
            later = np.min(u[t:nweeks, e])
            if not np.isfinite(later):
                continue
            lost = max(0.0, 1.0 - later / now)
            if lost <= 0.0:
                continue
            idx = (t - 1) * ep.nc + j
            if idx < ep.N:
                out[idx] = RATE * unit * (int(edge.tau) + 1) * lost
    return out


def add(ep, mode, ref):
    v = getattr(ep, "_d1", None)
    if v is None:
        v = _build(ep)
        v = np.where(np.isfinite(v), v, 0.0)
        ep._d1 = v
    return v.copy()
