"""Strait exposure: queueing and lane dispatch priced by how much the strait's openness drops later in the window.

Mechanism: cargo sent into a strait lane (x with a lane) or waiting in its queue Q is exposed to a closure the
point forecast treats as fixed. Lane dispatch pays 5 % of (edge freight + 1 % of value) per week of lane travel,
times the largest later fall of the open fraction o along the lane. Queues pay the strait's holding cost h_queue
times 0.15 times a ramp rising from 1 to 3 across the window, so late queueing is dearer than early. Effect: less
speculative committing to straits that are forecast to close, mild earlier clearing; tie-breaker scale.
"""

import numpy as np


RATE = 0.05
VALUE_SHARE = 0.01
QSHARE = 0.15


def _build(ep):
    inst, mk = ep.inst, ep.marks
    o = np.asarray(mk.o, dtype=float)
    hq = np.asarray(mk.h_queue, dtype=float)
    out = np.zeros(ep.N)
    nweeks = min(ep.T, o.shape[0])
    ordinal = inst.chokepoint_ordinal
    for key in sorted(ep.tm, key=repr):
        j = ep.tm[key]
        if key[0] == "x" and key[3] is not None:
            e, k, lane = key[1], key[2], key[3]
            cps = [ordinal[c] for c in inst.lanes[lane].chokepoints if c in ordinal]
            if not cps:
                continue
            ltau = sum(int(inst.edges[x].tau) for x in inst.lanes[lane].edges)
            unit = float(inst.edges[e].c0) + VALUE_SHARE * float(inst.commodities[k].v)
            for t in range(1, nweeks):
                drop = 0.0
                for c in cps:
                    now = o[t - 1, c]
                    if np.isfinite(now) and now > 0.0:
                        drop = max(drop, 1.0 - float(np.min(o[t:nweeks, c])) / now)
                idx = (t - 1) * ep.nc + j
                if drop > 0.0 and idx < ep.N:
                    out[idx] = RATE * unit * max(1, ltau) * drop
        elif key[0] == "Q":
            c, k = key[1], key[2]
            if c not in ordinal or k >= hq.shape[2]:
                continue
            co = ordinal[c]
            for t in range(1, nweeks + 1):
                h = float(hq[t - 1, co, k])
                ramp = 1.0 + 2.0 * (t - 1) / max(1, ep.T - 1)
                idx = (t - 1) * ep.nc + j
                if idx < ep.N and np.isfinite(h) and h > 0.0:
                    out[idx] = QSHARE * h * ramp
    return out


def add(ep, mode, ref):
    v = getattr(ep, "_d2", None)
    if v is None:
        v = _build(ep)
        v = np.where(np.isfinite(v), v, 0.0)
        ep._d2 = v
    return v.copy()
