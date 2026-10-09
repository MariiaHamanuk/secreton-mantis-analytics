"""Earliness tilt among parallel routes: a small price on the extra transit time of the slower alternative.

Mechanism: edges with the same (tail, head), and strait lanes with the same ends, are substitutes of different
travel time; the point forecast trusts the whole path equally for weeks ahead. Each slower alternative pays
1 % of the commodity's value per extra week over the fastest one (lane price on its first edge only), scaled by
a window factor 1 + (t-1)/T so far weeks lean more to quick arrival. Effect: fuel arrives earlier when the
extra freight is nearly equal, leaving less cargo exposed to late cuts. Tie-breaker: 1 % of value per week.
"""

import numpy as np


RATE = 0.01


def _build(ep):
    inst = ep.inst
    out = np.zeros(ep.N)
    best_edge = {}
    for e, ed in enumerate(inst.edges):
        if ed.coupling:
            continue
        g = (ed.tail, ed.head)
        best_edge[g] = min(best_edge.get(g, int(ed.tau)), int(ed.tau))
    best_lane = {}
    lane_tau, lane_group = {}, {}
    for li, ln in enumerate(inst.lanes):
        if not ln.edges:
            continue
        g = (inst.edges[ln.edges[0]].tail, inst.edges[ln.edges[-1]].head)
        lane_group[li] = g
        lane_tau[li] = sum(int(inst.edges[x].tau) for x in ln.edges)
        best_lane[g] = min(best_lane.get(g, lane_tau[li]), lane_tau[li])
    for key in sorted(ep.tm, key=repr):
        if key[0] != "x":
            continue
        e, k, lane = key[1], key[2], key[3]
        ed = inst.edges[e]
        v = float(inst.commodities[k].v)
        if lane is None:
            if ed.coupling:
                continue
            extra = int(ed.tau) - best_edge.get((ed.tail, ed.head), int(ed.tau))
        else:
            if lane not in lane_group or inst.lanes[lane].edges[0] != e:
                continue
            extra = lane_tau[lane] - best_lane[lane_group[lane]]
        if extra <= 0:
            continue
        j = ep.tm[key]
        for t in range(1, ep.T + 1):
            idx = (t - 1) * ep.nc + j
            if idx < ep.N:
                out[idx] = RATE * v * extra * (1.0 + (t - 1) / max(1, ep.T))
    return out


def add(ep, mode, ref):
    v = getattr(ep, "_d3", None)
    if v is None:
        v = _build(ep)
        v = np.where(np.isfinite(v), v, 0.0)
        ep._d3 = v
    return v.copy()
