"""Refinement of d3: tilt toward the faster route for fuel commodities only.

Parent d3 charged 1 % of value per extra transit week of the slower parallel route for every commodity. Here only
commodities that a grid burns (union of every grid's fuels) are tilted; chips and other goods keep the base
program's choice. The window factor 1 + (t-1)/T is kept. Expected: early fuel arrival (where a late cut empties a
grid) without disturbing the routing of the rest.
"""

import numpy as np


RATE = 0.01


def _build(ep):
    inst = ep.inst
    out = np.zeros(ep.N)
    fuels = set()
    for g in inst.grids:
        for k in inst.nodes[g].grid.fuels:
            fuels.add(k)
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
        if k not in fuels:
            continue
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
    v = getattr(ep, "_d3a", None)
    if v is None:
        v = _build(ep)
        v = np.where(np.isfinite(v), v, 0.0)
        ep._d3a = v
    return v.copy()
