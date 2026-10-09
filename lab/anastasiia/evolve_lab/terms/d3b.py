"""Refinement of d3: the earliness tilt depends on the destination grid's regimes in `mode`, not on the date.

Parent d3 scaled the tilt by 1 + (t-1)/T. Here, for a flow into a grid node, the weight is 2.0 when a closing week
(ON/SOFT/HULL) of that grid falls between the fast route's arrival and the slow route's (so only the fast one is in
time), 1.0 when a closing week lies later than both arrivals, and 0.25 when no closing week follows. Flows into
non-grid nodes get 0.5. Rate 1 % of value per extra week. Expected: speed matters only when a week is to be closed.
"""

import numpy as np


RATE = 0.01


def _entries(ep):
    inst = ep.inst
    gidx = {g: gi for gi, g in enumerate(inst.grids)}
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
    ents = []
    for key in sorted(ep.tm, key=repr):
        if key[0] != "x":
            continue
        e, k, lane = key[1], key[2], key[3]
        ed = inst.edges[e]
        v = float(inst.commodities[k].v)
        if lane is None:
            if ed.coupling:
                continue
            fast = best_edge.get((ed.tail, ed.head), int(ed.tau))
            slow = int(ed.tau)
            head = ed.head
        else:
            if lane not in lane_group or inst.lanes[lane].edges[0] != e:
                continue
            fast = best_lane[lane_group[lane]]
            slow = lane_tau[lane]
            head = lane_group[lane][1]
        if slow - fast <= 0:
            continue
        ents.append((ep.tm[key], v * (slow - fast), fast, slow, gidx.get(head, -1)))
    return ents


def add(ep, mode, ref):
    ents = getattr(ep, "_d3b", None)
    if ents is None:
        ents = ep._d3b = _entries(ep)
    grid = mode["grid"]
    closing = {}
    for gi in range(len(ep.inst.grids)):
        closing[gi] = [c for c in range(1, ep.T + 1) if grid.get((c, gi)) in ("ON", "SOFT", "HULL")]
    out = np.zeros(ep.N)
    for j, vx, fast, slow, gi in ents:
        for t in range(1, ep.T + 1):
            idx = (t - 1) * ep.nc + j
            if idx >= ep.N:
                continue
            if gi < 0:
                w = 0.5
            else:
                cl = closing[gi]
                if any(t + fast <= c <= t + slow for c in cl):
                    w = 2.0
                elif any(c > t + slow for c in cl):
                    w = 1.0
                else:
                    w = 0.25
            out[idx] = RATE * vx * w
    return np.where(np.isfinite(out), out, 0.0)
