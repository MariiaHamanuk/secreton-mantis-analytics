"""Combination of b3, d3 and a3: all three parent terms summed, each unchanged.

b3 prices OFF-week fuel burn before a closable week; d3 tilts parallel routes toward the faster one; a3 pays for
fuel in HULL/SOFT weeks and for scarce stock the week before. Expected: the effects add if real; a3's G half is
kept as in the parent (see a3a for the key fix).
"""

import numpy as np




SHARE_b3 = 0.06


def add_b3(ep, mode, ref):
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
                    out[ep.col("G", t, gi, k)] += SHARE_b3 * ga.voll * w
                    hit = True
    return out if hit else None





RATE_d3 = 0.01


def _build_d3(ep):
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
                out[idx] = RATE_d3 * v * extra * (1.0 + (t - 1) / max(1, ep.T))
    return out


def add_d3(ep, mode, ref):
    v = getattr(ep, "_d3", None)
    if v is None:
        v = _build_d3(ep)
        v = np.where(np.isfinite(v), v, 0.0)
        ep._d3 = v
    return v.copy()





GB_a3, IB_a3 = 0.04, 0.02


def _scarce_a3(ep):
    cached = getattr(ep, "_a3", None)
    if cached is not None:
        return cached
    inst = ep.inst
    scarce = set()
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        for k in ga.fuels:
            burn = float(ga.shares[k]) * float(ga.deliverable)
            if burn > 0 and (g, k) in inst.slot_index and float(ep.i0[ep.slot(g, k)]) < 2.0 * burn:
                scarce.add((gi, k))
    ep._a3 = scarce
    return scarce


def add_a3(ep, mode, ref):
    out = np.zeros(ep.N)
    inst, T = ep.inst, ep.T
    left = ep.info.get("left", 10**6) if getattr(ep, "info", None) else 10**6
    last = min(T, int(left) - 2)
    scarce = _scarce_a3(ep)
    closing = {key for key, gm in mode["grid"].items() if gm in ("HULL", "SOFT")}
    for t in range(1, last + 1):
        for gi, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            if ga.voll <= 0:
                continue
            for k in ga.fuels:
                if (t, gi) in closing and ep.has("G", t, gi, k):
                    out[ep.col("G", t, gi, k)] -= GB_a3 * float(ga.voll)
                if t < T and (t + 1, gi) in closing and (gi, k) in scarce and (g, k) in inst.slot_index:
                    s = ep.slot(g, k)
                    if ep.has("I", s):
                        out[ep.col("I", t, s)] -= IB_a3 * float(ga.voll)
    return out

def add(ep, mode, ref):
    out = np.zeros(ep.N)
    v = add_b3(ep, mode, ref)
    if v is not None:
        out = out + v
    v = add_d3(ep, mode, ref)
    if v is not None:
        out = out + v
    v = add_a3(ep, mode, ref)
    if v is not None:
        out = out + v
    return out
