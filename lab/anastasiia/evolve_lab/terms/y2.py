"""x1 plus the b3 price on MID weeks of a fab grid that are followed within 4 weeks by an ON/SOFT/HULL week,
at half the OFF price (share 0.06 * 0.5 * voll per unit of burnt fuel).
Mechanism: in a MID week the grid is capped but burns fuel that a closing week soon needs; FINDINGS says fuel and chips
are lost to burn in weeks that cannot make fab energy. Expected: same sign as b3, small extra gain.
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
            gm = grid.get((t, gi))
            if dist[t] <= 0:
                continue
            if gm == "OFF":
                w = 1.0 if dist[t] <= 4 else 0.5
            elif gm == "MID" and dist[t] <= 4:
                w = 0.5
            else:
                continue
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

def add(ep, mode, ref):
    out = np.zeros(ep.N)
    v = add_b3(ep, mode, ref)
    if v is not None:
        out = out + v
    v = add_d3(ep, mode, ref)
    if v is not None:
        out = out + v
    return out
