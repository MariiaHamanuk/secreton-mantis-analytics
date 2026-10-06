"""Print the network an agent acts on: grids, fabs, packaging plants, markets, and every action slot with its route.

    uv run python hub/eval/routes.py [small|full|tiny]

Everything printed here is also in ``config["static"]["instance"]``: an agent reads it from there, never hard-coded.
"""

import sys

from shockbench_flow.hosting.tasks import task_generator


inst, params = task_generator(sys.argv[1] if len(sys.argv) > 1 else "small")
N, E, K = inst.nodes, inst.edges, [c.id for c in inst.commodities]
init = {}
for n, k, q in inst.initial_state.stock:
    init[(n, k)] = init.get((n, k), 0.0) + q
print(
    f"T {inst.T} weeks; psi {inst.params.psi}; commodities (customs value): "
    + ", ".join(f"{c.id} {c.v:.0f}" for c in inst.commodities)
)
print(
    "\n== grids: fuel k -> share of output; weekly burn at full output; stock at reset; rationing "
    "threshold (psi * ibar, rationed fuel only); storage =="
)
for gi, g in enumerate(inst.grids):
    grid = N[g].grid
    need = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in inst.grid_fabs[gi])
    print(
        f"{N[g].id}: deliverable {grid.deliverable:.0f} GWh/week, base load {grid.base_load:.0f} "
        f"({100 * grid.base_load / grid.deliverable:.1f}%), priority {grid.priority}, "
        f"VOLL {grid.voll:.0f} USD/GWh; fabs {[N[inst.fabs[fo]].id for fo in inst.grid_fabs[gi]]} "
        f"need {need:.1f} GWh/week at full capacity; no-fuel share {grid.shares.get(None, 0):.2f}"
    )
    for k in grid.fuels:
        s = inst.slot_index[(g, k)]
        thr = f"{inst.params.psi * grid.ibar[k]:.0f}" if k == grid.rationed else "-"
        print(
            f"    {K[k]:8s} share {grid.shares[k]:.2f}  burn "
            f"{grid.shares[k] * grid.deliverable:8.0f}/week  stock at reset "
            f"{init.get((g, k), 0):10.0f}  rationing threshold {thr:>6}  storage "
            f"{inst.stock_slots[s].storage:10.0f}"
        )
print("\n== fabs: input -> product, capacity (lots/week), weeks in process, GWh per lot, grid ==")
for f in inst.fabs:
    fa = N[f].fab
    print(
        f"{N[f].id:20s} {K[fa.input]} -> {K[fa.product]:13s} cap0 {fa.cap0:9.0f}  tau {fa.tau}  e "
        f"{fa.e}  grid {N[fa.grid].id if fa.grid is not None else None}"
    )
print("\n== packaging plants (OSATs): raw -> packaged, throughput per week (shared pro rata by raw stock), weeks ==")
for o in inst.osats:
    oa = N[o].osat
    print(
        f"{N[o].id:10s} {', '.join(K[a] + ' -> ' + K[b] for a, b in oa.packages.items()):50s} thr "
        f"{oa.thr:9.0f}  tau {oa.tau}"
    )
print("\n== markets: mean weekly demand, shortage penalty per unit (all lost sales unless backlog) ==")
for d in inst.demands:
    print(f"{N[d.node].id:8s} {K[d.k]:9s} demand {d.dbar:9.0f}  penalty {d.pi:8.0f}  backlog {d.backlog}")
print(
    "\n== sources and materials: supply per week (refills the source's stock up to its storage; "
    "unused supply is lost) =="
)
for s in inst.stock_slots:
    if s.supply > 0:
        print(f"{N[s.node].id:16s} {K[s.k]:8s} supply {s.supply:10.0f}  storage {s.storage:10.0f}")
print(
    "\n== action slots: slot | first edge (capacity u0 per week) | lead time to the destination | "
    "destination | straits passed (smallest u0 on the lane) =="
)
for kk in range(len(K)):
    print(f"--- {K[kk]} ---")
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if k != kk:
            continue
        ed = E[e]
        if lane is None:
            print(f"  {s:3d} {ed.id:44s} u0 {ed.u0:10.0f} tau {ed.tau}  -> {N[ed.head].id}")
        else:
            path = [E[x] for x in inst.lanes[lane].edges]
            print(
                f"  {s:3d} {ed.id:44s} u0 {ed.u0:10.0f} tau {sum(x.tau for x in path)}  -> "
                f"{N[path[-1].head].id} via {[N[c].id for c in inst.lanes[lane].chokepoints]} (min u0 "
                f"{min(x.u0 for x in path):.0f})"
            )
