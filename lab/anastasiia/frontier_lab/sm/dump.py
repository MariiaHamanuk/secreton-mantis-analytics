"""Instance facts for the mechanics audit (read-only)."""
import sys
from collections import Counter, defaultdict

from shockbench_flow.hosting.tasks import task_generator

for task in ("small", "full"):
    inst, _params = task_generator(task)
    C = inst.commodities
    N = inst.nodes
    print(f"\n===== {task}: T={inst.T} nodes={len(N)} edges={len(inst.edges)} lanes={len(inst.lanes)} "
          f"slots={len(inst.action_slots)} ov={len(inst.override_slots)}")
    print("commodities:", [(i, c.id, c.pool, c.override, c.disposal_cost) for i, c in enumerate(C)])
    print("node types:", Counter(n.type for n in N))
    print("params: psi", inst.params.psi, "fleet_share", inst.params.fleet_share, "fleet_measure", inst.params.fleet_measure)
    # tau = 0 edges
    z = defaultdict(list)
    for e in inst.edges:
        z[(e.tau, N[e.tail].type, N[e.head].type)].append(e.id)
    print("edges by (tau, tail type, head type):")
    for key in sorted(z):
        print("   ", key, len(z[key]), z[key][:4] if key[0] == 0 else "")
    # action slot tails
    print("action slot tail types:", Counter(N[inst.edges[e].tail].type for e, k, l in inst.action_slots))
    print("action slots with lane:", sum(l is not None for e, k, l in inst.action_slots))
    # slots per (edge,k) with several lanes
    per = defaultdict(list)
    for s, (e, k, l) in enumerate(inst.action_slots):
        per[(e, k)].append(l)
    multi = {key: v for key, v in per.items() if len(v) > 1}
    print("(edge,k) with several lane slots:", len(multi), "of", len(per))
    for (e, k), v in list(multi.items())[:6]:
        print("    ", inst.edges[e].id, C[k].id, [inst.lanes[l].id for l in v])
    # chokepoints
    chk = set(inst.chokepoints)
    print("release pairs (override (c,k)):", sorted({(N[c].id, C[k].id) for c, k, e, l in inst.override_slots}))
    for c in inst.chokepoints:
        outs = inst.out_edges[c]
        ins = inst.in_edges[c]
        lanes_here = [(li, ln) for li, ln in enumerate(inst.lanes) if c in ln.chokepoints]
        print(f"  chk {N[c].id}: mu={N[c].chokepoint.mu} in={len(ins)} out={len(outs)} lanes={len(lanes_here)}")
        for e in outs:
            ed = inst.edges[e]
            print(f"      out {ed.id} -> {N[ed.head].id}({N[ed.head].type}) tau={ed.tau} u0={ed.u0} pool={ed.pool} K={[C[k].id for k in sorted(ed.K)]} alt={ed.alt_of}")
    # edges out of chokepoints with both pools
    both = [e.id for e in inst.edges if e.tail in chk and len({C[k].pool for k in e.K}) > 1]
    print("chokepoint out-edges with both pools:", both)
    both2 = [e.id for e in inst.edges if len({C[k].pool for k in e.K}) > 1]
    print("any edge with both pools:", both2[:10])
    # stores
    print("fabs:")
    for fi, f in enumerate(inst.fabs):
        fa = N[f].fab
        sw, sp_ = inst.slot_index[(f, fa.input)], inst.slot_index[(f, fa.product)]
        print(f"   {N[f].id} cls={fa.cls} cap0={fa.cap0:.0f} e={fa.e:.5f} tau={fa.tau} grid={N[fa.grid].id if fa.grid is not None else None} "
              f"wafer store={inst.stock_slots[sw].storage:.0f} ({inst.stock_slots[sw].storage/fa.cap0:.2f} wk) "
              f"raw store={inst.stock_slots[sp_].storage:.0f} ({inst.stock_slots[sp_].storage/fa.cap0:.2f} wk) "
              f"out edges={[(inst.edges[e].id, N[inst.edges[e].head].type, inst.edges[e].tau, inst.edges[e].u0) for e in inst.out_edges[f]]}")
    print("osats:")
    for oi, o in enumerate(inst.osats):
        oa = N[o].osat
        s = []
        for kr, kp in sorted(oa.packages.items()):
            s.append((C[kr].id, inst.stock_slots[inst.slot_index[(o, kr)]].storage, C[kp].id, inst.stock_slots[inst.slot_index[(o, kp)]].storage))
        print(f"   {N[o].id} thr={oa.thr:.0f} tau={oa.tau} stores={s} "
              f"out={[(inst.edges[e].id, N[inst.edges[e].head].type, inst.edges[e].tau, inst.edges[e].u0, inst.edges[e].mode) for e in inst.out_edges[o]]}")
    print("grids:")
    for gi, g in enumerate(inst.grids):
        ga = N[g].grid
        print(f"   {N[g].id} pri={ga.priority} y={ga.base_load:.0f} G={ga.deliverable:.0f} top={ga.deliverable-ga.base_load:.0f} "
              f"shares={ {(C[k].id if k is not None else None): round(v,4) for k,v in ga.shares.items()} } "
              f"ibar={ {C[k].id: round(v) for k,v in ga.ibar.items()} } rationed={C[ga.rationed].id if ga.rationed is not None else None} "
              f"stores={ {C[k].id: inst.stock_slots[inst.slot_index[(g,k)]].storage for k in ga.fuels} } "
              f"out_edges={len(inst.out_edges[g])} fabs={[N[inst.fabs[fi]].id for fi in inst.grid_fabs[gi]]}")
    # initial stock at grids
    init = defaultdict(float)
    for node, k, q in inst.initial_state.stock:
        init[(node, k)] += q
    for g in inst.grids:
        ga = N[g].grid
        print("   init", N[g].id, {C[k].id: round(init.get((g, k), 0.0)) for k in ga.fuels},
              "burn/wk", {C[k].id: round(ga.shares[k] * ga.deliverable) for k in ga.fuels})
    # terminals
    print("terminals:")
    for i, n in enumerate(N):
        if n.type == "terminal" or n.terminal is not None:
            print("   ", n.id, n.type, {C[sl.k].id: sl.storage for sl in inst.stock_slots if sl.node == i},
                  "out", [(inst.edges[e].id, inst.edges[e].tau, inst.edges[e].u0) for e in inst.out_edges[i]][:4])
    # sinks
    print("demands:", [(N[d.node].id, C[d.k].id, round(d.dbar), d.pi, d.backlog, inst.stock_slots[inst.slot_index[(d.node, d.k)]].storage) for d in inst.demands][:20])
    print("dup_items:", [(inst.edges[e].id, None if l is None else inst.lanes[l].id, dt) for e, l, dt in inst.dup_items][:40])
