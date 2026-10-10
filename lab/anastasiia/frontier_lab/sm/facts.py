import sys
from collections import Counter

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402
from shockbench_flow.dynamics.env import validate_action  # noqa: E402
from shockbench_flow.dynamics.sim import initial_state, step  # noqa: E402

for task in ("small", "full"):
    inst, marks = pc.world(task, 444, 0)
    Cn = [c.id for c in inst.commodities]
    N = inst.nodes
    print(f"== {task}")
    print("action slots by (tail type, commodity):", dict(Counter((N[inst.edges[e].tail].type, Cn[k]) for e, k, l in inst.action_slots)))
    c = inst.node_index["chk_malacca"]
    a = {"week": 1, "flows": None, "overrides": None, "hold": {"chokepoint": [c, c], "k": [Cn.index("wafer"), Cn.index("lng")]}}
    fl, ov, ho, bad = validate_action(inst, marks, 1, a)
    print("hold on (chk_malacca, wafer) and (chk_malacca, lng): accepted", [(N[x].id, Cn[k]) for x, k in ho], "| dropped:", bad)
    st = initial_state(inst)
    init = {}
    for node, k, q in inst.initial_state.stock:
        init[(node, k)] = init.get((node, k), 0.0) + q
    arr1 = {}
    for s in st.pipeline:
        if s.arrival_week == 1:
            arr1[(inst.edges[s.edge].head, s.k)] = arr1.get((inst.edges[s.edge].head, s.k), 0.0) + s.qty
    rows = []
    for f in inst.fabs:
        fa = N[f].fab
        rows.append((N[f].id[4:], round(fa.cap0 / 1e3), round(init.get((f, fa.input), 0.0) / 1e3), round(arr1.get((f, fa.input), 0.0) / 1e3)))
    print("fab: capacity k / wafers on hand at reset k / arriving in week 1 k:", rows)
    psi = inst.params.psi
    print("grid gas at reset / threshold / burn at the cap:", [(N[g].id[5:], round(init.get((g, N[g].grid.rationed), 0.0)), round(psi * N[g].grid.ibar[N[g].grid.rationed]), round(N[g].grid.shares[N[g].grid.rationed] * N[g].grid.deliverable)) for g in inst.grids if N[g].grid.rationed is not None])
    # week 1 with no action at all: lots started and gas burned
    rec = step(inst, marks, st, {}, None, frozenset())
    print("week 1 with an empty action: lots k", np.round(rec.lots_started / 1e3).tolist(), "| gas burned GWh", [round(rec.segment.get((gi, N[g].grid.rationed), 0.0)) for gi, g in enumerate(inst.grids) if N[g].grid.rationed is not None])
    ct_q = sum(q.qty for q in inst.initial_state.queue_lots if inst.commodities[q.k].pool == "ct")
    ct_p = sum(s.qty for s in inst.initial_state.pipeline if inst.commodities[s.k].pool == "ct" and inst.edges[s.edge].head in inst.chokepoint_ordinal)
    print(f"containers at reset: in strait queues {ct_q/1e3:.0f}k, on the way to straits {ct_p/1e3:.0f}k")
