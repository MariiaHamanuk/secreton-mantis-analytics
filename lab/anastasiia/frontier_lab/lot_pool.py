"""F1's accounting: the chips an agent's own play left unmade although the energy, the wafers and an exit were there.

    uv run python lab/anastasiia/frontier_lab/lot_pool.py w50as_s truthall_s truthall_h0_s --episodes=24

For every lot of the leading chip short of a fab's capacity in a week without energy to spare (weeks 1 to T - 14),
taken in turn from the kept play of a tag (``outputs/hazard_lab/play/<tag>_<task>_<entropy>.pkl``): wafers on hand or
within reach the week before; energy from the fuel the grid had already burned by that week, had it been bunched
into whole weeks (the burn of its scarcest gate fuel in weeks of its cap, minus the top-weeks its fabs got); room on a
direct edge to a plant the week the lot is out; an exit of that plant with room towards a market with unmet demand
within 12 weeks of packaging. Every capacity is used once. The value is the sale less the lot's energy at the price
of shed load, in bn USD an episode. Exits and capacities are the scenario's own (the future is used): an upper bound
on what better whole weeks could sell, not a forecast.

Printed: each tag's pool and cost, and the pool and cost of the first tag less each other's, with the correlation
of the two by episode. If a tag that sells more does not have a smaller pool, the accounting overstates.
"""

import importlib
import pickle
import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sample_omega = importlib.import_module("shockbench_flow.disruption.sampler").sample_omega
task_generator = importlib.import_module("shockbench_flow.hosting.tasks").task_generator
compute_marks = importlib.import_module("shockbench_flow.marks").compute_marks
sim = importlib.import_module("shockbench_flow.dynamics.sim")
VOLL, WAIT = 4125277.26, 12  # USD a GWh of shed load; weeks a packaged chip may wait for an exit


def main(*tags: str, task: str = "small", entropy: int = 444, episodes: int = 24, first: int = 0) -> None:
    kept = {tag: pickle.loads((ROOT / f"outputs/hazard_lab/play/{tag}_{task}_{entropy}.pkl").read_bytes()) for tag in tags}
    inst0, params = task_generator(task)
    C, N = inst0.commodities, inst0.nodes
    cid = {c.id: i for i, c in enumerate(C)}
    kle, kraw, kw = cid["chip_le"], cid["chip_le_raw"], cid["wafer"]

    def pool_of(inst, marks, r):
        stock, sent, lots, energy, lost, seg, shed = r["stock"], r["sent"], r["lots"], r["energy"], r["lost"], r["segment"], r["shed"]
        T = stock.shape[0]
        i0 = np.asarray(sim.initial_stock(inst))
        slots_from, edge_slots, slots_into = {}, {}, {}
        for s, (e, k, lane) in enumerate(inst.action_slots):
            slots_from.setdefault((inst.edges[e].tail, k), []).append(s)
            edge_slots.setdefault(e, []).append(s)
            if lane is None:
                slots_into.setdefault((inst.edges[e].head, k), []).append(s)
        left = lost.copy()
        room = {o: np.zeros(T) for o in inst.osats}
        for t in range(T):
            for o in inst.osats:
                for s in slots_from.get((o, kle), []):
                    e, k, lane = inst.action_slots[s]
                    if lane is not None or marks.prohibited[t][e, k]:
                        continue
                    d = [di for di, dm in enumerate(inst.demands) if dm.node == inst.edges[e].head and dm.k == kle and dm.dbar > 0]
                    ta = t + inst.edges[e].tau
                    if not d or ta >= T:
                        continue
                    x = min(max(0.0, marks.u[t][e] - sent[t, edge_slots[e]].sum()), left[ta, d[0]])
                    left[ta, d[0]] -= x; room[o][t] += x
        way, pool = {}, {}
        def held(src, w):
            if (src, w) not in pool:
                ss = inst.slot_index[(src, kw)]
                before = i0[ss] if w == 0 else stock[w - 1, ss]
                pool[(src, w)] = max(0.0, before - sent[w, slots_from[(src, kw)]].sum())
            return pool[(src, w)]
        val = 0.0
        for gi, g in enumerate(inst.grids):
            ga = N[g].grid
            members = [fi for fi in inst.grid_fabs[gi] if N[inst.fabs[fi]].fab.e > 0 and N[inst.fabs[fi]].fab.product == kraw]
            allm = [fi for fi in inst.grid_fabs[gi] if N[inst.fabs[fi]].fab.e > 0]
            if not members:
                continue
            top = sum(N[inst.fabs[fi]].fab.e * N[inst.fabs[fi]].fab.cap0 for fi in allm)
            gate = [k for k in ga.fuels if ga.shares[k] * ga.deliverable > top * 1.0001]
            cumw = np.min(np.stack([np.cumsum(seg[:, gi, k]) / (ga.shares[k] * ga.deliverable) for k in gate]), axis=0)
            cumtop = np.cumsum(energy[:, allm].sum(axis=1)) / top
            granted = 0.0
            for t in range(T - 14):
                nullcap = ga.shares.get(None, 0.0) * marks.G_bar[t, gi]
                if shed[t, gi] <= 1e-6 and seg[t, gi, -1] < nullcap * (1 - 1e-6):
                    continue
                for fi in sorted(members, key=lambda i: N[inst.fabs[i]].fab.e):
                    f = inst.fabs[fi]; fa = N[f].fab
                    Rf = max(marks.R[t, fi], 1e-9)
                    want = max(0.0, marks.alpha_bar[t, fi] * marks.R[t, fi] * fa.cap0 - lots[t, fi])
                    if want <= 1.0:
                        continue
                    w = min(want, stock[t, inst.slot_index[(f, kw)]])
                    if t >= 1:
                        for s in slots_into.get((f, kw), []):
                            e = inst.action_slots[s][0]
                            if inst.edges[e].tau != 1 or marks.prohibited[t - 1][e, kw] or w >= want:
                                continue
                            src = inst.edges[e].tail
                            q = min(max(0.0, marks.u[t - 1][e] - sent[t - 1, edge_slots[e]].sum()), held(src, t - 1), want - w)
                            pool[(src, t - 1)] -= q; w += q
                    w = min(w, max(0.0, (cumw[t] - cumtop[t]) * top - granted) * Rf / fa.e)
                    w1 = t + fa.tau + 1
                    sold, rest = 0.0, w
                    for s in slots_from.get((f, kraw), []):
                        e, _k, lane = inst.action_slots[s]
                        o = inst.edges[e].head
                        if lane is not None or N[o].osat is None or w1 >= T or marks.prohibited[w1][e, kraw] or rest <= 0:
                            continue
                        if (e, w1) not in way:
                            way[(e, w1)] = max(0.0, marks.u[w1][e] - sent[w1, edge_slots[e]].sum())
                        y = min(rest, way[(e, w1)])
                        w2 = w1 + inst.edges[e].tau + N[o].osat.tau + 1
                        z = 0.0
                        for wk in range(w2, min(w2 + WAIT, T)):
                            q = min(y - z, room[o][wk]); room[o][wk] -= q; z += q
                        way[(e, w1)] -= z
                        sold += z; rest -= y
                    granted += sold * fa.e / Rf
                    val += sold * (50.4e3 - fa.e / Rf * VOLL)
        return val / 1e9


    rows = []
    for n in range(first, first + episodes):
        if any(n not in kept[tag] for tag in tags):
            continue
        marks = compute_marks(inst0, sample_omega(inst0, params, entropy, n, "train"))
        inst = inst0.at_digest(marks.instance_digest)
        rows.append([pool_of(inst, marks, kept[tag][n]) for tag in tags] + [kept[tag][n]["J"] / 1e11 for tag in tags])
    a, k = np.array(rows), len(tags)
    print(f"{task} {entropy}, {len(a)} episodes from {first}; bn USD an episode")
    for i, tag in enumerate(tags):
        line = f"  {tag:16s} pool {a[:, i].mean():6.1f} (median {np.median(a[:, i]):5.1f})  cost {a[:, k + i].mean():8.1f}"
        if i:
            d_pool, d_cost = a[:, 0] - a[:, i], a[:, k] - a[:, k + i]
            line += (f"   {tags[0]} less it: pool {d_pool.mean():+6.1f}, cost {d_cost.mean():+6.1f}, "
                     f"correlation by episode {float(np.corrcoef(d_pool, d_cost)[0, 1]):+.2f}")
        print(line)


if __name__ == "__main__":
    fire.Fire(main)
