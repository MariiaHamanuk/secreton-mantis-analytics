"""How much of the freed-rules gap is executable: from a saved full-future descent, a chain of moves that are all
legal actions (a family's regime rows left out for one solve, the solution played by the simulator; cheap cargo
withheld from a strait's lanes; wafers withheld from a fab), each followed by the cell's own descent and kept only
when the simulator's cost falls. At the end: what the five families are still worth at the new trajectory."""
import copy
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
SCR = "outputs/frontier_lab/scratch/sm"
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

INF = float("inf")
eps = [int(a) for a in sys.argv[1].split(",")]
budget = float(sys.argv[2])
ALL = {"fuel", "fab", "osat", "lane", "disp", "lift"}


class Free(pc.Episode):
    free_fab = False

    def _lots(self, C, mode, ref, t, fi, gm, jr, rho0, gi):
        if self.free_fab and gm != "OFF":
            return
        return super()._lots(C, mode, ref, t, fi, gm, jr, rho0, gi)


res = {}
t_all = time.time()
for n in eps:
    t0 = time.time()
    ep = Free.of("small", 444, n)
    inst, T = ep.inst, ep.T
    Cn = [c.id for c in inst.commodities]
    acts0 = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/small_444_{n}_hybrid.pkl", "rb"))["acts"]
    recs0, J0 = ep.simulate(acts0)

    def cells(acts, kinds):
        recs, J = ep.simulate(acts)
        mode, ref = ep.regimes(recs)
        C = ep.cell(mode, ref)
        s0 = ep.solve(C)
        if s0["status"] == "Optimal":
            mode, ref = ep.regimes(recs, ep.hints(C, s0, mode))
            C = ep.cell(mode, ref)
            s0 = ep.solve(C)
        ep.free_fab = "fab" in kinds
        D0 = ep.cell(mode, ref)
        ep.free_fab = False
        D = copy.copy(D0)
        D.lb, D.ub, D.lo, D.hi = D0.lb.copy(), D0.ub.copy(), list(D0.lo), list(D0.hi)
        for kind, key, role, where, idx in D0.tags:
            if kind in kinds and kind != "grid":
                if where == "row":
                    D.lo[idx], D.hi[idx] = -INF, INF
                elif idx < ep.n0:
                    D.lb[idx], D.ub[idx] = ep.m.lb[idx], ep.m.ub[idx]
        return s0, ep.solve(D)

    def nudge(kinds):
        def move(acts):
            s0, s = cells(acts, kinds)
            return ep.actions(s["x"]) if s["status"] == "Optimal" else None
        return move

    blocks = [(a, min(T, a + 3)) for a in range(1, T + 1, 4)]
    week_block = {t: b for b, (a, z) in enumerate(blocks) for t in range(a, z + 1)}

    def throttle(groups):
        def move(acts):
            recs, J = ep.simulate(acts)
            slot_groups = {}
            for g, ss in groups.items():
                for s in ss:
                    slot_groups.setdefault(s, []).append(g)

            def build(mult):
                out = []
                for t, (fl, ov, ho) in enumerate(acts, start=1):
                    b = week_block[t]
                    new = {}
                    for s, q in fl.items():
                        f = min([mult.get((g, b), 1.0) for g in slot_groups.get(s, ())] or [1.0])
                        if q * f > 0:
                            new[s] = q * f
                    out.append((new, ov, ho))
                return out

            mult, best = {}, J
            for g, ss in groups.items():
                for b, (a, z) in enumerate(blocks):
                    if not any(acts[t - 1][0].get(s, 0.0) > 0 for t in range(a, z + 1) for s in ss):
                        continue
                    keep = 1.0
                    for f in (0.0, 0.5):
                        mult[(g, b)] = f
                        _r, Jn = ep.simulate(build(mult))
                        if Jn < best - 1e6:
                            best, keep = Jn, f
                    mult[(g, b)] = keep
            return build(mult) if best < J else None
        return move

    strait_groups, fab_groups = {}, {}
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if inst.commodities[k].pool != "ct":
            continue
        if lane is not None and Cn[k] != "chip_le_raw":
            for c in inst.lanes[lane].chokepoints:
                strait_groups.setdefault((inst.nodes[c].id, Cn[k]), []).append(s)
        if Cn[k] == "wafer":
            dest = inst.lane_destination(lane) if lane is not None else inst.edges[e].head
            if dest in inst.fab_ordinal:
                fab_groups.setdefault(inst.nodes[dest].id, []).append(s)
    moves = [("disposal+lift", nudge({"disp", "lift"})), ("lots", nudge({"fab"})), ("packaging", nudge({"osat"})),
             ("lanes", nudge({"lane"})), ("strait throttle", throttle(strait_groups)), ("wafer throttle", throttle(fab_groups))]
    s0, sa = cells(acts0, ALL)
    lp0 = (s0["J"] - sa["J"]) / 1e9
    cur, J, took = acts0, J0, {}
    ctrl = pc.descend(ep, acts0, iters=6)
    if ctrl["J"] < J - 1e8:
        took["plain descent"] = (J - ctrl["J"]) / 1e11
        cur, J = ctrl["acts"], ctrl["J"]
    for rd in range(3):
        moved = False
        for name, move in moves:
            if time.time() - t0 > budget:
                break
            a2 = move(cur)
            if a2 is None:
                continue
            d = pc.descend(ep, a2, iters=6)
            if d["J"] < J - 1e8:
                took[name] = took.get(name, 0.0) + (J - d["J"]) / 1e11
                cur, J, moved = d["acts"], d["J"], True
        if not moved:
            break
    s1, sb = cells(cur, ALL)
    lp1 = (s1["J"] - sb["J"]) / 1e9 if sb["status"] == "Optimal" and s1["status"] == "Optimal" else float("nan")
    recs1, _ = ep.simulate(cur)
    lost0, lost1 = np.sum([r.lost for r in recs0], axis=0), np.sum([r.lost for r in recs1], axis=0)
    sold = sum((a - b) * dm.pi for dm, a, b in zip(inst.demands, lost0, lost1)) / 1e9
    shed = (sum(r.costs.shed for r in recs0) - sum(r.costs.shed for r in recs1)) / 1e9
    print(f"ep {n}: {J0/1e11:.2f} -> {J/1e11:.2f} bn, executable gain {(J0 - J)/1e11:.2f} (sales {sold:+.2f}, shed {shed:+.2f}); by move {({k: round(v, 2) for k, v in took.items()})}; "
          f"five families' LP bound at the start {lp0:.2f}, at the result {lp1:.2f}; {time.time()-t0:.0f}s", flush=True)
    res[n] = {"J0": J0, "J": J, "took": took, "lp0": lp0, "lp1": lp1, "acts": cur, "sold": sold, "shed": shed}
    pickle.dump(res, open(f"{SCR}/chain_{'_'.join(map(str, eps))}.pkl", "wb"))
print(f"mean executable gain {np.mean([(r['J0'] - r['J']) / 1e11 for r in res.values()]):.2f} bn; LP bound {np.mean([r['lp0'] for r in res.values()]):.2f} -> {np.nanmean([r['lp1'] for r in res.values()]):.2f}; total {time.time()-t_all:.0f}s")
