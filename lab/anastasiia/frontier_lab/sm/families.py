"""Families of the simulator's automatic rules on the saved full-future descents: what the program gains with one
family's regime rows not written (the grid's base-load-first regime always written), what the simulator charges for
that solution played blind, and where the cell's own descent gets from that play. Control: the descent from the start."""
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
passes = int(sys.argv[2]) if len(sys.argv) > 2 else 6


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
    acts = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/small_444_{n}_hybrid.pkl", "rb"))["acts"]
    recs, J = ep.simulate(acts)
    mode, ref = ep.regimes(recs)
    C = ep.cell(mode, ref)
    s0 = ep.solve(C)
    hint = ep.hints(C, s0, mode)
    mode, ref = ep.regimes(recs, hint)
    C = ep.cell(mode, ref)
    s0 = ep.solve(C)
    ctrl = pc.descend(ep, acts, iters=passes)
    print(f"ep {n}: played {J/1e11:.2f} bn, cell {s0['J']/1e9:.2f}, control descent {(J - ctrl['J'])/1e11:+.2f}", flush=True)
    row = {"J": J, "control": (J - ctrl["J"]) / 1e11}

    def freed(kinds):
        ep.free_fab = "fab" in kinds
        D0 = ep.cell(mode, ref)
        ep.free_fab = False
        D = copy.copy(D0)
        D.lb, D.ub, D.lo, D.hi = D0.lb.copy(), D0.ub.copy(), list(D0.lo), list(D0.hi)
        for kind, key, role, where, idx in D0.tags:
            if kind not in kinds or (kind == "grid"):
                continue
            if where == "row":
                D.lo[idx], D.hi[idx] = -INF, INF
            elif idx < ep.n0:
                D.lb[idx], D.ub[idx] = ep.m.lb[idx], ep.m.ub[idx]
        return D

    for name, kinds in [("fuel", {"fuel"}), ("lots", {"fab"}), ("packaging", {"osat"}), ("lanes", {"lane"}),
                        ("disposal+lift", {"disp", "lift"}), ("all five", {"fuel", "fab", "osat", "lane", "disp", "lift"})]:
        s = ep.solve(freed(kinds))
        if s["status"] != "Optimal":
            print(f"   {name:14s} {s['status']}")
            continue
        gain = (s0["J"] - s["J"]) / 1e9
        a2 = ep.actions(s["x"])
        _r, J2 = ep.simulate(a2)
        blind = (J - J2) / 1e11
        d = pc.descend(ep, a2, iters=passes)
        after = (J - d["J"]) / 1e11
        row[name] = (gain, blind, after)
        print(f"   {name:14s} LP gain {gain:6.2f} | played blind {blind:+8.2f} | descent from that play {after:+7.2f} (to the saved descent, bn)", flush=True)
    res[n] = row
    print(f"   {time.time()-t0:.0f}s", flush=True)
    pickle.dump(res, open(f"{SCR}/families_{'_'.join(map(str, eps))}.pkl", "wb"))
names = ["fuel", "lots", "packaging", "lanes", "disposal+lift", "all five"]
print("\nmean over", len(res), "episodes: control", round(float(np.mean([r["control"] for r in res.values()])), 2))
for name in names:
    v = [r[name] for r in res.values() if name in r]
    if v:
        g, b, a = np.mean(v, axis=0)
        best = np.mean([max(r[name][2], r["control"], 0.0) for r in res.values() if name in r])
        print(f"   {name:14s} LP gain {g:6.2f} | played blind {b:+8.2f} | descent from that play {a:+7.2f} | the better of it and the control {best:+.2f}")
print(f"total {time.time()-t_all:.0f}s")
