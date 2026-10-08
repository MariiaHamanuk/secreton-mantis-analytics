"""Probe B: local search on the hub's played trajectory, moves only on KR-bound gas (orders, the term_kr valve) and on
lng releases at Hormuz / Malacca; the simulator is the judge (planner_LSF machinery, open loop, everything known).

    uv run python lab/anastasiia/mpc_lab/level1/probeB.py run --eps=45,57 --budget=10000 --secs=900 --n_jobs=1
    uv run python lab/anastasiia/mpc_lab/level1/probeB.py report --eps=45,57

Moves (strict first improvement, like planner_LSF):
- KR order slots (Qatar lanes to term_kr: main, east, lombok; Australia -> term_kr) and the term_kr -> grid_kr valve:
  planner_LS.MOVES on a positive dispatch; on a week the slot sent nothing (and is not prohibited): set it to 0.5 or 1.0
  of the route's smallest true edge capacity that week (valve: 500 / 1,500 / 3,000 GWh).
- lng override slots at Hormuz and Malacca, only in weeks where the hub already overrides that (strait, lng) pair (a new
  override would switch the default release of the whole pair off): scale a slot x0 / x0.5 / x1.5 / x2, or move half or
  all of a slot's release to a KR-bound slot of the same pair.
Episodes of root 111 are used here for this mechanism probe only (no tuning decision).
"""

import collections
import os
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT / "lab" / "anastasiia" / "mpc_lab" / "planners"), str(ROOT / "lab" / "anastasiia" / "mpc_lab")]
OUT = ROOT / "outputs" / "level1" / "probeB"
AGENT = str(ROOT / "agents" / "anastasiia_hybrid_hub")
BANDS = ((1, 13), (14, 26), (27, 39), (40, 52))


def setup():
    import planner_LSF as LSF

    LSF.configure("small", 111)
    LSF.AGENT = AGENT
    LSF.CACHE = str(OUT / "cache")
    os.makedirs(LSF.CACHE, exist_ok=True)
    return LSF


def kr_slots(inst):
    """{slot: kind} for lng slots bound for KR: 'order' (lane or edge ends at term_kr) or 'valve' (term_kr -> grid_kr)."""
    nid = {nd.id: i for i, nd in enumerate(inst.nodes)}
    lng = [c.id for c in inst.commodities].index("lng")
    out = {}
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if k != lng:
            continue
        last = inst.edges[inst.lanes[lane].edges[-1]] if lane is not None else inst.edges[e]
        if last.head == nid["term_kr"]:
            out[s] = "order"
        elif inst.edges[e].tail == nid["term_kr"] and inst.edges[e].head == nid["grid_kr"]:
            out[s] = "valve"
    return out


def route_cap(inst, marks, s, t):
    e, _k, lane = inst.action_slots[s]
    edges = inst.lanes[lane].edges if lane is not None else [e]
    return float(min(marks.u[t - 1, x] for x in edges))


def search(n, budget, secs, wide=False, wafers=False, pair=False):
    LSF = setup()
    t0 = time.time()
    d, R, rep = LSF.verify_one(n)
    out = dict(rep, before=R.J, after=R.J, moves=[], replays=0)
    if not rep["equal"] or rep["bad_weeks"]:
        out["error"] = "replay mismatch"
        return out
    inst, marks = R.inst, R.marks
    lng = [c.id for c in inst.commodities].index("lng")
    nid = {nd.id: i for i, nd in enumerate(inst.nodes)}
    ks = kr_slots(inst)
    if wide:  # also every other Qatar lng lane and Australia -> term_jp: the competitors for the fleet slack and supply
        for s, (e, k, lane) in enumerate(inst.action_slots):
            src = inst.nodes[inst.edges[e].tail].id
            if k == lng and s not in ks and src in ("src_qa_lng", "src_au_lng"):
                ks[s] = "other"
    if wafers:  # wafers into the KR memory fab: the extra gas only sells if the fab has wafers to start
        for s, (e, k, lane) in enumerate(inst.action_slots):
            last = inst.edges[inst.lanes[lane].edges[-1]] if lane is not None else inst.edges[e]
            if inst.commodities[k].id == "wafer" and inst.nodes[last.head].id == "fab_kr_memory_1":
                ks[s] = "wafer"
    ovs = {o: (c, e, lane) for o, (c, k, e, lane) in enumerate(inst.override_slots)
           if k == lng and inst.nodes[c].id in ("chk_hormuz", "chk_malacca")}
    kr_ov = {o for o, (c, e, lane) in ovs.items()
             if lane is not None and inst.edges[inst.lanes[lane].edges[-1]].head == nid["term_kr"]}
    comp0 = R.components()
    calls0, t1 = R.calls, time.time()

    def out_of():
        return R.calls - calls0 >= budget or time.time() - t1 > secs

    def trial_ov(t, newov):
        """J with week t's override dict replaced."""
        old = R.ov[t - 1]
        R.ov[t - 1] = newov
        try:
            J = R._run(t, R.flows, keep=False)
        finally:
            R.ov[t - 1] = old
        return J

    def accept_ov(t, newov, J):
        R.ov[t - 1] = newov
        J2 = R.commit(t)
        assert J2 == J, (J2, J)

    moves = []

    def record(t, kind, what, move, gain):
        moves.append(dict(week=t, kind=kind, what=what, move=move, gain=gain))

    for sweep in range(3):
        improved = False
        # flow candidates: KR slots, every week 1..45 (positive -> LS moves; zero -> set moves)
        cand = []
        for t in range(1, 46):
            for s in ks:
                q = R.flows[t - 1].get(s, 0.0)
                cand.append((-q, t, s))
        cand.sort()
        for _, t, s in cand:
            if out_of():
                break
            e, k, _ = inst.action_slots[s]
            if marks.prohibited[t - 1][e, k]:
                continue
            q = R.flows[t - 1].get(s, 0.0)
            if q > 1e-9:
                gen = LSF.moves_of(R, t, s)
            else:
                qs = (500.0, 1500.0, 3000.0) if ks[s] == "valve" else tuple(f * route_cap(inst, marks, s, t) for f in (0.5, 1.0))
                gen = ((f"set{int(v)}", {(t, s): v}) for v in qs if v > 1e-6)
            if pair and ks[s] == "order":  # order more AND open the valve by as much when it lands at term_kr
                valve = next(v for v, kind in ks.items() if kind == "valve")
                lane = inst.action_slots[s][2]
                lead = sum(inst.edges[x].tau for x in (inst.lanes[lane].edges if lane is not None else [e]))
                deltas = sorted({x for x in (0.5 * q, q, 0.5 * route_cap(inst, marks, s, t)) if x > 1.0})
                extra = []
                for dq in deltas:
                    for off in (0, 1, 2):
                        t2 = t + lead + off
                        if t2 <= R.T:
                            extra.append((f"pair{off}", {(t, s): q + dq, (t2, valve): R.flows[t2 - 1].get(valve, 0.0) + dq}))
                gen = list(gen) + extra
            for name, ch in gen:
                if out_of():
                    break
                J, flows = R.trial(ch)
                if J < R.J:
                    gain = R.J - J
                    R.accept(ch, flows, J)
                    record(t, ks[s], LSF.slot_name(inst, s), name, gain)
                    improved = True
                    break
        # override candidates: weeks where the hub overrides at Hormuz / Malacca
        for t in range(1, 46):
            if out_of():
                break
            cur = R.ov[t - 1] or {}
            mine = {o: q for o, q in cur.items() if o in ovs}
            if not mine:
                continue
            tries = []
            for o, q in mine.items():
                for f in (0.0, 0.5, 1.5, 2.0):
                    tries.append((f"x{f}", o, {**cur, o: q * f}))
                for o2 in kr_ov:
                    if o2 != o and ovs[o2][0] == ovs[o][0] and q > 1e-9:
                        for f in (0.5, 1.0):
                            tries.append((f"move{f}->{o2}", o, {**cur, o: q * (1 - f), o2: cur.get(o2, 0.0) + q * f}))
            for name, o, newov in tries:
                if out_of():
                    break
                J = trial_ov(t, newov)
                if J < R.J:
                    gain = R.J - J
                    accept_ov(t, newov, J)
                    c, e, lane = ovs[o]
                    record(t, "override", f"{inst.nodes[c].id} {inst.edges[e].id}", name, gain)
                    improved = True
                    break
        if not improved or out_of():
            break
    out.update(after=R.J, moves=moves, flows=R.flows, slots=sorted(ks), replays=R.calls - calls0, secs=time.time() - t1, sweeps=sweep + 1,
               comp0=comp0, comp1=R.components(), total_secs=time.time() - t0)
    seg = np.array([[r.segment.get((go, lng), 0.0) for go in range(len(inst.grids))] for r in R.records[1:]])
    out["kr_lng_burn"] = seg[:, inst.grid_ordinal[nid["grid_kr"]]]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"lsb{'w' if wide else ''}{'f' if wafers else ''}{'p' if pair else ''}_ep{n}.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"[ep {n}{' wide' if wide else ''}{' wafers' if wafers else ''}{' pair' if pair else ''}] dJ {(out['before'] - out['after']) / 1e11:.2f} bn, {len(moves)} moves, {out['replays']} replays, "
          f"{out['secs']:.0f} s", flush=True)
    return out


def run(eps, budget: int = 10000, secs: float = 900.0, n_jobs: int = 1, wide: bool = False, wafers: bool = False,
        pair: bool = False) -> None:
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    Parallel(n_jobs=n_jobs)(delayed(search)(n, budget, secs, wide, wafers, pair) for n in which)


def report(eps, wide: bool = False, wafers: bool = False, pair: bool = False) -> None:
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    res = []
    for n in which:
        p = OUT / f"lsb{'w' if wide else ''}{'f' if wafers else ''}{'p' if pair else ''}_ep{n}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                res.append(pickle.load(f))
    bn = 1e11
    print(f"{'ep':>3} {'dJ bn':>7} {'moves':>5} {'replays':>7} {'secs':>5}   shortage / shed change, bn   KR lng burned change GWh")
    for r in res:
        dsh = (r["comp1"]["shortage"] - r["comp0"]["shortage"]) / 1e9
        dsd = (r["comp1"]["shed"] - r["comp0"]["shed"]) / 1e9
        kr0 = np.load(OUT.parent / "probeA" / f"ep{r['n']}.npz")["hub_seg"][:, 1].sum()
        print(f"{r['n']:>3} {(r['before'] - r['after']) / bn:7.2f} {len(r['moves']):>5} {r['replays']:>7} {r['secs']:5.0f}"
              f"   {dsh:+7.1f} / {dsd:+7.1f}   {r['kr_lng_burn'].sum() - kr0:+8.0f}")
    g = np.array([(r["before"] - r["after"]) / bn for r in res])
    print(f"mean dJ {g.mean():.2f} bn/episode (median {np.median(g):.2f}); target 61.8 per L1 episode = "
          f"{61.8 * 25 / 8:.0f} per KR-deficit episode")
    allm = [m for r in res for m in r["moves"]]
    for key in ("kind", "what"):
        agg = collections.defaultdict(lambda: [0, 0.0])
        for m in allm:
            agg[m[key]][0] += 1
            agg[m[key]][1] += m["gain"] / bn / len(res)
        print(f"by {key}: " + "; ".join(f"{k} {v[0]} moves {v[1]:.2f} bn/ep" for k, v in sorted(agg.items(), key=lambda x: -x[1][1])))
    agg = collections.defaultdict(lambda: [0, 0.0])
    for m in allm:
        b = next(f"{lo}-{hi}" for lo, hi in BANDS if lo <= m["week"] <= hi)
        agg[b][0] += 1
        agg[b][1] += m["gain"] / bn / len(res)
    print("by week band: " + "; ".join(f"{k} {v[0]} moves {v[1]:.2f} bn/ep" for k, v in sorted(agg.items())))
    agg = collections.defaultdict(lambda: [0, 0.0])
    for m in allm:
        agg[m["move"]][0] += 1
        agg[m["move"]][1] += m["gain"] / bn / len(res)
    print("by move: " + "; ".join(f"{k} {v[0]} {v[1]:.2f}" for k, v in sorted(agg.items(), key=lambda x: -x[1][1])))


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report})
