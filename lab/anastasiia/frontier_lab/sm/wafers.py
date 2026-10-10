"""Legal emulation of 'free lot starts' and of a free split of the top between fabs: wafers withheld from one fab in
a block of weeks (the only handle on lot starts), judged by the simulator, then the cell's own descent; repeated.
Start: the saved full-future descents. Control: the same descent from the start."""
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
SCR = "outputs/frontier_lab/scratch/sm"
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

task = sys.argv[1]
eps = [int(a) for a in sys.argv[2].split(",")]
block, rounds, budget, passes = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
res = {}
t_all = time.time()
for n in eps:
    t0 = time.time()
    ep = pc.Episode.of(task, 444, n)
    inst, T = ep.inst, ep.T
    Cn = [c.id for c in inst.commodities]
    W = Cn.index("wafer")
    acts = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/{task}_444_{n}_hybrid.pkl", "rb"))["acts"]
    recs0, J0 = ep.simulate(acts)
    groups = {}
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if k != W:
            continue
        dest = inst.lane_destination(lane) if lane is not None else inst.edges[e].head
        if dest in inst.fab_ordinal:
            groups.setdefault(inst.nodes[dest].id, []).append(s)
    keys = sorted(groups)
    blocks = [(a, min(T, a + block - 1)) for a in range(1, T + 1, block)]
    week_block = {t: b for b, (a, z) in enumerate(blocks) for t in range(a, z + 1)}
    slot_group = {s: g for g in keys for s in groups[g]}

    def build(base, mult):
        out = []
        for t, (fl, ov, ho) in enumerate(base, start=1):
            b = week_block[t]
            new = {}
            for s, q in fl.items():
                f = mult.get((slot_group.get(s), b), 1.0)
                if q * f > 0:
                    new[s] = q * f
            out.append((new, ov, ho))
        return out

    cur_acts, cur_J, g_search, g_desc, moves = acts, J0, 0.0, 0.0, []
    for rd in range(rounds):
        if time.time() - t0 > budget:
            break
        mult, best = {}, cur_J
        for g in keys:
            for b in range(len(blocks)):
                a, z = blocks[b]
                if not any(cur_acts[t - 1][0].get(s, 0.0) > 0 for t in range(a, z + 1) for s in groups[g]):
                    continue
                keep = 1.0
                for f in (0.0, 0.5):
                    mult[(g, b)] = f
                    _r, J = ep.simulate(build(cur_acts, mult))
                    if J < best - 1e6:
                        best, keep = J, f
                mult[(g, b)] = keep
        new_acts = build(cur_acts, mult)
        gs = (cur_J - best) / 1e11
        d = pc.descend(ep, new_acts, iters=passes)
        gd = (best - d["J"]) / 1e11
        cut = {g: "".join("0" if mult.get((g, b), 1.0) == 0 else "h" if mult.get((g, b), 1.0) == 0.5 else "." for b in range(len(blocks))) for g in keys}
        cut = {g[4:]: v for g, v in cut.items() if set(v) != {"."}}
        print(f"   round {rd + 1}: wafers withheld {gs:+.2f} bn, then the cell's descent {gd:+.2f} bn -> {d['J']/1e11:.2f}; {cut}", flush=True)
        g_search += gs
        g_desc += gd
        if d["J"] >= cur_J - 1e8:
            if d["J"] < cur_J:
                cur_acts, cur_J = d["acts"], d["J"]
            break
        cur_acts, cur_J = d["acts"], d["J"]
    ctrl = pc.descend(ep, acts, iters=passes)
    recs1, _ = ep.simulate(cur_acts)
    lost0, lost1 = np.sum([r.lost for r in recs0], axis=0), np.sum([r.lost for r in recs1], axis=0)
    sold = {}
    for dm, a, b in zip(inst.demands, lost0, lost1):
        sold[Cn[dm.k]] = sold.get(Cn[dm.k], 0.0) + (a - b) * dm.pi / 1e9
    shed = (sum(r.costs.shed for r in recs0) - sum(r.costs.shed for r in recs1)) / 1e9
    lots = (np.sum([r.lots_started for r in recs1], axis=0) - np.sum([r.lots_started for r in recs0], axis=0)) / 1e3
    print(f"ep {n}: {J0/1e11:.2f} -> {cur_J/1e11:.2f} bn, gain {(J0 - cur_J)/1e11:.2f} (withheld {g_search:.2f}, descent after {g_desc:.2f}); control {(J0 - ctrl['J'])/1e11:+.2f}; "
          f"sales gained bn {({k: round(v, 2) for k, v in sold.items()})}; shed saved {shed:+.2f}; lots k by fab {np.round(lots).tolist()}; {time.time()-t0:.0f}s", flush=True)
    res[n] = {"J0": J0, "J": cur_J, "control": (J0 - ctrl["J"]) / 1e11, "shed": shed, "sold": sold, "lots": lots, "acts": cur_acts}
    pickle.dump(res, open(f"{SCR}/wafers_{task}_{'_'.join(map(str, eps))}.pkl", "wb"))
gain = np.array([(r["J0"] - r["J"]) / 1e11 for r in res.values()])
ctl = np.array([r["control"] for r in res.values()])
print(f"mean gain {gain.mean():.2f} bn, control {ctl.mean():.2f}, net of control {np.mean(np.maximum(gain - ctl, 0)):.2f}; shed saved {np.mean([r['shed'] for r in res.values()]):.2f}; total {time.time()-t_all:.0f}s")
