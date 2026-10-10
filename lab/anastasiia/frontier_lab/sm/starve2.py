"""Legal emulation of 'free release' at bound straits on saved full-future descents: throttle the cheap cargo sent
into a strait's lanes (by commodity and block of weeks), judged by the simulator, then let the cell's own descent run
from the result; repeat. Control: the same descent from the start."""
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
block = int(sys.argv[3])
rounds = int(sys.argv[4])
budget = float(sys.argv[5])
passes = int(sys.argv[6]) if len(sys.argv) > 6 else 6
t_all = time.time()
res = {}
for n in eps:
    t0 = time.time()
    ep = pc.Episode.of(task, 444, n)
    inst, T = ep.inst, ep.T
    Cn = [c.id for c in inst.commodities]
    acts = pickle.load(open(f"{ROOT}/outputs/regime_lab/hull/{task}_444_{n}_hybrid.pkl", "rb"))["acts"]
    recs0, J0 = ep.simulate(acts)
    t_play = time.time()
    ep.simulate(acts)
    t_play = time.time() - t_play
    groups = {}
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if lane is None or inst.commodities[k].pool != "ct" or Cn[k] == "chip_le_raw":
            continue
        for c in inst.lanes[lane].chokepoints:
            groups.setdefault((inst.nodes[c].id, Cn[k]), []).append(s)
    # only the straits where leading raw chips ever wait in the start
    waits = {}
    for r in recs0:
        for (c, k, lane), q in r.queue.items():
            if Cn[k] == "chip_le_raw":
                waits[inst.nodes[c].id] = waits.get(inst.nodes[c].id, 0.0) + q
    order = {"chip_mat": 0, "chip_mat_raw": 1, "wafer": 2}
    keys = sorted((g for g in groups if waits.get(g[0], 0.0) > 5e4), key=lambda g: (order[g[1]], g[0]))
    blocks = [(a, min(T, a + block - 1)) for a in range(1, T + 1, block)]
    week_block = {t: b for b, (a, z) in enumerate(blocks) for t in range(a, z + 1)}
    slot_groups = {}
    for g in keys:
        for s in groups[g]:
            slot_groups.setdefault(s, []).append(g)
    print(f"{task} 444 ep {n}: start {J0/1e11:.2f} bn, a play {t_play:.2f}s, leading raw chips waiting (thousand unit-weeks): "
          f"{ {k: round(v / 1e3) for k, v in waits.items() if v > 5e4} }, groups {len(keys)}", flush=True)

    def build(base, mult):
        out = []
        for t, (fl, ov, ho) in enumerate(base, start=1):
            b = week_block[t]
            new = {}
            for s, q in fl.items():
                f = 1.0
                for g in slot_groups.get(s, ()):
                    f = min(f, mult[(g, b)])
                if q * f > 0:
                    new[s] = q * f
            out.append((new, ov, ho))
        return out

    cur_acts, cur_J = acts, J0
    gain_search, gain_desc = 0.0, 0.0
    for rd in range(rounds):
        if not keys or time.time() - t0 > budget:
            break
        mult = {(g, b): 1.0 for g in keys for b in range(len(blocks))}
        best = cur_J
        for g in keys:
            for b in range(len(blocks)):
                if time.time() - t0 > budget:
                    break
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
        d = pc.descend(ep, new_acts, iters=passes) if time.time() - t0 < budget else {"J": best, "acts": new_acts}
        gd = (best - d["J"]) / 1e11
        print(f"   round {rd + 1}: throttle {gs:+.2f} bn, then the cell's descent {gd:+.2f} bn -> {d['J']/1e11:.2f} bn ({time.time()-t0:.0f}s); throttled: "
              + "; ".join(f"{g[0][4:]}/{g[1]} " + "".join("0" if mult[(g, b)] == 0 else "h" if mult[(g, b)] == 0.5 else "." for b in range(len(blocks)))
                          for g in keys if any(mult[(g, b)] != 1.0 for b in range(len(blocks)))), flush=True)
        gain_search += gs
        gain_desc += gd
        if d["J"] >= cur_J - 1e8:
            cur_acts, cur_J = (d["acts"], d["J"]) if d["J"] < cur_J else (cur_acts, cur_J)
            break
        cur_acts, cur_J = d["acts"], d["J"]
    ctrl = pc.descend(ep, acts, iters=passes) if time.time() - t0 < budget + 60 else {"J": J0}
    recs1, _ = ep.simulate(cur_acts)
    lost0, lost1 = np.sum([r.lost for r in recs0], axis=0), np.sum([r.lost for r in recs1], axis=0)
    le = sum((a - b) for d_, a, b in zip(inst.demands, lost0, lost1) if Cn[d_.k] == "chip_le")
    ma = sum((a - b) for d_, a, b in zip(inst.demands, lost0, lost1) if Cn[d_.k] == "chip_mat")
    shed = (sum(r.costs.shed for r in recs0) - sum(r.costs.shed for r in recs1)) / 1e9
    lots = (np.sum([r.lots_started for r in recs1]) - np.sum([r.lots_started for r in recs0])) / 1e3
    print(f"   ep {n}: {J0/1e11:.2f} -> {cur_J/1e11:.2f} bn, gain {(J0 - cur_J)/1e11:.2f} (throttle {gain_search:.2f}, descent after it {gain_desc:.2f}); "
          f"control descent from the start {(J0 - ctrl['J'])/1e11:+.2f}; sold more: chip_le {le/1e3:+.0f}k, chip_mat {ma/1e3:+.0f}k; shed saved {shed:+.2f} bn; lots {lots:+.0f}k; {time.time()-t0:.0f}s", flush=True)
    res[n] = {"J0": J0, "J": cur_J, "search": gain_search, "descent": gain_desc, "control": (J0 - ctrl["J"]) / 1e11, "acts": cur_acts}
    pickle.dump(res, open(f"{SCR}/starve2_{task}_{'_'.join(map(str, eps))}.pkl", "wb"))
print(f"mean gain {np.mean([(r['J0'] - r['J']) / 1e11 for r in res.values()]):.2f} bn over {len(res)} episodes; total {time.time()-t_all:.0f}s")
