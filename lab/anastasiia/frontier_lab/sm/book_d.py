"""The container lot book of kept plays, replayed from the kept dispatches: why leading raw chips wait at straits,
and what two counterfactuals deliver to the packaging plants in time to sell.

  fifo      the simulator's release on the kept dispatches (checked against the kept queue totals)
  starve    legal and causal: cheap cargo (wafers, mature chips) is not sent into a strait that is backlogged while
            leading raw chips wait there or are on the way to it; the release is the simulator's
  priority  not legal: the kept dispatches, leading raw chips released first (the relaxed program's freedom)
"""
import copy
import pickle
import sys
import time

import numpy as np
from shockbench_flow.dynamics import chokepoint as chk_mod
from shockbench_flow.dynamics.clip import dup_terms, fleet_caps
from shockbench_flow.dynamics.sim import initial_state
from shockbench_flow.dynamics.state import Lot, Shipment

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
SCR = "outputs/frontier_lab/scratch/sm"
sys.path.insert(0, f"{ROOT}/agents/anastasiia_plan_hazard4")
import plan_core as pc  # noqa: E402

tag, task = sys.argv[1], sys.argv[2]
first, count = int(sys.argv[3]), int(sys.argv[4])
budget = float(sys.argv[5]) if len(sys.argv) > 5 else 480.0
depth = float(sys.argv[6]) if len(sys.argv) > 6 else 9.9
kept = pickle.load(open(f"{ROOT}/outputs/hazard_lab/play/{tag}_{task}_444.pkl", "rb"))
t_all = time.time()
rows = {}
for n in range(first, first + count):
    if n not in kept or time.time() - t_all > budget:
        continue
    t0 = time.time()
    inst, marks = pc.world(task, 444, n)
    T = inst.T
    Cn = [c.id for c in inst.commodities]
    K = {name: k for k, name in enumerate(Cn)}
    LE = K["chip_le_raw"]
    cheap = {K["wafer"], K["chip_mat_raw"], K["chip_mat"]}
    ct = {k for k, c in enumerate(inst.commodities) if c.pool == "ct"}
    chk = inst.chokepoint_ordinal
    terms = dup_terms(inst)
    cap_fleet = fleet_caps(inst)[1]
    sent = kept[n]["sent"]
    stock = kept[n]["stock"]
    kap0 = np.array([inst.nodes[c].chokepoint.mu[1] * inst.nodes[c].chokepoint.k_c for c in inst.chokepoints])
    slots = [(s, e, k, lane) for s, (e, k, lane) in enumerate(inst.action_slots) if k in ct]
    lane_chk = {lane: set(inst.lanes[lane].chokepoints) for _s, _e, _k, lane in slots if lane is not None}
    osats, fabs = set(inst.osats), set(inst.fabs)

    def dupw(e, lane):
        return sum(dt for ln, dt in terms.get(e, ()) if ln is None or ln == lane)

    def replay(mode):
        st = initial_state(inst)
        lots = [copy.copy(lt) for lt in st.lots if lt.k in ct]
        pipe = [copy.copy(s) for s in st.pipeline if s.k in ct and inst.edges[s.edge].head in chk]
        nid = 10 ** 7
        deliv = {}  # (head kind, k) -> per arrival week
        wait = {"prohibited": 0.0, "edge": 0.0, "kappa": 0.0}
        waitc = {}
        dropped = {k: 0.0 for k in cheap}
        qerr = 0.0
        q_prev = {}
        for t in range(1, T + 1):
            ti = t - 1
            keep = []
            for sh in pipe:
                if sh.arrival_week == t:
                    head = inst.edges[sh.edge].head
                    lots.append(Lot(sh.lot_id if sh.lot_id is not None else nid, head, sh.k, sh.qty, sh.lane,
                                    inst.lane_next_edge(sh.lane, sh.edge), sh.dispatch_week, sh.edge, t))
                    nid += 1
                else:
                    keep.append(sh)
            pipe = keep
            u, kappa, proh = marks.u[ti].tolist(), marks.kappa[ti].tolist(), marks.prohibited[ti].tolist()
            if mode == "priority":
                rel = chk_mod.release(inst, [lt for lt in lots if lt.k == LE], u, kappa, proh, {}, frozenset())
                u2, k2 = list(u), [list(x) for x in kappa]
                for r in rel:
                    u2[r.edge] -= r.qty
                    k2[chk[r.lot.chokepoint]][1] -= r.qty
                k2 = [[max(0.0, a) for a in x] for x in k2]
                rel += chk_mod.release(inst, [lt for lt in lots if lt.k != LE], [max(0.0, a) for a in u2], k2, proh, {}, frozenset())
            else:
                rel = chk_mod.release(inst, lots, u, kappa, proh, {}, frozenset())
            # this week's dispatches, the legal counterfactual applied
            disp = []
            for s, e, k, lane in slots:
                q = float(sent[ti, s])
                if q <= 0:
                    continue
                if mode == "starve" and k in cheap and lane is not None:
                    for c in lane_chk[lane]:
                        ci = chk[c]
                        backlog = sum(v for (cc, _k), v in q_prev.items() if cc == c)
                        le_here = q_prev.get((c, LE), 0.0) > 0 or any(sh.k == LE and inst.edges[sh.edge].head == c for sh in pipe)
                        if le_here and backlog > float(marks.kappa_now[ti][ci][1]) and float(marks.kappa_now[ti][ci][1]) < depth * kap0[ci]:
                            dropped[k] += q
                            q = 0.0
                            break
                if q > 0:
                    disp.append((e, k, lane, q))
            D = sum(dupw(e, lane) * q for e, k, lane, q in disp)
            R = sum(dupw(r.edge, r.lane) * r.qty for r in rel)
            if R > 0 and D + R > cap_fleet * (1 + 1e-12):
                f = max(0.0, cap_fleet - D) / R
                for r in rel:
                    if dupw(r.edge, r.lane) > 0:
                        r.qty *= f
            used_e, used_c = {}, {}
            for r in rel:
                if r.qty <= 0:
                    continue
                r.lot.qty -= r.qty
                used_e[r.edge] = used_e.get(r.edge, 0.0) + r.qty
                used_c[r.lot.chokepoint] = used_c.get(r.lot.chokepoint, 0.0) + r.qty
                ed = inst.edges[r.edge]
                if ed.head in chk:
                    pipe.append(Shipment(r.edge, r.k, r.lane, r.qty, t, t + ed.tau, nid))
                    nid += 1
                else:
                    key = ("osat" if ed.head in osats else "fab" if ed.head in fabs else "sink", r.k)
                    deliv.setdefault(key, np.zeros(T + 12))[t + ed.tau] += r.qty
            lots = [lt for lt in lots if lt.qty > 1e-12]
            for e, k, lane, q in disp:
                ed = inst.edges[e]
                if ed.head in chk:
                    pipe.append(Shipment(e, k, lane, q, t, t + ed.tau, nid))
                    nid += 1
            q_prev = {}
            for lt in lots:
                q_prev[(lt.chokepoint, lt.k)] = q_prev.get((lt.chokepoint, lt.k), 0.0) + lt.qty
                if lt.k == LE:
                    if proh[lt.next_edge][lt.k]:
                        why = "prohibited"
                    elif used_e.get(lt.next_edge, 0.0) >= u[lt.next_edge] * (1 - 1e-9):
                        why = "edge"
                    else:
                        why = "kappa"
                    wait[why] += lt.qty
                    key = (inst.nodes[lt.chokepoint].id, why)
                    waitc[key] = waitc.get(key, 0.0) + lt.qty
            if mode == "fifo":
                for c in inst.chokepoints:
                    for k in ct:
                        if (c, k) in inst.slot_index:
                            qerr = max(qerr, abs(q_prev.get((c, k), 0.0) - float(stock[ti, inst.slot_index[(c, k)]])))
        end_le = sum(v for (c, k), v in q_prev.items() if k == LE)
        return deliv, wait, waitc, dropped, qerr, end_le

    def in_time(deliv, kind, k, last):
        a = deliv.get((kind, k))
        return 0.0 if a is None else float(a[: last + 1].sum())

    out = {}
    for mode in ("fifo", "starve", "priority"):
        deliv, wait, waitc, dropped, qerr, end_le = replay(mode)
        out[mode] = {
            "le": in_time(deliv, "osat", LE, T - 4), "le_all": in_time(deliv, "osat", LE, T + 11),
            "mat_raw": in_time(deliv, "osat", K["chip_mat_raw"], T - 4), "mat": in_time(deliv, "sink", K["chip_mat"], T),
            "wafer": in_time(deliv, "fab", K["wafer"], T - 14), "wait": wait, "waitc": waitc, "dropped": dropped,
            "qerr": qerr, "end_le": end_le,
        }
    b, s, p = out["fifo"], out["starve"], out["priority"]
    lost_le = float(sum(v for d, v in zip(inst.demands, kept[n]["lost"].sum(axis=0)) if Cn[d.k] == "chip_le"))
    rows[n] = out
    print(f"ep {n:2d}: replay error {b['qerr']:.1f}; le_raw waiting k unit-weeks: kappa {b['wait']['kappa']/1e3:7.0f} edge {b['wait']['edge']/1e3:7.0f} prohibited {b['wait']['prohibited']/1e3:7.0f}; "
          f"le to plants in time k: fifo {b['le']/1e3:6.0f}, starve {(s['le']-b['le'])/1e3:+6.0f} (mat_raw {(s['mat_raw']-b['mat_raw'])/1e3:+5.0f}, mat {(s['mat']-b['mat'])/1e3:+5.0f}, wafers {(s['wafer']-b['wafer'])/1e3:+6.0f}), "
          f"priority {(p['le']-b['le'])/1e3:+6.0f}; lost chip_le demand {lost_le/1e3:6.0f}k; {time.time()-t0:.1f}s", flush=True)
    pickle.dump(rows, open(f"{SCR}/book_{tag}_{task}_{first}_{count}_{depth}.pkl", "wb"))
ns = sorted(rows)
if ns:
    g = lambda mode, key: np.array([rows[n][mode][key] for n in ns])  # noqa: E731
    w = {why: np.mean([rows[n]["fifo"]["wait"][why] for n in ns]) / 1e3 for why in ("kappa", "edge", "prohibited")}
    ds, dp = g("starve", "le") - g("fifo", "le"), g("priority", "le") - g("fifo", "le")
    dm = (g("starve", "mat_raw") - g("fifo", "mat_raw")) + (g("starve", "mat") - g("fifo", "mat"))
    dw = g("starve", "wafer") - g("fifo", "wafer")
    print(f"\n{tag} {task} 444, {len(ns)} episodes ({ns[0]}..{ns[-1]}): leading raw chips waiting, thousand unit-weeks an episode: {w}")
    print(f"   starve (legal): leading raw chips at plants in time {ds.mean()/1e3:+.1f}k an episode = {ds.mean()*50.4e3/1e9:+.2f} bn at the penalty; "
          f"mature chips {dm.mean()/1e3:+.1f}k = {dm.mean()*10.3e3/1e9:+.2f} bn; wafers at fabs {dw.mean()/1e3:+.1f}k; episodes over 1 bn: {int((ds*50.4e3 > 1e9).sum())}; "
          f"top: {[(ns[i], round(ds[i]/1e3)) for i in np.argsort(-ds)[:6]]}")
    print(f"   priority (not legal): {dp.mean()/1e3:+.1f}k an episode = {dp.mean()*50.4e3/1e9:+.2f} bn; top: {[(ns[i], round(dp[i]/1e3)) for i in np.argsort(-dp)[:6]]}")
    print(f"   total {time.time()-t_all:.0f}s")
