"""One episode of a lab agent, as played and with one causal rule on top of its action: cheap container cargo is
not sent into a strait that is backlogged while leading raw chips wait there or are on the way to it."""
import pickle
import sys
import time

import numpy as np

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
agent, task, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
variants = sys.argv[4].split(",")


def play(rule: str):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=444)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    if hasattr(ag, "tell_truth"):
        ag.tell_truth({"marks": u.core._ep.marks, "events": None})
    inst, marks = u.core._ep.inst, u.core._ep.marks
    Cn = [c.id for c in inst.commodities]
    LE = Cn.index("chip_le_raw")
    cheap = {Cn.index("wafer"), Cn.index("chip_mat_raw"), Cn.index("chip_mat")}
    chk = inst.chokepoint_ordinal
    ct = [k for k, c in enumerate(inst.commodities) if c.pool == "ct"]
    lanes = {s: (k, set(inst.lanes[lane].chokepoints)) for s, (e, k, lane) in enumerate(inst.action_slots) if lane is not None and k in cheap}
    done, t, dropped = False, 0, {k: 0.0 for k in cheap}
    by_strait = {}
    t0 = time.process_time()
    while not done:
        action = ag.act(obs)
        t += 1
        if rule != "base":
            st = u.core._ep.state
            flows = np.array(action["flows"], dtype=float)
            blocked = set()
            for c in inst.chokepoints:
                backlog = sum(float(st.stock[inst.slot_index[(c, k)]]) for k in ct if (c, k) in inst.slot_index)
                le_q = float(st.stock[inst.slot_index[(c, LE)]]) if (c, LE) in inst.slot_index else 0.0
                le_way = any(sh.k == LE and inst.edges[sh.edge].head == c for sh in st.pipeline)
                kap = float(marks.kappa_now[t - 1][chk[c]][1])
                k0 = inst.nodes[c].chokepoint.mu[1] * inst.nodes[c].chokepoint.k_c
                deep = kap < float(rule.split(":")[1]) * k0 if ":" in rule else True
                if (le_q > 0 or le_way) and backlog > kap and deep:
                    blocked.add(c)
            for s, (k, cs) in lanes.items():
                if flows[s] > 0 and cs & blocked:
                    dropped[k] += flows[s]
                    key = (sorted(inst.nodes[c].id for c in cs & blocked)[0], Cn[k], inst.nodes[inst.lane_destination(inst.action_slots[s][2])].id)
                    by_strait[key] = by_strait.get(key, 0.0) + flows[s]
                    flows[s] = 0.0
            action = dict(action)
            action["flows"] = flows
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    recs = u.core._ep.traj.records
    J = int(u.core._ep.traj.J_cents)
    lost = np.sum([r.lost for r in recs], axis=0)
    by = {}
    for d, q in zip(inst.demands, lost):
        by[Cn[d.k]] = by.get(Cn[d.k], 0.0) + q * d.pi / 1e9
    shed = sum(r.costs.shed for r in recs) / 1e9
    global detail
    detail = {'lots': np.array([r.lots_started for r in recs]), 'shed': np.array([r.shed for r in recs]), 'lost': np.array([r.lost for r in recs]),
              'fabs': [inst.nodes[f].id for f in inst.fabs], 'grids': [inst.nodes[g].id for g in inst.grids], 'by_strait': by_strait,
              'demands': [(inst.nodes[d.node].id, Cn[d.k], d.pi) for d in inst.demands]}
    return J, by, shed, {Cn[k]: round(v / 1e3) for k, v in dropped.items()}, time.process_time() - t0


kept = pickle.load(open(f"{ROOT}/outputs/hazard_lab/play/{agent.rstrip('/').split('/')[-1]}_{task}_444.pkl", "rb"))
print(f"{agent} {task} 444 ep {n}: kept J {kept[n]['J']/1e11:.2f} bn" if n in kept else "not kept")
base = None
for rule in variants:
    J, by, shed, dropped, cpu = play(rule)
    if rule == "base":
        base = J
    ref = base if base is not None else kept[n]["J"]
    if rule != 'base' and n in kept:
        b = kept[n]
        dl = (detail['lots'].sum(axis=0) - b['lots'].sum(axis=0)) / 1e3
        ds = (b['shed'].sum(axis=0) - detail['shed'].sum(axis=0))
        dd = (b['lost'].sum(axis=0) - detail['lost'].sum(axis=0))
        print('   lots k by fab, rule - base:', {f[4:]: round(float(v)) for f, v in zip(detail['fabs'], dl) if abs(v) > 5})
        print('   shed saved GWh by grid:', {g[5:]: round(float(v)) for g, v in zip(detail['grids'], ds) if abs(v) > 20})
        print('   sold more, bn:', {(a[5:], b_[5:]): round(float(v) * pi / 1e9, 1) for (a, b_, pi), v in zip(detail['demands'], dd) if abs(v * pi) > 2e8})
        print('   withheld k by strait, cargo, destination:', {k_: round(v / 1e3) for k_, v in sorted(detail['by_strait'].items(), key=lambda kv: -kv[1])[:10]})
    print(f"   {rule:10s} J {J/1e11:.2f} bn ({(ref - J)/1e11:+.2f} to the base); lost sales bn {({k: round(v, 2) for k, v in by.items()})}; shed {shed:.2f}; withheld k {dropped}; cpu {cpu:.0f}s", flush=True)
