"""Play one episode with a lab agent up to a week and print what its plan holds for a plant's exits.

    uv run python lab/anastasiia/frontier_lab/chip_week.py outputs/hazard_lab/agents/w50as_s 9 14 18 osat_my

Arguments: the agent's folder, the episode (Small, root 444), the first and the last week to print, the plant. For
each printed week: the agent's note, the action on the plant's exits of packaged chips, the plan's first twelve
weeks on each exit beside the capacity, the prohibitions and the tariffs its window holds, the demand of the US
market in the window, and what the environment executed. Thousand units.
"""

import sys
import numpy as np
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_agent.convert import agent_config
from sbf_starter import env_id
from sbf_starter.agents import load, resolve

agent, n, first, last, plant = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
env = gym.make(env_id("small"), entropy=444)
obs, info = env.reset(options={"episode": n})
u = env.unwrapped
ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
inst = u.core._ep.inst
com = [c.id for c in inst.commodities]
name = [x.id for x in inst.nodes]
slots = list(inst.action_slots)
o = [x for x in inst.osats if name[x] == plant][0]
ex = [s for s, (e, k, l) in enumerate(slots) if inst.edges[e].tail == o and l is None and com[k] in ("chip_le", "chip_mat")]
lab = {s: f"{name[inst.edges[slots[s][0]].head][5:]}/{com[slots[s][1]][5:]}/{inst.edges[slots[s][0]].mode}" for s in ex}
kept = {}
window = ag.model.window
def spy(*a, **k):
    kept["w"] = window(*a, **k)
    return kept["w"]
ag.model.window = spy
done, week = False, 0
while not done and week < last:
    week += 1
    action = ag.act(obs)
    if week >= first:
        w = kept["w"]
        H = len(w.marks.u)
        st = w.state
        print(f"\n=== week {week}: note {str(getattr(ag, 'log', ['?'])[-1])}"[:260])
        print("   action:", {lab[s]: round(float(action['flows'][s]) / 1e3, 1) for s in ex})
        print("   window weeks", H, "| acts kept", None if ag.acts is None else len(ag.acts))
        for s in ex:
            e, k, _l = slots[s]
            plan = [round(ag.acts[h][0].get(s, 0.0) / 1e3) for h in range(min(12, len(ag.acts)))] if ag.acts else None
            print(f"   {lab[s]:14s} plan {plan} | window u {[round(float(w.marks.u[h][e]) / 1e3) for h in range(0, min(H, 12))]} ban {[int(w.marks.prohibited[h][e, k]) for h in range(min(H, 12))]} tariff {[round(float(w.marks.tariff[h][e, k]), 2) for h in range(min(H, 6))]}")
        for di, d in enumerate(w.inst.demands):
            if name[d.node] == "sink_us":
                print(f"   demand US {com[d.k]} in window: {[round(float(w.marks.demand[h][di]) / 1e3) for h in range(min(H, 12))]} pi {d.pi}")
    obs, _r, term, trunc, _i = env.step(action)
    done = term or trunc
    if week >= first:
        r = u.core._ep.traj.records[-1]
        print("   executed:", {lab[s]: round(float(r.x.get(slots[s], 0.0)) / 1e3, 1) for s in ex} if hasattr(r, "x") else [a for a in dir(r) if not a.startswith("_")])
