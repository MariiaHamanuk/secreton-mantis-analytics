"""Fuel stuck at straits behind a prohibited next edge (it releases nothing), per episode (scratch, root 111).

    python scratch/stuck_fuel.py <agent_folder> <first> <n>
"""
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

folder, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Agent = load(folder)
per_ep = []
for ep in range(first, first + n):
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    st, lay = cfg["static"], cfg["layout"]
    K = st["commodities"]["id"]
    agent = Agent(cfg)
    stuck_week, flow_week, ever = [], [], 0.0
    fuel_slots = [s for s, k in enumerate(st["action_slots"]["k"]) if K[k] in ("lng", "crude", "nucfuel")]
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        ql = np.where(obs["queue_lots.qty.observed"] == 1, obs["queue_lots.qty"], 0.0).sum(axis=1)
        prohibited = obs["graph_now.prohibited"] == 1
        s = 0.0
        for r_, key in enumerate(lay["lot_keys"]):
            c, k, lane, nxt = key
            if ql[r_] > 0 and K[k] in ("lng", "crude") and prohibited[nxt, k]:
                s += ql[r_]
        stuck_week.append(s)
        flow_week.append(float(obs["last_week.clip.executed"][fuel_slots].sum()))
    per_ep.append((ep, float(np.mean(stuck_week)), float(np.max(stuck_week)), float(np.mean(flow_week))))
print("episode  mean stuck GWh   max stuck GWh   mean fuel dispatched per week")
for ep, m, mx, f in per_ep:
    print(f"{ep:7d} {m:15,.0f} {mx:15,.0f} {f:14,.0f}")
print(f"mean over episodes: stuck {np.mean([p[1] for p in per_ep]):,.0f} GWh = {np.mean([p[1] for p in per_ep]) / np.mean([p[3] for p in per_ep]):.2f} weeks of dispatch")
