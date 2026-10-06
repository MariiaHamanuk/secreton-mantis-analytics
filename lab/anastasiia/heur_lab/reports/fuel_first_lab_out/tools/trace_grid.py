"""Week-by-week fuel trace of one grid in one episode (scratch).

    python scratch/trace_grid.py <agent_folder> <episode> <grid_suffix e.g. tw> [fuel e.g. lng]
"""
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

folder, ep, gs = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fuel = sys.argv[4] if len(sys.argv) > 4 else "lng"
import os

os.environ.setdefault("FUEL_TRACE", "1")
Agent = load(folder)
env = gym.make(env_id("small"), entropy=111)
obs, info = env.reset(options={"episode": ep})
cfg = agent_config_from_reset(env, obs, info)
names, K = cfg["static"]["nodes"]["id"], cfg["static"]["commodities"]["id"]
g = names.index(f"grid_{gs}")
k = K.index(fuel)
grids = cfg["layout"]["grids"]
agent = Agent(cfg)
if agent.fuel is not None:
    agent.fuel.P["trace"] = True
gi = grids.index(g)
done = False
print("week mode   grid_stock  terminal  transit  ship    tg   | shed   G_bar  burn_lng_est  load")
while not done:
    act = agent.act(obs)
    t = int(obs["week"][0])
    row = [r for r in (agent.fuel.trace if agent.fuel else []) if r["week"] == t and r["grid"] == g and r["k"] == k]
    obs, r, term, trunc, info = env.step(act)
    done = term or trunc
    shed = obs["last_week.shed.qty"][gi]
    r0 = row[0] if row else {}
    print(f"{t:4d} {r0.get('mode', '-'):5s} {r0.get('s0', 0):10d} {r0.get('term', 0):9d} {r0.get('transit', 0):8d} {r0.get('ship', 0):6d} {r0.get('tg', 0):6d} | {shed:8.0f} {obs['graph_now.grid.G_bar'][gi] if 'graph_now.grid.G_bar' in obs else 0:8.0f}  thr {r0.get('thr', 0)} b {r0.get('b', 0)}")
