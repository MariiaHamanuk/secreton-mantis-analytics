"""Nuclear fuel stock at each grid and what an agent ships, over one Full episode (scratch).

    SBF_CACHE_DIR=... python scratch/nuc_full.py <agent_folder> <episode>
"""
import os
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

folder, ep = sys.argv[1], int(sys.argv[2])
task = os.environ.get("TASK", "full")
Agent = load(folder)
env = gym.make(env_id(task), entropy=111)
obs, info = env.reset(options={"episode": ep})
cfg = agent_config_from_reset(env, obs, info)
st = cfg["static"]
names, K = st["nodes"]["id"], st["commodities"]["id"]
nuc = K.index("nucfuel")
rows = [(i, names[n]) for i, (n, k) in enumerate(cfg["layout"]["stock_slots"]) if k == nuc and names[n].startswith("grid_")]
slots = [s for s, k in enumerate(st["action_slots"]["k"]) if k == nuc]
agent = Agent(cfg)
hist, shipped = [], np.zeros(len(slots))
done = False
while not done:
    t = int(obs["week"][0])
    if t in (1, 13, 26, 39, 52, 65, 78, 91, 104):
        hist.append((t, [float(obs["stock.qty"][i]) for i, _ in rows]))
    obs, r, term, trunc, info = env.step(agent.act(obs))
    done = term or trunc
    shipped += obs["last_week.clip.executed"][slots]
print("grids:", [n for _, n in rows])
for t, v in hist:
    print(f"week {t:3d} nuc stock", " ".join(f"{x:10.0f}" for x in v))
print("nuc shipped over the episode per slot:", {st['action_slots']['edge'][s]: round(float(x)) for s, x in zip(slots, shipped)})
print("total shipped", round(float(shipped.sum())))
