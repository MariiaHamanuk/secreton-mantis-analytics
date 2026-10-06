"""Mean executed fuel flow per slot over an episode, for several agent folders side by side (scratch).

    python scratch/flows_ep.py <episode> <folder_a> [<folder_b> ...]
"""
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

ep = int(sys.argv[1])
folders = sys.argv[2:]
res = {}
for folder in folders:
    Agent = load(folder)
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    agent = Agent(cfg)
    names = cfg["static"]["nodes"]["id"]
    K = cfg["static"]["commodities"]["id"]
    slots = cfg["static"]["action_slots"]
    ed = cfg["static"]["edges"]
    ex, shed, rationed = [], [], []
    u_hist = []
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        ex.append(obs["last_week.clip.executed"].copy())
        shed.append(obs["last_week.shed.qty"].copy())
        u_hist.append(obs["graph_now.u"].copy())
    res[folder] = (np.array(ex[:-1]), np.array(shed))
u_med = np.median(np.array(u_hist), axis=0)
print("slot  commodity   edge/route (median u on first edge)" + "".join(f" {f.rstrip('/').split('/')[-1]:>10}" for f in folders))
for s in range(len(slots["edge"])):
    k = slots["k"][s]
    if K[k] not in ("lng", "crude", "nucfuel"):
        continue
    vals = [res[f][0][:, s].mean() for f in folders]
    if max(vals) < 1:
        continue
    e = slots["edge"][s]
    lane = slots["lane"][s]
    lname = cfg["static"]["lanes"]["id"][lane] if lane is not None else ed["id"][e]
    print(f"{s:4d}  {K[k]:8s} {lname[:52]:52s} u~{u_med[e]:8.0f}" + "".join(f" {v:10.0f}" for v in vals))
print("shed per grid (GWh/wk):", {f.rstrip('/').split('/')[-1]: [round(float(x)) for x in res[f][1].mean(axis=0)] for f in folders})
