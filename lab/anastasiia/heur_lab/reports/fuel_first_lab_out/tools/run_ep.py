"""Play episodes of root 111 with an agent folder and print a compact fuel summary (scratch).

    python scratch/run_ep.py <agent_folder> <first_episode> <n_episodes> [trace]
"""
import sys
import time

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

np.set_printoptions(linewidth=220, suppress=True, precision=1)

import os

folder = sys.argv[1]
first = int(sys.argv[2])
n = int(sys.argv[3])
trace = len(sys.argv) > 4 and sys.argv[4] == "trace"
task = os.environ.get("TASK", "small")
Agent = load(folder)
tot = {}
for ep in range(first, first + n):
    env = gym.make(env_id(task), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    names = cfg["static"]["nodes"]["id"]
    K = cfg["static"]["commodities"]["id"]
    grids = cfg["layout"]["grids"]
    agent = Agent(cfg)
    t0 = time.time()
    costs = np.zeros(8)
    shed = []
    stocks = []
    done = False
    cpu = 0.0
    while not done:
        t1 = time.process_time()
        act = agent.act(obs)
        cpu = max(cpu, time.process_time() - t1)
        obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        if "last_week.cost_components" in obs:
            costs += obs["last_week.cost_components"]
        shed.append(obs["last_week.shed.qty"].copy())
        stocks.append(obs["stock.qty"].copy())
    shed = np.array(shed)
    print(f"ep {ep}: {time.time() - t0:.1f}s, max act cpu {cpu * 1000:.0f} ms, cost bn: total {costs.sum() / 1e9:.1f} "
          + " ".join(f"{c}={v / 1e9:.1f}" for c, v in zip(cfg["layout"]["cost_components"], costs) if v > 1e8)
          + " | shed/wk " + " ".join(f"{names[g][5:]}={shed[:, i].mean():.0f}({100 * (shed[:, i] > 1e-6).mean():.0f}%)" for i, g in enumerate(grids)))
    if trace and hasattr(agent, "fuel") and agent.fuel is not None:
        for row in agent.fuel.trace[-40:]:
            print(row)
