"""Share of weeks each grid's rationed fuel spends in each mode, over several episodes (scratch).

    python scratch/modes.py <agent_folder> <first> <n>
"""
import sys
from collections import Counter, defaultdict

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

folder, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Agent = load(folder)
count = defaultdict(Counter)
shed_by_mode = defaultdict(lambda: defaultdict(list))
for ep in range(first, first + n):
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    names, K = cfg["static"]["nodes"]["id"], cfg["static"]["commodities"]["id"]
    grids = cfg["layout"]["grids"]
    agent = Agent(cfg)
    agent.fuel.P["trace"] = True
    done = False
    while not done:
        t = int(obs["week"][0])
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        for row in agent.fuel.trace:
            if row["week"] == t and K[row["k"]] == "lng":
                count[names[row["grid"]]][row["mode"]] += 1
                gi = grids.index(row["grid"])
                shed_by_mode[names[row["grid"]]][row["mode"]].append(obs["last_week.shed.qty"][gi])
        agent.fuel.trace.clear()
for g, c in count.items():
    tot = sum(c.values())
    print(g, {m: f"{100 * v / tot:.0f}%" for m, v in sorted(c.items())}, " mean shed by mode:", {m: round(float(np.mean(v))) for m, v in shed_by_mode[g].items()})
