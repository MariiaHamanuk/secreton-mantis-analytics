"""On the observations fuel_first meets, pull's flows and fuel_first's flows must agree on every non-fuel slot (scratch)."""
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

task = sys.argv[1] if len(sys.argv) > 1 else "small"
episodes = [int(x) for x in sys.argv[2:]] or [0, 1, 2]
Mine = load("/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.claude/worktrees/agent-a598484e6f2fc03eb/agents/fuel_first")
Pull = load("/Users/anastasiiamazur/Projects/secreton-mantis-analytics/agents/pull")
worst = 0.0
n_checked = 0
for ep in episodes:
    env = gym.make(env_id(task), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    mine, pull = Mine(cfg), Pull(cfg)
    fuel_slots = sorted(mine.fuel.slot)
    other = np.setdiff1d(np.arange(len(mine.edge)), fuel_slots)
    done = False
    while not done:
        a = mine.act(obs)
        b = pull.act(obs)
        diff = np.abs(np.asarray(a["flows"])[other] - np.asarray(b["flows"])[other]).max()
        worst = max(worst, float(diff))
        n_checked += 1
        assert set(a) == {"flows"}, "fuel_first must return only flows"
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
print(f"{task}: {n_checked} weeks, {len(other)} non-fuel slots of {len(mine.edge)}; largest difference to pull on a non-fuel slot: {worst}")
