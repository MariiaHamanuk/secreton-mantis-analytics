"""Grid priority, sliver premium and which grids pulse, as the final agent computes them from config (scratch)."""
import os
import sys

import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

for task in sys.argv[1:] or ["small"]:
    Agent = load("/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.claude/worktrees/agent-a598484e6f2fc03eb/agents/fuel_first")
    env = gym.make(env_id(task), entropy=111)
    obs, info = env.reset(options={"episode": 0})
    cfg = agent_config_from_reset(env, obs, info)
    names = cfg["static"]["nodes"]["id"]
    fr = Agent(cfg).fuel
    print(f"{task}: priority order {[names[g] for g in fr.priority]}")
    for g in fr.priority:
        print(f"   {names[g]:9s} premium {fr.premium[g]:6.3f}  pulses (>= {fr.P['pulse_min']}): {fr.premium[g] >= fr.P['pulse_min']}  has terminal valve for the rationed fuel: "
              f"{any((g, k) in fr.tg for k in fr.grids[g]['fuels'] if k == fr.grids[g]['rationed'])}")
    print(f"   stocked (nuclear-like) pairs: {sorted((names[g], k) for g, k in fr.stocked)}")
