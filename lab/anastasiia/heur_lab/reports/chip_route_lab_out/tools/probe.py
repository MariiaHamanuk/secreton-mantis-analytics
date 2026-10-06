"""Scratch: play an episode with an agent folder up to a week, then hand back (agent, obs) for inspection.

    from probe import run_to; agent, obs, env = run_to(folder, ep, week)
"""
import importlib.util
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_agent.convert import agent_config

from sbf_starter import env_id


def load_mod(folder, name="probe_agent"):
    spec = importlib.util.spec_from_file_location(name, f"{folder}/agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_to(folder, ep, week, task="small", entropy=111):
    mod = load_mod(folder)
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    agent = mod.Agent(cfg)
    for _ in range(week - 1):
        obs, *_ = env.step(agent.act(obs))
    return agent, obs, env, mod
