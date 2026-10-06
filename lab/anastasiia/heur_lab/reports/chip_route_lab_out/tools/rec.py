"""Scratch: play one episode of an agent folder and pickle the record (obs per week, costs, events)."""
import pickle
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401

from sbf_starter import env_id
from sbf_starter.agents import load
from shockbench_flow_gym.dashboard import record_episode


def play(agent: str, n: int, task: str = "small", entropy: int = 111):
    env = gym.make(env_id(task), entropy=entropy)
    return record_episode(env, load(agent), options={"episode": n})


if __name__ == "__main__":
    agent, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rec = play(agent, n)
    with open(out, "wb") as f:
        pickle.dump(rec, f)
    print({k: (np.asarray(v).shape if not isinstance(v, (dict, list)) else type(v).__name__) for k, v in rec.items()})
    print(sorted(rec["obs"].keys()))
    print(rec["meta"])
