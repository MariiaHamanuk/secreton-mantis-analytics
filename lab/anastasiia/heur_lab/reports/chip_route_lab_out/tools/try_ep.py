"""Scratch: play one episode in-process, report cost components, served chips and the slowest act()."""
import sys
import time

import numpy as np

sys.path.insert(0, "lab_scratch")

import gymnasium as gym
import shockbench_flow_gym  # noqa: F401
from sbf_starter import env_id
from sbf_starter.agents import load
from shockbench_flow_gym.dashboard import record_episode


class Timed:
    def __init__(self, cls):
        self.cls = cls
        self.times = []

    def __call__(self, config):
        inner = self.cls(config)
        outer = self

        class W:
            def act(self, obs):
                t0 = time.perf_counter()
                out = inner.act(obs)
                outer.times.append(time.perf_counter() - t0)
                return out

        return W()


if __name__ == "__main__":
    agent, n = sys.argv[1], int(sys.argv[2])
    env = gym.make(env_id("small"), entropy=111)
    cls = Timed(load(agent))
    rec = record_episode(env, cls, options={"episode": n})
    costs = np.asarray(rec["costs"]).sum(axis=0)
    names = ["freight", "war", "tariff", "holding", "queue", "shortage", "disposal", "shed"]
    print({k: round(v / 1e9, 2) for k, v in zip(names, costs)}, "total", round(costs.sum() / 1e9, 1))
    served = np.asarray(rec["obs"]["last_week.sinks.served"])[1:]
    print("served per week (k): ", (served.sum(axis=0) / 52 / 1e3).round(1))
    print("act time: mean %.4f max %.4f s" % (np.mean(cls.times), np.max(cls.times)))
