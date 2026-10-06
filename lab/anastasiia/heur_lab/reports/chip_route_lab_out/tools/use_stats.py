"""Scratch: how useful are the fabs' chips over episodes: share of weeks a fab's usefulness is < 0.05 / < 0.5, mean."""
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "lab_scratch")


def one(folder, ep):
    from probe import load_mod
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config
    from sbf_starter import env_id

    mod = load_mod(folder, f"use_agent_{ep}")
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    agent = mod.Agent(cfg)
    rows = []
    for t in range(1, 53):
        c = agent._context(obs)
        u = agent.usefulness(c)
        rows.append([u[n] for n in agent.fabs])
        obs, *_ = env.step(agent.act(obs))
    names = [info["static"]["nodes"]["id"][n] for n in agent.fabs]
    return np.array(rows), names


if __name__ == "__main__":
    folder, n = sys.argv[1], int(sys.argv[2])
    res = Parallel(n_jobs=3)(delayed(one)(folder, e) for e in range(n))
    names = res[0][1]
    U = np.stack([r[0] for r in res])  # (eps, T, fabs)
    print(f"{n} episodes: per fab, mean usefulness, share of weeks < 0.05, < 0.5, and share of episodes < 0.05 in week 1")
    for i, nm in enumerate(names):
        x = U[:, :, i]
        print(f"  {nm:18s} mean {x.mean():.2f}  <0.05 {100 * (x < 0.05).mean():3.0f}%  <0.5 {100 * (x < 0.5).mean():3.0f}%  week-1 <0.05 {100 * (x[:, 0] < 0.05).mean():3.0f}%")
