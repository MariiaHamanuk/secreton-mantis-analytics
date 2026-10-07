"""Play one episode of the rollout agent through gymnasium; per-week CPU, picks, and exceptions (diagnostics).

    uv run python lab/anastasiia/mpc_lab/rollout_probe.py --task=small --entropy=111 --n=0
"""
import collections
import os
import sys
import time
import traceback

import fire
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("SBF_CACHE_DIR", os.path.join(ROOT, "hub", "refcache"))


def main(task="small", entropy=111, n=0, agent=os.path.join(HERE, "agents", "hyb_rollout"), debug=True):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    u = env.unwrapped
    obs, info = env.reset(options={"episode": n})
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    c0 = time.process_time()
    ag = load(agent)(config)
    init = time.process_time() - c0
    roll = getattr(ag, "roll", None)
    if debug and roll is not None:  # surface exceptions instead of silently falling back
        orig = roll.choose

        def wrapped(*a, **k):
            try:
                return orig(*a, **k)
            except Exception:
                traceback.print_exc()
                raise
        roll.choose = wrapped
    cpu, done, rows = [], False, []
    while not done:
        c0 = time.process_time()
        a = ag.act(obs)
        cpu.append(time.process_time() - c0)
        obs, _r, term, trunc, inf = env.step(a)
        if roll is not None and roll.week1:
            rows.append((int(obs["week"][0]) - 1, -inf["reward_cents"] / 100, roll.week1[0]))
            roll.week1 = []
        done = term or trunc
    J = u.core.trajectory.J_cents / 100
    if rows:
        r = np.array(rows)
        err = (r[:, 2] - r[:, 1]) / np.maximum(np.abs(r[:, 1]), 1)
        print(f"week-1 fidelity (as-is candidate vs realized cost, only weeks with >1 candidate): median rel err {np.median(err):+.3f},"
              f" |err| p50 {np.median(np.abs(err)):.3f} p90 {np.percentile(np.abs(err), 90):.3f}")
    cpu = np.array(cpu)
    cpu[0] += init
    print(f"{task} root {entropy} ep {n}: J {J / 1e9:.3f} bn USD; init {init:.2f}s; week1 {cpu[0]:.3f} median {np.median(cpu):.3f}"
          f" p95 {np.percentile(cpu, 95):.3f} max {cpu.max():.3f} s; wins {roll.wins if roll else None}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
