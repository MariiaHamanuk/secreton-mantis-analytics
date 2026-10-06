"""How far is the observed graph_now.u / kappa from what the simulator uses that week? (scratch, root 111 only)

The observation of week t holds the instantaneous value at t-1 (marks.u_now); the simulator's clip and releases use the
week average over [t-1, t) (marks.u). Compared on fuel edges (and on the tanker/bulk strait throughput).

    python scratch/obs_vs_sim.py <first> <n>
"""
import os
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset
from shockbench_flow_agent.scoring import _world

from sbf_starter import env_id
from sbf_starter.agents import load

first, n = int(sys.argv[1]), int(sys.argv[2])
Agent = load("/Users/anastasiiamazur/Projects/secreton-mantis-analytics/agents/pull")
rows = []
for ep in range(first, first + n):
    inst, omega, marks, fallback = _world("small", 111, ep, 1000, os.environ["SBF_CACHE_DIR"])
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    st = cfg["static"]
    K = st["commodities"]["id"]
    fuel_edges = [i for i, ks in enumerate(st["edges"]["K"]) if ks and all(K[k] in ("lng", "crude", "nucfuel") for k in ks)]
    agent = Agent(cfg)
    done = False
    while not done:
        t = int(obs["week"][0])
        u_obs = np.asarray(obs["graph_now.u"])[fuel_edges]
        u_sim = np.asarray(marks.u[t - 1])[fuel_edges]
        k_obs = np.asarray(obs["graph_now.kappa.tb"])
        k_sim = np.asarray(marks.kappa[t - 1])[:, 0]
        u0 = np.array([st["edges"]["u0"][i] for i in fuel_edges], float)
        rows.append((ep, t, u_obs, u_sim, u0, k_obs, k_sim))
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
u_obs = np.stack([r[2] for r in rows])
u_sim = np.stack([r[3] for r in rows])
u0 = np.stack([r[4] for r in rows])
k_obs = np.stack([r[5] for r in rows])
k_sim = np.stack([r[6] for r in rows])
weeks = np.array([r[1] for r in rows])
rel = np.abs(u_obs - u_sim) / np.maximum(u0, 1.0)
print(f"{len(rows)} weeks x {u_obs.shape[1]} fuel edges ({n} episodes)")
print(f"edge-weeks where observed u differs from the simulator's week value by more than 1% of nominal: {100 * (rel > 0.01).mean():.2f}%  (more than 10%: {100 * (rel > 0.10).mean():.2f}%)")
print(f"  of which the observed capacity is HIGHER than the simulator's: {100 * ((u_obs - u_sim) / np.maximum(u0, 1.0) > 0.01).mean():.2f}% of edge-weeks, LOWER: {100 * ((u_sim - u_obs) / np.maximum(u0, 1.0) > 0.01).mean():.2f}%")
print(f"  week 1: {100 * (rel[weeks == 1] > 0.01).mean():.2f}%   later weeks: {100 * (rel[weeks > 1] > 0.01).mean():.2f}%")
relk = np.abs(k_obs - k_sim) / np.maximum(k_sim.max(axis=0, keepdims=True), 1.0)
print(f"strait-weeks (tanker/bulk pool) where observed kappa differs by more than 1% of its largest value: {100 * (relk > 0.01).mean():.2f}%")
print(f"sum over fuel edges of observed u / simulator u, all weeks: {u_obs.sum() / u_sim.sum():.4f}")
