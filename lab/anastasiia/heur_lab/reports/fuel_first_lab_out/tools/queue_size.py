"""How much fuel waits at straits under an agent, per week, as a share of the fuel dispatched (scratch, root 111).

    python scratch/queue_size.py <agent_folder> <first> <n>
"""
import sys

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

folder, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Agent = load(folder)
queued, sent, weeks_with = [], [], 0
total_weeks = 0
for ep in range(first, first + n):
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    st, lay = cfg["static"], cfg["layout"]
    K = st["commodities"]["id"]
    fuel_rows = [r for r, key in enumerate(lay["lot_keys"]) if K[key[1]] in ("lng", "crude")]
    fuel_slots = [s for s, k in enumerate(st["action_slots"]["k"]) if K[k] in ("lng", "crude", "nucfuel")]
    agent = Agent(cfg)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        ql = np.where(obs["queue_lots.qty.observed"] == 1, obs["queue_lots.qty"], 0.0)[fuel_rows].sum()
        flow = float(obs["last_week.clip.executed"][fuel_slots].sum()) if "last_week.clip.executed" in obs else 0.0
        queued.append(ql)
        sent.append(flow)
        total_weeks += 1
        weeks_with += ql > 0.5 * max(flow, 1.0)
queued, sent = np.array(queued), np.array(sent)
print(f"{total_weeks} weeks: mean fuel waiting at straits {queued.mean():,.0f} GWh (mean fuel dispatched per week {sent.mean():,.0f}); "
      f"ratio of the means {queued.mean() / max(sent.mean(), 1):.2f} weeks of dispatch; weeks with more than half a week of dispatch waiting: {100 * weeks_with / total_weeks:.0f}%")
