"""How often is a strait throttled with cargo for more than one grid in its fuel queue? (scratch, root 111)

    python scratch/contested_straits.py <agent_folder> <first> <n>
A (strait, week) is 'throttled' when the fuel waiting there exceeds the tanker/bulk throughput kappa of the week; it is
'contested' when, besides, the waiting fuel is bound for two or more grids (the case where releasing one grid's cargo first
could matter: release_mode 1 with override_qty).
"""
import sys
from collections import defaultdict

import gymnasium as gym
import numpy as np
import shockbench_flow_gym  # noqa: F401
from shockbench_flow_gym import agent_config_from_reset

from sbf_starter import env_id
from sbf_starter.agents import load

folder, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Agent = load(folder)
strait_weeks = throttled = contested = 0
excess_gwh = contested_excess = 0.0
for ep in range(first, first + n):
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    st, lay = cfg["static"], cfg["layout"]
    K = st["commodities"]["id"]
    ed, ln = st["edges"], st["lanes"]
    chk_rows = {int(c): i for i, c in enumerate(lay["chokepoints"])}
    grid_of = {}
    for g in lay["grids"]:
        grid_of[int(g)] = int(g)
    # terminal -> grid via the terminal-to-grid edges
    term_grid = {}
    for e in range(len(ed["id"])):
        if ed["id"][e].startswith("tg."):
            term_grid[int(ed["tail"][e])] = int(ed["head"][e])
    agent = Agent(cfg)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        ql = np.where(obs["queue_lots.qty.observed"] == 1, obs["queue_lots.qty"], 0.0).sum(axis=1)
        kap = obs["graph_now.kappa.tb"]
        by_c = defaultdict(lambda: defaultdict(float))
        for r_, key in enumerate(lay["lot_keys"]):
            c, k, lane, nxt = key
            if ql[r_] > 0 and K[k] in ("lng", "crude"):
                dest = int(ed["head"][ln["edges"][lane][-1]])
                by_c[c][term_grid.get(dest, dest)] += ql[r_]
        for c, row in chk_rows.items():
            strait_weeks += 1
            tot = sum(by_c[c].values()) if c in by_c else 0.0
            if tot > kap[row] + 1e-6:
                throttled += 1
                excess_gwh += tot - kap[row]
                if len(by_c[c]) >= 2:
                    contested += 1
                    contested_excess += tot - kap[row]
print(f"{strait_weeks} strait-weeks: throttled {100 * throttled / strait_weeks:.1f}% (mean fuel above one week's throughput {excess_gwh / max(throttled, 1):,.0f} GWh), "
      f"contested (cargo for 2+ grids) {100 * contested / strait_weeks:.1f}% (mean excess {contested_excess / max(contested, 1):,.0f} GWh)")
