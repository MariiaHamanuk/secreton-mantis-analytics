"""Lots started by each fab in weeks its grid runs complete (lng mode run/on with no shed) vs other weeks (scratch).

    python scratch/runweeks.py <agent_folder> <first> <n>
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
rows = defaultdict(list)  # (fab, class) -> list of (lots/cap, wafers/cap)
for ep in range(first, first + n):
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": ep})
    cfg = agent_config_from_reset(env, obs, info)
    st = cfg["static"]
    names, K = st["nodes"]["id"], st["commodities"]["id"]
    inst = st["instance"]
    nodes_by_id = {x["id"]: x for x in inst["nodes"]}
    grids = cfg["layout"]["grids"]
    fabs = cfg["layout"]["fabs"]
    srow = {(int(a), int(b)): i for i, (a, b) in enumerate(cfg["layout"]["stock_slots"])}
    wafer = K.index("wafer")
    agent = Agent(cfg)
    agent.fuel.P["trace"] = True
    done = False
    while not done:
        t = int(obs["week"][0])
        wafers = {f: obs["stock.qty"][srow[(f, wafer)]] for f in fabs}
        obs, r, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        mode = {}
        for row in agent.fuel.trace:
            if row["week"] == t and K[row["k"]] == "lng":
                mode[row["grid"]] = row["mode"]
        agent.fuel.trace.clear()
        if t > 40:
            continue
        shed = obs["last_week.shed.qty"]
        for fi, f in enumerate(fabs):
            fab = nodes_by_id[names[f]]["fab"]
            g = names.index(fab["grid"])
            gi = grids.index(g)
            tau = fab["tau"]
            lots = 0.0
            live = np.asarray(obs["wip.qty.observed"]) == 1
            for node, q, ow in zip(obs["wip.node"][live], obs["wip.qty"][live], obs["wip.out_week"][live]):
                if int(node) == f and int(ow) == t + tau:
                    lots += q
            cls = "run" if mode.get(g) in ("run",) else ("on" if mode.get(g) == "on" else "hold/prime")
            full = shed[gi] < 1e-6
            rows[(names[f], cls, "noshed" if full else "shed")].append((lots / fab["cap0"], wafers[f] / fab["cap0"]))
print(f"{'fab':18s} {'grid mode':11s} {'shed?':7s} {'weeks':>6s} {'lots/cap mean':>13s} {'share >=90%':>12s} {'wafers/cap mean':>16s} {'wafers<cap share':>17s}")
for key in sorted(rows):
    a = np.array(rows[key])
    print(f"{key[0]:18s} {key[1]:11s} {key[2]:7s} {len(a):6d} {a[:, 0].mean():13.2f} {(a[:, 0] >= 0.9).mean():12.2f} {a[:, 1].mean():16.2f} {(a[:, 1] < 1.0).mean():17.2f}")
