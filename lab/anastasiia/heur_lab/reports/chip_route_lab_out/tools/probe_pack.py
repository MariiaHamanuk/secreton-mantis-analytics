import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from probe import run_to

folder, ep, week = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
agent, obs, env, mod = run_to(folder, ep, week)
c = agent._context(obs)
print("week", c.t)
names = agent.static_nodes if hasattr(agent, "static_nodes") else None
st = env.unwrapped
nid = [n for n in range(len(agent.dest))]
import json

inst_nodes = agent.node_id if hasattr(agent, "node_id") else None
from shockbench_flow_agent.convert import agent_config

cid = ["lng", "crude", "nucfuel", "wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"]
# the packaged slots
node_name = {}
for k in (6, 7):
    print("==", cid[k])
    for s in agent.pack_slots[k]:
        print(f"  slot {s:3d} tail {agent.tail[s]} dest {agent.dest[s]} lead {c.lead[s]} cap {c.cap[s]:9.0f} cost {c.cost[s]:6.1f} mask {c.mask[s]} strait {c.strait[s]}")
    for m in sorted({int(agent.dest[s]) for s in agent.pack_slots[k]}):
        for lead in (1, 2, 3, 4):
            print(f"  market {m} lead {lead}: need {agent._need(c, m, k, lead):10.0f}  stock {agent.stock_of(c, m, k):9.0f} arrivals<= {agent.arrivals(c, m, k, c.t + lead):9.0f}")
    for o in sorted({int(agent.tail[s]) for s in agent.pack_slots[k]}):
        print(f"  plant {o} stock {agent.stock_of(c, o, k):9.0f}")
flows = agent._pull_flows(obs)
agent._pack_flows(c, flows)
for k in (6, 7):
    print("flows", cid[k], {s: round(float(flows[s])) for s in agent.pack_slots[k] if flows[s] > 0})
