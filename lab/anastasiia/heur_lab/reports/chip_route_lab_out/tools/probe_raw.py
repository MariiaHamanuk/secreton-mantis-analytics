import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from probe import run_to

folder, ep, week = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
agent, obs, env, mod = run_to(folder, ep, week)
c = agent._context(obs)
print("week", c.t, "open", c.open.round(2), "kappa", c.kappa.round(0))
cid = ["lng", "crude", "nucfuel", "wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"]
for n in agent.fabs:
    r = agent.fab[n]["out"]
    print(f"fab node {n} out {cid[r]} stock {agent.stock_of(c, n, r):10.0f}")
    for s in agent.raw_slots[n]:
        print(f"   slot {s:3d} -> {agent.dest[s]} lead {c.lead[s]} cap {c.cap[s]:9.0f} u_first {c.u[agent.edge[s]]:9.0f} mask {c.mask[s]} strait {c.strait[s]} cost {c.cost[s]:7.1f}")
for o in agent.osats:
    for r in (4, 5):
        if r in agent.osat[o]["pack"]:
            for lead_in in (1,):
                print(f"plant {o} raw {cid[r]} room {agent._room(c, o, r, lead_in):12.0f} outlet {agent._outlet(c, o, agent.osat[o]['pack'][r]):10.0f}")
flows = agent._pull_flows(obs)
agent._raw_flows(c, flows)
for n in agent.fabs:
    print(n, {s: round(float(flows[s])) for s in agent.raw_slots[n] if flows[s] > 0})
