import sys

import numpy as np

sys.path.insert(0, "lab_scratch")
from probe import run_to

folder = "/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.claude/worktrees/agent-a9fc6562ebf34a489/agents/chip_route"
agent, obs, env, mod = run_to(folder, 2, 12)
v = agent.chip_value(obs)
names = env.unwrapped.instance.nodes if hasattr(env.unwrapped, "instance") else None
print("chip_value by grid node:")
for g, rows in v.items():
    print("  grid", g, [(round(d / 1e6, 1), round(gwh, 1)) for d, gwh in rows])
# fill_chip_flows on someone else's flows: fuel entries must be untouched
flows = np.full(agent.n_slots, 7.0)
before = flows.copy()
agent.fill_chip_flows(obs, flows)
changed = np.flatnonzero(flows != before)
print("slots changed:", len(changed), "all chip-chain:", bool(np.all(agent.kind[changed] > 0)), "fuel untouched:", bool(np.all(flows[agent.kind == 0] == 7.0)))
ref = agent._pull_flows(obs)
a = agent.act(obs)["flows"]
print("fuel slots equal pull's:", bool(np.allclose(a[agent.kind == 0], ref[agent.kind == 0])))
print("usefulness:", {int(k): round(x, 2) for k, x in agent.usefulness(agent._context(obs)).items()})
