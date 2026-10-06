"""Disposal cost by commodity and node type for an agent folder over several episodes (scratch).

    python scratch/disposal.py <agent_folder> <first> <n>
"""
import os
import sys
from collections import defaultdict

import numpy as np
from shockbench_flow.dynamics.env import rollout
from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
from shockbench_flow_agent.scoring import _policy_seed, _world
from shockbench_flow_agent.shim import AgentShim, load_agent_class

from sbf_starter import ROOT  # noqa: F401  (sets SBF_CACHE_DIR)
from sbf_starter.agents import resolve

folder, first, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
factory = load_agent_class(resolve(folder), "disposal_probe")
cost = defaultdict(float)
for ep in range(first, first + n):
    inst, omega, marks, fallback = _world("small", 111, ep, 1000, os.environ["SBF_CACHE_DIR"])
    traj = rollout(inst, AgentShim(factory), omega, "standard", _policy_seed(111, ep, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    for rec in traj.records:
        for s, q in enumerate(rec.disposal):
            if q > 0:
                sl = inst.stock_slots[s]
                c = inst.commodities[sl.k]
                cost[(inst.nodes[sl.node].type, c.id)] += q * c.disposal_cost
tot = sum(cost.values())
print(f"disposal cost per episode, USD bn (mean over {n} episodes): total {tot / n / 1e9:.2f}")
for (nt, cid), v in sorted(cost.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {nt:9s} {cid:13s} {v / n / 1e9:7.2f}")
