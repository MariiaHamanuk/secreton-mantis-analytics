"""Tanker fleet slack (7) in the hub's played episodes: weekly use against the cap, and which (edge, lane) use it."""

import collections
import sys

import numpy as np

sys.path.insert(0, "lab/anastasiia/mpc_lab/level1")
from probeB import setup  # noqa: E402

LSF = setup()
from shockbench_flow.dynamics.clip import dup_terms, fleet_caps  # noqa: E402

for n in [int(x) for x in sys.argv[1].split(",")]:
    d, R, rep = LSF.verify_one(n)
    inst = R.inst
    cap = fleet_caps(inst)
    terms = dup_terms(inst)
    pools = inst.commodity_pool
    use = np.zeros((inst.T, 2))
    who = collections.defaultdict(float)
    for t, r in enumerate(R.records[1:]):
        for (e, k, lane), q in r.x.items():
            for ln, dt in terms.get(e, ()):
                if ln is None or ln == lane:
                    use[t, pools[k]] += dt * q
                    if pools[k] == 0 and 13 <= t < 39:
                        who[(inst.edges[e].id, inst.commodities[k].id)] += dt * q / 26
    tb = use[:, 0]
    print(f"ep {n}: tanker fleet cap {cap[0]:.0f}; weeks 14-39 mean use {tb[13:39].mean():.0f}, weeks at >=99% of cap "
          f"{(tb[13:39] >= 0.99 * cap[0]).sum()}/26")
    for k, v in sorted(who.items(), key=lambda x: -x[1])[:6]:
        print(f"     {k[0]:40s} {k[1]:6s} {v:7.0f} /week")
