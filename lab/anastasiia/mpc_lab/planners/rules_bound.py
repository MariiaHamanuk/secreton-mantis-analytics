import sys, numpy as np
sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
import package_baselines as pb
from joblib import Parallel, delayed
def one(n):
    import sbf_starter
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.oracle.lp import build_lp, solve_oracle
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world
    inst, omega, marks, fb = _world("small", 111, n, REPLICATIONS, str(default_cache_dir()))
    out = []
    for rules in (False, True):
        r = solve_oracle(build_lp(inst, marks, planning_rules=rules))
        out.append(r.J_cents)
    return out
if __name__ == "__main__":
    from sbf_starter import scoring
    refs = list(scoring.episode_set("small", 64, entropy=111, verbose=False).references)
    res = Parallel(n_jobs=8)(delayed(one)(n) for n in range(64))
    for i, name in enumerate(("LP without the simulator's rules (the clairvoyant)", "LP with the simulator's rules")):
        costs = [r[i] for r in res]
        if any(c is None for c in costs): print(name, "unsolved:", sum(c is None for c in costs)); costs = [c if c is not None else refs[j]["J_naive_cents"] for j, c in enumerate(costs)]
        score, lv = pb.rss(refs, costs); print(f"{name:52s} {score:.4f}", [round(v, 3) for v in lv.values()])
