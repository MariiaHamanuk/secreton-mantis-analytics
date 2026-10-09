"""The support of the week's optimum: can the same answer come out of a program ten times smaller?

    uv run python lab/anastasiia/next_lab/support.py --task=full --entropy=111 --episode=0 --weeks=6
    uv run python lab/anastasiia/next_lab/support.py --task=small --entropy=111 --episode=0 --weeks=8

A vertex of a linear program has at most as many nonzeros as the program has rows, and a time-expanded
multicommodity flow carries flow on far fewer routes than it has columns. If the optimum of the week's cell lives on
a few thousand of its fifty thousand columns, then the method for this problem class is not a factorisation of the
whole matrix (simplex or interior point) but column generation: solve a restricted master over the columns that
carried flow last week plus a priced-in candidate set, then certify with reduced costs that no excluded column would
enter. That is exact - not a neighbourhood heuristic - and its cost is shortest paths, not factorisations.

This measures, for each week's exact cell: columns, rows, nonzeros of the optimum, and how the support of this week's
optimum overlaps the support of last week's (moved on by a week). The second number decides whether last week's
support is a good starting set.
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def main(task: str = "full", entropy: int = 111, episode: int = 0, weeks: int = 6,
         folder: str = "anastasiia_plan_hull", tol: float = 1e-7) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    u = env.unwrapped
    ag = load(str(resolve(folder).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    nc = None
    last = None
    for week in range(1, weeks + 1):
        action = ag.act(obs)
        if getattr(ag, "last", None) is not None:
            ep, d = ag.last[0], ag.last[1]
            recs = d.get("recs")
            if recs:
                mode, ref = ep.regimes(recs)
                C = ep.cell(mode, ref)
                A, lo, hi = ep.rows(C)
                t0 = time.process_time()
                sol = ep.solve(C, method="ipm", time_limit=20.0)
                secs = time.process_time() - t0
                if sol.get("x") is not None:
                    x = np.asarray(sol["x"])
                    nz = np.flatnonzero(np.abs(x) > tol)
                    nc = nc or ep.nc
                    # the support moved on by a week: a column of week t becomes the column of week t-1
                    moved = {int(j) - nc for j in nz if j >= nc}
                    text = ""
                    if last is not None:
                        inter = len(moved & last)
                        text = (f"  of last week's support {100*inter/max(len(last),1):5.1f}% is used again; "
                                f"new columns {len(set(map(int, nz)) - last):5d}")
                    print(f"week {week:3d}  cols {ep.N:6d}  rows {A.shape[0]:6d}  nonzeros {len(nz):6d} "
                          f"({100*len(nz)/ep.N:5.2f}% of cols, {100*len(nz)/A.shape[0]:5.1f}% of rows)  "
                          f"ipm {secs:5.2f}s{text}", flush=True)
                    last = set(map(int, nz))
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break


if __name__ == "__main__":
    sys.path[:0] = [str(HERE)]
    fire.Fire(main)
