"""Does restricting the week's program to last week's support lose the optimum, and does it save the time?

    uv run python lab/anastasiia/next_lab/restrict.py --task=full --entropy=111 --episode=0 --weeks=6

``support.py`` measured that the optimum of the exact cell lives on a quarter of its columns and that 86 % of last
week's support carries flow again. The method that follows is column generation: solve a master restricted to the
columns that carried flow, then certify with reduced costs. This tests the two things that decide whether it is worth
building, on the real cells the agent solves:

1. **Does it lose anything?** The restricted optimum's objective against the full optimum's, relative. A gap of 0
   means last week's support already contained this week's optimum and no pricing round is needed at all.
2. **Does it save time?** Columns outside the support are fixed at zero, which lets the solver's presolve remove them,
   so the time is the time of the smaller program.

A loss with a saving means pricing rounds are needed and the question becomes how many. A loss with no saving kills
the method for the interior point and leaves it only for a simplex.
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def main(task: str = "full", entropy: int = 111, episode: int = 0, weeks: int = 6,
         folder: str = "anastasiia_plan_hull", tol: float = 1e-7, method: str = "ipm", pad: float = 0.0, singletons: bool = True) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    u = env.unwrapped
    ag = load(str(resolve(folder).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    keep = None  # last week's support, already moved on by a week
    for week in range(1, weeks + 1):
        action = ag.act(obs)
        if getattr(ag, "last", None) is not None:
            ep, d = ag.last[0], ag.last[1]
            recs = d.get("recs")
            if recs:
                mode, ref = ep.regimes(recs)
                C = ep.cell(mode, ref)
                t0 = time.process_time()
                full = ep.solve(C, method=method, time_limit=30.0)
                t_full = time.process_time() - t0
                x = np.asarray(full["x"]) if full.get("x") is not None else None
                if x is None:
                    continue
                if keep is not None:
                    free = np.zeros(ep.N, dtype=bool)
                    free[keep[keep < ep.N]] = True
                    if pad:  # the columns of the weeks nearest the present stay free whatever last week used
                        free[: int(pad * ep.nc)] = True
                    R = ep.cell(mode, ref)  # a fresh cell: fixing is destructive
                    if singletons:
                        # a column with at most one row is a slack, a disposal or a scrap: zero in the optimum but
                        # the rows' equalities have no solution without it, so the support alone is infeasible
                        A = ep.rows(R)[0].tocsc()
                        free |= np.diff(A.indptr) <= 1
                    out = ~free
                    R.ub[out] = np.minimum(R.ub[out], 0.0)
                    R.lb[out] = np.maximum(R.lb[out], 0.0)
                    t0 = time.process_time()
                    small = ep.solve(R, method=method, time_limit=30.0)
                    t_small = time.process_time() - t0
                    if small["status"] == "Optimal":
                        gap = (small["J"] - full["J"]) / max(abs(full["J"]), 1.0)
                        print(f"week {week:3d}  kept {int(free.sum()):6d}/{ep.N} cols  "
                              f"full {t_full:5.2f}s  restricted {t_small:5.2f}s ({t_small/max(t_full,1e-9):4.2f}x)  "
                              f"objective gap {gap:+.3e}", flush=True)
                    else:
                        print(f"week {week:3d}  kept {int(free.sum()):6d}/{ep.N} cols  "
                              f"full {t_full:5.2f}s  restricted {small['status']} in {t_small:5.2f}s", flush=True)
                keep = np.flatnonzero(np.abs(x) > tol) - ep.nc  # this week's support, moved on by a week
                keep = keep[keep >= 0]
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break


if __name__ == "__main__":
    sys.path[:0] = [str(HERE)]
    fire.Fire(main)
