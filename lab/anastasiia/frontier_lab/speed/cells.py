"""Keep the programs the model solves in the first weeks of an episode, to time the solver on them outside a play.

    uv run python lab/anastasiia/frontier_lab/speed/cells.py outputs/speed_lab/t_ref --episode=3 --weeks=12

Plays as ``weeks.py play`` does (root 444, no clock, no solver limits, the folder's ``regime.json`` with ``numbers``
over it) and writes every program handed to ``Episode.solve`` to
``outputs/speed_lab/cells/<task>_<entropy>_<episode>/w<week>_<n>_<what>.npz``: the matrix (column-wise), the row and
column bounds, the costs, the offset, the solver's name and options as the model sets them, and the solution it got
(status, objective, x), so that a bench can say whether another way of solving gives the same x.
"""

import json
import shutil
import tempfile
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "speed_lab" / "cells"
NO_LIMITS = {"warm_share": 0, "solve_share": 0, "solve_seconds": 600}


def main(folder: str, task: str = "full", episode: int = 3, weeks: int = 12, entropy: int = 444,
         numbers: dict | None = None, tag: str = "") -> None:
    import sys

    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    out = OUT / f"{task}_{entropy}_{episode}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    source = Path(folder).resolve()
    with tempfile.TemporaryDirectory(prefix="cells_") as tmp:
        target = Path(tmp) / f"{source.name}_{task}"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((target / "regime.json").read_text()) | NO_LIMITS | (numbers or {})
        (target / "regime.json").write_text(json.dumps(regime))
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        cls = load(str(target))
        core = sys.modules[cls.__module__]._core
        solve, state = core.Episode.solve, {"week": 0, "n": 0}

        def kept(ep, C, method="simplex", time_limit=600.0, basis=None, crossover=True, what="", ipm_tol=None,
                 big="ipm"):
            sol = solve(ep, C, method=method, time_limit=time_limit, basis=basis, crossover=crossover, what=what,
                        ipm_tol=ipm_tol, big=big)
            A, lo, hi = ep.rows(C)
            name = big if method == "auto" and ep.N > core.BIG else "simplex" if method == "auto" else method
            state["n"] += 1
            np.savez(
                out / f"w{state['week']:03d}_{state['n']}_{what.replace(' ', '_')}.npz",
                indptr=A.indptr, indices=A.indices, data=A.data, shape=np.array(A.shape), lo=lo, hi=hi, lb=C.lb,
                ub=C.ub,
                cost=ep.obj if C.cost is None else ep.obj + C.cost, offset=float(ep.offset), method=name,
                crossover=bool(crossover), ipm_tol=np.nan if ipm_tol is None else float(ipm_tol), status=sol["status"],
                J=float(sol["J"]), x=sol.get("x", np.zeros(0)), T=int(ep.T), nc=int(ep.nc), n0=int(ep.n0),
                n1=int(ep.n1), n2=int(ep.n2), N=int(ep.N), nub=int(ep.nub), neq=int(ep.neq), G=int(ep.G), S=int(ep.S),
                cell_cost=np.zeros(0) if C.cost is None else C.cost, base_rows=int(ep.base.shape[0]),
            )  # fmt: skip
            return sol

        core.Episode.solve = kept
        agent = cls(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        done = False
        while not done and state["week"] < weeks:
            state["week"] += 1
            action = agent.act(obs)
            obs, _r, term, trunc, _i = env.step(action)
            done = term or trunc
    print(f"{out}: {state['n']} programs of {state['week']} weeks")


if __name__ == "__main__":
    fire.Fire(main)
