"""Does a changed ``plan_core.py`` build the same cells and the same matrices as the base one, call for call?

    uv run python lab/anastasiia/frontier_lab/speed/twin.py outputs/speed_lab/t_x3 --episode=3 --weeks=12

``same.py check`` compares the actions; this compares what is handed to the solver. The folder plays as in
``weeks.py`` (root 444, no clock, no solver limits) and at every call of ``Episode.cell`` and ``Episode.rows`` the
base sources' function (``outputs/speed_lab/t_base/plan_core.py``, loaded beside) is run on the same arguments: the
cell's bounds, costs, rows, row bounds, tags and hull marks, and the matrix's three arrays with the row bounds, must
be equal to the bit and of the same types. Prints the number of calls compared and the first difference, if any.
"""

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "outputs" / "speed_lab" / "t_base" / "plan_core.py"
NO_LIMITS = {"warm_share": 0, "solve_share": 0, "solve_seconds": 600}


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def main(folder: str, task: str = "full", episode: int = 3, weeks: int = 12, entropy: int = 444,
         numbers: dict | None = None) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    source = Path(folder).resolve()
    with tempfile.TemporaryDirectory(prefix="twin_") as tmp:
        target = Path(tmp) / f"{source.name}_{task}"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((target / "regime.json").read_text()) | NO_LIMITS | (numbers or {})
        (target / "regime.json").write_text(json.dumps(regime))
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        cls = load(str(target))
        core = sys.modules[cls.__module__]._core
        spec = importlib.util.spec_from_file_location("twin_base_core", BASE)
        base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base)
        base.PKG = core.PKG
        cell, rows = core.Episode.cell, core.Episode.rows
        seen = {"cell": 0, "rows": 0, "bad": []}

        def cell2(ep, mode, ref, anchor=None, price=None, bonus=None):
            snap = {k: dict(v) for k, v in mode.items()}
            new = cell(ep, mode, ref, anchor, price, bonus)
            old = base.Episode.cell(ep, snap, ref, anchor, price, bonus)
            seen["cell"] += 1
            kinds = [type(x) for x in new.r[-9:]] == [type(x) for x in old.r[-9:]]  # plain ints, not numpy's
            ok = (_same(new.lb, old.lb) and _same(new.ub, old.ub) and _same(new.cost, old.cost) and new.r == old.r
                  and new.c == old.c and new.v == old.v and new.lo == old.lo and new.hi == old.hi
                  and new.tags == old.tags and new.hull == old.hull and kinds
                  and {type(x) for x in new.lo} == {type(x) for x in old.lo})  # fmt: skip
            if not ok:
                seen["bad"].append(("cell", seen["cell"]))
            return new

        def rows2(ep, C):
            A, lo, hi = rows(ep, C)
            A0, lo0, hi0 = base.Episode.rows(ep, C)
            seen["rows"] += 1
            ok = (A.shape == A0.shape and _same(A.indptr, A0.indptr) and _same(A.indices, A0.indices)
                  and _same(A.data, A0.data) and _same(lo, lo0) and _same(hi, hi0))  # fmt: skip
            if not ok:
                seen["bad"].append(("rows", seen["rows"]))
            return A, lo, hi

        core.Episode.cell, core.Episode.rows = cell2, rows2
        agent = cls(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        done, week = False, 0
        while not done and week < weeks:
            week += 1
            action = agent.act(obs)
            obs, _r, term, trunc, _i = env.step(action)
            done = term or trunc
    verdict = "all the same to the bit" if not seen["bad"] else f"DIFFERENT: {seen['bad'][:6]}"
    print(f"{folder}: {week} weeks of {task} {entropy} episode {episode}, {seen['cell']} cells and {seen['rows']} "
          f"matrices against the base sources: {verdict}")


if __name__ == "__main__":
    fire.Fire(main)
