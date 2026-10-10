"""Does ``01_solve_handoff``'s other path act the same: the solver's interface without ``passModel``'s form with arrays?

    uv run python lab/anastasiia/frontier_lab/speed/fallback.py outputs/speed_lab/t_x3 --against=ref26a --weeks=6

``Episode.solve`` hands the program over as flat arrays and, where the interface refuses that form with a
``TypeError``, fills a ``HighsLp`` as before. No build met here refuses it, so that path never runs in a play. Here
the folder plays as in ``weeks.py`` (root 444, no clock, no solver limits) behind a solver whose ``passModel`` takes
a model object alone, and its weeks are set against the kept weeks of ``against`` (a tag of ``weeks.py play`` on the
same episode): the action arrays to the bit and the notes. Prints how many programs went the other path.
"""

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
KEPT = ROOT / "outputs" / "speed_lab" / "weeks"
NO_LIMITS = {"warm_share": 0, "solve_share": 0, "solve_seconds": 600}


def main(folder: str, against: str, task: str = "full", episode: int = 3, weeks: int = 6, entropy: int = 444,
         numbers: dict | None = None) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    kept = pickle.loads((KEPT / f"{against}_{task}_{entropy}_{episode}.pkl").read_bytes())["rows"]
    source = Path(folder).resolve()
    with tempfile.TemporaryDirectory(prefix="fallback_") as tmp:
        target = Path(tmp) / f"{source.name}_{task}"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((target / "regime.json").read_text()) | NO_LIMITS | (numbers or {})
        (target / "regime.json").write_text(json.dumps(regime))
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        cls = load(str(target))
        core = sys.modules[cls.__module__]._core
        real, seen = core.highs, {"refused": 0, "object": 0}

        def refusing():
            hs, Highs = real()

            class Old:  # the solver with ``passModel`` of a model object alone
                def __init__(self) -> None:
                    self._h = Highs()

                def __getattr__(self, name: str):
                    return getattr(self._h, name)

                def passModel(self, *args):  # noqa: N802 - the interface's own name
                    if len(args) != 1:
                        seen["refused"] += 1
                        raise TypeError("passModel(): incompatible function arguments")
                    seen["object"] += 1
                    return self._h.passModel(*args)

            return hs, Old

        core.highs = refusing
        agent = cls(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        first, week, done = None, 0, False
        while not done and week < min(weeks, len(kept)):
            action = agent.act(obs)
            a, b = kept[week]["action"], {k: np.array(v) for k, v in action.items()}
            week += 1
            moved = [k for k in sorted(set(a) | set(b)) if k not in a or k not in b or not np.array_equal(a[k], b[k])]
            if first is None and (moved or kept[week - 1]["note"] != tuple(agent.log[-1][1:])):
                first = (week, moved)
            obs, _r, term, trunc, _i = env.step(action)
            done = term or trunc
    verdict = f"the same actions and notes as {against}"
    if first is not None:
        verdict = f"DIFFERS from week {first[0]}: {first[1]}"
    print(f"{folder}: {week} weeks of {task} {entropy} episode {episode} with the form with arrays refused "
          f"({seen['refused']} times; {seen['object']} programs handed over as objects): {verdict}")


if __name__ == "__main__":
    fire.Fire(main)
