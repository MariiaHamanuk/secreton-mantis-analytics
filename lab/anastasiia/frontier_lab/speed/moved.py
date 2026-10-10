"""How far does a week's start lie from what last week's plan expected of it?

    uv run python lab/anastasiia/frontier_lab/speed/moved.py outputs/speed_lab/t_ref --episode=3 --weeks=30

The exact cell could be skipped in a week where neither the forecast nor the stock moved against last week's plan:
the carried plan would then be played as it is. Here the model plays as in ``weeks.py`` (root 444, no clock, no
solver limits) and every week the carried plan's replay on this week's window (weeks 1 to H - 1) is set against the
same weeks of the trajectory last week's plan was kept with (its weeks 2 to H): the largest difference of a stock,
as a share of that stock slot's largest level in the two trajectories, the difference of the weeks' summed cost as a
share of it, and the same for the first four weeks alone; then what the week's new plan gained over the carried one
in the model's cost, as a share of the window's cost. A week is "quiet" at a threshold when all of the first three
are under it.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
NO_LIMITS = {"warm_share": 0, "solve_share": 0, "solve_seconds": 600}


def main(folder: str, task: str = "full", episode: int = 3, weeks: int = 30, entropy: int = 444) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    source = Path(folder).resolve()
    with tempfile.TemporaryDirectory(prefix="moved_") as tmp:
        target = Path(tmp) / f"{source.name}_{task}"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((target / "regime.json").read_text()) | NO_LIMITS
        (target / "regime.json").write_text(json.dumps(regime))
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        cls = load(str(target))
        core = sys.modules[cls.__module__]._core
        simulate, first = core.Episode.simulate, {}

        def kept(ep, acts):
            out = simulate(ep, acts)
            first.setdefault("recs", out[0])  # the week's first play is the carried plan's
            return out

        core.Episode.simulate = kept
        agent = cls(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        rows, last, done, week = [], None, False, 0
        while not done and week < weeks:
            week += 1
            first.clear()
            action = agent.act(obs)
            note = agent.log[-1]
            if last is not None and "recs" in first and str(note[3]).startswith("carried"):
                now, then = first["recs"][:-1], last[1:]
                n = min(len(now), len(then))
                a = np.array([r.stock for r in now[:n]], dtype=float)
                b = np.array([r.stock for r in then[:n]], dtype=float)
                scale = np.maximum(1.0, np.maximum(np.abs(a).max(axis=0), np.abs(b).max(axis=0)))
                stock = float((np.abs(a - b) / scale).max())
                stock4 = float((np.abs(a[:4] - b[:4]) / scale).max())
                ca, cb = sum(r.cost_cents for r in now[:n]), sum(r.cost_cents for r in then[:n])
                cost = abs(ca - cb) / max(1.0, abs(cb))
                gain = (note[1] - note[2]) / max(1e-9, abs(note[1]))
                rows.append((week, stock, cost, stock4, gain))
                print(f"week {week:3d}: stock moved {stock:.2e} (first four weeks {stock4:.2e}), cost moved "
                      f"{cost:.2e}, the new plan gained {gain:.2e} of the window's cost  {note[3]}", flush=True)
            last = agent.last[1]["recs"] if agent.acts is not None else None
            obs, _r, term, trunc, _i = env.step(action)
            done = term or trunc
    a = np.array(rows)
    for thr in (1e-6, 1e-4, 1e-3, 1e-2):
        quiet = (a[:, 1] < thr) & (a[:, 2] < thr)
        print(f"quiet at {thr:g}: {int(quiet.sum())} of {len(a)} weeks" + (f", the new plan's gain there median "
              f"{np.median(a[quiet, 4]):.2e} largest {a[quiet, 4].max():.2e}" if quiet.any() else ""))  # fmt: skip
    print(f"all weeks: the new plan's gain median {np.median(a[:, 4]):.2e}, 90th percentile "
          f"{np.percentile(a[:, 4], 90):.2e}")


if __name__ == "__main__":
    fire.Fire(main)
