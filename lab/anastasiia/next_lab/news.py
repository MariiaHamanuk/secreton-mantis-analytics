"""How much of a week's program is news: the premise of incremental re-optimisation, measured before it is built.

    uv run python lab/anastasiia/next_lab/news.py --task=small --entropy=111 --episodes=4
    uv run python lab/anastasiia/next_lab/news.py --task=full --entropy=111 --episodes=2 --weeks=30

Each week the planner builds a window of H weeks from the observation and solves it from scratch. If the window of
this week is almost the window of last week moved on by one, then almost every column of the program carries the same
data as last week, and the cost of a week should be proportional to what changed, not to the size of the program.

Per week this prints, between the window of last week moved on by a week and the window of this week:

- ``u``: the share of finite (edge, week) capacities that differ by more than ``tol`` relative, and the share of the
  window's capacity mass that sits on them;
- ``open``, ``kappa``: the same for the straits;
- ``prohibited``: cells that changed;
- ``edges``: how many distinct edges carry any change at all, out of the instance's edges. This is the one that sizes
  a neighbourhood: the columns of an incremental solve are the slots of those edges (and what they feed) across the
  window, not every slot of every week.

The forecast is the planner's own (``sim_model.Model.forecast``), so this measures exactly what the agent sees.
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def _agent(task: str, entropy: int, episode: int, folder: str):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    u = env.unwrapped
    Agent = load(str(resolve(folder).resolve()))
    return env, u, Agent(agent_config(info["static"], info["policy_seed"], u.layout, obs)), obs


def _changed(now: np.ndarray, before: np.ndarray, tol: float) -> tuple[int, int, float, float]:
    """Cells of two arrays of the same shape that differ by more than ``tol`` relative: count, total, mass moved,
    total mass. Infinite and zero cells count as equal when both sides agree."""
    a, b = np.asarray(now, dtype=float), np.asarray(before, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    scale = np.maximum(np.abs(b), 1.0)
    diff = finite & (np.abs(a - b) > tol * scale)
    mass = float(np.sum(np.abs(a[finite])))
    return int(diff.sum()), int(finite.sum()), float(np.sum(np.abs(a[diff] - b[diff]))), mass


def main(task: str = "small", entropy: int = 111, episodes: int = 4, first: int = 0, weeks: int = 0,
         folder: str = "anastasiia_plan_hull", tol: float = 1e-6, horizon: int = 26) -> None:
    rows = []
    for episode in range(first, first + episodes):
        env, u, ag, obs = _agent(task, entropy, episode, folder)
        T = u.core._ep.inst.T
        last = None
        n_edges = len(ag.model.inst.edges)
        for week in range(1, (weeks or T) + 1):
            H = min(horizon, T - week + 1)
            fc = {k: np.array(v) for k, v in ag.model.forecast(obs, H, False).items()}
            if last is not None:
                h = min(fc["u"].shape[0], last["u"].shape[0] - 1)
                out = {}
                for name in ("u", "o", "kappa", "prohibited"):
                    if name not in fc or name not in last:
                        continue
                    a, b = fc[name][:h], last[name][1 : 1 + h]
                    if a.shape != b.shape:
                        continue
                    out[name] = _changed(a, b, tol)
                a, b = fc["u"][:h], last["u"][1 : 1 + h]
                moved = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > tol * np.maximum(np.abs(b), 1.0))
                hot = int(np.unique(np.nonzero(moved)[1]).size) if moved.ndim == 2 else 0
                cells, total, dmass, mass = out.get("u", (0, 1, 0.0, 1.0))
                rows.append((episode, week, cells, total, dmass / max(mass, 1e-9), hot, n_edges, out))
                print(f"ep {episode:3d} week {week:3d}  u changed {cells:6d}/{total:7d} ({100*cells/max(total,1):5.2f}%)"
                      f"  mass {100*dmass/max(mass,1e-9):5.2f}%  edges touched {hot:4d}/{n_edges}"
                      f" ({100*hot/n_edges:5.1f}%)"
                      + "".join(f"  {k} {v[0]}" for k, v in out.items() if k != "u" and v[0]), flush=True)
            last = fc
            obs, _r, term, trunc, _i = env.step(ag.act(obs))
            if term or trunc:
                break
    if not rows:
        return
    cells = np.array([r[2] / max(r[3], 1) for r in rows])
    mass = np.array([r[4] for r in rows])
    hot = np.array([r[5] / max(r[6], 1) for r in rows])
    print(f"\n{task}, root {entropy}, episodes {first}..{first + episodes - 1}, {len(rows)} weeks")
    for name, v in (("cells of u changed", cells), ("capacity mass moved", mass), ("edges touched", hot)):
        print(f"  {name:22s} median {100*np.median(v):6.2f}%  mean {100*v.mean():6.2f}%  p95 {100*np.quantile(v, .95):6.2f}%  max {100*v.max():6.2f}%")


if __name__ == "__main__":
    sys.path[:0] = [str(HERE)]
    fire.Fire(main)
