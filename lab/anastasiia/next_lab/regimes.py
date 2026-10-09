"""How much of the week's cell is news, not of the network but of the regimes: the premise `news.py` did not test.

    uv run python lab/anastasiia/next_lab/regimes.py --task=small --entropy=111 --episodes=2 --weeks=14

``news.py`` measures the window's forecast and finds the median week changes nothing. But the cell the planner solves
is not the forecast: its rows are the regimes read off the reference trajectory (``Episode.regimes``), and that
trajectory is replayed every week from a state the simulator moved. The neighbouring session measured the
consequence - a warm simplex on Full is worse than a cold interior point, because the cell is not close to last
week's - so the premise "nothing changed, skip the solve" has to be tested on the regimes, not on the network.

Per week this prints, between the regimes of last week's cell moved on by a week and this week's:

- ``grid``, ``fuel``, ``fab``, ``osat`` (whatever ``regimes`` returns): labels that differ, out of those comparable;
- ``all``: the same over every group together.

A week with no network news and no regime news is a week whose cell is last week's cell: there a plan carried from
last week could be played with no solve at all. A week with no network news but with regime news is a week where the
state, not the scenario, moved the program - and then the saving has to come from somewhere else.
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def _moved(table: dict, weeks: int = 1) -> dict:
    """Last week's labels in this window's weeks: the key's first entry is the week."""
    return {(k[0] - weeks, *k[1:]): v for k, v in table.items() if k[0] > weeks}


def main(task: str = "small", entropy: int = 111, episodes: int = 2, first: int = 0, weeks: int = 14,
         folder: str = "anastasiia_plan_hull") -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    shares, by_group = [], {}
    for episode in range(first, first + episodes):
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        u = env.unwrapped
        ag = load(str(resolve(folder).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
        T = u.core._ep.inst.T
        last = None
        for week in range(1, min(weeks, T) + 1):
            obs_next = None
            action = ag.act(obs)
            mode = None
            if getattr(ag, "last", None) is not None:  # the agent keeps (ep, d, ...) of this week for a lab script
                ep, d = ag.last[0], ag.last[1]
                recs = d.get("recs")
                if recs:
                    mode = ep.regimes(recs)[0]
            if mode is not None and last is not None:
                same = diff = 0
                parts = []
                for group, table in mode.items():
                    if not isinstance(table, dict) or group not in last:
                        continue
                    before = _moved(last[group])
                    keys = set(table) & set(before)
                    if not keys:
                        continue
                    d_g = sum(table[k] != before[k] for k in keys)
                    same, diff = same + len(keys) - d_g, diff + d_g
                    parts.append(f"{group} {d_g}/{len(keys)}")
                    by_group.setdefault(group, []).append(d_g / max(len(keys), 1))
                total = same + diff
                if total:
                    shares.append(diff / total)
                    print(f"ep {episode:3d} week {week:3d}  regimes changed {diff:5d}/{total:6d} "
                          f"({100*diff/total:5.2f}%)   " + "  ".join(parts), flush=True)
            last = mode
            obs_next, _r, term, trunc, _i = env.step(action)
            obs = obs_next
            if term or trunc:
                break
    if not shares:
        print("no cells seen: does the agent keep `self.last`?")
        return
    v = np.array(shares)
    print(f"\n{task}, root {entropy}, episodes {first}..{first + episodes - 1}, {len(v)} weeks")
    print(f"  regimes changed   median {100*np.median(v):6.2f}%  mean {100*v.mean():6.2f}%  "
          f"p95 {100*np.quantile(v, .95):6.2f}%  max {100*v.max():6.2f}%   weeks with none: "
          f"{int((v == 0).sum())}/{len(v)}")
    for group, xs in by_group.items():
        a = np.array(xs)
        print(f"  {group:10s}      median {100*np.median(a):6.2f}%  mean {100*a.mean():6.2f}%  max {100*a.max():6.2f}%")


if __name__ == "__main__":
    sys.path[:0] = [str(HERE)]
    fire.Fire(main)
