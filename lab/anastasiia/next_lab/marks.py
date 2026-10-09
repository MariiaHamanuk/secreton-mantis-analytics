"""How often the whole-week marks change: whether the week's first linear program has to be built at all.

    uv run python lab/anastasiia/next_lab/marks.py --task=small --entropy=111 --episodes=3 --weeks=20
    uv run python lab/anastasiia/next_lab/marks.py --task=full --entropy=111 --episodes=2 --weeks=16

The hull cell exists for one output: the set of (week, grid) cells ``_rounded`` writes as whole, about 98 numbers on
Full, and it costs ~40 % of the week to get them. ``regimes.py`` showed the cell is never last week's cell (4 % of
labels move every week, no week with none), so a plan cannot simply be carried with no solve. This asks the narrower
question: do the *marks* move? If last week's marks, moved on by a week, are this week's marks in most weeks, then
building the cell every week is waste, and the trigger should be a cheap test, not a fixed period (``hull_every 2``
is a fixed period and costs 0.005).

Per week, between last week's marks moved on by a week and this week's: how many cells only one of them has, and
whether the two sets agree exactly. Only cells of weeks both windows cover are compared, so a mark that merely fell
off the end of the window is not counted as a change.

Run it with a folder whose ``descend`` returns ``out["whole"]`` (``anastasiia_plan_hull_tilt`` does, and with its two
numbers at 0 it is ``anastasiia_plan_hull`` to the cent).
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent


def main(task: str = "small", entropy: int = 111, episodes: int = 3, first: int = 0, weeks: int = 20,
         folder: str = "lab/anastasiia/next_lab/agents/anastasiia_plan_hull_tilt", horizon: int = 26) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    same_weeks, moved_shares, sizes = 0, [], []
    seen = 0
    for episode in range(first, first + episodes):
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        u = env.unwrapped
        ag = load(str(resolve(folder).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
        T = u.core._ep.inst.T
        last = None
        for week in range(1, min(weeks, T) + 1):
            action = ag.act(obs)
            now = None
            if getattr(ag, "last", None) is not None:
                now = ag.last[1].get("whole")
            if now is not None and last is not None:
                before = {(k[0] - 1, *k[1:]) for k in last if k[0] > 1}
                # compare only on the weeks both windows cover: a mark that fell off the end is not a change
                span = min(max((k[0] for k in now), default=0), max((k[0] for k in before), default=0))
                a = {k for k in now if k[0] <= span}
                b = {k for k in before if k[0] <= span}
                union = len(a | b)
                diff = len(a ^ b)
                seen += 1
                sizes.append(len(a))
                moved_shares.append(diff / max(union, 1))
                same_weeks += int(diff == 0)
                print(f"ep {episode:3d} week {week:3d}  marks {len(a):3d} (was {len(b):3d})  "
                      f"differ {diff:3d}/{union:3d} ({100*diff/max(union,1):5.1f}%)"
                      f"{'   SAME' if diff == 0 else ''}", flush=True)
            last = now
            obs, _r, term, trunc, _i = env.step(action)
            if term or trunc:
                break
    if not seen:
        print("no marks seen: does the folder's descend return out['whole']?")
        return
    v = np.array(moved_shares)
    s = np.array(sizes)
    print(f"\n{task}, root {entropy}, episodes {first}..{first + episodes - 1}, {seen} weeks")
    print(f"  marks a week          median {np.median(s):.1f}  max {s.max()}")
    print(f"  marks that differ     median {100*np.median(v):6.2f}%  mean {100*v.mean():6.2f}%  "
          f"p95 {100*np.quantile(v, .95):6.2f}%")
    print(f"  weeks with the same set: {same_weeks}/{seen} ({100*same_weeks/seen:.0f}%)")


if __name__ == "__main__":
    sys.path[:0] = [str(HERE)]
    fire.Fire(main)
