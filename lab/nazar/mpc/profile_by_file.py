"""CPU of the first weeks of an episode, summed by source file (and by package), to see where a week goes.

    uv run python lab/nazar/mpc/profile_by_file.py lab/nazar/agents/h2_opt --task=small --weeks=8

``cProfile`` on process time. Own time (``tottime``) is added up per file, so a file's number is the work done in its own
functions, not in what they call. The HiGHS solves are C code called from Python: they show as the Python function that
calls them (``scipy.optimize``/``highspy``) when that file is listed. Read-only.
"""

import cProfile
import pstats
import time
from collections import defaultdict
from pathlib import Path

import fire


def main(agent: str, task: str = "small", entropy: int = 222, episode: int = 0, weeks: int = 8, top: int = 16) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    ag = load(agent)(cfg)
    prof = cProfile.Profile(time.process_time)
    done = 0
    t_all = 0.0
    for _ in range(weeks):
        t0 = time.process_time()
        prof.enable()
        action = ag.act(obs)
        prof.disable()
        t_all += time.process_time() - t0
        done += 1
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    stats = pstats.Stats(prof).stats
    by_file, by_pkg = defaultdict(float), defaultdict(float)
    for (fn, _line, name), (_cc, _nc, tt, _ct, _callers) in stats.items():
        path = Path(fn)
        label = path.name if "site-packages" not in fn else "/".join(path.parts[path.parts.index("site-packages") + 1 : path.parts.index("site-packages") + 3])
        if fn.startswith("~") or fn == "":
            label = "<built-in C functions>"
        by_file[label] += tt
        by_pkg[label.split("/")[0] if "/" in label else label] += tt
    total = sum(by_file.values())
    print(f"{agent}: {task} root {entropy} ep {episode}, {done} weeks, CPU per week {t_all / done:.2f} s (profiled; {total:.1f} s own time summed)")
    print(f"\n  {'file':<34}{'own time s':>11}{'share':>8}")
    for label, tt in sorted(by_file.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {label:<34}{tt:11.2f}{100 * tt / total:7.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
