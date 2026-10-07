"""Where an agent's week goes: one episode played under cProfile, the time summed by function.

    uv run python lab/anastasiia/plan_lab/profile_week.py outputs/plan_lab/agents/hullr_2031 --task=full --episode=1 --weeks=40

Prints the CPU seconds of every week (median, 95th percentile, maximum; this machine, not the container) and the
functions that hold most of the time, cumulative, per week.
"""

import cProfile
import pstats
import time

import fire
import numpy as np


def main(agent: str, task: str = "full", entropy: int = 444, episode: int = 1, weeks: int = 40, top: int = 40,
         params: str = "") -> None:
    import builtins
    import json

    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    import sbf_starter.agents as agents_mod
    from sbf_starter import env_id
    from sbf_starter.agents import resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    builtins.PLAN_LAB_PARAMS = json.loads(params) if params else {}
    ag = agents_mod.load(str(resolve(agent)))(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
    prof, cpu = cProfile.Profile(), []
    for _ in range(weeks):
        t0 = time.process_time()
        prof.enable()
        action = ag.act(obs)
        prof.disable()
        cpu.append(time.process_time() - t0)
        obs, _reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    cpu = np.array(cpu)
    print(f"{agent}, {task} {entropy} episode {episode}, {len(cpu)} weeks: CPU s a week under the profiler: median "
          f"{np.median(cpu):.2f}, p95 {np.percentile(cpu, 95):.2f}, max {cpu.max():.2f}; week 1 {cpu[0]:.2f}")
    stats = pstats.Stats(prof)
    rows = []
    for (file, line, name), (_cc, calls, tt, ct, _callers) in stats.stats.items():
        rows.append((ct / len(cpu), tt / len(cpu), calls / len(cpu), f"{file.split('/')[-1]}:{line} {name}"))
    print(f"{'cum s/week':>10} {'own s/week':>10} {'calls/week':>10}  function")
    for ct, tt, calls, label in sorted(rows, reverse=True)[:top]:
        print(f"{ct:10.3f} {tt:10.3f} {calls:10.1f}  {label}")
    log = getattr(ag, "log", None)
    if log:
        notes = {}
        for row in log:
            notes[str(row[-1])[:40]] = notes.get(str(row[-1])[:40], 0) + 1
        print("notes:", sorted(notes.items(), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    fire.Fire(main)
