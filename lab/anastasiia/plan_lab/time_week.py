"""Where an agent's week goes, by its own steps: named functions of the agent and its core timed as they run.

    uv run python lab/anastasiia/plan_lab/time_week.py outputs/plan_lab/agents/hullr_2031 --task=full --episode=1 --weeks=30

For an agent built by ``lab/anastasiia/regime_lab/build.py`` (``agent.py``, ``plan_core.py``, ``sim_model.py`` in one
folder). CPU seconds a week, this machine, not the container: the whole ``act`` (median, 95th percentile, maximum)
and the mean seconds and calls a week of each step, inclusive of what it calls (the steps nest: ``descend`` holds
``cell``, ``solve`` and ``simulate``).
"""

import inspect
import sys
import time

import fire
import numpy as np


STEPS = {
    "agent": ("_plan", "_roll", "_planned"),
    "core:Episode": ("__init__", "regimes", "cell", "solve", "simulate", "actions", "hints", "clean", "vector", "rows"),
    "core": ("descend", "switch_step", "_rounded", "_whole_weeks"),
    "model:Model": ("window", "restart", "flat", "wire", "forecast"),
    "rules": ("fill_chip_flows", "fill"),
}


def main(agent: str, task: str = "full", entropy: int = 444, episode: int = 1, weeks: int = 30) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    import sbf_starter.agents as agents_mod
    from sbf_starter import env_id
    from sbf_starter.agents import resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    folder = resolve(agent)
    ag = agents_mod.load(str(folder))(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
    spent, calls = {}, {}

    def timed(label, fn):
        def run(*args, **kwargs):
            t0 = time.process_time()
            try:
                return fn(*args, **kwargs)
            finally:
                spent[label] = spent.get(label, 0.0) + time.process_time() - t0
                calls[label] = calls.get(label, 0) + 1

        return run

    def wrap(owner, attr, label):
        """``owner.attr`` timed in place, a static or class method kept as one."""
        if not hasattr(owner, attr):
            return
        raw = inspect.getattr_static(owner, attr)
        if isinstance(raw, staticmethod):
            setattr(owner, attr, staticmethod(timed(label, raw.__func__)))
        elif isinstance(raw, classmethod):
            setattr(owner, attr, classmethod(timed(label, raw.__func__)))
        else:
            setattr(owner, attr, timed(label, getattr(owner, attr)))

    name = folder.name
    core, model = sys.modules.get(f"{name}_core"), sys.modules.get(f"{name}_model")
    for attr in STEPS["agent"]:
        wrap(type(ag), attr, f"agent.{attr}")
    if core is not None:
        for attr in STEPS["core:Episode"]:
            wrap(core.Episode, attr, f"Episode.{attr}")
        for attr in STEPS["core"]:
            wrap(core, attr, f"core.{attr}")
    if model is not None:
        for attr in STEPS["model:Model"]:
            wrap(model.Model, attr, f"Model.{attr}")
    for part, attr in ((type(ag.chips), "fill_chip_flows"), (type(ag.fuel), "fill"), (type(ag.strait), "fill")):
        wrap(part, attr, f"{part.__name__}.{attr}")
    cpu = []
    for _ in range(weeks):
        t0 = time.process_time()
        action = ag.act(obs)
        cpu.append(time.process_time() - t0)
        obs, _reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    cpu = np.array(cpu)
    n = len(cpu)
    print(f"{agent}, {task} {entropy} episode {episode}, {n} weeks: act CPU s a week: mean {cpu.mean():.2f}, median "
          f"{np.median(cpu):.2f}, p95 {np.percentile(cpu, 95):.2f}, max {cpu.max():.2f}")
    print(f"{'s a week':>9} {'calls':>7}  step (inclusive)")
    for label in sorted(spent, key=lambda k: -spent[k]):
        print(f"{spent[label] / n:9.3f} {calls[label] / n:7.1f}  {label}")
    log = getattr(ag, "log", None)
    if log:
        for i in np.argsort(cpu)[::-1][:4]:  # the slowest weeks and what the agent noted in them
            print(f"   week {i + 1}: {cpu[i]:.2f} s, the agent's note {log[i][1:] if i < len(log) else None}")
        notes = {}
        for row in log:
            notes[str(row[-1])[:44]] = notes.get(str(row[-1])[:44], 0) + 1
        print("notes:", sorted(notes.items(), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    fire.Fire(main)
