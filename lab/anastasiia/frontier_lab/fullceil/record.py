"""Play an agent folder once per episode and keep every week's action as the environment took it (the tanker releases
included: ``hazard_lab/play.py`` keeps the dispatches only, and a kept play replayed without its releases costs
hundreds of bn more on Full).

    uv run python lab/anastasiia/frontier_lab/fullceil/record.py outputs/hazard_lab/agents/tah0_f --tag=tah0_f --only=2,3,9

A lab agent with ``tell_truth`` is handed the scenario exactly as ``hazard_lab/play.py`` hands it. One file an episode
in ``outputs/frontier_lab/fullceil/starts/``; an episode already there is not played again. No CPU limit.
"""

import importlib.util
import pickle
import time
from pathlib import Path

import common as K
import fire
import numpy as np


def _hazard_play():
    spec = importlib.util.spec_from_file_location("hazard_play", K.LAB / "hazard_lab" / "play.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def start_path(tag: str, task: str, entropy: int, n: int) -> Path:
    return K.OUT / "starts" / f"{tag}_{task}_{entropy}_{n}.pkl"


def main(agent: str, tag: str, task: str = "full", entropy: int = 444, only: str | tuple | int = "", first: int = 0,
         episodes: int = 16) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import action_to_wire, agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    hp = _hazard_play()
    cls = load(str(resolve(agent).resolve()))
    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else range(first, first + episodes)
    for n in ns:
        path = start_path(tag, task, entropy, n)
        if path.is_file():
            continue
        t0 = time.process_time()
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": n})
        u = env.unwrapped
        ag = cls(agent_config(info["static"], info["policy_seed"], u.layout, obs))
        if hasattr(ag, "tell_truth"):
            wants = getattr(ag, "p", {}).get("truth_events") or getattr(ag, "p", {}).get("events_seen")
            ag.tell_truth({"marks": u.core._ep.marks, "events": hp.events_of(u.core._ep.inst, u._omega(n)) if wants else None})
        actions, cpu = [], []
        while True:
            t1 = time.process_time()
            action = ag.act(obs)
            cpu.append(time.process_time() - t1)
            week = int(np.asarray(obs["week"]).ravel()[0])
            actions.append(action_to_wire(u.layout, week, action))
            obs, _r, term, trunc, _i = env.step(action)
            if term or trunc:
                break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"actions": actions, "J": int(u.core._ep.traj.J_cents), "cpu": cpu}))
        print(f"{tag} ep {n}: {len(actions)} weeks, {u.core._ep.traj.J_cents / K.BN:.2f} bn, {time.process_time() - t0:.0f} s CPU", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
