"""An agent's played actions kept in the layout ``regime_lab/plan.py``'s ``start_hybrid`` reads, so a descent with
the whole future can start from the agent's own trajectory (paradigm_lab P7).

    uv run python lab/anastasiia/next_lab/record_start.py agents/anastasiia_plan_hull3 --task=small --entropy=444 --episodes=8

``play.py`` keeps a week's outcomes and not its action, and ``plan.py``'s start reads
``outputs/search_lab/played/<agent>_<task>_<entropy>_<n>.pkl`` with an ``"actions"`` key, so neither of them can hand
a trajectory of a new model to the offline descent. This writes exactly that file and nothing else: the agent is
played once per episode in the package's environment, and every week's action is kept as the environment took it.

Episodes already written are skipped, so a stopped run resumes.
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PLAYED = ROOT / "outputs" / "search_lab" / "played"


def one(agent: str, task: str, entropy: int, n: int) -> tuple[int, int]:
    """One episode played: the count of weeks kept and the episode's cost in cents."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from shockbench_flow_agent.convert import action_to_wire

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    actions = []
    while True:
        action = ag.act(obs)
        # The agent speaks the gym Dict action; the protocol's wire action (what `validate_action` and so
        # `Episode.validated` read) carries the week and names the flows by slot. The wrapper converts, so the same
        # converter is used here rather than a hand-rolled one
        week = int(np.asarray(obs["week"]).ravel()[0])
        actions.append(action_to_wire(u.layout, week, action))
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    name = Path(agent).name
    PLAYED.mkdir(parents=True, exist_ok=True)
    path = PLAYED / f"{name}_{task}_{entropy}_{n}.pkl"
    path.write_bytes(pickle.dumps({"actions": actions, "J": int(u.core._ep.traj.J_cents)}))
    return len(actions), int(u.core._ep.traj.J_cents)


def main(agent: str, task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0,
         n_jobs: int = 2) -> None:
    from joblib import Parallel, delayed

    name = Path(agent).name
    todo = [n for n in range(first, first + episodes)
            if not (PLAYED / f"{name}_{task}_{entropy}_{n}.pkl").is_file()]
    if not todo:
        print(f"{name}: every episode of {task} root {entropy} is already written")
        return
    print(f"{name}: playing {len(todo)} episode(s) of {task} root {entropy}: {todo}", flush=True)
    done = Parallel(n_jobs=n_jobs)(delayed(one)(agent, task, entropy, n) for n in todo)
    for n, (weeks, J) in zip(todo, done):
        print(f"  ep {n}: {weeks} weeks, {J / 1e11:8.1f} (1e9 USD)", flush=True)
    print(f"written to {PLAYED.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    fire.Fire(main)
