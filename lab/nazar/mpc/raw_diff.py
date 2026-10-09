"""Where the program sends raw chips differently from the rules, along the rules' own trajectory.

    uv run python lab/nazar/mpc/raw_diff.py --episodes=8

Plays the rules (``lab/nazar/agents/probe2_base``: rules_v2 plus the program) and every week records the rules'
raw-chip flows (commodities 4 and 5) and the program's, per action slot. Prints, per fab, the slots with total flow
of each over the episodes (mean per episode), the share going to each destination, and the weeks where they differ most.
Read-only; the played trajectory is the rules' own, so the program sees the states the rules produce.
"""

import importlib.util
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
AGENT = ROOT / "lab" / "nazar" / "agents" / "probe2_base"


def play(n: int, task: str, entropy: int):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id

    sys.path.insert(0, str(AGENT))
    spec = importlib.util.spec_from_file_location(f"raw_diff_agent_{n}", AGENT / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = mod.Agent(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    inst = ag.mpc.inst
    k = np.array([s[1] for s in inst.action_slots])
    raw = np.flatnonzero(np.isin(k, [4, 5]))
    rules, lp = [], []
    done = False
    while not done:
        action = ag.rules.act(obs)
        plan = ag.mpc.act(obs)
        rules.append(np.asarray(action["flows"])[raw])
        lp.append(np.asarray(plan["flows"])[raw])
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    meta = []  # plain data only: the program's own classes cannot be unpickled in the parent process
    for s in raw:
        e, kk, _lane = inst.action_slots[int(s)]
        meta.append((int(s), int(inst.edges[e].tail), str(inst.edges[e].id), int(kk)))
    return raw, np.array(rules), np.array(lp), meta


def main(episodes: int = 8, task: str = "small", entropy: int = 111, n_jobs: int = 8) -> None:
    from joblib import Parallel, delayed

    runs = Parallel(n_jobs=n_jobs)(delayed(play)(n, task, entropy) for n in range(episodes))
    raw, _r, _l, meta = runs[0]
    rules = np.mean([r for _x, r, _l, _i in runs], axis=0)  # (T, slots)
    lp = np.mean([flow for _x, _r, flow, _i in runs], axis=0)
    print(
        f"{task} root {entropy}, {episodes} episodes: raw-chip slots, mean flow per episode (units), rules vs program"
    )
    print(f"  {'slot':>4} {'fab(tail)':>9} {'edge':<34}{'k':>2}{'rules':>12}{'program':>12}{'diff':>12}")
    rows = []
    for j, (s, tail, eid, kk) in enumerate(meta):
        rows.append((rules[:, j].sum() - lp[:, j].sum(), s, tail, eid, kk, rules[:, j].sum(), lp[:, j].sum()))
    for d, s, tail, eid, kk, a, b in sorted(rows, key=lambda r: -abs(r[0]))[:24]:
        print(f"  {s:>4} {tail:>9} {eid[:33]:<34}{kk:>2}{a:12,.0f}{b:12,.0f}{a - b:12,.0f}")
    print("\n  totals by fab (tail): rules / program")
    for tail in sorted({r[2] for r in rows}):
        a = sum(r[5] for r in rows if r[2] == tail)
        b = sum(r[6] for r in rows if r[2] == tail)
        print(f"  fab {tail}: {a:12,.0f} / {b:12,.0f}")


if __name__ == "__main__":
    fire.Fire(main)
