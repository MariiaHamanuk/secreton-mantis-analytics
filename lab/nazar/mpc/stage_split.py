"""How much of a week is the solver and how much is everything else (the part a clock cannot stop).

    uv run python lab/nazar/mpc/stage_split.py lab/nazar/agents/h2_opt lab/nazar/agents/h2c75 --task=small --weeks=30

Plays the first weeks of one episode; per week the CPU of ``act`` (the agent's own ``log``) and the CPU of every solver
run of the week (``detail["solves"][i]["cpu"]``); the rest is "outside the solver": building the window, replaying the
rules on the model, assembling the program, the fuel and strait rules. Prints the medians and the share of weeks in
which the outside part alone exceeds a given share of the budget. Read-only; the agents have to be played one at a time
on a quiet machine, since their clock reads CPU time.
"""

import statistics as st
import time

import fire

BUDGET = {"small": 2.0, "full": 4.0}


def one(agent: str, task: str, entropy: int, episode: int, weeks: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    ag = load(agent)(cfg)
    rows = []
    for _ in range(weeks):
        t = time.process_time()
        action = ag.act(obs)
        total = time.process_time() - t
        solves = ag.detail[-1]["solves"] if getattr(ag, "detail", None) else []
        solver = sum(s["cpu"] for s in solves)
        rows.append((total, solver, len(solves)))
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    return {"rows": rows, "budget": BUDGET[task]}


def main(*agents: str, task: str = "small", entropy: int = 222, episode: int = 0, weeks: int = 30) -> None:
    for agent in agents:
        res = one(agent, task, entropy, episode, weeks)
        rows, b = res["rows"], res["budget"]
        tot = [r[0] for r in rows]
        sol = [r[1] for r in rows]
        out = [max(r[0] - r[1], 0.0) for r in rows]
        print(f"\n{agent}: {task} root {entropy} ep {episode}, {len(rows)} weeks, budget {b} s")
        print(f"  act CPU:        median {st.median(tot):.2f}, p90 {sorted(tot)[int(0.9 * len(tot))]:.2f}, max {max(tot):.2f}; over the budget in {sum(t > b for t in tot)} weeks")
        print(f"  solver CPU:     median {st.median(sol):.2f}, max {max(sol):.2f} ({st.mean(r[2] for r in rows):.1f} runs a week)")
        print(f"  outside solver: median {st.median(out):.2f}, p90 {sorted(out)[int(0.9 * len(out))]:.2f}, max {max(out):.2f}")
        for share in (0.5, 0.75, 1.0):
            print(f"  weeks whose outside-solver part alone exceeds {share:.2f} x budget: {sum(o > share * b for o in out)} of {len(out)}")


if __name__ == "__main__":
    fire.Fire(main)
