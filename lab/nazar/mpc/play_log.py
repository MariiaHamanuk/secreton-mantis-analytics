"""Play the first weeks of one episode with an agent; save its actions and the CPU seconds of every week.

    uv run python lab/nazar/mpc/play_log.py <agent> --task=full --weeks=12 --out=outputs/a.npz
    uv run python lab/nazar/mpc/play_log.py <agent B> --task=full --weeks=12 --out=outputs/b.npz --ref=outputs/a.npz

With ``--ref`` the actions are compared with the saved ones week by week (exact equality of every entry) and the
CPU seconds are compared; the agents are in separate processes because two agents that ship an ``sbfv/`` copy would
share the first one imported. Read-only.
"""

import time

import fire
import numpy as np


def main(agent: str, out: str, task: str = "full", entropy: int = 222, episode: int = 0, weeks: int = 12, ref: str | None = None) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    t0 = time.process_time()
    ag = load(agent)(cfg)
    init = time.process_time() - t0
    flows, over, mode, cpu = [], [], [], []
    for _ in range(weeks):
        t = time.process_time()
        action = ag.act(obs)
        cpu.append(time.process_time() - t)
        flows.append(np.asarray(action["flows"], dtype=float))
        over.append(np.asarray(action.get("override_qty", []), dtype=float))
        mode.append(np.asarray(action.get("release_mode", []), dtype=float))
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    np.savez(out, flows=np.array(flows), over=np.array(over), mode=np.array(mode), cpu=np.array(cpu), init=init)
    print(f"{agent}: {len(cpu)} weeks, init {init:.2f} s, CPU per week: " + " ".join(f"{c:.2f}" for c in cpu) + f"  (sum {sum(cpu):.1f})")
    if ref:
        r = np.load(ref)
        same = all(np.array_equal(a, b) for key in ("flows", "over", "mode") for a, b in zip(r[key], np.load(out)[key]))
        n = min(len(r["cpu"]), len(cpu))
        print(f"actions identical to {ref} in all {n} weeks: {same}")
        print(f"CPU sum {float(np.sum(r['cpu'][:n])):.1f} s -> {float(np.sum(cpu[:n])):.1f} s ({100 * (1 - np.sum(cpu[:n]) / np.sum(r['cpu'][:n])):+.1f} % saved)")


if __name__ == "__main__":
    fire.Fire(main)
