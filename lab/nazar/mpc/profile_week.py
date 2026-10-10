"""Where a week's CPU goes: cProfile of the first weeks of one episode, plus the CPU seconds of every week.

    uv run python lab/nazar/mpc/profile_week.py agents/anastasiia_plan_hull2 --task=full --weeks=10

Plays ``weeks`` weeks of one episode with the agent under cProfile (process time, not wall time), prints the CPU
seconds of each ``act`` and the functions with the most own time and the most cumulative time. Read-only.
"""

import cProfile
import io
import pstats
import time

import fire


def main(agent: str, task: str = "full", entropy: int = 222, episode: int = 0, weeks: int = 10, top: int = 28) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    cfg = agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs)
    prof = cProfile.Profile(time.process_time)
    prof.enable()
    ag = load(agent)(cfg)
    prof.disable()
    times = []
    for _ in range(weeks):
        t0 = time.process_time()
        prof.enable()
        action = ag.act(obs)
        prof.disable()
        times.append(time.process_time() - t0)
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    print(f"{agent}: {task} root {entropy} episode {episode}; CPU seconds of weeks 1..{len(times)}:")
    print("  " + " ".join(f"{t:.2f}" for t in times) + f"   (sum {sum(times):.1f}, mean {sum(times) / len(times):.2f})")
    for key in ("tottime", "cumulative"):
        out = io.StringIO()
        pstats.Stats(prof, stream=out).sort_stats(key).print_stats(top)
        text = out.getvalue()
        print(f"\n=== by {key}")
        print("\n".join(line[:170] for line in text.splitlines()[:top + 8]))


if __name__ == "__main__":
    fire.Fire(main)
