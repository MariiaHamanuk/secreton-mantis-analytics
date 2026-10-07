"""What thr_min / the early lng buffer change in the fuel rules: one played episode per (agent, episode), fuel trace on.

    uv run python lab/anastasiia/mpc_lab/planners/thrmin_diag.py --episodes=0,1,2,3 --entropy=111

Per agent and grid (lng, the rationed fuel): weeks whose last closing stock is under the threshold, the throttle's
target ration (mean, share of weeks under 0.92), closing grid stock in weeks 1-13, the valve mode counts, and shed in
weeks 1-6 / 7-13 / 14-T; plus the episode cost J. Read-only; prints a table.
"""

import os
import sys
from collections import defaultdict

import fire
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
AGENTS = {
    "base": "agents/anastasiia_hybrid_chiplp",
    "thrmin": "lab/anastasiia/mpc_lab/agents/hyb_thrmin",
    "early": "lab/anastasiia/mpc_lab/agents/hyb_early",
    "both": "lab/anastasiia/mpc_lab/agents/hyb_thrmin_early",
}


def play(agent_dir, task, entropy, n):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_gym.dashboard import record_episode

    from sbf_starter import env_id
    from sbf_starter.agents import load

    cls = load(os.path.join(ROOT, agent_dir))
    box = {}

    def factory(config):
        a = cls(config)
        a.fuel.P = {**a.fuel.P, "trace": True}
        box["a"], box["config"] = a, config
        return a

    env = gym.make(env_id(task), entropy=entropy)
    rec = record_episode(env, factory, options={"episode": n})
    obs = rec["obs"]
    shed = np.asarray(obs["last_week.shed.qty"][1:])  # (T, G)
    return rec["meta"]["J_cents"] / 100, shed, box["a"].fuel, box["config"]


def main(episodes="0,1,2,3", entropy: int = 111, task: str = "small", agents="base,thrmin,early,both"):
    eps = [int(x) for x in str(episodes).split(",")] if not isinstance(episodes, tuple) else list(episodes)
    names = agents.split(",") if isinstance(agents, str) else list(agents)
    rows = defaultdict(list)
    for n in eps:
        for name in names:
            J, shed, fuel, config = play(AGENTS[name], task, entropy, n)
            ids = config["static"]["nodes"]["id"]
            grids = [int(g) for g in config["layout"]["grids"]]
            tr = [r for r in fuel.trace if r["k"] == fuel.grids[r["grid"]]["rationed"]]
            for gi, g in enumerate(grids):
                rg = [r for r in tr if r["grid"] == g]
                if not rg:
                    continue
                thr = fuel.grids[g]["thr"][fuel.grids[g]["rationed"]]
                under = sum(r["s0"] < thr - 1e-6 for r in rg)
                rho = np.array([r.get("rho", 1.0) for r in rg])
                s_early = np.mean([r["s0"] for r in rg if 2 <= r["week"] <= 14])  # closing stock of weeks 1-13
                modes = defaultdict(int)
                for r in rg:
                    modes[r["mode"]] += 1
                rows[(name, ids[g])].append(
                    [
                        J / 1e9,
                        under,
                        rho.mean(),
                        (rho < 0.92).mean(),
                        s_early,
                        shed[:6, gi].sum(),
                        shed[6:13, gi].sum(),
                        shed[13:, gi].sum(),
                        modes["hold"] + modes["prime"],
                    ]
                )
            print(f"episode {n} {name}: J {J / 1e9:.2f} bn", flush=True)
    print(f"\n{task}, root {entropy}, episodes {eps}: means over episodes")
    print(
        f"{'agent':7} {'grid':9} {'J bn':>8} {'wks<thr':>8} {'rho':>6} {'rho<.92':>8} {'s0 1-13':>8} "
        f"{'shed1-6':>8} {'shed7-13':>9} {'shed14+':>8} {'hold+prime':>10}"
    )
    for (name, gid), v in sorted(rows.items(), key=lambda kv: (kv[0][1], names.index(kv[0][0]))):
        m = np.mean(v, axis=0)
        print(
            f"{name:7} {gid:9} {m[0]:8.2f} {m[1]:8.1f} {m[2]:6.3f} {m[3]:8.2f} {m[4]:8.0f} "
            f"{m[5]:8.0f} {m[6]:9.0f} {m[7]:8.0f} {m[8]:10.1f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
