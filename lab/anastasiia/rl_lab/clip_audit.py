"""How much of what the rules ask for the simulator actually executes, by commodity.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/clip_audit.py --task=small --episodes=8

The claim this settles: a multiplicative correction on a request that is already clipped changes nothing. If the
executed share is near 1, the request decides the flow and a multiplier has room; if it is far below 1, capacity
and stock decide it and scaling the ask up is wasted. The observation reports both numbers for the week just gone
(``last_week.clip.requested`` and ``last_week.clip.executed``), so no instrumentation of the simulator is needed.

Reported per commodity: the share of asked quantity that moved, the share of slots whose ask was cut at all, and —
the number that matters for a multiplier — what an ask raised by 25 % would have delivered, computed from the
slots that were not cut (those are the only ones a higher ask can move).
"""

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym  # noqa: E402
import residual as R  # noqa: E402
from shockbench_flow_gym import agent_config_from_reset  # noqa: E402
from shockbench_flow_gym.wrappers import ScenarioPool  # noqa: E402

from sbf_starter import env_id  # noqa: E402


EPS = 1e-9


def main(task="small", episodes=8, entropy=666, rules="agents/anastasiia_rules_v2"):
    folder = Path(rules) if Path(rules).is_absolute() else ROOT / rules
    module = R.load_rules(folder)
    env = ScenarioPool(gym.make(env_id(task), regime="standard"), max(episodes, 1), entropy)

    asked = {}
    moved = {}
    cut_slots = {}
    all_slots = {}
    for ep in range(episodes):
        obs, info = env.reset(options={"pool_index": ep})
        agent = R.Residual(agent_config_from_reset(env, obs, info), module, None)
        names = agent.layout.commodities
        while True:
            action = agent.rules_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            req = np.asarray(obs["last_week.clip.requested"], dtype=np.float64)
            exe = np.asarray(obs["last_week.clip.executed"], dtype=np.float64)
            live = req > EPS
            for k in np.unique(agent.layout.slot_k[live]):
                sel = live & (agent.layout.slot_k == k)
                name = names[int(k)]
                asked[name] = asked.get(name, 0.0) + float(req[sel].sum())
                moved[name] = moved.get(name, 0.0) + float(exe[sel].sum())
                ratio = exe[sel] / np.maximum(req[sel], EPS)
                cut_slots[name] = cut_slots.get(name, 0) + int((ratio < 0.999).sum())
                all_slots[name] = all_slots.get(name, 0) + int(sel.sum())
            if terminated or truncated:
                break

    print(f"the rules of {folder.name} on {task}, {episodes} episode(s) of root {entropy}")
    print(
        f"{'commodity':14s} {'asked slots':>11s} {'executed share':>15s} {'slots cut':>10s} "
        f"{'headroom of a +25% ask':>23s}"
    )
    for name in asked:
        share = moved[name] / max(asked[name], EPS)
        cut = cut_slots[name] / max(all_slots[name], 1)
        # only an uncut slot can move more; a cut one is already at its capacity or stock
        headroom = (1.0 - cut) * 0.25
        print(f"{name:14s} {all_slots[name]:11d} {share:15.3f} {cut:10.1%} {headroom:22.1%}")
    print("\nexecuted share near 1 with few slots cut: the ask decides the flow, a multiplier has room.")
    print("executed share well below 1: capacity and stock decide it, and asking for more is wasted.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
