"""Where a correction's extra flow ends up: does it sell, or does it only pile up?

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/effect_audit.py \
        --run=outputs/rl_lab/q4_chips_only --task=small --episodes=8

Plays the plain rules and the network on the same scenarios and compares what the simulator recorded: quantity
despatched, demand served, demand lost, load shed, and the week's cost by component. A correction that moves 26 %
more finished chips and changes the cost by 0.04 % must show where the difference went — if served and lost are
unchanged, the extra cargo never became a sale and the request was never the binding constraint.
"""

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym  # noqa: E402
import nets  # noqa: E402
import residual as R  # noqa: E402
import torch  # noqa: E402
from shockbench_flow_gym import agent_config_from_reset  # noqa: E402
from shockbench_flow_gym.wrappers import ScenarioPool  # noqa: E402

from sbf_starter import env_id  # noqa: E402


COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def _episode(env, index, module, policy, settings):
    obs, info = env.reset(options={"pool_index": index})
    agent = R.Residual(
        agent_config_from_reset(env, obs, info),
        module,
        policy,
        groups=settings.get("groups") if settings else None,
        lo=(settings or {}).get("lo", 0.5),
        hi=(settings or {}).get("hi", 2.0),
        families=(settings or {}).get("families"),
    )
    totals = {k: 0.0 for k in ("despatched", "served", "lost", "shed", "cost")}
    costs = np.zeros(len(COMPONENTS))
    while True:
        action = agent.act(obs) if policy is not None else agent.rules_action(obs)
        totals["despatched"] += float(np.asarray(action["flows"]).sum())
        obs, reward, terminated, truncated, info = env.step(action)
        totals["served"] += float(np.asarray(obs["last_week.sinks.served"]).sum())
        totals["lost"] += float(np.asarray(obs["last_week.sinks.lost"]).sum())
        totals["shed"] += float(np.asarray(obs["last_week.shed.qty"]).sum())
        totals["cost"] += -float(reward)
        costs += np.asarray(obs["last_week.cost_components"], dtype=np.float64)
        if terminated or truncated:
            break
    return totals, costs


def main(run, task="small", episodes=8, entropy=666, policy="policy.pt"):
    run = Path(run) if Path(run).is_absolute() else ROOT / run
    settings = json.loads((run / "settings.json").read_text())
    folder = Path(settings["rules"])
    if not folder.is_absolute():
        folder = ROOT / folder
    module = R.load_rules(folder)
    net = nets.build(settings["kind"], settings["sizes"], settings["width"], settings["rounds"])
    net.load_state_dict(torch.load(run / policy, map_location="cpu", weights_only=True))
    net.eval()

    env = ScenarioPool(gym.make(env_id(task), regime="standard"), max(episodes, 1), entropy)
    base, mine = [], []
    base_c, mine_c = [], []
    for i in range(episodes):
        t, c = _episode(env, i, module, None, None)
        base.append(t)
        base_c.append(c)
        t, c = _episode(env, i, module, net, settings)
        mine.append(t)
        mine_c.append(c)

    print(f"{run.name} against the plain rules, {task}, {episodes} episode(s) of root {entropy}")
    print(f"{'quantity':14s} {'rules':>16s} {'network':>16s} {'change':>10s}")
    for key in ("despatched", "served", "lost", "shed", "cost"):
        a = float(np.mean([t[key] for t in base]))
        b = float(np.mean([t[key] for t in mine]))
        rel = (b - a) / abs(a) if abs(a) > 0 else 0.0
        print(f"{key:14s} {a:16.4g} {b:16.4g} {rel:+9.2%}")
    print(f"\n{'cost component':14s} {'rules':>16s} {'network':>16s} {'change':>10s}")
    a_all, b_all = np.mean(base_c, axis=0), np.mean(mine_c, axis=0)
    for name, a, b in zip(COMPONENTS, a_all, b_all):
        rel = (b - a) / abs(a) if abs(a) > 1e-9 else 0.0
        print(f"{name:14s} {a:16.4g} {b:16.4g} {rel:+9.2%}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
