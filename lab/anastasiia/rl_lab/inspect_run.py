"""What a trained network actually does: the correction it applies, by commodity family and by week.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/inspect_run.py --run=outputs/rl_lab/<run>

A run whose held-out curve goes down is not a mystery to be guessed at: either the network multiplies everything
by less than one (cutting this week's freight and paying for it fourteen weeks later), or it has found something
on a few slots. This prints which.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym  # noqa: E402
import nets  # noqa: E402
import residual as R  # noqa: E402
from shockbench_flow_gym import agent_config_from_reset  # noqa: E402
from shockbench_flow_gym.wrappers import ScenarioPool  # noqa: E402

from sbf_starter import env_id  # noqa: E402


FAMILIES = ("lng", "crude", "nucfuel", "wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat")


def main(run, task="small", episodes=4, entropy=666, policy="policy.pt"):
    run = Path(run) if Path(run).is_absolute() else ROOT / run
    settings = json.loads((run / "settings.json").read_text())
    rules_folder = Path(settings["rules"])
    if not rules_folder.is_absolute():
        rules_folder = ROOT / rules_folder

    pol = nets.build(settings["kind"], settings["sizes"], settings["width"], settings["rounds"])
    pol.load_state_dict(torch.load(run / policy, map_location="cpu", weights_only=True))
    pol.eval()
    rules = R.load_rules(rules_folder)

    env = ScenarioPool(gym.make(env_id(task), regime="standard"), max(episodes, 1), entropy)
    by_k = {k: [] for k in range(len(FAMILIES))}
    by_week = []
    for ep in range(episodes):
        obs, info = env.reset(options={"pool_index": ep})
        agent = R.Residual(
            agent_config_from_reset(env, obs, info),
            rules,
            pol,
            groups=settings["groups"],
            lo=settings["lo"],
            hi=settings["hi"],
            families=settings.get("families"),
        )
        while True:
            base = agent.rules_action(obs)
            flows = np.asarray(base["flows"], dtype=np.float64)
            with torch.inference_mode():
                raw, _ = pol(R.one_week_batch(*agent.tables(obs, flows), agent.index))
            out = agent.apply(base, raw.numpy())
            live = flows > 0
            factor = np.ones_like(flows)
            factor[live] = out["flows"][live] / flows[live]
            for s in np.flatnonzero(live):
                by_k[int(agent.layout.slot_k[s])].append(factor[s])
            by_week.append(factor[live].mean() if live.any() else 1.0)
            obs, reward, terminated, truncated, info = env.step(out)
            if terminated or truncated:
                break

    print(f"{run.name} on {task}, {episodes} episode(s), weights {policy}")
    print(f"{'family':14s} {'slots seen':>10s} {'mean':>7s} {'p10':>7s} {'p50':>7s} {'p90':>7s}")
    for k, name in enumerate(FAMILIES):
        v = np.array(by_k[k])
        if not v.size:
            continue
        print(
            f"{name:14s} {v.size:10d} {v.mean():7.3f} "
            f"{np.percentile(v, 10):7.3f} {np.percentile(v, 50):7.3f} {np.percentile(v, 90):7.3f}"
        )
    w = np.array(by_week)
    print(
        f"\nmean factor over a week: {w.mean():.4f}  (first quarter {w[: len(w) // 4].mean():.4f}, "
        f"last quarter {w[-len(w) // 4 :].mean():.4f})"
    )
    print("a mean far from 1.0 means the network moved every slot the same way, not a few of them")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
