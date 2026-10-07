"""Where a hindsight knob search found its money, by cost component.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/knob_effect.py \
        --run=outputs/rl_lab/knobs_hindsight_small --task=small

Replays every episode twice — once on the shipped numbers, once on the numbers the search found for that very
episode — and compares the simulator's own cost components. Since shed load and unmet demand are 99 % of the
cost, the only interesting question about any gain is which of the two it came out of.
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
import tuned_rules as T  # noqa: E402
from shockbench_flow_gym import agent_config_from_reset  # noqa: E402
from shockbench_flow_gym.wrappers import ScenarioPool  # noqa: E402

from sbf_starter import env_id  # noqa: E402


COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def _play(env, parts, index, values, segments):
    obs, info = env.reset(options={"pool_index": index})
    config = agent_config_from_reset(env, obs, info)
    agent = parts.agent(config, values) if segments < 2 else parts.agent(config, schedule=values)
    costs = np.zeros(len(COMPONENTS))
    extra = {"served": 0.0, "lost": 0.0, "shed_qty": 0.0}
    while True:
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        costs += np.asarray(obs["last_week.cost_components"], dtype=np.float64)
        extra["served"] += float(np.asarray(obs["last_week.sinks.served"]).sum())
        extra["lost"] += float(np.asarray(obs["last_week.sinks.lost"]).sum())
        extra["shed_qty"] += float(np.asarray(obs["last_week.shed.qty"]).sum())
        if terminated or truncated:
            break
    return costs, extra


def main(run, task="small", entropy=111, episodes=None):
    run = Path(run) if Path(run).is_absolute() else ROOT / run
    settings = json.loads((run / "settings.json").read_text())
    rows = json.loads((run / "result.json").read_text())
    segments = int(settings.get("segments", 1))
    episodes = episodes or len(rows)
    rows = sorted(rows, key=lambda r: r["episode"])[:episodes]

    env = ScenarioPool(gym.make(env_id(task), regime="standard"), int(settings["episodes"]), entropy)
    parts = T.Parts()
    base_c, best_c = [], []
    base_x, best_x = [], []
    for r in rows:
        c, e = _play(env, parts, int(r["episode"]), None, 1)
        base_c.append(c)
        base_x.append(e)
        c, e = _play(env, parts, int(r["episode"]), r["values"], segments)
        best_c.append(c)
        best_x.append(e)

    a, b = np.mean(base_c, axis=0), np.mean(best_c, axis=0)
    total = a.sum()
    print(f"{run.name}: {len(rows)} episode(s) of {task}, root {entropy}, {segments} segment(s)")
    print(f"{'cost component':14s} {'shipped':>15s} {'searched':>15s} {'saved':>14s} {'of the saving':>14s}")
    saved = a - b
    for name, x, y, s in zip(COMPONENTS, a, b, saved):
        print(f"{name:14s} {x:15.4e} {y:15.4e} {s:14.4e} {s / max(saved.sum(), 1e-9):13.1%}")
    print(f"{'TOTAL':14s} {total:15.4e} {b.sum():15.4e} {saved.sum():14.4e} ({saved.sum() / total:+.2%} of the cost)")
    print(f"\n{'quantity':14s} {'shipped':>15s} {'searched':>15s} {'change':>10s}")
    for key in ("served", "lost", "shed_qty"):
        x = float(np.mean([e[key] for e in base_x]))
        y = float(np.mean([e[key] for e in best_x]))
        print(f"{key:14s} {x:15.4e} {y:15.4e} {(y - x) / max(abs(x), 1e-9):+9.2%}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
