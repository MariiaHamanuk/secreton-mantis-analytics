"""The scenario of each episode as the simulator has it: the network by week and the list of events.

    uv run python lab/anastasiia/frontier_lab/diag/world.py --task=full --entropy=444 --episodes=16

Writes ``outputs/frontier_lab/diag/world_<task>_<entropy>.pkl``: per episode the weekly marks (edge capacity, strait
openness and throughput, prohibitions, grid output and base load, supply; week averages and the instants an
observation shows) and the events with what each alone does (``hazard_lab/play.py::events_of``). Nothing is solved.
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "frontier_lab" / "diag"
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "hazard_lab"))
KEEP = ("u", "u_now", "prohibited", "o", "o_now", "kappa", "kappa_now", "supply", "supply_now", "G_bar", "G_bar_now",
        "y_bar", "R", "R_osat", "demand", "tariff", "wr_class")  # fmt: skip


def main(task: str = "full", entropy: int = 444, episodes: int = 16, first: int = 0) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from play import events_of

    from sbf_starter import env_id

    path = OUT / f"world_{task}_{entropy}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
    env = gym.make(env_id(task), entropy=entropy)
    for n in range(first, first + episodes):
        if n in kept:
            continue
        env.reset(options={"episode": n})
        u = env.unwrapped
        marks = u.core._ep.marks
        kept[n] = {"marks": {name: np.asarray(getattr(marks, name)) for name in KEEP},
                   "events": events_of(u.core._ep.inst, u._omega(n))}  # fmt: skip
    path.write_bytes(pickle.dumps(kept))
    print(f"{path}: {len(kept)} episodes")


if __name__ == "__main__":
    fire.Fire(main)
