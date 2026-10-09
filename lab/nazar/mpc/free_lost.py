"""When does a grid lose its free (no-fuel) segment: the weeks and the situations, per grid.

    uv run python lab/nazar/mpc/free_lost.py agents/anastasiia_rules_v2 --episodes=12

The free segment's output is lost when the grid offers more than its load takes (the load factor then lowers every
segment). Per grid, the weeks 1..38 are split by what happened: no shed and the fabs had energy ("complete"), shed and
no energy ("short"), no shed and no energy ("idle": base load only), shed with some energy. Prints the share of weeks and
the free-segment GWh lost in each. Reuses ``play`` of heur_lab2/tools/account.py. Read-only.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "heur_lab2" / "tools"))
from account import play  # noqa: E402


def static(task: str, entropy: int):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    env.reset(options={"episode": 0})
    inst = env.unwrapped.instance
    out = []
    for go, g in enumerate(inst.grids):
        grid = inst.nodes[g].grid
        out.append((str(inst.nodes[g].id), float(grid.shares.get(None, 0.0)), [int(f) for f in inst.grid_fabs[go]]))
    return out, inst.T


def main(
    agent: str, task: str = "small", entropy: int = 444, episodes: int = 12, n_jobs: int = 8, useful: int = 38
) -> None:
    grids, T = static(task, entropy)
    runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
    w = slice(0, min(useful, T))
    print(f"{Path(agent).name}: {task} root {entropy}, {episodes} episodes, weeks 1..{useful}")
    print(f"  {'grid':<10}{'case':<28}{'% weeks':>8}{'free GWh lost / wk':>20}{'share of the lost':>19}")
    for go, (name, share, fabs) in enumerate(grids):
        if share <= 0 or not fabs:
            continue
        lost, shed, en = [], [], []
        for r in runs:
            lost.append(share * r["G_bar"][w, go] - r["segment"][w, go, -1])
            shed.append(r["shed"][w, go] > 1e-6)
            en.append(r["energy"][w][:, fabs].sum(1))
        lost, shed, en = np.concatenate(lost), np.concatenate(shed), np.concatenate(en)
        masks = {
            "no shed, fabs have energy": (~shed) & (en > 1e-6),
            "shed, fabs have no energy": shed & (en <= 1e-6),
            "no shed, fabs have no energy": (~shed) & (en <= 1e-6),
            "shed and fabs have energy": shed & (en > 1e-6),
        }
        total_lost = lost.sum()
        for label, m in masks.items():
            print(
                f"  {name:<10}{label:<28}{100 * m.mean():8.1f}{lost[m].sum() / len(lost):20.1f}"
                f"{100 * lost[m].sum() / max(total_lost, 1e-9):18.1f}%"
            )


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
