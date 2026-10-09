"""Raw-chip slots: what the rules ask for and what the simulator executes, per slot, mean per episode.

    uv run python lab/nazar/mpc/raw_clip.py agents/anastasiia_rules_v2 --episodes=8

Reuses ``play`` of heur_lab2/tools/account.py (its ``asked`` and ``sent`` arrays). Read-only.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "heur_lab2" / "tools"))
from account import play  # noqa: E402


def slots(task: str, entropy: int):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    env.reset(options={"episode": 0})
    inst = env.unwrapped.instance
    out = []
    for i, (e, k, lane) in enumerate(inst.action_slots):
        if k in (4, 5):
            out.append((i, int(inst.edges[e].tail), str(inst.edges[e].id), int(k), int(e)))
    return out


def main(agent: str, task: str = "small", entropy: int = 111, episodes: int = 8, n_jobs: int = 8) -> None:
    meta = slots(task, entropy)
    runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
    print(f"{Path(agent).name}: {task} root {entropy}, {episodes} episodes, raw-chip slots (mean per episode)")
    print(f"  {'slot':>4} {'fab':>4} {'edge':<34}{'k':>2}{'asked':>12}{'executed':>12}{'executed/asked':>16}")
    for i, tail, eid, k, e in meta:
        a = np.mean([r["asked"][:, i].sum() for r in runs])
        s = np.mean([r["sent"][:, i].sum() for r in runs])
        print(f"  {i:>4} {tail:>4} {eid[:33]:<34}{k:>2}{a:12,.0f}{s:12,.0f}{(s / a if a > 0 else 0):16.2f}")


if __name__ == "__main__":
    fire.Fire(main)
