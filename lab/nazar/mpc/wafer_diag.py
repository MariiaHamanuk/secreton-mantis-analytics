"""Where do the wafers go: requested, executed and on hand at the fabs, week by week, for two agents on the same episodes.

    uv run python lab/nazar/mpc/wafer_diag.py lab/nazar/agents/mpc_std agents/anastasiia_rules_v2 --episodes=16

Reuses ``play`` of lab/anastasiia/heur_lab2/tools/account.py (the simulator's own weekly records). Read-only.
Per week band: wafers requested (all wafer slots), executed (what the simulator let through), the wafer stock at the
fabs at the end of the week, lots started, and the fabs' capacity share that had no wafers.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "heur_lab2" / "tools"))
from account import play  # noqa: E402


def wafer_ids(task: str, entropy: int):
    """(wafer action slots, fab wafer stock slots, fab capacity per week) of the task."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    env.reset(options={"episode": 0})
    inst = env.unwrapped.instance
    k = inst.commodity_index["wafer"]
    slots = [i for i, (_e, kk, _l) in enumerate(inst.action_slots) if kk == k]
    stock = [inst.slot_index[(int(n), k)] for n in inst.fabs if (int(n), k) in inst.slot_index]
    cap = np.array([inst.nodes[int(n)].fab.cap0 if hasattr(inst.nodes[int(n)].fab, "cap0") else 0.0 for n in inst.fabs])
    return slots, stock, cap, inst.T


def summarise(runs: list[dict], slots, stock, T: int, bands) -> list[list[float]]:
    rows = []
    for lo, hi in bands:
        w = slice(lo - 1, min(hi, T))
        asked = np.mean([r["asked"][w][:, slots].sum() / (min(hi, T) - lo + 1) for r in runs])
        sent = np.mean([r["sent"][w][:, slots].sum() / (min(hi, T) - lo + 1) for r in runs])
        held = np.mean([r["stock"][w][:, stock].sum(1).mean() for r in runs])
        lots = np.mean([r["lots"][w].sum(1).mean() for r in runs])
        empty = np.mean([(r["stock"][w][:, stock] < 1e-6).mean() for r in runs])
        rows.append([lo, hi, asked, sent, held, lots, empty])
    return rows


def main(*agents: str, task: str = "small", entropy: int = 444, episodes: int = 16, n_jobs: int = 8) -> None:
    slots, stock, _cap, T = wafer_ids(task, entropy)
    bands = [(1, 10), (11, 20), (21, 30), (31, T)]
    print(f"{task} root {entropy}, episodes 0..{episodes - 1}; {len(slots)} wafer slots, {len(stock)} fabs")
    for agent in agents:
        runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
        print(f"\n{Path(agent).name}: mean J {np.mean([r['J'] for r in runs]) / 1e9:.1f} bn")
        print("  weeks      asked/wk   executed/wk   stock at fabs   lots/wk   share of fab-weeks with no wafer stock")
        for lo, hi, asked, sent, held, lots, empty in summarise(runs, slots, stock, T, bands):
            print(f"  {lo:>3}-{hi:<3} {asked:12,.0f} {sent:13,.0f} {held:15,.0f} {lots:9,.0f} {empty:12.2f}")


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
