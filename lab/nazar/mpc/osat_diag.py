"""Why packaged chips pile up at the packaging plants and are disposed of there.

    uv run python lab/nazar/mpc/osat_diag.py lab/nazar/agents/nazar_rules_lpraw agents/anastasiia_rules_v2

Per plant and packaged chip, over weeks 1..52 of the episodes: the stock against the storage, the chips disposed of,
and in the weeks of disposal what was asked of the outlets (the slots out of the plant) against what the simulator
executed and what stock was there. Then per outlet slot: asked, executed, weeks with no request. Then per market: its
stock against its storage and the share of demand served. Reuses ``play`` of heur_lab2/tools/account.py. Read-only.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "heur_lab2" / "tools"))
from account import play  # noqa: E402


def layout(task: str, entropy: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    env.reset(options={"episode": 0})
    inst = env.unwrapped.instance
    names = [c.id for c in inst.commodities]
    out = {"plants": [], "markets": []}
    for o in inst.osats:
        for k in (names.index("chip_le"), names.index("chip_mat")):
            s = inst.slot_index.get((int(o), k))
            if s is None:
                continue
            outlets = [
                (i, str(inst.edges[e].id))
                for i, (e, kk, _l) in enumerate(inst.action_slots)
                if kk == k and int(inst.edges[e].tail) == int(o)
            ]
            out["plants"].append((str(inst.nodes[int(o)].id), names[k], s, float(inst.stock_slots[s].storage), outlets))
    for d, dd in enumerate(inst.demands):
        s = inst.slot_index.get((int(dd.node), int(dd.k)))
        if s is not None and dd.pi > 0:
            out["markets"].append((f"{inst.nodes[dd.node].id} {names[dd.k]}", d, s, float(inst.stock_slots[s].storage)))
    return out


def main(*agents: str, task: str = "small", entropy: int = 444, episodes: int = 12, n_jobs: int = 8) -> None:
    lay = layout(task, entropy)
    for agent in agents:
        runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
        print(f"\n=== {Path(agent).name}: {task} root {entropy}, {episodes} episodes ===")
        print(
            f"{'plant':<10}{'chip':<9}{'disposed/ep':>12}{'wks disp':>9}{'stock/stor':>11} | in disposal weeks: {'stock':>9}{'asked':>10}{'executed':>10}{'asked/stock':>12}"
        )
        for name, chip, s, storage, outlets in lay["plants"]:
            ix = [i for i, _ in outlets]
            disp = np.array([r["disposal"][:, s] for r in runs])
            stock = np.array([r["stock"][:, s] for r in runs])
            asked = np.array([r["asked"][:, ix].sum(1) for r in runs]) if ix else np.zeros_like(disp)
            sent = np.array([r["sent"][:, ix].sum(1) for r in runs]) if ix else np.zeros_like(disp)
            m = disp > 1.0
            if disp.sum() <= 0:
                print(f"{name:<10}{chip:<9}{0:12.0f}{0:9.1f}{stock.mean() / storage:11.2f} |")
                continue
            print(
                f"{name:<10}{chip:<9}{disp.sum(1).mean():12,.0f}{m.sum(1).mean():9.1f}{stock.mean() / storage:11.2f} | "
                f"{'':>20}{stock[m].mean():9,.0f}{asked[m].mean():10,.0f}{sent[m].mean():10,.0f}{asked[m].mean() / max(stock[m].mean(), 1):12.2f}"
            )
        print("\n  outlet slots out of plants with disposal (mean per episode):")
        for name, chip, s, storage, outlets in lay["plants"]:
            if not any(r["disposal"][:, s].sum() > 1.0 for r in runs):
                continue
            for i, eid in outlets:
                a = np.mean([r["asked"][:, i].sum() for r in runs])
                e = np.mean([r["sent"][:, i].sum() for r in runs])
                z = np.mean([(r["asked"][:, i] <= 1e-9).mean() for r in runs])
                print(
                    f"    {name:<8}{chip:<9}{eid[:38]:<40}asked {a:11,.0f}  executed {e:11,.0f}  weeks asked 0: {100 * z:4.0f} %"
                )
        print("\n  markets: mean stock / storage, demand served share")
        for label, d, s, storage in lay["markets"]:
            fill = np.mean([r["stock"][:, s].mean() for r in runs]) / max(storage, 1)
            served = np.mean([r["served"][:, d].sum() / max(r["demand"][:, d].sum(), 1e-9) for r in runs])
            print(f"    {label:<22} stock/storage {fill:5.2f}   served {100 * served:5.1f} %")


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
