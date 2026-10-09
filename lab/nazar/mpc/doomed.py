"""Which fabs make chips that cannot leave the network: lots started whose raw chips go nowhere.

    uv run python lab/nazar/mpc/doomed.py agents/anastasiia_rules_v2 --episodes=20

A plant is "blocked" in an episode when its packaged chips leave at under ``block`` of the median episode's outflow.
Per fab and episode: lots started, raw chips shipped to plants and the part shipped to blocked ones, and raw chips
disposed of at the fab itself. A fab-episode is "doomed" when most of its raw output ends in one of those two places.
Prints per fab how many episodes are doomed, the lots started in them, and the lots of the other episodes.
Reuses ``play`` of heur_lab2/tools/account.py. Read-only.
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
    raw_k = [names.index("chip_le_raw"), names.index("chip_mat_raw")]
    pack_k = [names.index("chip_le"), names.index("chip_mat")]
    fabs = {int(n): i for i, n in enumerate(inst.fabs)}
    osats = {int(o) for o in inst.osats}
    raw_slots, pack_slots = [], []
    for i, (e, k, lane) in enumerate(inst.action_slots):
        tail = int(inst.edges[e].tail)
        dest = int(inst.lane_destination(lane)) if lane is not None else int(inst.edges[e].head)
        if k in raw_k and tail in fabs:
            raw_slots.append((i, tail, dest, k))
        if k in pack_k and tail in osats:
            pack_slots.append((i, tail, k))
    stock = {(int(n), k): inst.slot_index[(int(n), k)] for n in fabs for k in raw_k if (int(n), k) in inst.slot_index}
    return {
        "fab_name": {n: str(inst.nodes[n].id) for n in fabs},
        "fab_ord": fabs,
        "osat_name": {o: str(inst.nodes[o].id) for o in osats},
        "pack_of": {raw_k[0]: pack_k[0], raw_k[1]: pack_k[1]},  # raw chip commodity -> its packaged chip
        "raw_slots": raw_slots,
        "pack_slots": pack_slots,
        "raw_stock": stock,
    }


def main(
    agent: str, task: str = "small", entropy: int = 444, episodes: int = 20, n_jobs: int = 8, block: float = 0.15
) -> None:
    lay = layout(task, entropy)
    runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
    plants = sorted(lay["osat_name"])
    kinds = sorted(set(lay["pack_of"].values()))
    keys = [(p, k) for p in plants for k in kinds]
    out = np.array(
        [[sum(r["sent"][:, i].sum() for i, t, kk in lay["pack_slots"] if (t, kk) == key) for key in keys] for r in runs]
    )
    median = np.median(out, axis=0)
    blocked = out < block * np.maximum(median, 1.0)  # (episodes, (plant, packaged chip))
    print(f"{Path(agent).name}: {task} root {entropy}, {episodes} episodes")
    print(
        "blocked plant outlets by episode (a packaged chip leaves at under %.0f %% of the median episode):"
        % (100 * block)
    )
    for j, (p, k) in enumerate(keys):
        eps = [n for n in range(episodes) if blocked[n, j]]
        print(
            f"  {lay['osat_name'][p]:<8} chip {k}  median outflow {median[j]:>12,.0f}   blocked in {len(eps):>2} episodes: {eps}"
        )
    print(
        f"\n  {'fab':<18}{'doomed eps':>11}{'lots in doomed':>16}{'lots in others':>16}{'raw to blocked':>16}{'raw disposed':>14}"
    )
    for fab, fo in lay["fab_ord"].items():
        doomed_lots, other_lots, doomed_n, to_blocked, disposed = [], [], 0, 0.0, 0.0
        for n, r in enumerate(runs):
            shipped = sum(r["sent"][:, i].sum() for i, t, d, k in lay["raw_slots"] if t == fab)
            to_b = sum(
                r["sent"][:, i].sum()
                for i, t, d, k in lay["raw_slots"]
                if t == fab and d in plants and blocked[n, keys.index((d, lay["pack_of"][k]))]
            )
            disp = sum(r["disposal"][:, s].sum() for (nn, k), s in lay["raw_stock"].items() if nn == fab)
            lots = r["lots"][:, fo].sum()
            bad = (to_b + disp) / max(shipped + disp, 1.0)
            to_blocked += to_b / episodes
            disposed += disp / episodes
            if bad > 0.5:
                doomed_n += 1
                doomed_lots.append(lots)
            else:
                other_lots.append(lots)
        print(
            f"  {lay['fab_name'][fab]:<18}{doomed_n:>11}{(np.mean(doomed_lots) if doomed_lots else 0):16,.0f}"
            f"{(np.mean(other_lots) if other_lots else 0):16,.0f}{to_blocked:16,.0f}{disposed:14,.0f}"
        )


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
