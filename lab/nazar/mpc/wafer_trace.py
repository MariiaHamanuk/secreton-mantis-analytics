"""Where the wafers go: per wafer stock slot, over whole episodes, for two agents on the same episodes.

    uv run python lab/nazar/mpc/wafer_trace.py lab/nazar/agents/mpc_std agents/anastasiia_rules_v2 --episodes=8

Per node holding wafers (sources and fabs): wafers on hand at the end of weeks 1, 10, 26 and the last, wafers disposed
of over the episode, and for fabs the lots started and the wafers that arrived (stock change + lots + disposal; one
wafer a lot is assumed, so it is a check, not a measure). Reuses ``play`` of heur_lab2/tools/account.py. Read-only.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "heur_lab2" / "tools"))
from account import play  # noqa: E402


def layout(task: str, entropy: int):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    env.reset(options={"episode": 0})
    inst = env.unwrapped.instance
    k = inst.commodity_index["wafer"]
    slots = {n: s for (n, kk), s in inst.slot_index.items() if kk == k}
    fabs = {int(n): i for i, n in enumerate(inst.fabs)}
    return inst, slots, fabs


def main(*agents: str, task: str = "small", entropy: int = 444, episodes: int = 8, n_jobs: int = 8) -> None:
    inst, slots, fabs = layout(task, entropy)
    print(f"{task} root {entropy}, episodes 0..{episodes - 1}; wafer stock at {len(slots)} nodes")
    for agent in agents:
        runs = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(episodes))
        print(f"\n{Path(agent).name}")
        print(
            f"  {'node':<18}{'wk1':>10}{'wk10':>10}{'wk26':>10}{'last':>10}{'disposed':>11}{'lots':>11}{'arrived':>11}"
        )
        for node, s in slots.items():
            st = np.mean([r["stock"][:, s] for r in runs], axis=0)
            disp = np.mean([r["disposal"][:, s].sum() for r in runs])
            if fabs.get(node) is not None:
                lots = np.mean([r["lots"][:, fabs[node]].sum() for r in runs])
                arrived = np.mean(
                    [
                        (np.diff(r["stock"][:, s], prepend=0) + r["lots"][:, fabs[node]] + r["disposal"][:, s]).sum()
                        for r in runs
                    ]
                )
                tail = f"{lots:11,.0f}{arrived:11,.0f}"
            else:
                tail = f"{'':>11}{'':>11}"
            name = inst.nodes[node].id
            print(f"  {name:<18}{st[0]:10,.0f}{st[9]:10,.0f}{st[25]:10,.0f}{st[-1]:10,.0f}{disp:11,.0f}{tail}")


if __name__ == "__main__":
    fire.Fire(main)
