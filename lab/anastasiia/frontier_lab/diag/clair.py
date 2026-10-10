"""Solve the board's clairvoyant plan of an episode and keep its weekly quantities, with the scenario's network.

    uv run python lab/anastasiia/frontier_lab/diag/clair.py --task=full --entropy=444 --episodes=16

One process, about 70 s of CPU an episode on Full. Per episode, ``outputs/frontier_lab/diag/clair_<task>_<entropy>.pkl``
keeps the plan in the layout of ``hazard_lab/play.py`` (costs by item, lots, shed, lost, served, energy, stock,
segment, sent, all by week) and the scenario's network by week (edge capacity, prohibitions, strait openness, grid
output, base load, supply), so a played episode can be laid beside the plan and beside what happened to the network.
Episodes already kept are not solved again.
"""

import pickle
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "frontier_lab" / "diag"
ITEMS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def episode(task: str, entropy: int, n: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow.oracle.lp import build_lp, lp_cents, lp_costs, lp_flow_key, solve_oracle

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    env.reset(options={"episode": n})
    ep = env.unwrapped.core._ep
    inst, marks = ep.inst, ep.marks
    T, F, G, D = inst.T, len(inst.fabs), len(inst.grids), len(inst.demands)
    model = build_lp(inst, marks)
    plan = solve_oracle(model)
    x, index = plan.x, model.index

    def var(name, *key):
        return np.array([x[index[q]] if (q := (name, t, *key)) in index else 0.0 for t in range(1, T + 1)])

    weekly, _credit = lp_costs(model, x)
    seg = np.zeros((T, G, len(inst.commodities) + 1))  # last column: the no-fuel segment, as play.py keeps it
    for go in range(G):
        seg[:, go, -1] = var("G", go, None)
        for k in range(len(inst.commodities)):
            seg[:, go, k] = var("G", go, k)
    sent = np.stack([var("x", *lp_flow_key(inst, e, k, lane)) for e, k, lane in inst.action_slots], axis=1)
    stock = np.stack([var("I", s) for s in range(len(inst.stock_slots))], axis=1)
    keep = (
        "u",
        "u_now",
        "prohibited",
        "o",
        "o_now",
        "kappa",
        "supply",
        "G_bar",
        "G_bar_now",
        "y_bar",
        "R",
        "R_osat",
        "demand",
    )
    return {
        "n": n, "J": int(lp_cents(model, x)), "J_plan": int(plan.J_cents),
        "costs": np.array([[w.as_dict()[c] for c in ITEMS] for w in weekly]),
        "lots": np.stack([var("p", fo) for fo in range(F)], axis=1),
        "shed": np.stack([var("ysh", go) for go in range(G)], axis=1),
        "lost": np.stack([var("U", do) for do in range(D)], axis=1),
        "served": np.stack([var("D", do) for do in range(D)], axis=1),
        "energy": np.stack([var("E", fo) for fo in range(F)], axis=1),
        "stock": stock, "segment": seg, "sent": sent,
        "marks": {name: np.asarray(getattr(marks, name)) for name in keep if hasattr(marks, name)},
    }  # fmt: skip


def main(task: str = "full", entropy: int = 444, episodes: int = 16, first: int = 0) -> None:
    path = OUT / f"clair_{task}_{entropy}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
    for n in range(first, first + episodes):
        if n in kept:
            continue
        kept[n] = episode(task, entropy, n)
        path.write_bytes(pickle.dumps(kept))
        print(f"episode {n}: J {kept[n]['J'] / 1e11:.1f} bn", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
