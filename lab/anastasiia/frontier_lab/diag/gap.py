"""The model's gap to the clairvoyant plan on Full 444 episodes 0-15, split by what was told, by item, system and time.

    uv run python lab/anastasiia/frontier_lab/diag/gap.py

Three steps of one ladder, the same planner in each (``hazard_lab/play.py`` tags, played on the rented machines):
the model (``h3c_f``), told the true network of its 26-week window (``truthallc_f``), told everything to the
episode's end (``tah0c_f``), and the board's clairvoyant plan (``clair.py``). Prints the steps per episode and per
harm level, each step by cost item, the shed load by grid and the unmet demand by market, and both by 13-week block.
"""

import fire
import numpy as np
from common import BN, ITEMS, Names, clair, kept, level_weights, paired, refs


def main(model: str = "h3c_f", window: str = "truthallc_f", whole: str = "tah0c_f", episodes: int = 16) -> None:
    nm = Names("full")
    inst, N = nm.inst, nm.inst.nodes
    R, C = refs(), clair()
    E = list(range(episodes))
    lvl = np.array([R[n]["stratum"] for n in E])
    naive = np.array([R[n]["J_naive_cents"] for n in E]) / 1e11
    best = np.array([R[n]["J_oracle_cents"] for n in E]) / 1e11
    room = naive - best
    w = level_weights(lvl)
    voll = np.array([N[g].grid.voll for g in inst.grids])
    plays = {"model": kept(model), "window": kept(window), "whole": kept(whole), "clairvoyant": C}
    J = {name: np.array([d[n]["J"] for n in E]) / 1e11 for name, d in plays.items()}
    steps = (("model", "window", "the window's network"), ("window", "whole", "the rest of the episode"),
             ("whole", "clairvoyant", "left with everything told"))  # fmt: skip

    print(
        f"1. Per episode, bn USD: room, the model's gap, and its three steps ({model} > {window} > {whole} > clairvoyant)"
    )
    print(" ep lvl   room    gap | window  whole   left | score of the model")
    for n in E:
        print(
            f" {n:2d}  {lvl[n]}  {room[n]:6.0f} {J['model'][n] - best[n]:6.0f} | {J['model'][n] - J['window'][n]:6.0f} "
            f"{J['window'][n] - J['whole'][n]:6.0f} {J['whole'][n] - best[n]:6.0f} | {(naive[n] - J['model'][n]) / room[n]:.3f}"
        )
    print("\n2. Per harm level (plain mean of its episodes), and the board's weighting of the levels")
    print(" lvl  n    room    gap | window  whole   left | model's score, told the window, told everything")
    for s in sorted(set(lvl.tolist())):
        m = lvl == s
        print(
            f"  {s}  {m.sum():2d}  {room[m].mean():6.0f} {(J['model'] - best)[m].mean():6.0f} | {(J['model'] - J['window'])[m].mean():6.0f} "
            f"{(J['window'] - J['whole'])[m].mean():6.0f} {(J['whole'] - best)[m].mean():6.0f} | "
            + " ".join(f"{(naive - J[x])[m].sum() / room[m].sum():.4f}" for x in ("model", "window", "whole"))
        )
    print(
        f" weighted  {np.sum(w * room):6.0f} {np.sum(w * (J['model'] - best)):6.0f} | {np.sum(w * (J['model'] - J['window'])):6.0f} "
        f"{np.sum(w * (J['window'] - J['whole'])):6.0f} {np.sum(w * (J['whole'] - best)):6.0f} | "
        + " ".join(f"{np.sum(w * (naive - J[x])) / np.sum(w * room):.4f}" for x in ("model", "window", "whole"))
    )
    for a, b, what in steps[:2]:
        point, lo, hi = paired(J[a] - J[b], lvl, room)
        print(f"   {what}: {point:+.4f} ({lo:+.4f} to {hi:+.4f})")

    def items(d) -> np.ndarray:
        return np.array([d[n]["costs"] for n in E]) / BN  # (episodes, T, 8)

    def shed(d) -> np.ndarray:
        return np.array([d[n]["shed"] * voll for n in E]) / BN  # (episodes, T, G)

    def lost(d) -> np.ndarray:
        return np.array([d[n]["lost"] * nm.penalty for n in E]) / BN  # (episodes, T, D)

    blocks = [(i, i + 13) for i in range(0, 104, 13)]
    le, mat = nm.demand_k == nm.K.index("chip_le"), nm.demand_k == nm.K.index("chip_mat")
    print("\n3. Each step by cost item, bn USD an episode (plain mean; weighted by the board's levels)")
    for a, b, what in steps:
        d = items(plays[a]) - items(plays[b])
        tot = d.sum(axis=1)
        print(
            f" {what} ({a} less {b}): "
            + ", ".join(f"{ITEMS[i]} {tot[:, i].mean():+.1f} ({np.sum(w * tot[:, i]):+.1f})" for i in (5, 7))
            + f", the other six {np.delete(tot, (5, 7), axis=1).sum(axis=1).mean():+.1f}"
        )
        for i in (5, 7):
            print(
                f"    {ITEMS[i]:8s} by 13-week block: "
                + " ".join(f"{d[:, lo:hi, i].sum(axis=1).mean():6.1f}" for lo, hi in blocks)
            )
        ds, dl = (shed(plays[a]) - shed(plays[b])).sum(axis=1), (lost(plays[a]) - lost(plays[b])).sum(axis=1)
        print(
            "    shed by grid: "
            + ", ".join(f"{x[5:]} {v:+.1f}" for x, v in zip(nm.grids, ds.mean(axis=0)) if abs(v) >= 0.05)
        )
        print(
            f"    unmet demand: chip_le {dl[:, le].sum(axis=1).mean():+.1f}, chip_mat {dl[:, mat].sum(axis=1).mean():+.1f}; by market: "
            + ", ".join(
                f"{m[5:]}:{k[5:]} {v:+.1f}"
                for (m, k), v in sorted(zip(nm.demands, dl.mean(axis=0)), key=lambda x: -abs(x[1]))[:8]
            )
        )

    print(
        "\n4. The clairvoyant plan's own use of what the simulator forbids: lots started on a grid that sheds base load that week"
    )
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
    for o in inst.osats:
        for raw, packed in N[o].osat.packages.items():
            pi[raw] = max(pi.get(raw, 0.0), pi[packed])
    worth = np.array([pi[N[f].fab.product] for f in inst.fabs])
    base = np.array([N[inst.grids[g]].grid.base_load for g in nm.fab_grid])
    share, value = [], []
    for n in E:
        on_shed = C[n]["shed"][:, nm.fab_grid] > 1e-6 * base
        share.append((C[n]["lots"] * on_shed).sum() / C[n]["lots"].sum())
        value.append(((C[n]["lots"] * worth - C[n]["energy"] * voll[nm.fab_grid]) * on_shed).sum() / BN)
    share, value = np.array(share), np.array(value)
    for s in sorted(set(lvl.tolist())):
        m = lvl == s
        print(
            f"  level {s}: {100 * share[m].mean():.0f} % of its lots; their chips less their energy at the price of shed load {value[m].mean():.0f} bn an episode"
        )
    left = J["whole"] - best
    print(
        f"  correlation over the episodes with what is left when everything is told: {np.corrcoef(value, left)[0, 1]:.2f}; "
        f"with the model's gap: {np.corrcoef(value, J['model'] - best)[0, 1]:.2f}"
    )
    print(
        "  the plays' shed load less the clairvoyant's, bn an episode: "
        + ", ".join(
            f"{name} {(shed(plays[name]).sum(axis=(1, 2)) - shed(C).sum(axis=(1, 2))).mean():+.1f}"
            for name in ("model", "window", "whole")
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
