"""When the play told the true network of its window sheds less than the model: before, after or away from a change.

    uv run python lab/anastasiia/frontier_lab/diag/timing.py

For every grid and week of Full 444 episodes 0-15 the supply of each fuel is summed over the slots that feed the grid
or its terminal: the nominal capacity of the slot's first edge times the worst factor along its route that week (edge
capacity over nominal, a prohibition, a strait's openness), in weeks of the grid's burn of that fuel. A fall of 0.15
weeks of burn from one week to the next is an onset, a rise of 0.15 an end. The model's shed load less the told play's
(both pairs of plays: the rented machines' and this Mac's) is then split by where the week stands:

- "after an onset": the 12 weeks from an onset on a route of the grid (what a plan that saw the cut coming had ready);
- "after an end": the 8 weeks from an end (what a plan that knew the end did with the route's last cut weeks);
- "output cut": the grid's own output is under its nominal that week or was in the 4 weeks before;
- "quiet": none of these.
"""

import pickle

import fire
import numpy as np
from common import BN, OUT, Names, kept, level_weights, refs


def supply_index(nm: Names, marks: dict) -> np.ndarray:
    """(T, G, K): the routes' capacity for each fuel into each grid, in weeks of the grid's burn of it."""
    inst, N = nm.inst, nm.inst.nodes
    T = marks["u"].shape[0]
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
    choke = {c: i for i, c in enumerate(inst.chokepoints)}
    out = np.zeros((T, len(inst.grids), len(inst.commodities)))
    for go, g in enumerate(inst.grids):
        gr = N[g].grid
        homes = {g} | {inst.edges[e].tail for e in inst.in_edges[g] if N[inst.edges[e].tail].id.startswith("term")}
        for e, k, lane in inst.action_slots:
            if k not in gr.fuels or inst.edges[e].tail in homes:
                continue
            path = [e] if lane is None else list(inst.lanes[lane].edges)
            if inst.edges[path[-1]].head not in homes or not np.isfinite(u0[e]):
                continue
            factor = np.ones(T)
            for x in path:
                if np.isfinite(u0[x]):
                    factor = np.minimum(factor, marks["u"][:, x] / u0[x])
                factor = factor * (1.0 - marks["prohibited"][:, x, k])
                if inst.edges[x].tail in choke:
                    factor = np.minimum(factor, marks["o"][:, choke[inst.edges[x].tail]])
            out[:, go, k] += u0[e] * factor / (gr.shares[k] * gr.deliverable)
    return out


def main(pairs: str = "h3c_f:truthallc_f,h3_f:truthall_f", episodes: int = 16, step: float = 0.15) -> None:
    nm = Names("full")
    inst, N = nm.inst, nm.inst.nodes
    world = pickle.loads((OUT / "world_full_444.pkl").read_bytes())
    R = refs()
    lvl = np.array([R[n]["stratum"] for n in range(episodes)])
    w = level_weights(lvl)
    voll = np.array([N[g].grid.voll for g in inst.grids])
    kinds = ("after an onset", "after an end", "output cut", "quiet")
    for pair in pairs.split(","):
        a, b = pair.split(":")
        money = np.zeros((episodes, len(kinds), len(inst.grids)))
        weeks = np.zeros((episodes, len(kinds)))
        for n in range(episodes):
            M = world[n]["marks"]
            idx = np.minimum(supply_index(nm, M), 3.0)  # a route far above the burn is no scarcer for a part of it lost
            d = (kept(a)[n]["shed"] - kept(b)[n]["shed"]) * voll / BN  # (T, G), bn USD
            T = d.shape[0]
            for go, g in enumerate(inst.grids):
                change = np.diff(idx[:, go, :], axis=0, prepend=idx[:1, go, :])  # (T, K)
                onset = np.flatnonzero((change < -step).any(axis=1))
                end = np.flatnonzero((change > step).any(axis=1))
                low = M["G_bar"][:, go] < 0.999 * N[g].grid.deliverable
                kind = np.full(T, 3)
                for t in np.flatnonzero(low):
                    kind[t : t + 5] = 2
                for t in end:
                    kind[t : t + 8] = 1
                for t in onset:
                    kind[t : t + 12] = 0
                for i in range(len(kinds)):
                    money[n, i, go] = d[kind == i, go].sum()
                    weeks[n, i] += (kind == i).sum()
        print(f"== shed load, {a} less {b}, Full 444 episodes 0-{episodes - 1}, bn USD an episode")
        print(
            f"{'weeks that are':16s} {'grid-weeks':>10s} {'plain':>8s} {'weighted':>9s}   by grid: "
            + " ".join(f"{x[5:]:>6s}" for x in nm.grids)
        )
        for i, name in enumerate(kinds):
            tot = money[:, i].sum(axis=1)
            print(
                f"{name:16s} {weeks[:, i].mean():10.0f} {tot.mean():8.1f} {np.sum(w * tot):9.1f}             "
                + " ".join(f"{v:6.1f}" for v in money[:, i].mean(axis=0))
            )
        tot = money.sum(axis=(1, 2))
        print(f"{'all':16s} {weeks.sum(axis=1).mean():10.0f} {tot.mean():8.1f} {np.sum(w * tot):9.1f}")
        print("   after an onset, by episode: " + " ".join(f"{v:.0f}" for v in money[:, 0].sum(axis=1)))
        print("   after an end, by episode:   " + " ".join(f"{v:.0f}" for v in money[:, 1].sum(axis=1)))


if __name__ == "__main__":
    fire.Fire(main)
