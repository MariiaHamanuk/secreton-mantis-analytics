"""Fuel the model throws away while its grid's stores have room, and the shed that follows the next cut of the route.

    uv run python lab/anastasiia/frontier_lab/diag/buffers.py

For a grid whose fuel comes by direct edges only (no strait on the way), what arrives each week is what was sent a
lead time earlier, so the kept plays give the disposal of the pair "terminal and grid's store" exactly:
stock before + arrivals - burn - stock after. Printed per play: the fuel disposed of over the episode, the part of it
disposed of in weeks that end with room in the pair, the pair's mean fill, and the shed load of the grid in weeks
when that fuel's segment runs short. Then, for every onset of a cut on a grid's fuel routes (``timing.py``), the
stores the model and the told play held the week before and the shed the told play saves in the 12 weeks after.
"""

import pickle

import fire
import numpy as np
from common import BN, OUT, Names, kept, level_weights, refs
from timing import supply_index


def direct_pairs(nm: Names) -> list:
    """(grid ordinal, fuel, stock slots of the pair, [(action slot, lead time)]) where every route is a direct edge."""
    inst, N = nm.inst, nm.inst.nodes
    out = []
    for go, g in enumerate(inst.grids):
        homes = {g} | {inst.edges[e].tail for e in inst.in_edges[g] if N[inst.edges[e].tail].id.startswith("term")}
        for k in N[g].grid.fuels:
            if not any((x, k) in inst.slot_index for x in homes - {g}):
                continue  # no terminal: nothing to choose between
            routes, direct = [], True
            for s, (e, kk, lane) in enumerate(inst.action_slots):
                if kk != k or inst.edges[e].tail in homes:
                    continue
                path = [e] if lane is None else list(inst.lanes[lane].edges)
                if inst.edges[path[-1]].head in homes:
                    direct &= lane is None
                    routes.append((s, int(inst.edges[e].tau)))
            if direct and routes:
                out.append(
                    (go, k, [inst.slot_index[(x, k)] for x in sorted(homes) if (x, k) in inst.slot_index], routes)
                )
    return out


def main(tags: str = "h3c_f,h3_f,truthallc_f,truthall_f,tah0c_f", episodes: int = 16, also: str = "") -> None:
    nm = Names("full")
    inst, N, S = nm.inst, nm.inst.nodes, nm.inst.stock_slots
    names = [t for t in (tags + ("," + also if also else "")).split(",") if t]
    print(
        "1. Grids fed by direct edges only: fuel disposed of at the terminal and the grid's store, thousand units an episode"
    )
    for go, k, slots, routes in direct_pairs(nm):
        gr = N[inst.grids[go]].grid
        burn, room_all = gr.shares[k] * gr.deliverable, sum(S[s].storage for s in slots)
        print(
            f" {nm.grids[go]} {nm.K[k]}: burn {burn:.0f} a week, the two stores hold {room_all / burn:.1f} weeks of it"
        )
        for tag in names:
            waste, with_room, fill, short = [], [], [], []
            for n in range(episodes):
                e = kept(tag)[n]
                B = e["stock"][:, slots].sum(axis=1)
                arrive = np.zeros(len(B))
                for s, tau in routes:
                    arrive[tau:] += e["sent"][: len(B) - tau, s]
                d = np.maximum(0.0, B[:-1] + arrive[1:] - e["segment"][1:, go, k] - B[1:])  # weeks 2..T
                waste.append(d.sum())
                with_room.append(d[B[1:] < room_all - 0.25 * burn].sum())
                fill.append(B.mean() / room_all)
                short.append(
                    (e["shed"][:, go] * (e["segment"][:, go, k] < 0.98 * gr.shares[k] * gr.deliverable)).sum()
                    * gr.voll
                    / BN
                )
            print(
                f"   {tag:12s} disposed of {np.mean(waste) / 1e3:7.1f}, of it while the stores had room {np.mean(with_room) / 1e3:7.1f} "
                f"({np.mean(with_room) / burn:.1f} weeks of burn); mean fill {100 * np.mean(fill):3.0f} %; shed in weeks short of this fuel "
                f"{np.mean(short):6.1f} bn"
            )

    world = pickle.loads((OUT / "world_full_444.pkl").read_bytes())
    R = refs()
    lvl = np.array([R[n]["stratum"] for n in range(episodes)])
    w = level_weights(lvl)
    rows = []
    for n in range(episodes):
        idx = np.minimum(supply_index(nm, world[n]["marks"]), 3.0)
        for go, g in enumerate(inst.grids):
            gr = N[g].grid
            homes = [g] + sorted(
                {inst.edges[e].tail for e in inst.in_edges[g] if N[inst.edges[e].tail].id.startswith("term")}
            )
            change = np.diff(idx[:, go, :], axis=0, prepend=idx[:1, go, :])
            onsets = []
            for t in np.flatnonzero((change < -0.15).any(axis=1)):
                if not onsets or t - onsets[-1] >= 12:
                    onsets.append(int(t))
            for t0 in onsets:
                k = next(k for k in gr.fuels if change[t0, k] < -0.15)
                slots = [inst.slot_index[(x, k)] for x in homes if (x, k) in inst.slot_index]
                burn = gr.shares[k] * gr.deliverable
                saved, held = [], []
                for a, b in (("h3c_f", "truthallc_f"), ("h3_f", "truthall_f")):
                    ea, eb = kept(a)[n], kept(b)[n]
                    saved.append((ea["shed"][t0 : t0 + 12, go] - eb["shed"][t0 : t0 + 12, go]).sum() * gr.voll / BN)
                    held.append(
                        (eb["stock"][max(t0 - 1, 0), slots].sum() - ea["stock"][max(t0 - 1, 0), slots].sum()) / burn
                    )
                rows.append(
                    (
                        float(np.mean(saved)),
                        n,
                        nm.grids[go],
                        t0 + 1,
                        nm.K[k],
                        idx[max(t0 - 1, 0), go, k],
                        idx[t0, go, k],
                        float(np.mean(held)),
                        w[n],
                    )
                )
    rows.sort(reverse=True)
    print(
        f"\n2. Onsets of a cut on a grid's fuel routes: {len(rows)} in {episodes} episodes. Shed the told play saves in the 12 weeks after "
        "(mean of the two pairs of plays), and the weeks of burn it held more than the model the week before"
    )
    for r in rows[:12]:
        print(
            f"   {r[0]:6.1f} bn  ep {r[1]:2d} {r[2]:9s} week {r[3]:3d} {r[4]:7s} routes {r[5]:.2f} > {r[6]:.2f} weeks of burn a week; told held {r[7]:+.1f} weeks more"
        )
    total = sum(r[0] for r in rows)
    more = [r for r in rows if r[7] > 1.0]
    print(
        f"   all onsets: {total / episodes:.1f} bn an episode plain, {sum(r[0] * r[8] for r in rows):.1f} weighted; "
        f"those before which the told play held over a week of burn more ({len(more)}): {sum(r[0] for r in more) / episodes:.1f} plain, "
        f"{sum(r[0] * r[8] for r in more):.1f} weighted"
    )


if __name__ == "__main__":
    fire.Fire(main)
