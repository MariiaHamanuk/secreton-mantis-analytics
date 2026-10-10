"""Lots started, chips sold and chips left over by the model and the told plays on Full 444 episodes 0-15.

    uv run python lab/anastasiia/frontier_lab/diag/chips.py

1. Lots started (at the chip's penalty) and sales by 13-week block, fab energy and shed load beside them.
2. What the episode ends with: packaged and raw chips at fabs, plants and in the straits' queues.
3. Per episode: raw leading chips in the straits' queues at the end, the lots of the fabs that feed them, and what the
   play told everything sells more than the model.
"""

import fire
import numpy as np
from common import BN, Names, clair, kept, refs


def main(tags: str = "h3c_f,truthallc_f,tah0c_f", episodes: int = 16, also: str = "") -> None:
    nm = Names("full")
    inst, N = nm.inst, nm.inst.nodes
    E = list(range(episodes))
    R = refs()
    lvl = np.array([R[n]["stratum"] for n in E])
    names = [t for t in (tags + ("," + also if also else "")).split(",") if t]
    plays = {t: kept(t) for t in names} | {"clairvoyant": clair()}
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
    for o in inst.osats:
        for raw, packed in N[o].osat.packages.items():
            pi[raw] = max(pi.get(raw, 0.0), pi[packed])
    worth = np.array([pi[N[f].fab.product] for f in inst.fabs])
    voll = np.array([N[g].grid.voll for g in inst.grids])
    blocks = [(i, i + 13) for i in range(0, 104, 13)]

    print("1. By 13-week block, bn USD an episode (fab energy in GWh)")
    for what, get in (("lots started, at the chip's penalty", lambda e: e["lots"] * worth), ("sold", lambda e: e["served"] * nm.penalty),
                      ("fab energy, GWh", lambda e: e["energy"] * BN), ("shed load", lambda e: e["shed"] * voll)):  # fmt: skip
        print(f" {what}")
        for t, d in plays.items():
            v = np.array([get(d[n]) for n in E]) / BN
            print(
                f"   {t:12s} "
                + " ".join(f"{v[:, lo:hi].sum(axis=(1, 2)).mean():7.0f}" for lo, hi in blocks)
                + f"   all {v.sum(axis=(1, 2)).mean():7.0f}"
            )

    straits = set(inst.chokepoints)
    k_of = np.array([s.k for s in inst.stock_slots])
    place = np.array(["strait" if s.node in straits else N[s.node].id.split("_")[0] for s in inst.stock_slots])
    print("\n2. Chips at the end of the episode, thousand units an episode (and bn USD at the penalty)")
    for t in names:
        line = f"   {t:12s}"
        for name in ("chip_le", "chip_le_raw", "chip_mat", "chip_mat_raw"):
            k = nm.K.index(name)
            for where in ("fab", "osat", "strait"):
                m = (k_of == k) & (place == where)
                v = np.mean([plays[t][n]["stock"][-1, m].sum() for n in E]) if m.any() else 0.0
                if v > 20e3:
                    line += f" {name}@{where} {v / 1e3:.0f} ({v * pi[k] / BN:.0f})"
        print(line)

    k = nm.K.index("chip_le_raw")
    q = (k_of == k) & (place == "strait")
    le = nm.demand_k == nm.K.index("chip_le")
    base, told = plays[names[0]], plays[names[-1]]
    print(
        f"\n3. Per episode: raw leading chips in the straits' queues at the end (thousand), and what {names[-1]} sells more than {names[0]} (bn)"
    )
    for n in E:
        stuck = [plays[t][n]["stock"][-1, q].sum() / 1e3 for t in names]
        more = ((base[n]["lost"] - told[n]["lost"]) * nm.penalty).sum() / BN
        more_le = ((base[n]["lost"] - told[n]["lost"]) * nm.penalty)[:, le].sum() / BN
        lots = [(plays[t][n]["lots"] * worth).sum() / BN for t in (names[0], names[-1])]
        print(
            f"   ep {n:2d} L{lvl[n]}: in queues "
            + " / ".join(f"{v:6.0f}" for v in stuck)
            + f"; sold more {more:6.1f} (leading {more_le:6.1f}); "
            f"lots at the penalty {lots[0]:5.0f} against {lots[1]:5.0f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
