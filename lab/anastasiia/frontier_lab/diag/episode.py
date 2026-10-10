"""One episode of two kept plays side by side: where the shed load differs, and the fuel stores of that grid by week.

uv run python lab/anastasiia/frontier_lab/diag/episode.py h3_f dg_hold_f 41
uv run python lab/anastasiia/frontier_lab/diag/episode.py h3_f dg_hold_f 41 --grid=grid_jp --weeks=30,60
"""

import fire
import numpy as np
from common import BN, Names, kept


def main(base: str, new: str, n: int, grid: str = "", weeks: str = "", task: str = "full", entropy: int = 444) -> None:
    nm = Names(task)
    inst, N = nm.inst, nm.inst.nodes
    a, b = kept(base, task, entropy)[n], kept(new, task, entropy)[n]
    voll = np.array([N[g].grid.voll for g in inst.grids])
    d = (b["shed"] - a["shed"]) * voll / BN  # what the new play sheds more, (T, G)
    print(
        f"episode {n}: {new} sheds more than {base} by grid, bn: "
        + ", ".join(f"{x[5:]} {v:+.1f}" for x, v in zip(nm.grids, d.sum(axis=0)))
    )
    go = nm.grids.index(grid) if grid else int(np.argmax(np.abs(d.sum(axis=0))))
    g = inst.grids[go]
    gr = N[g].grid
    big = np.flatnonzero(np.abs(d[:, go]) > 0.5)
    print(f"{nm.grids[go]}: weeks with over 0.5 bn of difference: " + " ".join(f"{t + 1}:{d[t, go]:+.1f}" for t in big))
    if weeks:
        lo, hi = (int(x) for x in str(weeks).strip("()").split(","))
    else:
        lo, hi = (max(1, int(big[0]) + 1 - 14), min(d.shape[0], int(big[-1]) + 3)) if len(big) else (1, 20)
    homes = [g] + sorted({inst.edges[e].tail for e in inst.in_edges[g] if N[inst.edges[e].tail].id.startswith("term")})
    fabs = np.flatnonzero(nm.fab_grid == go)
    cols = [(k, [inst.slot_index[(x, k)] for x in homes if (x, k) in inst.slot_index]) for k in gr.fuels]
    print(
        "week | per play: shed % of base; per fuel: generation % of its share, grid store + terminal in weeks of burn; lots % of capacity"
    )
    print("     | fuels: " + ", ".join(nm.K[k] for k, _ in cols))
    for t in range(lo - 1, hi):
        row = f"{t + 1:4d} |"
        for e in (a, b):
            row += f" {100 * e['shed'][t, go] / gr.base_load:5.1f}"
            for k, slots in cols:
                burn = gr.shares[k] * gr.deliverable
                row += f"  {100 * e['segment'][t, go, k] / burn:3.0f} " + "+".join(
                    f"{e['stock'][t, s] / burn:4.1f}" for s in slots
                )
            row += f"  {100 * np.mean([e['lots'][t, f] / N[inst.fabs[f]].fab.cap0 for f in fabs]) if len(fabs) else 0:3.0f} |"
        print(row)


if __name__ == "__main__":
    fire.Fire(main)
