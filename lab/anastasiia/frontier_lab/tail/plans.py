"""The plan of the long window against the plan of the short one from the same state (``probe.py run``'s files).

    uv run python lab/anastasiia/frontier_lab/tail/plans.py w20 --episodes=3,12,15

Per probed week both plans cover the short window's weeks: the first week's requests by kind of slot, the lots, the
shed load and the lost sales by part of the window, what is held at the short window's last week, and (with the
probe's ``--duals``) what the long window's program pays for a unit held there, in shares of the flat end credit.
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "outputs" / "tail_lab" / "probe"
END_FUEL, END_CHIP = 3.3e6, 0.6


def main(name: str = "w20", task: str = "full", entropy: int = 444, episodes: str | tuple = "3", since: int = 1,
         until: int = 104, by_week: bool = False) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    ns = [int(n) for n in (episodes.split(",") if isinstance(episodes, str) else episodes)]
    names = [c.id for c in inst.commodities]
    supply, chk = set(inst.supply_nodes), set(inst.chokepoints)
    slot_k = np.array([k for _e, k, _lane in inst.action_slots])
    first = np.array([e for e, _k, _lane in inst.action_slots])
    order = np.array([inst.edges[e].tail in supply for e in first])
    pi = np.zeros(len(inst.commodities))
    for d in inst.demands:
        pi[d.k] = max(pi[d.k], d.pi)
    for o in inst.osats:
        for raw, packed in inst.nodes[o].osat.packages.items():
            pi[raw] = max(pi[raw], pi[packed])
    prod = np.array([inst.nodes[f].fab.product for f in inst.fabs])
    voll = np.array([inst.nodes[g].grid.voll for g in inst.grids])
    dpi = np.array([d.pi for d in inst.demands])
    dk = np.array([d.k for d in inst.demands])

    def kind(node: int) -> str:
        nd = inst.nodes[node]
        return ("grid" if nd.grid is not None else "plant" if nd.osat is not None else "fab" if nd.fab is not None
                else "source" if node in supply else "strait" if node in chk else "market" if any(d.node == node for d in inst.demands)
                else "port")  # fmt: skip

    where = np.array([f"{names[sl.k]}@{kind(sl.node)}" for sl in inst.stock_slots])
    credit = np.array([0.0 if sl.node in supply else (END_FUEL if sl.k <= 2 else END_CHIP * pi[sl.k]) for sl in inst.stock_slots])
    groups = sorted(set(where))
    rows, price_rows = [], []
    for n in ns:
        data = pickle.loads((OUT / f"{name}_{task}_{entropy}_{n}.pkl").read_bytes())
        for week, row in sorted(data["weeks"].items()):
            if "shadow_plan" not in row or row["plan"] is None or row["shadow_plan"] is None or not since <= week <= until:
                continue
            a, b = row["plan"], row["shadow_plan"]  # the long window's, the short one's
            Hs = b["H"]
            if a["H"] <= Hs:  # the same window: the episode's last weeks
                continue
            df = row["shadow_flows"] - row["flows"]
            out = {"n": n, "week": week}
            for k in (0, 1, 2):
                sel = (slot_k == k) & order
                out[f"order {names[k]}"] = (df[sel].sum(), np.abs(df[sel]).sum(), row["flows"][sel].sum())
                sel = (slot_k == k) & ~order
                out[f"moved on {names[k]}"] = (df[sel].sum(), np.abs(df[sel]).sum(), row["flows"][sel].sum())
            for k in (3, 4, 5, 6, 7):
                sel = slot_k == k
                out[f"sent {names[k]}"] = (df[sel].sum(), np.abs(df[sel]).sum(), row["flows"][sel].sum())
            parts = [(0, 6), (6, 12), (12, Hs)]
            for k in (4, 5):
                for lo, hi in parts:
                    la, lb = a["lots"][lo:hi][:, prod == k].sum(), b["lots"][lo:hi][:, prod == k].sum()
                    out[f"lots {names[k]} w{lo + 1}-{hi}"] = (lb - la, abs(lb - la), la)
            for lo, hi in parts:
                sa, sb = (a["shed"][lo:hi] * voll).sum() / 1e9, (b["shed"][lo:hi] * voll).sum() / 1e9
                out[f"shed bn w{lo + 1}-{hi}"] = (sb - sa, abs(sb - sa), sa)
                for k in (6, 7):
                    xa, xb = (a["lost"][lo:hi] * dpi)[:, dk == k].sum() / 1e9, (b["lost"][lo:hi] * dpi)[:, dk == k].sum() / 1e9
                    out[f"lost {names[k]} bn w{lo + 1}-{hi}"] = (xb - xa, abs(xb - xa), xa)
            ca, cb = a["costs"][:Hs].sum() / 1e9, b["costs"][:Hs].sum() / 1e9
            out["cost of the short window's weeks, bn"] = (cb - ca, abs(cb - ca), ca)
            for g in groups:
                sel = where == g
                ha, hb = a["stock"][Hs - 1][sel].sum(), b["stock"][Hs - 1][sel].sum()
                out[f"held {g}"] = (hb - ha, abs(hb - ha), ha)
            rows.append(out)
            if row.get("prices") is not None:
                p = row["prices"]
                pr = {"n": n, "week": week}
                for g in groups:
                    sel = (where == g) & (credit > 0)
                    if sel.any():
                        pr[g] = (p["worth"][sel] / credit[sel], p["held"][sel])
                price_rows.append(pr)
    print(f"{name}: {len(rows)} probed weeks of episodes {ns}; the short window's plan minus the long one's, same state")
    print(f"{'':42s} {'mean':>12s} {'mean |.|':>12s} {'long plan':>12s}")
    keys = [k for k in rows[0] if k not in ("n", "week")]
    for key in keys:
        arr = np.array([r[key] for r in rows])
        print(f"{key:42s} {arr[:, 0].mean():+12.4g} {arr[:, 1].mean():12.4g} {arr[:, 2].mean():12.4g}")
    if by_week:
        for r in rows:
            print(f"  ep {r['n']} week {r['week']:3d}: " + "  ".join(
                f"{key} {r[key][0]:+.3g}" for key in ("order lng", "order nucfuel", "sent wafer", "shed bn w1-6", "shed bn w7-12", "shed bn w13-20",
                                                    "lots chip_le_raw w7-12", "lots chip_le_raw w13-20", "cost of the short window's weeks, bn") if key in r))
    if price_rows:
        print(f"\nwhat the long window's program pays for a unit held at the short window's last week, in shares of the flat end credit "
              f"({len(price_rows)} weeks; weighted by what is held / plain mean over slots; share of slot-weeks at 0, under 0.5, 0.5-1.5, above)")
        for g in groups:
            got = [r[g] for r in price_rows if g in r]
            if not got:
                continue
            w = np.concatenate([x[0] for x in got])
            h = np.concatenate([x[1] for x in got])
            hw = (w * h).sum() / h.sum() if h.sum() > 0 else float("nan")
            print(f"  {g:22s} {hw:8.2f} / {w.mean():8.2f}   {np.mean(np.abs(w) < 0.02):.2f} {np.mean((np.abs(w) >= 0.02) & (w < 0.5)):.2f} "
                  f"{np.mean((w >= 0.5) & (w <= 1.5)):.2f} {np.mean(w > 1.5):.2f}   quartiles {np.percentile(w, 25):.2f} {np.percentile(w, 50):.2f} {np.percentile(w, 75):.2f}")


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    fire.Fire(main)
