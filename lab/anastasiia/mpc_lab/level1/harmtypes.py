"""What separates level-1 episodes: harm by event type and a few scenario summaries, beside the agent's loss.

    uv run python lab/anastasiia/mpc_lab/level1/harmtypes.py --rows=outputs/level1/<dt>/rows.json
"""

import json
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def main(rows: str, entropy: int = 111) -> None:
    from shockbench_flow.disruption.harm import totals
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.omega.codes import EVENT_TYPES

    inst, params = task_generator("small")
    data = [r for r in json.loads((ROOT / rows).read_text()) if r["excluded"] is None]
    G0 = np.array([inst.nodes[g].grid.Gbar0 if hasattr(inst.nodes[g].grid, "Gbar0") else np.nan for g in inst.grids])
    wafer = [i for i, s in enumerate(inst.stock_slots) if inst.commodities[s.k].id == "wafer"]
    feats, names = [], []
    for r in data:
        n = r["episode"]
        omega = sample_omega(inst, params, entropy, n, "train")
        _, H, by = totals(inst, omega)
        mk = compute_marks(inst, omega)
        gen = np.asarray(mk.G_bar) / np.asarray(mk.G_bar).max(axis=0, keepdims=True)
        sup = np.asarray(mk.supply)[:, wafer]
        supf = sup.sum(axis=1) / max(sup.sum(axis=1).max(), 1e-9)
        f = [by[c] / 1e9 for c in range(len(EVENT_TYPES))]
        f += [100 * (gen < 0.999).mean(), float(supf.mean()), float(np.asarray(mk.o).mean())]
        f += [float(np.asarray(mk.prohibited).any(axis=2).mean() * 100)]
        feats.append(f)
    names = [f"H_{t}" for t in EVENT_TYPES] + ["gen_dip_%wk", "wafer_sup_rel", "strait_open", "prohib_%edge_wk"]
    X = np.array(feats)
    lv = np.array([r["stratum"] for r in data])
    jn = np.array([r["J_naive_cents"] for r in data]) / 1e11
    jc = np.array([r["J_clairvoyant_cents"] for r in data]) / 1e11
    ja = np.array([r["J_policy_cents"] for r in data]) / 1e11
    loss, gap = ja - jc, jn - jc
    m1, m23 = lv == 1, (lv == 2) | (lv == 3)
    print(f"{'feature':18s} {'L1 mean':>9} {'L2-3 mean':>9} {'corr(loss) all':>14} {'corr(loss) L1':>13} "
          f"{'corr(loss/gap) all':>18}")
    for j, nm in enumerate(names):
        x = X[:, j]
        c_all = np.corrcoef(x, loss)[0, 1] if x.std() > 0 else np.nan
        c_1 = np.corrcoef(x[m1], loss[m1])[0, 1] if x[m1].std() > 0 else np.nan
        c_r = np.corrcoef(x, loss / gap)[0, 1] if x.std() > 0 else np.nan
        print(f"{nm:18s} {x[m1].mean():9.2f} {x[m23].mean():9.2f} {c_all:14.2f} {c_1:13.2f} {c_r:18.2f}")
    print("\nper level-1 episode: ep loss | " + " ".join(n[:10] for n in names))
    for i in np.flatnonzero(m1):
        print(f"  {data[i]['episode']:3d} {loss[i]:6.1f} | " + " ".join(f"{v:10.2f}" for v in X[i]))


if __name__ == "__main__":
    fire.Fire(main)
