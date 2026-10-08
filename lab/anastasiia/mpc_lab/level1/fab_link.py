"""How much of the shortage gap to the base-first plan follows each fab's lot deficit (per episode, ldiag npz)."""

import json
import sys

import numpy as np
from shockbench_flow.hosting.tasks import task_generator

inst, _ = task_generator("small")
N = inst.nodes
rows = {r["episode"]: r["stratum"] for r in json.load(open(sys.argv[1]))}
import glob
eps = sorted(int(p.split("ep")[-1][:-4]) for p in glob.glob("outputs/level1/diag/ep*.npz"))
Z = {n: np.load(f"outputs/level1/diag/ep{n}.npz") for n in eps}
for name, sel in (("L1", [n for n in eps if rows[n] == 1]), ("L2-3", [n for n in eps if rows[n] in (2, 3)])):
    short = np.array([(Z[n]["a_costs"][:, 5].sum() - Z[n]["b_costs"][:, 5].sum()) / 1e9 for n in sel])
    shed = np.array([(Z[n]["a_costs"][:, 7].sum() - Z[n]["b_costs"][:, 7].sum()) / 1e9 for n in sel])
    print(f"{name}: {len(sel)} ep, shortage a-bf mean {short.mean():.1f}, shed a-bf {shed.mean():.1f}")
    for i, f in enumerate(inst.fabs):
        cap = N[f].fab.cap0
        d = np.array([(Z[n]["b_lots"][:40, i].sum() - Z[n]["a_lots"][:40, i].sum()) for n in sel])  # lots, wk 1-40
        c = np.corrcoef(d, short)[0, 1]
        pos = d.clip(0).mean()
        print(f"   {N[f].id:18s} lot deficit bf-a {d.mean() / 1e6:6.2f} M/ep (positive part {pos / 1e6:5.2f}), "
              f"episodes with bf > a by >10% cap-weeks: {(d > 0.1 * cap * 40).sum():2d}, corr with shortage gap {c:5.2f}")
    go = inst.grid_ordinal[N[inst.fabs[2]].fab.grid]
    kr = np.array([(Z[n]["a_shed_usd"][:, go].sum() - Z[n]["b_shed_usd"][:, go].sum()) / 1e9 for n in sel])
    dkr = np.array([(Z[n]["b_lots"][:40, 2].sum() - Z[n]["a_lots"][:40, 2].sum()) for n in sel])
    A = np.c_[np.ones(len(sel)), dkr / 1e6]
    coef = np.linalg.lstsq(A, short, rcond=None)[0]
    print(f"   KR shed a-bf {kr.mean():.1f} bn/ep; regression shortage gap ~ {coef[0]:.1f} + {coef[1]:.1f} x KR lot deficit (M)"
          f" -> KR part {coef[1] * dkr.mean() / 1e6:.1f} bn/ep")

print("\nKR-deficit episodes (bf starts > a by 10% of capacity x 40 weeks at fab_kr_memory_1):")
cap = N[inst.fabs[2]].fab.cap0
go = inst.grid_ordinal[N[inst.fabs[2]].fab.grid]
for name, sel in (("L1", [n for n in eps if rows[n] == 1]), ("L2-3", [n for n in eps if rows[n] in (2, 3)])):
    kd = [n for n in sel if Z[n]["b_lots"][:40, 2].sum() - Z[n]["a_lots"][:40, 2].sum() > 0.1 * cap * 40]
    tot = sum((Z[n]["a_J"] - Z[n]["b_J"]) for n in kd) / 1e9 / len(sel)
    sh = sum((Z[n]["a_costs"][:, 5].sum() - Z[n]["b_costs"][:, 5].sum()) for n in kd) / 1e9 / len(sel)
    krs = sum((Z[n]["a_shed_usd"][:, go].sum() - Z[n]["b_shed_usd"][:, go].sum()) for n in kd) / 1e9 / len(sel)
    alla = sum((Z[n]["a_J"] - Z[n]["b_J"]) for n in sel) / 1e9 / len(sel)
    print(f"  {name}: {len(kd)} of {len(sel)} episodes {kd}: a-bf in them {tot:.1f} bn per episode of the level "
          f"(shortage {sh:.1f}, KR shed {krs:.1f}) of a-bf {alla:.1f}")
