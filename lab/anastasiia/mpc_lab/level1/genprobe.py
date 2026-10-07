"""Generation per grid and fuel segment: the agent against the base-first plan (GWh per episode, by week band).

    uv run python lab/anastasiia/mpc_lab/level1/genprobe.py --eps=26,45 --n_jobs=2
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "stats_lab"))
BANDS = ((0, 13), (13, 26), (26, 39), (39, 52))


def one(agent, entropy, n, time_limit):
    from account_eps import play
    from plan_stats import base_first_plan
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp

    inst, params = task_generator("small")
    z = play(agent, "small", entropy, n)
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "train"))
    model = build_lp(inst, marks, planning_rules=True)
    x, _status, _gap = base_first_plan(inst, marks, model, time_limit, 2e-3)
    T, nc, tmpl = model.T, model.meta["nc"], model.meta["template"]
    w = np.asarray(x).reshape(T, nc)
    K = len(inst.commodities)
    seg = np.zeros((T, len(inst.grids), K + 1))
    for go, g in enumerate(inst.grids):
        for k in inst.nodes[g].grid.fuels + (None,):
            if ("G", go, k) in tmpl:
                seg[:, go, -1 if k is None else k] = w[:, tmpl[("G", go, k)]]
    return z["segment"], seg


def main(eps, agent: str = "agents/anastasiia_hybrid_hub", entropy: int = 111, time_limit: float = 150.0,
         n_jobs: int = 2) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("small")
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    res = Parallel(n_jobs=n_jobs)(delayed(one)(str(ROOT / agent), entropy, n, time_limit) for n in which)
    a = np.mean([r[0] for r in res], axis=0)
    b = np.mean([r[1] for r in res], axis=0)
    names = [c.id for c in inst.commodities] + ["free"]
    print(f"episodes {which}: generation a-bf, GWh per episode, by band 1-13 14-26 27-39 40-52 (and total; agent total)")
    for go, g in enumerate(inst.grids):
        for k in list(inst.nodes[g].grid.fuels) + [None]:
            kk = -1 if k is None else k
            d = [a[lo:hi, go, kk].sum() - b[lo:hi, go, kk].sum() for lo, hi in BANDS]
            print(f"  {inst.nodes[g].id:8s} {names[kk]:8s} " + " ".join(f"{v:8.0f}" for v in d)
                  + f"  | {sum(d):8.0f}  agent {a[:, go, kk].sum():9.0f}")


if __name__ == "__main__":
    fire.Fire(main)
