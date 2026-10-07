"""Packaged chips disposed of at OSATs: when, where, and was the OSAT's outlet full or shut.

    uv run python lab/anastasiia/mpc_lab/level1/osat_probe.py --eps=1,2,5 --n_jobs=1

For each OSAT and packaged chip: units disposed of per week band; in the weeks with a disposal, the OSAT's outgoing
edges for that chip: open capacity (marks.u, not prohibited) and what the agent sent on them (executed slots whose
first edge leaves the OSAT), summed.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
BANDS = ((0, 13), (13, 26), (26, 39), (39, 52))


def one(agent: str, entropy: int, n: int) -> dict:
    from account_eps import play
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.marks import compute_marks

    z = play(agent, "small", entropy, n)
    inst, params = task_generator("small")
    mk = compute_marks(inst, sample_omega(inst, params, entropy, n, "train"))
    u, banned = np.asarray(mk.u), np.asarray(mk.prohibited)
    out = {}
    osats = [i for i, nd in enumerate(inst.nodes) if nd.id.startswith("osat")]
    for s, slot in enumerate(inst.stock_slots):
        if slot.node not in osats or inst.commodities[slot.k].id not in ("chip_le", "chip_mat"):
            continue
        k = slot.k
        edges = [e for e, ed in enumerate(inst.edges) if ed.tail == slot.node and k in ed.K]
        cap = sum(np.where(banned[:, e, k], 0.0, u[:, e]) for e in edges) if edges else np.zeros(inst.T)
        slots = [i for i, (e, kk, _l) in enumerate(inst.action_slots) if kk == k and e in edges]
        sent = z["sent"][:, slots].sum(axis=1) if slots else np.zeros(inst.T)
        # the edge's capacity is shared with the other commodities it carries: all its flows
        allk = [i for i, (e, _kk, _l) in enumerate(inst.action_slots) if e in edges]
        sent_all = z["sent"][:, allk].sum(axis=1) if allk else np.zeros(inst.T)
        cap_all = sum(u[:, e] for e in edges) if edges else np.zeros(inst.T)
        out[(inst.nodes[slot.node].id, inst.commodities[k].id)] = {
            "disp": z["disposal"][:, s],
            "stock": z["stock"][:, s],
            "storage": slot.storage,
            "cap": cap,
            "sent": sent,
            "sent_all": sent_all,
            "cap_all": cap_all,
        }
    return out


def main(eps, agent: str = "agents/anastasiia_hybrid_hub", entropy: int = 111, n_jobs: int = 1) -> None:
    root = Path(__file__).resolve().parents[4]
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    res = Parallel(n_jobs=n_jobs)(delayed(one)(str(root / agent), entropy, n) for n in which)
    keys = sorted(res[0])
    print(f"episodes {which}")
    print("osat / chip: disposed k units per episode by band 1-13 14-26 27-39 40-52 | in weeks with disposal: "
          "open out-cap of the chip, agent sent of the chip, all flows on those edges / their total cap (per week)")
    for key in keys:
        d = np.stack([r[key]["disp"] for r in res])
        if d.sum() <= 0:
            continue
        bands = " ".join(f"{d[:, lo:hi].sum(axis=1).mean() / 1e3:7.1f}" for lo, hi in BANDS)
        m = d > 0
        cap = np.stack([r[key]["cap"] for r in res])[m].mean()
        sent = np.stack([r[key]["sent"] for r in res])[m].mean()
        sa = np.stack([r[key]["sent_all"] for r in res])[m].mean()
        ca = np.stack([r[key]["cap_all"] for r in res])[m].mean()
        eps_with = [n for n, r in zip(which, res) if r[key]["disp"].sum() > 1e3]
        print(f"{key[0]:8s} {key[1]:9s} {d.sum(axis=1).mean() / 1e3:7.1f} = {bands} | cap {cap:9.0f} sent {sent:9.0f}"
              f" | all {sa:9.0f} / {ca:9.0f} | weeks {100 * m.mean():4.1f}%  eps>1k {eps_with}")
    print("per episode chip_le disposed at OSATs, k units:")
    for n, r in zip(which, res):
        print(f"  {n:3d} " + " ".join(f"{k[0]}:{r[k]['disp'].sum() / 1e3:.0f}" for k in keys if k[1] == "chip_le"))


if __name__ == "__main__":
    fire.Fire(main)
