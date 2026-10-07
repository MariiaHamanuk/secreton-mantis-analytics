"""Where an agent's gain over a base comes from, as a ledger of three channels that add up to the cost difference.

    uv run python lab/anastasiia/plan_lab/ledger.py --task=small --base=outputs/plan_lab/20261007_step0/hub_small_444.npz \
        truth=outputs/plan_lab/20261007_ledger/truth_small_444.npz lean=outputs/plan_lab/20261007_gate3/small_lean_h26_end_x16.npz

Every file holds weekly arrays of the same episodes 0.. of one root (``harness.py --save``, ``regime.py run --save``);
the first ``--episodes`` of each are compared (all the agents have, when 0). No episode is played.

An episode's cost is almost all shed base load and unserved demand, and both follow from where the electricity goes:

- generation: a grid's output is its fuel burned plus its free segment; more of it is less shed, at the grid's VOLL;
- energy to the fabs: what the fabs take is not there for the base load, at the same VOLL;
- sales: the chips that energy becomes, when they reach a market.

Shed = base load - (generation - energy to the fabs), so the first two rows are the change of shed cost exactly, per
grid; the third is the change of the unserved-demand cost; "other" is the rest (freight, holding, disposal). Under
the money: the quantities each channel moved (fuel burned, energy to the fabs, lots, chips sold and thrown away,
grid-weeks with the base load fully served). All per episode, the agent minus the base; negative money is a gain.
"""

import fire
import numpy as np


def channels(inst, A: dict, B: dict, n: int) -> dict:
    """The ledger of agent arrays ``A`` against base arrays ``B`` on their first ``n`` episodes."""
    N, K = inst.nodes, [c.id for c in inst.commodities]
    bn = 1e9
    voll = np.array([N[g].grid.voll for g in inst.grids])
    cost = (A["cost"][:n] - B["cost"][:n]).sum(axis=1).mean(axis=0) / bn  # by component
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    seg = (A["segment"][:n] - B["segment"][:n]).sum(axis=1).mean(axis=0)  # (grid, commodity + free)
    fuels = sorted({k for g in inst.grids for k in N[g].grid.fuels})
    fab_e = np.zeros(len(inst.grids))
    for gi in range(len(inst.grids)):
        members = list(inst.grid_fabs[gi])
        if members:
            fab_e[gi] = (A["energy"][:n][:, :, members] - B["energy"][:n][:, :, members]).sum() / n
    pi = np.zeros(len(K))
    for d in inst.demands:
        pi[d.k] = max(pi[d.k], d.pi)
    packed_of = {raw: pk for o in inst.osats for raw, pk in N[o].osat.packages.items()}
    fabs = set(inst.fabs)
    plants = {x for x in range(len(N)) if getattr(N[x], "osat", None) is not None}
    waste = [
        s
        for s, sl in enumerate(inst.stock_slots)
        if (sl.k in packed_of and sl.node in fabs) or (pi[sl.k] > 0 and sl.node in plants)
    ]
    fab_grids = [gi for gi in range(len(inst.grids)) if inst.grid_fabs[gi]]
    out = {
        "total": float((A["J"][:n] - B["J"][:n]).mean() / bn),
        "generation": float(-(seg.sum(axis=1) * voll).sum() / bn),
        "generation: fuel burned": float(-(seg[:, :-1].sum(axis=1) * voll).sum() / bn),
        "generation: free segment": float(-(seg[:, -1] * voll).sum() / bn),
        "energy to the fabs": float((fab_e * voll).sum() / bn),
        "shed (the two above)": float(cost[comp.index("shed")]),
        "sales": float(cost[comp.index("shortage")]),
    }
    out["other"] = out["total"] - out["shed (the two above)"] - out["sales"]
    out |= {
        "fuel burned, thousand units": {K[k]: float(seg[:, k].sum() / 1e3) for k in fuels},
        "energy to the fabs, TWh": float(fab_e.sum() / 1e3),
        "lots, M": float((A["lots"][:n] - B["lots"][:n]).sum() / n / 1e6),
        "chips sold, M": {
            K[k]: float(
                sum(
                    (A["served"][:n][:, :, di] - B["served"][:n][:, :, di]).sum()
                    for di, d in enumerate(inst.demands)
                    if d.k == k
                )
                / n
                / 1e6
            )
            for k in sorted({d.k for d in inst.demands})
        },
        "chips thrown away, M": float(
            (A["disposal"][:n][:, :, waste] - B["disposal"][:n][:, :, waste]).sum() / n / 1e6
        ),
        "complete weeks of grids with fabs": float(
            ((A["shed"][:n][:, :, fab_grids] < 1e-6).sum() - (B["shed"][:n][:, :, fab_grids] < 1e-6).sum()) / n
        ),
        "base": {
            "lots, M": float(B["lots"][:n].sum() / n / 1e6),
            "chips thrown away, M": float(B["disposal"][:n][:, :, waste].sum() / n / 1e6),
            "energy to the fabs, TWh": float(
                sum(B["energy"][:n][:, :, list(inst.grid_fabs[gi])].sum() for gi in fab_grids) / n / 1e3
            ),
            "complete weeks of grids with fabs": float((B["shed"][:n][:, :, fab_grids] < 1e-6).sum() / n),
            "shed, bn": float((B["cost"][:n][:, :, comp.index("shed")]).sum() / n / bn),
            "unserved demand, bn": float((B["cost"][:n][:, :, comp.index("shortage")]).sum() / n / bn),
        },
    }
    if (
        out["energy to the fabs, TWh"] != 0
    ):  # what a GWh taken from the fabs cost in sales, M USD (VOLL is what it saved)
        out["sales per GWh moved off the fabs, M USD"] = float(
            out["sales"] * bn / (-out["energy to the fabs, TWh"] * 1e3) / 1e6
        )
    return out


def main(*agents: str, task: str = "small", base: str = "", episodes: int = 0) -> None:
    """``agents`` are ``name=path`` of weekly arrays; ``base`` the arrays they are set against."""
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    B = dict(np.load(base))
    loaded = {a.split("=", 1)[0]: dict(np.load(a.split("=", 1)[1])) for a in agents}
    n = episodes or min([len(B["J"])] + [len(A["J"]) for A in loaded.values()])
    ledgers = {name: channels(inst, A, B, n) for name, A in loaded.items()}
    names = list(ledgers)
    first = ledgers[names[0]]
    voll = sorted({inst.nodes[g].grid.voll for g in inst.grids})
    print(f"{task}: {n} episodes, per episode, the agent minus the base; VOLL {voll[0] / 1e6:.1f} M USD per GWh")
    print(f"base: {first['base']}")
    width = max(len(x) for x in names) + 2
    print(f"\n{'bn USD (negative: a gain)':40}" + "".join(f"{x:>{width}}" for x in names))
    for key in (
        "total",
        "generation",
        "generation: fuel burned",
        "generation: free segment",
        "energy to the fabs",
        "shed (the two above)",
        "sales",
        "other",
    ):
        print(f"{key:40}" + "".join(f"{ledgers[x][key]:>{width}.1f}" for x in names))
    print(f"\n{'quantities':40}" + "".join(f"{x:>{width}}" for x in names))
    for key in (
        "energy to the fabs, TWh",
        "lots, M",
        "chips thrown away, M",
        "complete weeks of grids with fabs",
        "sales per GWh moved off the fabs, M USD",
    ):
        print(f"{key:40}" + "".join(f"{ledgers[x].get(key, float('nan')):>{width}.2f}" for x in names))
    for key in ("fuel burned, thousand units", "chips sold, M"):
        for sub in first[key]:
            print(f"{key + ': ' + sub:40}" + "".join(f"{ledgers[x][key][sub]:>{width}.2f}" for x in names))


if __name__ == "__main__":
    fire.Fire(main)
