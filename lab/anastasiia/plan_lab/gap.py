"""An agent beside the plan with "base load first", by quarter of the episode: where the money and the chips are.

    uv run python lab/anastasiia/plan_lab/gap.py outputs/plan_lab/20261007_step0/hub_small_444.npz \
        outputs/plan_stats/20261006_040041/episodes.npz --task=small
    uv run python lab/anastasiia/plan_lab/gap.py outputs/plan_lab/20261007_step0/hub_full_444.npz \
        outputs/heur4/lead/plan_full_444_x20/episodes.npz --task=full --skip=3

The first file is the agent's weekly arrays (``lab/anastasiia/heur_lab2/tools/account.py --save``), the second the
plan's (``lab/anastasiia/stats_lab/plan_stats.py``), both on episodes 0.. of the same root; ``--skip`` names the
episodes the plan file leaves out. No episode is played. ``lab/anastasiia/heur_lab3/tools/plan_gap.py`` prints the
money by quarter and the totals of the chain; this prints the chain by quarter too (lots, fab energy, raw chips out of
the fabs, packaged chips out of the plants, chips sold) and what the agent throws away or leaves at the end, stage by
stage. Money is USD bn per episode, the agent minus the plan; quantities are per episode.
"""

import fire
import numpy as np


def main(agent: str, plan: str, task: str = "small", skip: str = "") -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    N, E, K, T = inst.nodes, inst.edges, [c.id for c in inst.commodities], inst.T
    a, p = np.load(agent), np.load(plan)
    left_out = {int(x) for x in str(skip).split(",") if str(x).strip() != ""}
    keep = [n for n in range(len(a["J"])) if n not in left_out][: len(p["J_plan"])]
    A = {k: a[k][keep] for k in a.files}
    n_ep, bn = len(keep), 1e9
    quarters = [(q * T // 4, (q + 1) * T // 4) for q in range(4)]
    head = "  ".join(f"wk {lo + 1}-{hi}".rjust(15) for lo, hi in quarters)

    def by_quarter(x) -> list[float]:
        """(episode, week) -> the mean over episodes of each quarter's sum."""
        return [float(x[:, lo:hi].sum() / n_ep) for lo, hi in quarters]

    def pair(name: str, x, y, scale: float = 1.0, digits: int = 2) -> None:
        xs, ys = by_quarter(x), by_quarter(y)
        cells = "  ".join(f"{u / scale:7.{digits}f}|{v / scale:7.{digits}f}" for u, v in zip(xs, ys))
        print(f"   {name:26} {cells}  {sum(xs) / scale:8.{digits}f}|{sum(ys) / scale:8.{digits}f}")

    gap = (A["J"] - p["J_plan"]).mean() / bn
    print(f"{task}: {n_ep} episodes; the agent minus the plan {gap:.1f} bn per episode (agent | plan below)")

    pi = np.array([d.pi for d in inst.demands])
    voll = np.array([N[g].grid.voll for g in inst.grids])
    short = (p["plan_served"] - A["served"]) * pi / bn
    shed = (A["shed"] - p["plan_shed"]) * voll / bn
    chips = sorted({K[d.k] for d in inst.demands})
    print(f"\n1. Money, the agent minus the plan, bn per episode\n   {'':26} {head}            total")
    total = np.zeros(4)
    for chip in chips:
        m = np.array([K[d.k] == chip for d in inst.demands])
        row = np.array(by_quarter(short[:, :, m].sum(axis=2)))
        total += row
        print(f"   {'unsold ' + chip:26} " + "  ".join(f"{x:15.1f}" for x in row) + f"  {row.sum():15.1f}")
    for gi, g in enumerate(inst.grids):
        row = np.array(by_quarter(shed[:, :, gi]))
        total += row
        print(f"   {'shed ' + N[g].id:26} " + "  ".join(f"{x:15.1f}" for x in row) + f"  {row.sum():15.1f}")
    row = np.array(by_quarter(shed.sum(axis=2)))
    print(f"   {'shed, all grids':26} " + "  ".join(f"{x:15.1f}" for x in row) + f"  {row.sum():15.1f}")
    print(f"   {'chips and shed':26} " + "  ".join(f"{x:15.1f}" for x in total) + f"  {total.sum():15.1f}")
    print(f"   {'everything else':26} {'':66}  {gap - total.sum():15.1f}")

    print(f"\n2. Lots started, millions (agent | plan)\n   {'':26} {head}            total")
    for fo, f in enumerate(inst.fabs):
        pair(N[f].id, A["lots"][:, :, fo], p["plan_lots"][:, :, fo], 1e6)
    pair("all fabs", A["lots"].sum(axis=2), p["plan_lots"].sum(axis=2), 1e6)
    print(f"\n3. Energy to the fabs, TWh (agent | plan)\n   {'':26} {head}            total")
    for gi, g in enumerate(inst.grids):
        members = list(inst.grid_fabs[gi])
        if members:
            pair(N[g].id, A["energy"][:, :, members].sum(axis=2), p["plan_fab_energy"][:, :, members].sum(axis=2), 1e3)
    pair("all grids", A["energy"].sum(axis=2), p["plan_fab_energy"].sum(axis=2), 1e3)

    fabs = set(inst.fabs)
    plants = {n for n in range(len(N)) if getattr(N[n], "osat", None) is not None}
    slots = list(enumerate(inst.action_slots))
    print(f"\n4. The chip chain, millions (agent | plan)\n   {'':26} {head}            total")
    for chip in chips:
        raw = f"{chip}_raw"
        out = [s for s, (e, k, _l) in slots if K[k] == raw and E[e].tail in fabs]
        to_market = [s for s, (e, k, _l) in slots if K[k] == chip and E[e].tail in plants]
        m = np.array([K[d.k] == chip for d in inst.demands])
        made = [fo for fo, f in enumerate(inst.fabs) if K[N[f].fab.product] == raw]
        pair(f"{chip}: lots", A["lots"][:, :, made].sum(axis=2), p["plan_lots"][:, :, made].sum(axis=2), 1e6)
        pair(f"{chip}: raw out of fabs", A["sent"][:, :, out].sum(axis=2), p["plan_sent"][:, :, out].sum(axis=2), 1e6)
        pair(
            f"{chip}: out of plants",
            A["sent"][:, :, to_market].sum(axis=2),
            p["plan_sent"][:, :, to_market].sum(axis=2),
            1e6,
        )
        pair(f"{chip}: sold", A["served"][:, :, m].sum(axis=2), p["plan_served"][:, :, m].sum(axis=2), 1e6)

    # what the agent throws away and leaves at the end, by stage (the plan's disposal is not in its file)
    plain = [s for s, st in enumerate(inst.stock_slots) if st.node not in inst.chokepoint_ordinal]
    where = {s: i for i, s in enumerate(plain)}
    markets = {d.node for d in inst.demands}
    print("\n5. Thrown away by the agent (storage full) by quarter, and stock at the end (agent | plan), millions")
    for chip in chips:
        for kind, stage in ((f"{chip}_raw", "fab"), (f"{chip}_raw", "plant"), (chip, "plant"), (chip, "market")):
            nodes = fabs if stage == "fab" else plants if stage == "plant" else markets
            ss = [s for s, st in enumerate(inst.stock_slots) if K[st.k] == kind and st.node in nodes]
            if not ss:
                continue
            row = by_quarter(A["disposal"][:, :, ss].sum(axis=2))
            end_a = A["stock"][:, -1, ss].sum() / n_ep
            end_p = p["plan_stock"][:, -1, [where[s] for s in ss]].sum() / n_ep
            print(
                f"   {kind + ' at ' + stage:26} "
                + "  ".join(f"{x / 1e6:15.2f}" for x in row)
                + f"  {sum(row) / 1e6:8.2f}   end {end_a / 1e6:6.2f}|{end_p / 1e6:6.2f}"
            )
        waiting = [s for s, st in enumerate(inst.stock_slots) if K[st.k].startswith(chip) and s not in where]
        if waiting:
            print(f"   {chip + ' (any) at straits':26} {'':66}            end {A['stock'][:, -1, waiting].sum() / n_ep / 1e6:6.2f}|     -")
    wafers = {N[f].fab.input for f in inst.fabs}
    ss = [s for s, st in enumerate(inst.stock_slots) if st.k in wafers and st.node in fabs]
    row = by_quarter(A["disposal"][:, :, ss].sum(axis=2))
    print(f"   {'wafers at fabs':26} " + "  ".join(f"{x / 1e6:15.2f}" for x in row) + f"  {sum(row) / 1e6:8.2f}")


if __name__ == "__main__":
    fire.Fire(main)
