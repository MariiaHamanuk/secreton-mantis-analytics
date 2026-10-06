"""An agent beside the plan with "base load first", quantity by quantity, on Small or Full.

    uv run python lab/anastasiia/heur_lab3/tools/plan_gap.py outputs/heur3/full_ceiling/v2_full_444.npz \
        outputs/heur3/full_ceiling/run8/episodes.npz --task=full --skip=3
    uv run python lab/anastasiia/heur_lab3/tools/plan_gap.py outputs/heur3/data/v2_small_444.npz \
        outputs/plan_stats/20261006_040041/episodes.npz --task=small

The first file is the agent's weekly arrays (``lab/anastasiia/heur_lab2/tools/account.py --save``), the second the
plan's (``lab/anastasiia/stats_lab/plan_stats.py``), both on episodes 0.. of the same root; ``--skip`` names the
episodes the plan file leaves out. No episode is played. All money is USD bn per episode, the agent minus the plan:

1. unserved demand by chip and shed by grid, by quarter of the episode;
2. lots by fab, and the chips of each kind made, shipped out of the fabs and sold;
3. fuel moved into each grid while it can still be burned (every week but the last).

The plan's own cost is not reached by replaying its orders (see ``hub/FINDINGS.md``): its totals say where the room
is, its timing of lots and of the burn is not something the simulator would do.
"""

import fire
import numpy as np


def main(agent: str, plan: str, task: str = "full", skip: str = "") -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    N, E, K, T = inst.nodes, inst.edges, [c.id for c in inst.commodities], inst.T
    a, p = np.load(agent), np.load(plan)
    left_out = {int(x) for x in str(skip).split(",") if str(x).strip() != ""}
    keep = [n for n in range(len(a["J"])) if n not in left_out][: len(p["J_plan"])]
    A = {k: a[k][keep] for k in a.files}
    n_ep, bn = len(keep), 1e9
    print(f"{task}: {n_ep} episodes; the agent minus the plan {(A['J'] - p['J_plan']).mean() / bn:.0f} bn per episode")

    quarters = [(q * T // 4, (q + 1) * T // 4) for q in range(4)]
    head = "  ".join(f"wk {lo + 1}-{hi}".rjust(10) for lo, hi in quarters)
    pi = np.array([d.pi for d in inst.demands])
    short = (p["plan_served"] - A["served"]) * pi / bn  # (episode, week, demand): the agent's extra unserved demand
    print(f"\n1. Unserved demand and shed, the agent minus the plan, bn per episode\n   {'':22} {head}       total")
    for chip in sorted({K[d.k] for d in inst.demands}):
        m = np.array([K[d.k] == chip for d in inst.demands])
        row = [short[:, lo:hi][:, :, m].sum() / n_ep for lo, hi in quarters]
        print(f"   {chip:22} " + "  ".join(f"{x:10.1f}" for x in row) + f"  {sum(row):10.1f}")
    voll = np.array([N[g].grid.voll for g in inst.grids])
    shed = (A["shed"] - p["plan_shed"]) * voll / bn
    for gi, g in enumerate(inst.grids):
        row = [shed[:, lo:hi, gi].sum() / n_ep for lo, hi in quarters]
        print(f"   shed {N[g].id:17} " + "  ".join(f"{x:10.1f}" for x in row) + f"  {sum(row):10.1f}")
    row = [shed[:, lo:hi].sum() / n_ep for lo, hi in quarters]
    print(f"   {'shed, all grids':22} " + "  ".join(f"{x:10.1f}" for x in row) + f"  {sum(row):10.1f}")

    print("\n2. Fabs, millions of units per episode: lots (agent | plan), energy GWh (agent | plan)")
    for fo, f in enumerate(inst.fabs):
        print(
            f"   {N[f].id:20} lots {A['lots'][:, :, fo].sum() / n_ep / 1e6:6.2f} | "
            f"{p['plan_lots'][:, :, fo].sum() / n_ep / 1e6:6.2f}   energy {A['energy'][:, :, fo].sum() / n_ep:8.0f} | "
            f"{p['plan_fab_energy'][:, :, fo].sum() / n_ep:8.0f}"
        )
    fabs, plants = set(inst.fabs), {n for n in range(len(N)) if getattr(N[n], "osat", None) is not None}

    def head_of(e, lane):
        return E[e].head if lane is None else E[inst.lanes[lane].edges[-1]].head

    print("   chips, millions per episode (agent | plan):")
    for chip in sorted({K[d.k] for d in inst.demands}):
        raw = f"{chip}_raw"
        out = [s for s, (e, k, _l) in enumerate(inst.action_slots) if K[k] == raw and E[e].tail in fabs]
        to_market = [s for s, (e, k, _l) in enumerate(inst.action_slots) if K[k] == chip and E[e].tail in plants]
        m = np.array([K[d.k] == chip for d in inst.demands])
        print(
            f"   {chip:9} raw shipped out of fabs {A['sent'][:, :, out].sum() / n_ep / 1e6:6.2f} | "
            f"{p['plan_sent'][:, :, out].sum() / n_ep / 1e6:6.2f}   packaged shipped out of plants "
            f"{A['sent'][:, :, to_market].sum() / n_ep / 1e6:6.2f} | "
            f"{p['plan_sent'][:, :, to_market].sum() / n_ep / 1e6:6.2f}   sold "
            f"{A['served'][:, :, m].sum() / n_ep / 1e6:6.2f} | {p['plan_served'][:, :, m].sum() / n_ep / 1e6:6.2f}"
        )

    print("\n3. Fuel moved into each grid in weeks 1..T-1, thousands of units per episode (agent | plan | difference)")
    for g in inst.grids:
        for fuel in N[g].grid.fuels:
            into = [s for s, (e, k, lane) in enumerate(inst.action_slots) if k == fuel and head_of(e, lane) == g]
            if not into:
                continue
            x, y = A["sent"][:, : T - 1, into].sum() / n_ep / 1e3, p["plan_sent"][:, : T - 1, into].sum() / n_ep / 1e3
            print(f"   {N[g].id:10} {K[fuel]:8} {x:8.1f} | {y:8.1f} | {y - x:+7.1f}")


if __name__ == "__main__":
    fire.Fire(main)
