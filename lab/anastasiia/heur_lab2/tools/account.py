"""Where an agent loses: every lot and every unit of fuel accounted for from the simulator's own weekly records.

    uv run python lab/anastasiia/heur_lab2/tools/account.py agents/anastasiia_rules_fuelchip --episodes=16
    uv run python lab/anastasiia/heur_lab2/tools/account.py <agent> --entropy=444 --episodes=40 --plan
    uv run python lab/anastasiia/heur_lab2/tools/account.py <agent> --task=full --episodes=8

Read-only. About 2 s per episode of Small and 4 s of Full, per worker. ``--plan`` compares with the plan that knows the
future AND obeys the simulator's "base load first" rule: a mixed-integer program solved once for the first 40 episodes
of root 444 by ``lab/anastasiia/stats_lab/plan_stats.py``, which keeps its weekly arrays in
``outputs/plan_stats/<date_time>/episodes.npz`` (local, not in git; ``--plan_file`` names another file). Its own cost
is the realistic ceiling (score about 0.95).

Sections:
1. costs by component;
2. fabs: lots started, raw chips shipped, raw chips disposed of at the fab (storage full), wafers disposed of;
3. why lots were not started, as a share of nominal capacity: no wafers at the fab, no power (grid short of its base
   load, or the sliver above the base load only partly available);
4. grids: weeks without shed, how much of the fabs' need was powered, each fuel at its cap, the free (no-fuel) segment's
   output lost because the grid offered more than the load took, fuel disposed of and left at the end;
5. chips and wafers disposed of, by stock slot;
6. (--plan) the gap to the plan: chips by market, shed by grid, lots by fab.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / "outputs/plan_stats/20261006_040041/episodes.npz"
COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def play(agent: str, task: str, entropy: int, n: int) -> dict:
    """One episode played by the agent; the simulator's records as arrays."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    inst = u.instance
    ag = load(agent)(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    done = False
    while not done:
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done = term or trunc
    ep = u.core._ep  # the trusted side's episode: the weekly records and the marks
    recs, marks = ep.traj.records, ep.marks
    T, G = inst.T, len(inst.grids)
    seg = np.zeros((T, G, len(inst.commodities) + 1))  # last column: the no-fuel segment
    sent = np.zeros((T, len(inst.action_slots)))
    asked = np.zeros((T, len(inst.action_slots)))
    for t, r in enumerate(recs):
        for (gi, k), q in r.segment.items():
            seg[t, gi, -1 if k is None else k] = q
        for s, q in r.executed.items():
            sent[t, s] = q
        for s, q in r.requested.items():
            asked[t, s] = q
    return {
        "J": ep.traj.J_cents / 100,
        "cost": np.array([[getattr(r.costs, c) for c in COMPONENTS] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]),  # (T, F)
        "scrap": np.array([r.scrapped for r in recs]),
        "energy": np.array([r.energy for r in recs]),  # (T, F) GWh given to each fab
        "disposal": np.array([r.disposal for r in recs]),  # (T, stock slots)
        "stock": np.array([r.stock for r in recs]),  # (T, stock slots) end of week
        "shed": np.array([r.shed for r in recs]),  # (T, G)
        "served": np.array([r.served for r in recs]),  # (T, D)
        "demand": np.array([r.demand for r in recs]),
        "segment": seg,  # (T, G, K + 1) GWh generated per fuel segment
        "sent": sent,
        "asked": asked,
        "alpha": np.asarray(marks.alpha_bar)[:T],  # (T, F) capacity factors of the fabs
        "R": np.asarray(marks.R)[:T],
        "G_bar": np.asarray(marks.G_bar)[:T],  # (T, G) deliverable output this week
        "y_bar": np.asarray(marks.y_bar)[:T],  # (T, G) base load this week
    }


def table(header: list[str], rows: list[list]) -> None:
    """Print rows under a header, columns as wide as their widest cell; numbers right-aligned."""

    def cell(v) -> str:
        return v if isinstance(v, str) else f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"

    body = [[cell(v) for v in row] for row in rows]
    width = [max(len(x) for x in col) for col in zip(header, *body)]
    for row, raw in zip([header, *body], [header, *rows]):
        print("   " + "  ".join(x.ljust(w) if isinstance(v, str) else x.rjust(w) for x, w, v in zip(row, width, raw)))


def main(
    agent: str,
    task: str = "small",
    episodes: int = 16,
    entropy: int = 111,
    first: int = 0,
    n_jobs: int = 2,
    plan: bool = False,
    plan_file: str = str(PLAN),
    save: str = "",
) -> None:
    """Print the accounting of ``agent`` on ``episodes`` episodes (see the module docstring).

    Args:
        agent: an agent folder, or a name of agents/.
        task: small or full.
        episodes: how many episodes, starting at ``first``.
        entropy: the scenarios' root (111 tuning; 444 with --plan).
        first: the first episode index.
        n_jobs: workers.
        plan: also print the gap to the best plan (root 444, episodes 0..39 of Small only).
        plan_file: the plan's weekly arrays (``episodes.npz`` of plan_stats.py).
        save: a path to write the arrays to (.npz), optional.

    """
    from shockbench_flow.hosting.tasks import task_generator

    from sbf_starter.agents import resolve

    agent = str(resolve(agent))
    inst, _ = task_generator(task)
    N, E, K = inst.nodes, inst.edges, [c.id for c in inst.commodities]
    eps = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n) for n in range(first, first + episodes))
    A = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    if save:
        np.savez_compressed(save, **A)
    T, M, slot = inst.T, 1e6, inst.slot_index
    pi, pk = {}, {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
    for o in inst.osats:
        pk.update(N[o].osat.packages)
    useful = max(T - 14, 1)  # a lot started later is not sold before the end
    w = slice(0, useful)
    grid_fuels = {k for g in inst.grids for k in N[g].grid.fuels}

    print(f"{Path(agent).name}: {task}, root {entropy}, episodes {first}..{first + episodes - 1}")
    costs = ", ".join(f"{c} {v:.1f}" for c, v in zip(COMPONENTS, A["cost"].sum(1).mean(0) / 1e9))
    print(f"\n1. Cost per episode, USD bn: {costs}; total J {A['J'].mean() / 1e9:.1f}")

    print("\n2. Fabs, millions of units per episode (energy in GWh)")
    rows = []
    for fo, f in enumerate(inst.fabs):
        fab = N[f].fab
        wip0 = sum(x.qty for x in inst.initial_state.fab_wip if x.node == f)
        out = [s for s, (e, k, lane) in enumerate(inst.action_slots) if E[e].tail == f and k == fab.product]
        disp = A["disposal"][:, :, slot[(f, fab.product)]].sum(1).mean()
        rows.append(
            [
                N[f].id,
                fab.cap0 * T / M,
                A["lots"][:, :, fo].sum(1).mean() / M,
                wip0 / M,
                A["sent"][:, :, out].sum((1, 2)).mean() / M,
                disp / M,
                A["disposal"][:, :, slot[(f, fab.input)]].sum(1).mean() / M,
                float(A["energy"][:, :, fo].sum(1).mean()),
                disp * fab.e,
            ]
        )
    header = ["fab", "capacity x T", "lots", "in process at reset", "raw shipped", "raw disposed"]
    table([*header, "wafers disposed", "energy", "energy of the disposed"], rows)

    print(f"\n3. Why lots were not started, weeks 1..{useful}, share of nominal capacity")
    rows = []
    for fo, f in enumerate(inst.fabs):
        fab = N[f].fab
        go = inst.grid_ordinal[fab.grid]
        cap = A["alpha"][:, w, fo] * A["R"][:, w, fo] * fab.cap0
        lots = A["lots"][:, w, fo]
        phat = np.minimum(cap, A["stock"][:, w, slot[(f, fab.input)]] + lots)  # wafers on hand before the starts
        shedding = A["shed"][:, w, go] > 1e-6
        nopow, c0 = phat - lots, fab.cap0
        rows.append(
            [
                N[f].id,
                lots.mean() / c0,
                (cap - phat).mean() / c0,
                (nopow * shedding).mean() / c0,
                (nopow * ~shedding).mean() / c0,
                lots[~shedding].sum() / max(cap[~shedding].sum(), 1),
            ]
        )
    header = ["fab", "started", "no wafers", "no power: grid short of base load", "no power: sliver partly there"]
    table([*header, "lots / capacity in weeks without shed"], rows)

    print(f"\n4. Grids, weeks 1..{useful} (GWh per week; fuels: burned over the segment's cap)")
    rows, names = [], sorted(grid_fuels)
    for go, g in enumerate(inst.grids):
        grid = N[g].grid
        need = sum(N[inst.fabs[fo]].fab.cap0 * N[inst.fabs[fo]].fab.e for fo in inst.grid_fabs[go])
        e_fab = sum((A["energy"][:, w, fo] for fo in inst.grid_fabs[go]), np.zeros_like(A["shed"][:, w, go]))
        free = grid.shares.get(None, 0.0) * A["G_bar"][:, w, go]
        burned = []
        for k in names:
            if k in grid.fuels:
                burned.append(float(A["segment"][:, w, go, k].mean() / (grid.shares[k] * A["G_bar"][:, w, go]).mean()))
            else:
                burned.append("-")
        rows.append(
            [
                N[g].id,
                100 * (A["shed"][:, w, go] <= 1e-6).mean(),
                float(A["shed"][:, w, go].mean()),
                e_fab.mean() / max(need, 1e-9),
                float((free - A["segment"][:, w, go, -1]).mean()),
                *burned,
            ]
        )
    header = ["grid", "weeks without shed %", "shed", "fabs' energy / need", "free segment lost"]
    table([*header, *[K[k] for k in names]], rows)
    print("   fuel disposed of (storage full) and left at the end, units per episode:")
    for s, st in enumerate(inst.stock_slots):
        if st.k in grid_fuels and N[st.node].type in ("terminal", "grid"):
            d, left = A["disposal"][:, :, s].sum(1).mean(), A["stock"][:, -1, s].mean()
            if d > 1 or left > 1:
                print(f"     {N[st.node].id:12s} {K[st.k]:8s} disposed {d:9,.0f}   left at the end {left:9,.0f}")
    q = [s for s, st in enumerate(inst.stock_slots) if st.node in inst.chokepoint_ordinal and st.k in grid_fuels]
    print(f"     fuel waiting at straits at the end: {A['stock'][:, -1, q].sum(1).mean():,.0f}")

    print("\n5. Chips and wafers disposed of, by stock slot (units per episode; sales value if they were sellable)")
    d = A["disposal"].sum(1).mean(0)
    val = np.array([pi.get(pk.get(st.k, st.k), 0.0) for st in inst.stock_slots])
    for s in np.argsort(-d * val)[:12]:
        st = inst.stock_slots[s]
        if d[s] > 1 and val[s] > 0:
            where = f"{N[st.node].id:20s} {K[st.k]:13s}"
            print(f"     {where} {d[s]:12,.0f}   {d[s] * val[s] / 1e9:7.1f} bn   (storage {st.storage:,.0f})")
    served = []
    for do, dd in enumerate(inst.demands):
        if A["demand"][:, :, do].sum() > 0:
            share = 100 * A["served"][:, :, do].sum() / A["demand"][:, :, do].sum()
            served.append(f"{N[dd.node].id} {K[dd.k]} {A['served'][:, :, do].mean() / 1e3:.1f} ({share:.0f}%)")
    print("\n   Demand served per week, thousands: " + "; ".join(served))

    if not plan:
        return
    if task != "small" or entropy != 444 or first != 0 or episodes > 40:
        sys.exit("--plan needs --task=small --entropy=444 --first=0 and at most 40 episodes")
    P = np.load(plan_file)
    n = episodes
    voll = np.array([N[g].grid.voll for g in inst.grids])
    pis = np.array([dd.pi for dd in inst.demands])
    dj = (A["J"] - P["J_plan"][:n]) / 1e9
    short = ((P["plan_served"][:n] - A["served"]) * pis).sum(1) / 1e9  # (episodes, D)
    shed = ((A["shed"] - P["plan_shed"][:n]) * voll).sum(1) / 1e9  # (episodes, G)
    other = (dj - short.sum(1) - shed.sum(1)).mean()
    replay = (P["J_replay_plan"][:n] - P["J_plan"][:n]).mean() / 1e9
    print(
        f"\n6. Gap to the best plan's own cost, USD bn per episode: total {dj.mean():.1f} = chips "
        f"{short.sum(1).mean():.1f} + shed {shed.sum(1).mean():.1f} + other {other:.1f}"
    )
    print(f"   (the plan replayed blindly in the simulator is itself {replay:.1f} above its own cost)")
    by_market = [f"{N[dd.node].id} {K[dd.k]} {short[:, do].mean():.1f}" for do, dd in enumerate(inst.demands)]
    print("   chips by market: " + "; ".join(x for x, dd in zip(by_market, inst.demands) if dd.pi > 0))
    print("   shed by grid: " + "; ".join(f"{N[g].id} {shed[:, go].mean():.1f}" for go, g in enumerate(inst.grids)))
    rows = []
    for fo, f in enumerate(inst.fabs):
        fab = N[f].fab
        a, p = A["lots"][:, w, fo].sum(1), P["plan_lots"][:n, w, fo].sum(1)
        more, less = (p - a).clip(0).mean(), (a - p).clip(0).mean()
        rows.append([N[f].id, a.mean() / M, p.mean() / M, more / M, more * pi[pk[fab.product]] / 1e9, less / M])
        rows[-1].append(less * fab.e)
    header = ["fab", "lots, agent", "lots, plan", "where the plan starts more", "worth, bn"]
    table([*header, "where the plan starts fewer", "GWh those cost"], rows)
    e_a, e_p = A["energy"].sum((1, 2)).mean(), P["plan_fab_energy"][:n].sum((1, 2)).mean()
    lost = 0.0
    for go, g in enumerate(inst.grids):
        free = N[g].grid.shares.get(None, 0.0) * A["G_bar"][:, :, go]
        lost += (free - A["segment"][:, :, go, -1]).sum(1).mean()
    print(f"   energy to fabs, GWh per episode: agent {e_a:,.0f}, plan {e_p:,.0f}")
    print(f"   free segment lost by the agent: {lost:,.0f} GWh per episode (the plan: 0); 1,000 GWh is 4.1 bn")
    order = np.argsort(-dj)
    print("   episodes by gap (bn): " + ", ".join(f"{i}:{dj[i]:.0f}" for i in order[:12]) + " ...")


if __name__ == "__main__":
    fire.Fire(main)
