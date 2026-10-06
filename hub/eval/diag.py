"""Where an agent's plan differs physically from the clairvoyant plan: costs, power shed, lots started, demand served.

    uv run python hub/eval/diag.py path/to/agent_folder --episodes=12
    uv run python hub/eval/diag.py path/to/agent_folder --episodes=12 --nobest   # faster: no LP solve

Each episode is played once by the agent (under gymnasium, policy seed 0) and, unless ``--nobest``, solved once as the
clairvoyant plan's linear program. Read-only: nothing is written. About 2 s per episode for a rule agent plus 5 s for
the clairvoyant plan, per worker.

The clairvoyant plan is a relaxation: it may power a fab while its grid sheds base load, which the simulator never
does (base load is served first). The column "lots the plan starts while shedding" says how much of its output that is.
"""

import fire
import numpy as np
from joblib import Parallel, delayed


def episode(agent: str, task: str, entropy: int, n: int, best: bool) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
    from shockbench_flow.dynamics.state import COST_COMPONENTS
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_gym.dashboard import record_episode

    from sbf_starter import env_id
    from sbf_starter.agents import load

    inst, params = task_generator(task)
    T, F, G, D = inst.T, len(inst.fabs), len(inst.grids), len(inst.demands)
    env = gym.make(env_id(task), entropy=entropy)
    rec = record_episode(env, load(agent), options={"episode": n}, naive_replications=REPLICATIONS)
    obs = rec["obs"]
    out = {
        "costs": np.asarray(rec["costs"]).sum(axis=0),  # (8,) USD
        "naive_costs": np.asarray(rec["naive"]["costs"]).sum(axis=0),
        "J": rec["meta"]["J_cents"] / 100,
        "naive_J": int(rec["meta"]["naive_J_cents"]) / 100,
        "shed": np.asarray(obs["last_week.shed.qty"][1:]),  # (T, G)
        "served": np.asarray(obs["last_week.sinks.served"][1:]),  # (T, D)
        "demand": np.asarray(obs["last_week.sinks.demand"][1:]),
        "asked": np.asarray(obs["last_week.clip.requested"][1:]),  # (T, slots)
        "sent": np.asarray(obs["last_week.clip.executed"][1:]),
    }
    plain = [i for i, s in enumerate(inst.stock_slots) if s.node not in inst.chokepoint_ordinal]
    row = {(inst.stock_slots[i].node, inst.stock_slots[i].k): j for j, i in enumerate(plain)}
    stock = np.asarray(obs["stock.qty"])  # (T + 1, plain slots): the state at the start of weeks 1..T+1
    below = np.zeros((T, G))
    for go, g in enumerate(inst.grids):
        grid = inst.nodes[g].grid
        k = grid.rationed
        if k is not None and (g, k) in row:
            below[:, go] = stock[:T, row[(g, k)]] < inst.params.psi * grid.ibar[k]
    out["rationed"] = below  # (T, G) 1 where the rationed fuel's stock is under the rationing threshold
    tau = {f: inst.nodes[f].fab.tau for f in inst.fabs}
    starts = np.zeros((T, F))
    for t in range(1, T + 1):  # lots started in week t: the WIP entries of the next observation that mature tau later
        live = np.asarray(obs["wip.qty.observed"][t]) == 1
        for node, q, ow in zip(obs["wip.node"][t][live], obs["wip.qty"][t][live], obs["wip.out_week"][t][live]):
            node = int(node)
            if node in inst.fab_ordinal and int(ow) == t + tau[node]:
                starts[t - 1, inst.fab_ordinal[node]] += q
    out["lots"] = starts
    if best:
        from shockbench_flow.disruption.sampler import sample_omega
        from shockbench_flow.marks import compute_marks
        from shockbench_flow.oracle.lp import build_lp, lp_costs, lp_flow_key, solve_oracle

        omega = sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train")
        model = build_lp(inst, compute_marks(inst, omega))
        plan = solve_oracle(model)
        x = plan.x

        def var(*key):
            keys = [(key[0], t, *key[1:]) for t in range(1, T + 1)]
            return np.array([x[model.index[q]] if q in model.index else 0.0 for q in keys])

        weekly, _credit = lp_costs(model, x)
        out["best_costs"] = np.array([[w.as_dict()[c] for c in COST_COMPONENTS] for w in weekly]).sum(axis=0)
        out["best_J"] = plan.J_cents / 100
        out["best_lots"] = np.stack([var("p", fo) for fo in range(F)], axis=1)
        out["best_shed"] = np.stack([var("ysh", go) for go in range(G)], axis=1)
        out["best_served"] = np.stack([var("D", do) for do in range(D)], axis=1)
        slots = rec["static"]["action_slots"]
        sent = np.zeros((T, len(slots["edge"])))
        for s, (e, k, lane) in enumerate(zip(slots["edge"], slots["k"], slots["lane"])):
            sent[:, s] = var("x", *lp_flow_key(inst, e, k, lane))
        out["best_sent"] = sent
    return out


def main(
    agent: str,
    task: str = "small",
    episodes: int = 12,
    entropy: int = 111,
    first: int = 0,
    best: bool = True,
    n_jobs: int = 3,
) -> None:
    """Print the agent's costs and physical flows beside the naive rule's and the clairvoyant plan's.

    Args:
        agent: an agent folder (or a name of agents/ in the main checkout).
        task: tiny, small or full.
        episodes: how many episodes, starting at ``first``.
        entropy: the scenarios' root (111, the tuning root).
        first: the first episode index.
        best: also solve the clairvoyant plan of each episode (--nobest skips it).
        n_jobs: workers.

    """
    from shockbench_flow.dynamics.state import COST_COMPONENTS
    from shockbench_flow.hosting.tasks import task_generator

    from sbf_starter.agents import resolve

    path = str(resolve(agent))
    eps = Parallel(n_jobs=n_jobs)(
        delayed(episode)(path, task, entropy, n, best) for n in range(first, first + episodes)
    )
    z = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    inst, _ = task_generator(task)
    N, K = inst.nodes, [c.id for c in inst.commodities]
    T = inst.T
    bn = 1e9
    jn, ja = z["naive_J"].mean(), z["J"].mean()
    print(f"{agent}: {task}, episodes {first}..{first + episodes - 1} of root {entropy}")
    if best:
        jb = z["best_J"].mean()
        print(
            f"unweighted score on these episodes: {(jn - ja) / (jn - jb):.4f}   (not the board's "
            "weighting; use compare.py for decisions)"
        )
    print("\n1. Cost per episode, USD bn (mean)")
    print(f"{'component':14s} {'naive':>9} {'agent':>9}" + (f" {'clairv.':>9} {'agent - clairv.':>16}" if best else ""))
    for c, name in enumerate(COST_COMPONENTS):
        row = f"{name:14s} {z['naive_costs'][:, c].mean() / bn:9.1f} {z['costs'][:, c].mean() / bn:9.1f}"
        if best:
            row += (
                f" {z['best_costs'][:, c].mean() / bn:9.1f} "
                f"{(z['costs'][:, c].mean() - z['best_costs'][:, c].mean()) / bn:16.1f}"
            )
        print(row)
    print(
        f"{'total J':14s} {jn / bn:9.1f} {ja / bn:9.1f}" + (f" {jb / bn:9.1f} {(ja - jb) / bn:16.1f}" if best else "")
    )

    print(
        "\n2. Grids: power shed, GWh per week (share of weeks with any shed %); weeks with the "
        "rationed fuel under its threshold %"
    )
    for go, g in enumerate(inst.grids):
        sh = z["shed"][:, :, go]
        row = (
            f"{N[g].id:10s} agent {sh.mean():8.1f} ({100 * (sh > 1e-6).mean():3.0f}%)  rationed "
            f"{100 * z['rationed'][:, :, go].mean():3.0f}%"
        )
        if best:
            b = z["best_shed"][:, :, go]
            row += f"   clairv. {b.mean():8.1f} ({100 * (b > 1e-6).mean():3.0f}%)"
        print(row)

    print("\n3. Fabs: lots started per week as % of nominal capacity (share of weeks with no start %)")
    w = slice(0, max(T - 8, 1))  # lots of the last weeks never reach a market
    for fo, f in enumerate(inst.fabs):
        cap = N[f].fab.cap0
        a = z["lots"][:, w, fo]
        row = f"{N[f].id:18s} agent {100 * a.mean() / cap:5.1f}% ({100 * (a < 0.01 * cap).mean():3.0f}%)"
        if best:
            b = z["best_lots"][:, w, fo]
            dry = z["best_shed"][:, w, inst.grid_ordinal[N[f].fab.grid]] > 1e-6
            row += (
                f"   clairv. {100 * b.mean() / cap:5.1f}% ({100 * (b < 0.01 * cap).mean():3.0f}%), of "
                f"which started while shedding {100 * b[dry].sum() / max(b.sum(), 1e-9):3.0f}%"
            )
        print(row)
    print(
        f"all fabs, lots per week: agent {z['lots'][:, w].sum(axis=2).mean():,.0f}"
        + (f", clairv. {z['best_lots'][:, w].sum(axis=2).mean():,.0f}" if best else "")
    )

    print("\n4. Markets: demand served per week, thousands (share of demand %)")
    for do, d in enumerate(inst.demands):
        dem = z["demand"][:, :, do].mean()
        if dem <= 0:
            continue
        a = z["served"][:, :, do].mean()
        row = f"{N[d.node].id:8s} {K[d.k]:9s} demand {dem / 1e3:7.1f}  agent {a / 1e3:7.1f} ({100 * a / dem:3.0f}%)"
        if best:
            b = z["best_served"][:, :, do].mean()
            row += f"  clairv. {b / 1e3:7.1f} ({100 * b / dem:3.0f}%)"
        print(row)

    print("\n5. Dispatched per week by commodity (action slots only, units of the commodity)")
    slots_k = np.array([k for _e, k, _lane in inst.action_slots])
    print(f"{'commodity':13s} {'asked':>12} {'executed':>12}" + (f" {'clairv.':>12}" if best else ""))
    for k, name in enumerate(K):
        m = slots_k == k
        if not m.any():
            continue
        row = (
            f"{name:13s} {z['asked'][:, :, m].sum(axis=2).mean():12,.0f} {z['sent'][:, :, m].sum(axis=2).mean():12,.0f}"
        )
        if best:
            row += f" {z['best_sent'][:, :, m].sum(axis=2).mean():12,.0f}"
        print(row)
    if best:
        print("\n6. Slots where the agent ships most differently from the clairvoyant plan (mean units per week)")
        a, b = z["sent"].mean(axis=(0, 1)), z["best_sent"].mean(axis=(0, 1))
        for s in np.argsort(-np.abs(a - b) * np.array([inst.commodities[k].v for k in slots_k]))[:14]:
            e, k, lane = inst.action_slots[s]
            where = inst.edges[e].id + ("" if lane is None else f" (lane {inst.lanes[lane].id})")
            print(f"  slot {s:3d} {K[k]:12s} agent {a[s]:10,.0f}  clairv. {b[s]:10,.0f}   {where}")


if __name__ == "__main__":
    fire.Fire(main)
