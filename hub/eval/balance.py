"""Chip mass balance of an agent, stage by stage: where the chips it makes stop before a market.

    uv run python hub/eval/balance.py path/to/agent_folder --episodes=12

Per episode (means over the episodes, millions of units), for leading-edge and mature chips apart:
fab output (lots that matured) -> shipped fab to plant -> shipped plant to market, per market: shipped, served,
left in the market's stock at the end, and the rest (disposed above storage, or still on the way).
Read-only.
"""

import fire
import numpy as np
from joblib import Parallel, delayed


def episode(agent: str, task: str, entropy: int, n: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_gym.dashboard import record_episode

    from sbf_starter import env_id
    from sbf_starter.agents import load

    inst, _ = task_generator(task)
    rec = record_episode(
        gym.make(env_id(task), entropy=entropy), load(agent), options={"episode": n}, naive_replications=REPLICATIONS
    )
    o = rec["obs"]
    return {
        "sent": np.asarray(o["last_week.clip.executed"][1:]).sum(axis=0),  # (slots,) the episode's total
        "asked": np.asarray(o["last_week.clip.requested"][1:]).sum(axis=0),
        "served": np.asarray(o["last_week.sinks.served"][1:]).sum(axis=0),
        "demand": np.asarray(o["last_week.sinks.demand"][1:]).sum(axis=0),
        "stock0": np.asarray(o["stock.qty"][0]),
        "stock1": np.asarray(o["stock.qty"][-1]),
        "stock_mean": np.asarray(o["stock.qty"]).mean(axis=0),
    }


def main(
    agent: str, task: str = "small", episodes: int = 12, entropy: int = 111, first: int = 0, n_jobs: int = 3
) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    from sbf_starter.agents import resolve

    inst, _ = task_generator(task)
    N, E, K = inst.nodes, inst.edges, [c.id for c in inst.commodities]
    eps = Parallel(n_jobs=n_jobs)(
        delayed(episode)(str(resolve(agent)), task, entropy, n) for n in range(first, first + episodes)
    )
    z = {k: np.stack([e[k] for e in eps]).mean(axis=0) for k in eps[0]}
    plain = [s for s in inst.stock_slots if s.node not in inst.chokepoint_ordinal]
    row = {(s.node, s.k): j for j, s in enumerate(plain)}
    dest = []
    for e, k, lane in inst.action_slots:
        dest.append(E[e].head if lane is None else E[inst.lanes[lane].edges[-1]].head)
    slots = list(enumerate(inst.action_slots))
    M = 1e6
    print(f"{agent}: {task}, episodes {first}..{first + episodes - 1} of root {entropy}; millions of units per episode")
    packaged = {}
    for o in inst.osats:
        packaged.update(N[o].osat.packages)  # raw -> packaged
    for raw, pk in sorted(set(packaged.items())):
        print(f"\n== {K[raw]} -> {K[pk]} ==")
        print("fab -> plants: asked, shipped; raw stock at the fab at reset -> mean -> end")
        for f in inst.fabs:
            if N[f].fab.product != raw:
                continue
            mine = [s for s, (e, k, lane) in slots if k == raw and E[e].tail == f]
            j = row[(f, raw)]
            by = {}
            for s in mine:
                by[N[dest[s]].id] = by.get(N[dest[s]].id, 0.0) + z["sent"][s]
            print(
                f"  {N[f].id:18s} asked {z['asked'][mine].sum() / M:7.2f}  shipped "
                f"{z['sent'][mine].sum() / M:6.2f}  "
                f"{', '.join(f'{a} {b / M:.2f}' for a, b in by.items()):44s} stock "
                f"{z['stock0'][j] / M:.2f} -> {z['stock_mean'][j] / M:.2f} -> {z['stock1'][j] / M:.2f}"
            )
        print(
            "plant -> markets: raw received, packaged chips shipped; chip stock at the plant at "
            "reset -> mean -> end; raw stock mean"
        )
        for o in inst.osats:
            if raw not in N[o].osat.packages:
                continue
            into = [s for s, (e, k, lane) in slots if k == raw and dest[s] == o]
            out = [s for s, (e, k, lane) in slots if k == pk and E[e].tail == o]
            j, jr = row[(o, pk)], row[(o, raw)]
            by = {}
            for s in out:
                by[N[dest[s]].id] = by.get(N[dest[s]].id, 0.0) + z["sent"][s]
            print(
                f"  {N[o].id:10s} received {z['sent'][into].sum() / M:6.2f}  shipped "
                f"{z['sent'][out].sum() / M:6.2f} "
                f"({', '.join(f'{a} {b / M:.2f}' for a, b in by.items())})  stock "
                f"{z['stock0'][j] / M:.2f} -> {z['stock_mean'][j] / M:.2f} -> "
                f"{z['stock1'][j] / M:.2f}  raw {z['stock_mean'][jr] / M:.2f}"
            )
        print(
            "markets: demand, shipped to it, served, stock at reset -> end, shipped but neither "
            "served nor in stock (disposed or on the way)"
        )
        tot = np.zeros(4)
        for do, d in enumerate(inst.demands):
            if d.k != pk:
                continue
            into = [s for s, (e, k, lane) in slots if k == pk and dest[s] == d.node]
            j = row[(d.node, pk)]
            shipped, served = z["sent"][into].sum(), z["served"][do]
            lost = shipped + z["stock0"][j] - served - z["stock1"][j]
            tot += [z["demand"][do], shipped, served, lost]
            print(
                f"  {N[d.node].id:8s} demand {z['demand'][do] / M:6.2f}  shipped {shipped / M:6.2f}  "
                f"served {served / M:6.2f} ({100 * served / max(z['demand'][do], 1):3.0f}%)  stock "
                f"{z['stock0'][j] / M:.2f} -> {z['stock1'][j] / M:.2f}  unaccounted {lost / M:6.2f}"
            )
        print(
            f"  {'all':8s} demand {tot[0] / M:6.2f}  shipped {tot[1] / M:6.2f}  served "
            f"{tot[2] / M:6.2f} ({100 * tot[2] / tot[0]:3.0f}%)  unaccounted {tot[3] / M:6.2f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
