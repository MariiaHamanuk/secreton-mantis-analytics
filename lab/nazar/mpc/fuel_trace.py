"""Fuel on its way to one grid, week by week (4-week bins): requests, executed flow, the straits, the queues, the stocks.

    uv run python lab/nazar/mpc/fuel_trace.py lab/nazar/agents/nazar_rules_lpraw --entropy=555 --episode=22 --grid=grid_eu

For each fuel (lng, crude): what was asked and executed on the slots that end at the grid or its terminal, split by
the strait the slot's first edge enters; the openness (graph_now.open) and tanker/bulk throughput (kappa) of each
strait; the cargo queued at each strait; the terminal's and the grid's stock; what the grid burned.
Read-only. A full episode is played once.
"""

from pathlib import Path

import fire
import numpy as np


def main(
    agent: str, grid: str = "grid_eu", task: str = "small", entropy: int = 555, episode: int = 22, bins: int = 4
) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": episode})
    u = env.unwrapped
    inst = u.instance
    cfg = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    layout = cfg["layout"]
    ag = load(agent)(cfg)
    names = [c.id for c in inst.commodities]
    node = lambda n: str(inst.nodes[int(n)].id)  # noqa: E731
    chk_nodes = [int(c) for c in layout["chokepoints"]]
    chk_names = [node(c) for c in chk_nodes]
    stock_slots = [(int(n), int(k)) for n, k in layout["stock_slots"]]
    lot_keys = layout.get("lot_keys")
    gnode = next(int(g) for g in inst.grids if node(g) == grid)
    # the terminals that feed this grid: nodes with an edge into it carrying fuel
    feeders = {
        int(inst.edges[e].tail)
        for e in range(len(inst.edges))
        if int(inst.edges[e].head) == gnode and set(inst.edges[e].K) & {0, 1}
    }
    targets = {gnode} | feeders
    obs_log = []
    done = False
    while not done:
        obs_log.append(
            {
                "open": np.array(obs["graph_now.open"], dtype=float),
                "kappa": np.array(obs["graph_now.kappa.tb"], dtype=float),
                "stock": np.array(obs["stock.qty"], dtype=float),
                "queue": np.array(obs["queue_lots.qty"], dtype=float).sum(1) if lot_keys is not None else None,
            }
        )
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done = term or trunc
    recs = u.core._ep.traj.records
    T = len(recs)
    slots = []
    for i, (e, k, lane) in enumerate(inst.action_slots):
        if k not in (0, 1):
            continue
        dest = int(inst.lane_destination(lane)) if lane is not None else int(inst.edges[e].head)
        if dest in targets:
            via = next(
                (
                    node(inst.edges[x].head)
                    for x in ([e] if lane is None else [e])
                    if node(inst.edges[x].head).startswith("chk")
                ),
                "direct",
            )
            slots.append((i, int(k), node(dest), via, str(inst.edges[e].id)))
    print(
        f"{Path(agent).name}: root {entropy} episode {episode}, {grid} (fed through {[node(f) for f in feeders]}); {len(slots)} fuel slots end there"
    )
    for k in (0, 1):
        ks = [s for s in slots if s[1] == k]
        if not ks:
            continue
        print(
            f"\n=== {names[k]} into {grid}: per {bins}-week bin: asked / executed by route (first strait of the slot)"
        )
        vias = sorted({s[3] for s in ks})
        head = "  weeks    " + "".join(f"{v[:16]:>22}" for v in vias)
        print(head)
        for lo in range(0, T, bins):
            w = slice(lo, min(lo + bins, T))
            cells = []
            for v in vias:
                ix = [s[0] for s in ks if s[3] == v]
                a = sum(sum(r.requested.get(i, 0.0) for i in ix) for r in recs[w])
                e = sum(sum(r.executed.get(i, 0.0) for i in ix) for r in recs[w])
                cells.append(f"{a:10,.0f}/{e:<10,.0f}")
            print(f"  {lo + 1:>2}-{min(lo + bins, T):<5}  " + "".join(f"{c:>22}" for c in cells))
    print("\n=== straits: mean openness (graph_now.open) and tanker throughput factor (kappa tb) per bin")
    print("  weeks    " + "".join(f"{n[4:12]:>14}" for n in chk_names))
    for lo in range(0, T, bins):
        w = slice(lo, min(lo + bins, T))
        o = np.mean([x["open"] for x in obs_log[w]], axis=0)
        kp = np.mean([x["kappa"] for x in obs_log[w]], axis=0)
        print(f"  {lo + 1:>2}-{min(lo + bins, T):<5}  " + "".join(f"{f'{a:.2f}/{b:.2f}':>14}" for a, b in zip(o, kp)))
    print("\n=== queued cargo at the straits (all fuel) and stocks at the terminals / the grid, per bin")
    rows = [i for i, (n, k) in enumerate(stock_slots) if n in targets and k in (0, 1)]
    print(
        "  weeks    "
        + "".join(f"{(node(stock_slots[i][0])[:8] + ' ' + names[stock_slots[i][1]][:5]):>16}" for i in rows)
        + "   queue total"
    )
    for lo in range(0, T, bins):
        w = slice(lo, min(lo + bins, T))
        st = np.mean([[x["stock"][i] for i in rows] for x in obs_log[w]], axis=0)
        q = np.mean([x["queue"].sum() if x["queue"] is not None else 0.0 for x in obs_log[w]])
        print(f"  {lo + 1:>2}-{min(lo + bins, T):<5}  " + "".join(f"{v:16,.0f}" for v in st) + f"{q:14,.0f}")


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
