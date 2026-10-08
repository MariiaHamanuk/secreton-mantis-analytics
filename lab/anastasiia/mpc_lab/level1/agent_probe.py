"""Play any agent folder on chosen root-111 episodes and print the KR-gas picture of each episode: cost, shortage,
shed, KR lots and burned gas, lng parked at the straits, and grid_kr's lng stock against its rationing threshold.

Made to answer two questions about a new model (e.g. anastasiia_plan_hull) on the 8 KR-deficit episodes of
reports/level1.md: does it burn the gas in Korea that the full-knowledge plan burns (hypothesis "insure the fuel
block"), and does it still park gas behind visibly cut edges the way the hybrid's buffer does (worth +0.009 there).

    uv run python lab/anastasiia/mpc_lab/level1/agent_probe.py agents/anastasiia_plan_hull --eps=23,26,32,33,38,45,57,61
"""

from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def play(folder, n):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": int(n)})
    u = env.unwrapped
    inst = u.instance
    ag = load(str(folder))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    done = False
    while not done:
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done = term or trunc
    recs = u.core._ep.traj.records
    nid = {nd.id: i for i, nd in enumerate(inst.nodes)}
    lng = [c.id for c in inst.commodities].index("lng")
    kr_f = next(i for i, f in enumerate(inst.fabs) if inst.nodes[f].id == "fab_kr_memory_1")
    slot = lambda nm: next(i for i, st in enumerate(inst.stock_slots) if st.node == nid[nm] and st.k == lng)  # noqa: E731
    stock = np.array([r.stock for r in recs])  # (T, stock slots), end of week
    queues = sum(stock[:, slot(nm)].mean() for nm in ("chk_hormuz", "chk_malacca", "chk_taiwan"))
    return dict(
        J=u.core._ep.traj.J_cents / 1e11,
        shortage=sum(r.costs.shortage for r in recs) / 1e9,
        shed=sum(r.costs.shed for r in recs) / 1e9,
        kr_lots=sum(r.lots_started[kr_f] for r in recs[:40]) / 1e6,
        kr_burn=sum(r.segment.get((1, lng), 0.0) for r in recs),
        queues_lng=queues,
        grid_kr=stock[13:39, slot("grid_kr")].mean(),
    )


def main(agent, eps="23,26,32,33,38,45,57,61", against=None) -> None:
    folder = Path(agent) if Path(agent).is_absolute() else ROOT / agent
    base = Path(against) if against else None
    if base is not None and not base.is_absolute():
        base = ROOT / base
    which = [int(e) for e in str(eps).split(",")]
    print(f"{folder.name} on small 111 episodes {which}" + (f", against {base.name}" if base else ""))
    print(f"{'ep':>3} {'J bn':>8} {'short':>7} {'shed':>7} {'KRlots M':>8} {'KRburn':>7} {'queues':>7} {'grid_kr':>8}"
          + ("  | base: J / dJ / KRburn / queues / grid_kr" if base else ""))
    tot = []
    for n in which:
        a = play(folder, n)
        row = (f"{n:3d} {a['J']:8.1f} {a['shortage']:7.1f} {a['shed']:7.1f} {a['kr_lots']:8.2f} "
               f"{a['kr_burn']:7.0f} {a['queues_lng']:7.0f} {a['grid_kr']:8.0f}")
        if base is not None:
            b = play(base, n)
            tot.append(b["J"] - a["J"])
            row += (f"  | {b['J']:8.1f} / {b['J'] - a['J']:+6.1f} / {a['kr_burn'] - b['kr_burn']:+6.0f} / "
                    f"{a['queues_lng'] - b['queues_lng']:+6.0f} / {a['grid_kr'] - b['grid_kr']:+6.0f}")
        print(row, flush=True)
    if tot:
        print(f"mean dJ (base - agent, positive = agent cheaper): {np.mean(tot):+.1f} bn/ep, "
              f"better in {sum(x > 0 for x in tot)}/{len(tot)}")


if __name__ == "__main__":
    fire.Fire(main)
