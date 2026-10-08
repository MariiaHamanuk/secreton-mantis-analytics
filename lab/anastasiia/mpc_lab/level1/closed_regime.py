"""Closed-loop regime probe: the hub plays, but in weeks w0..w1 every KR-bound Qatar lane (lane ids given) gets +delta
GWh a week on top of what the hub asked (if the slot is open); optional +delta on the term_kr -> grid_kr valve. The
hub's own logic reacts to the state everywhere else (wafers, chips, other fuel). Root 111 episodes: mechanism probe only.

    uv run python lab/anastasiia/mpc_lab/level1/closed_regime.py --eps=26,45 --deltas=0,250,500
"""

from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
AGENT = str(ROOT / "agents" / "anastasiia_hybrid_hub")


def pick_lane(inst, obs):
    """The KR bypass a visible cut calls for this week (None: no cut on KR's main lane, or nothing to bypass with).

    Main lane: Hormuz -> Malacca -> Taiwan -> term_kr. Taiwan -> term_kr below half: the east lane (Malacca -> term_kr);
    Malacca strait or Malacca -> Taiwan below half: the lombok lane (Hormuz -> Taiwan); Hormuz itself: no Qatar bypass.
    """
    nid = {nd.id: i for i, nd in enumerate(inst.nodes)}
    E = inst.edges
    u = np.asarray(obs["graph_now.u"], dtype=float)
    op = np.asarray(obs["graph_now.open"], dtype=float)

    def frac(a, b):
        e = next(i for i, x in enumerate(E) if x.tail == nid[a] and x.head == nid[b])
        return u[e] / E[e].u0 if np.isfinite(u[e]) else 1.0

    if op[inst.chokepoint_ordinal[nid["chk_hormuz"]]] < 0.5:
        return None
    if frac("chk_taiwan", "term_kr") < 0.5 or op[inst.chokepoint_ordinal[nid["chk_taiwan"]]] < 0.5:
        return "lane.src_qa_lng.term_kr.east"
    if op[inst.chokepoint_ordinal[nid["chk_malacca"]]] < 0.5 or frac("chk_malacca", "chk_taiwan") < 0.5:
        return "lane.src_qa_lng.term_kr.lombok"
    return None


def play(n, delta, lanes, w0, w1, valve_too, strict_weeks=0, stock_lt=0.0):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    inst = u.instance
    slots = [s for s, (e, k, lane) in enumerate(inst.action_slots) if lane is not None and inst.lanes[lane].id in lanes]
    valve = next(s for s, (e, k, lane) in enumerate(inst.action_slots)
                 if inst.edges[e].id.endswith("term_kr.grid_kr") and inst.commodities[k].id == "lng")
    ag = load(AGENT)(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    nid = {nd.id: i for i, nd in enumerate(inst.nodes)}
    gk_lng = next(i for i, st in enumerate(inst.stock_slots)
                  if st.node == nid["grid_kr"] and inst.commodities[st.k].id == "lng")
    done, t, cut_run, fired = False, 1, 0, 0
    while not done:
        a = {k: np.array(v, copy=True) for k, v in ag.act(obs).items()}
        if lanes == ("auto",):
            want = pick_lane(inst, obs)
            cut_run = cut_run + 1 if want else 0
        if lanes == ("auto",) and w0 <= t <= w1 and delta > 0:
            mask = np.asarray(obs["action_mask"]) == 1
            # strict gate: the cut must have persisted and KR must actually be starving (stock under rationing)
            armed = cut_run >= strict_weeks and (stock_lt <= 0.0 or float(obs["stock.qty"][gk_lng]) < stock_lt)
            for s in range(len(inst.action_slots)):
                lane = inst.action_slots[s][2]
                if armed and want and lane is not None and inst.lanes[lane].id == want and mask[s]:
                    a["flows"][s] += delta
                    fired += 1
        elif w0 <= t <= w1 and delta > 0:
            mask = np.asarray(obs["action_mask"]) == 1
            for s in slots:
                if mask[s]:
                    a["flows"][s] += delta / len(slots)
            if valve_too and mask[valve]:
                a["flows"][valve] += delta
        obs, _r, term, trunc, _i = env.step(a)
        done = term or trunc
        t += 1
    recs = u.core._ep.traj.records
    kr_f = next(i for i, f in enumerate(inst.fabs) if inst.nodes[f].id == "fab_kr_memory_1")
    lng = [c.id for c in inst.commodities].index("lng")
    return dict(
        fired=fired,
        J=u.core._ep.traj.J_cents / 1e11,
        shortage=sum(r.costs.shortage for r in recs) / 1e9,
        shed=sum(r.costs.shed for r in recs) / 1e9,
        kr_lots=sum(r.lots_started[kr_f] for r in recs[:40]) / 1e6,
        kr_burn=sum(r.segment.get((1, lng), 0.0) for r in recs),
    )


def main(eps, deltas="0,250,500,750", lanes="lane.src_qa_lng.term_kr.lombok", w0: int = 10, w1: int = 39,
         valve_too: bool = False, strict_weeks: int = 0, stock_lt: float = 0.0) -> None:
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    ds = [float(d) for d in deltas] if isinstance(deltas, (list, tuple)) else [float(d) for d in str(deltas).split(",")]
    ln = tuple(lanes) if isinstance(lanes, (list, tuple)) else tuple(str(lanes).split(","))
    print(f"lanes {ln}, weeks {w0}-{w1}, valve too: {valve_too}, strict_weeks {strict_weeks}, stock_lt {stock_lt}; "
          f"dJ = hub - variant (positive = better), bn/ep")
    tot = {d: [] for d in ds}
    for n in which:
        base = None
        row = f"{n:3d}"
        for d in ds:
            r = play(n, d, ln, w0, w1, valve_too, strict_weeks, stock_lt)
            if base is None:
                base = r
            tot[d].append(base["J"] - r["J"])
            row += (f" | +{d:.0f}: dJ {base['J'] - r['J']:+6.1f} (fired {r['fired']:2d}, short "
                    f"{r['shortage'] - base['shortage']:+6.1f}, shed "
                    f"{r['shed'] - base['shed']:+6.1f}, KR lots {r['kr_lots'] - base['kr_lots']:+.2f} M, KR burn "
                    f"{r['kr_burn'] - base['kr_burn']:+6.0f})")
        print(row, flush=True)
    print("mean dJ: " + ", ".join(f"+{d:.0f}: {np.mean(v):+.1f} (better in {sum(x > 0 for x in v)}/{len(v)})"
                                  for d, v in tot.items()))


if __name__ == "__main__":
    fire.Fire(main)
