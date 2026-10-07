"""Probe A: where the base-first plan's extra KR gas comes from, against the hub, and was its capacity visible.

    uv run python lab/anastasiia/mpc_lab/level1/probeA.py run --eps=23,26 --n_jobs=1
    uv run python lab/anastasiia/mpc_lab/level1/probeA.py report --eps=23,26,32,33,38,45,57,61

``run`` plays the hub (gym) and solves the base-first MILP (plan_stats.base_first_plan) for each episode and saves, per
week, the lng flow on every lng edge (plan: LP columns summed over lanes; hub: the simulator's executed x, dispatches
and chokepoint releases), lng queues at straits, lng stock at term_kr / grid_kr, and the edges' true capacity
``marks.u`` beside the capacity an agent saw that week ``marks.u_now``, and the straits' tanker throughput
(``kappa`` / ``kappa_now``). Files: outputs/level1/probeA/ep<n>.npz.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "stats_lab"))
OUT = ROOT / "outputs" / "level1" / "probeA"
BANDS = ((0, 13), (13, 26), (26, 39), (39, 52))


def one(n: int, agent: str, entropy: int, time_limit: float) -> str:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from plan_stats import base_first_plan
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp, lp_cents
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    path = OUT / f"ep{n}.npz"
    if path.exists():
        return f"ep{n}: kept"
    inst, params = task_generator("small")
    lng = [c.id for c in inst.commodities].index("lng")
    T, E = inst.T, len(inst.edges)
    env = gym.make(env_id("small"), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = load(agent)(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    done = False
    while not done:
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done = term or trunc
    recs = u.core._ep.traj.records
    J_hub = u.core._ep.traj.J_cents / 100
    hub_flow = np.zeros((T, E))
    hub_lane = {}  # (edge, lane) -> (T,)
    for t, r in enumerate(recs):
        for (e, k, lane), q in r.x.items():
            if k == lng:
                hub_flow[t, e] += q
                hub_lane.setdefault((e, -1 if lane is None else lane), np.zeros(T))[t] += q
    hub_stock = np.array([r.stock for r in recs])
    hub_seg = np.array([[r.segment.get((go, lng), 0.0) for go in range(len(inst.grids))] for r in recs])

    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "train"))
    model = build_lp(inst, marks, planning_rules=True)
    x, status, gap = base_first_plan(inst, marks, model, time_limit, 2e-3)
    nc, tmpl = model.meta["nc"], model.meta["template"]
    w = np.asarray(x).reshape(T, nc)
    plan_flow = np.zeros((T, E))
    plan_lane = {}
    for key, j in tmpl.items():
        if key[0] == "x" and key[2] == lng:
            plan_flow[:, key[1]] += w[:, j]
            plan_lane.setdefault((key[1], -1 if key[3] is None else key[3]), np.zeros(T))[:] += w[:, j]
    plan_stock = np.zeros((T, len(inst.stock_slots)))
    for s in range(len(inst.stock_slots)):
        if ("I", s) in tmpl:
            plan_stock[:, s] = w[:, tmpl[("I", s)]]
    plan_seg = np.stack([w[:, tmpl[("G", go, lng)]] if ("G", go, lng) in tmpl else np.zeros(T)
                         for go in range(len(inst.grids))], axis=1)
    pool = inst.commodity_pool[lng]
    np.savez(
        path,
        J_hub=J_hub, J_plan=lp_cents(model, x) / 100, status=status, gap=gap,
        hub_flow=hub_flow, plan_flow=plan_flow, hub_stock=hub_stock, plan_stock=plan_stock,
        hub_seg=hub_seg, plan_seg=plan_seg,
        u=np.asarray(marks.u)[:T], u_now=np.asarray(marks.u_now)[:T],
        kappa=np.asarray(marks.kappa)[:T, :, pool], kappa_now=np.asarray(marks.kappa_now)[:T, :, pool],
        hub_lane_keys=np.array(list(hub_lane)), hub_lane_vals=np.array(list(hub_lane.values())),
        plan_lane_keys=np.array(list(plan_lane)), plan_lane_vals=np.array(list(plan_lane.values())),
    )
    return f"ep{n}: hub {J_hub / 1e9:.1f}  plan {lp_cents(model, x) / 1e11:.1f} (status {status}, gap {gap:.4f})"


def run(eps, agent: str = "agents/anastasiia_hybrid_hub", entropy: int = 111, time_limit: float = 150.0,
        n_jobs: int = 1) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    for msg in Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(one)(n, str(ROOT / agent), entropy, time_limit) for n in which
    ):
        print(msg, flush=True)


def report(eps) -> None:
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("small")
    N, E = inst.nodes, inst.edges
    nid = {nd.id: i for i, nd in enumerate(N)}
    lng = [c.id for c in inst.commodities].index("lng")
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    Z = {n: dict(np.load(OUT / f"ep{n}.npz")) for n in which}

    def edge(a, b):
        return next(i for i, e in enumerate(E) if e.tail == nid[a] and e.head == nid[b] and lng in e.K)

    named = [
        ("Qatar dispatch (all lanes)", edge("src_qa_lng", "chk_hormuz")),
        ("Hormuz -> Malacca (TW+KR lanes)", edge("chk_hormuz", "chk_malacca")),
        ("Hormuz -> Taiwan (KR lombok bypass)", edge("chk_hormuz", "chk_taiwan")),
        ("Hormuz -> term_tw (TW lombok)", edge("chk_hormuz", "term_tw")),
        ("Malacca -> Taiwan (KR main)", edge("chk_malacca", "chk_taiwan")),
        ("Malacca -> term_kr (KR east bypass)", edge("chk_malacca", "term_kr")),
        ("Malacca -> term_tw", edge("chk_malacca", "term_tw")),
        ("Taiwan -> term_kr (KR main+lombok)", edge("chk_taiwan", "term_kr")),
        ("Australia -> term_kr", edge("src_au_lng", "term_kr")),
        ("Australia -> term_jp", edge("src_au_lng", "term_jp")),
        ("term_kr -> grid_kr (valve)", edge("term_kr", "grid_kr")),
    ]
    print(f"episodes {which}: lng GWh per episode, plan - hub by week band 1-13 / 14-26 / 27-39 / 40-52 (hub total)")
    for name, e in named:
        d = [np.mean([Z[n]["plan_flow"][lo:hi, e].sum() - Z[n]["hub_flow"][lo:hi, e].sum() for n in which])
             for lo, hi in BANDS]
        h = np.mean([Z[n]["hub_flow"][:, e].sum() for n in which])
        print(f"  {name:38s} " + " ".join(f"{v:8.0f}" for v in d) + f"   | {sum(d):8.0f}  (hub {h:8.0f})")
    gk = inst.grid_ordinal[nid["grid_kr"]]
    d = [np.mean([Z[n]["plan_seg"][lo:hi, gk].sum() - Z[n]["hub_seg"][lo:hi, gk].sum() for n in which]) for lo, hi in BANDS]
    print(f"  {'KR lng burned':38s} " + " ".join(f"{v:8.0f}" for v in d) + f"   | {sum(d):8.0f}")
    # Qatar dispatch per lane (the first edge keeps its lane in both)
    eq = edge("src_qa_lng", "chk_hormuz")
    print("  Qatar dispatch by lane, plan - hub (hub total):")
    for lane in sorted({int(k[1]) for n in which for k in Z[n]["plan_lane_keys"] if k[0] == eq}
                       | {int(k[1]) for n in which for k in Z[n]["hub_lane_keys"] if k[0] == eq}):
        def lane_sum(n, who, lo, hi, lane=lane):
            keys, vals = Z[n][f"{who}_lane_keys"], Z[n][f"{who}_lane_vals"]
            m = [(int(k[0]) == eq and int(k[1]) == lane) for k in keys]
            return vals[np.array(m)][:, lo:hi].sum() if any(m) else 0.0

        d = [np.mean([lane_sum(n, "plan", lo, hi) - lane_sum(n, "hub", lo, hi) for n in which]) for lo, hi in BANDS]
        h = np.mean([lane_sum(n, "hub", 0, 52) for n in which])
        print(f"    {inst.lanes[lane].id if lane >= 0 else '-':36s} " + " ".join(f"{v:8.0f}" for v in d)
              + f"   | {sum(d):8.0f}  (hub {h:8.0f})")
    print("  lng queue at straits, mean over weeks (plan / hub) and stock at term_kr, grid_kr:")
    for nm in ("chk_hormuz", "chk_malacca", "chk_taiwan", "term_kr", "grid_kr"):
        s = next(i for i, st in enumerate(inst.stock_slots) if st.node == nid[nm] and st.k == lng)
        p = np.mean([Z[n]["plan_stock"][:, s].mean() for n in which])
        h = np.mean([Z[n]["hub_stock"][:, s].mean() for n in which])
        print(f"    {nm:12s} plan {p:8.0f}   hub {h:8.0f}")

    # visibility of the capacity the plan's extra flow used, on the KR-bound edges
    print("\n  Visibility (weeks 1-45): plan's extra flow over the hub on each KR-bound edge, split by whether it fits"
          "\n  under the capacity an agent saw at dispatch time (u_now at week t - lead, lead = weeks from Qatar/AU"
          " to this edge),\n  under the true capacity only, or neither (plan used queue release capacity / other):")
    lead = {}
    for name, e in named:
        tail = N[E[e].tail].id
        lead[e] = {"src_qa_lng": 0, "chk_hormuz": 1, "chk_malacca": 2, "chk_taiwan": 3}.get(tail, 0)
    for name, e in named[1:9]:
        tot = vis = true_only = 0.0
        for n in which:
            z = Z[n]
            for t in range(45):
                extra = z["plan_flow"][t, e] - z["hub_flow"][t, e]
                if extra <= 0:
                    continue
                t0 = max(t - lead[e], 0)
                seen = z["u_now"][t0, e] - z["hub_flow"][t, e]  # room an agent at dispatch time believed in
                room = z["u"][t, e] - z["hub_flow"][t, e]
                a = min(extra, max(seen, 0.0))
                vis += a
                true_only += min(extra - a, max(room - a, 0.0))
                tot += extra
        if tot > 0:
            k = len(which)
            print(f"    {name:38s} extra {tot / k:8.0f}  visible {100 * vis / tot:5.1f}%  true-only "
                  f"{100 * true_only / tot:5.1f}%  other {100 * (tot - vis - true_only) / tot:5.1f}%")
    print("\n  Capacity seen vs true on KR edges, mean over weeks 14-39 (u_now at t-lead / u at t / hub flow / plan flow):")
    for name, e in named[1:9]:
        s = np.mean([[Z[n]["u_now"][max(t - lead[e], 0), e] for t in range(13, 39)] for n in which])
        tr = np.mean([Z[n]["u"][13:39, e].mean() for n in which])
        h = np.mean([Z[n]["hub_flow"][13:39, e].mean() for n in which])
        p = np.mean([Z[n]["plan_flow"][13:39, e].mean() for n in which])
        print(f"    {name:38s} {s:8.0f} {tr:8.0f} {h:8.0f} {p:8.0f}")
    print("\n  Strait tanker throughput, weeks 14-39 (kappa_now at t-1 / kappa at t):")
    for c in ("chk_hormuz", "chk_malacca", "chk_taiwan"):
        co = inst.chokepoint_ordinal[nid[c]]
        s = np.mean([Z[n]["kappa_now"][12:38, co].mean() for n in which])
        tr = np.mean([Z[n]["kappa"][13:39, co].mean() for n in which])
        print(f"    {c:12s} {s:8.0f} {tr:8.0f}")
    print("\n  per episode: plan - hub KR lng burned (GWh), AU->KR, Taiwan->KR, Malacca->KR, Hormuz->Taiwan; MILP gap")
    for n in which:
        z = Z[n]
        f = [z["plan_seg"][:, gk].sum() - z["hub_seg"][:, gk].sum()] + [
            z["plan_flow"][:, e].sum() - z["hub_flow"][:, e].sum() for e in (named[8][1], named[7][1], named[5][1], named[2][1])]
        print(f"    {n:3d} " + " ".join(f"{v:8.0f}" for v in f) + f"   {float(z['gap']):.4f}")


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report})
