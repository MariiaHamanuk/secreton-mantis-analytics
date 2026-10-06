"""What the best plan an agent could follow does: the clairvoyant plan under the simulator's energy rule, and its flows.

    uv run python lab/anastasiia/stats_lab/plan_stats.py
    uv run python lab/anastasiia/stats_lab/plan_stats.py --task=small --episodes=48 --entropy=444 --time_limit=60

The board's clairvoyant plan (score 1) is a relaxation: it may power a fab while its grid sheds base load, which the
simulator never does (every grid serves its base load first). Here each episode is also solved with that rule, as a
mixed-integer program: one binary per grid and week, 1 where the base load is fully served and the fabs may draw power,
0 where the fabs draw nothing. It is built on the oracle's rows with the planning rules (fuel burned pro rata every
week), at the true cost. Its flows are a plan a rule can imitate; the relaxed plan's fuel flows are not.

With ``--replay`` each plan's own orders are also played in the simulator, open loop: the cost an agent would pay by
following them blindly (a floor on what the plan is worth in practice; a closed-loop rule can do better).

Writes ``summary.md`` and ``episodes.npz`` under ``outputs/plan_stats/<date_time>/``. A solve stopped by the time limit
keeps its best plan so far: the plan is feasible, its cost an upper bound (the summary counts them). An episode with
no plan at all within the limit is tried once more with three times the limit, then left out and named in the summary.
"""

import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HOLD, OVERRIDE = 2, 1  # release_mode codes


def base_first_plan(inst, marks, model, time_limit: float, gap: float):
    """(x, status, mip gap) of ``model`` with base load first at every grid and week; x is None without a plan."""
    import scipy.sparse as sp
    from scipy.optimize import Bounds, LinearConstraint, milp

    T, nc, tmpl = model.T, model.meta["nc"], model.meta["template"]
    ncol, G = T * nc, len(inst.grids)
    rows, cols, vals, rhs = [], [], [], []
    for t in range(T):
        for go in range(G):
            z, r = ncol + t * G + go, len(rhs)
            y_bar = float(marks.y_bar[t, go])
            rows += [r, r]  # shed <= base load (1 - z)
            cols += [t * nc + tmpl[("ysh", go)], z]
            vals += [1.0, y_bar]
            rhs.append(y_bar)
            for fo in inst.grid_fabs[go]:
                if ("E", fo) not in tmpl:
                    continue
                fab = inst.nodes[inst.fabs[fo]].fab
                most = fab.e * float(marks.alpha_bar[t, fo]) * fab.cap0 * (1 + 1e-6) + 1e-9
                r = len(rhs)
                rows += [r, r]  # the fab's energy <= its largest draw x z
                cols += [t * nc + tmpl[("E", fo)], z]
                vals += [1.0, -most]
                rhs.append(0.0)
    extra = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), ncol + T * G))

    def pad(a):
        return sp.hstack([a, sp.csr_matrix((a.shape[0], T * G))], format="csr")

    constraints = [LinearConstraint(extra, -np.inf, np.array(rhs))]
    if model.A_eq.shape[0]:
        constraints.append(LinearConstraint(pad(model.A_eq), model.b_eq, model.b_eq))
    if model.A_ub.shape[0]:
        constraints.append(LinearConstraint(pad(model.A_ub), -np.inf, model.b_ub))
    res = milp(
        np.concatenate([model.cost - model.salvage, np.zeros(T * G)]),  # the true J: no planning prices
        constraints=constraints,
        integrality=np.concatenate([np.zeros(ncol), np.ones(T * G)]),
        bounds=Bounds(np.concatenate([model.lb, np.zeros(T * G)]), np.concatenate([model.ub, np.ones(T * G)])),
        options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False},
    )
    if res.x is None:
        return None, int(res.status), float("nan")
    return res.x[:ncol], int(res.status), float(getattr(res, "mip_gap", float("nan")))


class Replay:
    """Asks every week for a plan's flows and tanker releases of that week (an agent for ``record_episode``)."""

    def __init__(self, config, inst, model, x, marks):
        self.inst, self.model, self.x, self.marks = inst, model, x, marks
        layout, action = config["layout"], config["spaces"]["action"]
        self.n_slots, self.n_override = action["flows"]["shape"][0], action["override_qty"]["shape"][0]
        self.pair = {(int(c), int(k)): i for i, (c, k) in enumerate(layout["release_pairs"])}
        slots = config["static"]["override_slots"]
        self.override_pair = [self.pair[(c, k)] for c, k in zip(slots["chokepoint"], slots["k"])]

    def act(self, observation):
        from shockbench_flow.policies import lp_common as L

        t, nc = int(observation["week"][0]), self.model.meta["nc"]
        week = np.zeros_like(self.model.lb)
        week[:nc] = np.maximum(self.x[(t - 1) * nc : t * nc], 0.0)
        banned = np.asarray(self.marks.prohibited[t - 1], dtype=bool)
        wire = L.week1_action(self.inst, self.model, week, {"week": t}, banned)
        flows = np.zeros(self.n_slots)
        flows[wire["flows"]["slot"]] = wire["flows"]["qty"]
        override_qty = np.zeros(self.n_override)
        release_mode = np.zeros(len(self.pair), dtype=np.int64)
        for slot, qty in zip(*((wire["overrides"] or {}).get(name, []) for name in ("slot", "qty"))):
            override_qty[slot] = qty
            release_mode[self.override_pair[slot]] = OVERRIDE
        for c, k in zip(*((wire["hold"] or {}).get(name, []) for name in ("chokepoint", "k"))):
            release_mode[self.pair[(c, k)]] = HOLD
        override_qty[[i for i, p in enumerate(self.override_pair) if release_mode[p] != OVERRIDE]] = 0.0
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}


def plan_arrays(inst, model, x) -> dict:
    """A plan's weekly quantities: slot flows, stocks, lots, power, demand served, each (T, ...)."""
    from shockbench_flow.oracle.lp import lp_flow_key

    T, nc, tmpl = model.T, model.meta["nc"], model.meta["template"]
    w = np.asarray(x).reshape(T, nc)

    def col(*key):
        return w[:, tmpl[key]] if key in tmpl else np.zeros(T)

    plain = [s for s, st in enumerate(inst.stock_slots) if st.node not in inst.chokepoint_ordinal]
    return {
        "sent": np.stack([col("x", *lp_flow_key(inst, e, k, lane)) for e, k, lane in inst.action_slots], axis=1),
        "stock": np.stack([col("I", s) for s in plain], axis=1),
        "lots": np.stack([col("p", fo) for fo in range(len(inst.fabs))], axis=1),
        "fab_energy": np.stack([col("E", fo) for fo in range(len(inst.fabs))], axis=1),
        "shed": np.stack([col("ysh", go) for go in range(len(inst.grids))], axis=1),
        "served": np.stack([col("D", do) for do in range(len(inst.demands))], axis=1),
    }


def episode(task: str, entropy: int, n: int, time_limit: float, gap: float, replay: bool) -> dict:
    """One episode: the relaxed plan, the plan with base load first, their flows and, with ``replay``, their replays."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp, lp_cents, solve_oracle
    from shockbench_flow_gym.dashboard import record_episode

    from sbf_starter import env_id

    inst, params = task_generator(task)
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train"))
    relaxed_model = build_lp(inst, marks)
    relaxed = solve_oracle(relaxed_model)
    model = build_lp(inst, marks, planning_rules=True)
    start = time.perf_counter()
    x, status, mip_gap = base_first_plan(inst, marks, model, time_limit, gap)
    if x is None:  # no plan yet: one more try, three times as long
        x, status, mip_gap = base_first_plan(inst, marks, model, 3 * time_limit, gap)
    if x is None:
        return None
    out = {
        "J_relaxed": relaxed.J_cents / 100,
        "status": status,
        "mip_gap": mip_gap,
        "seconds": time.perf_counter() - start,
    }
    out |= {f"relaxed_{k}": v for k, v in plan_arrays(inst, relaxed_model, relaxed.x).items()}
    out["J_plan"] = lp_cents(model, x) / 100
    out |= {f"plan_{k}": v for k, v in plan_arrays(inst, model, x).items()}
    slot_edges = [e for e, _k, _lane in inst.action_slots]
    out["capacity"] = np.asarray(marks.u)[:, slot_edges]  # (T, slots) the first edge's capacity
    out["banned"] = np.array(  # (T, slots) 1 where an edge of the slot's route is prohibited for its commodity
        [
            [
                any(marks.prohibited[t][x][k] for x in ([e] if lane is None else inst.lanes[lane].edges))
                for e, k, lane in inst.action_slots
            ]
            for t in range(inst.T)
        ],
        dtype=float,
    )
    out["open"] = np.asarray(marks.o)  # (T, straits)
    if replay:
        for name, m, plan in (("relaxed", relaxed_model, relaxed.x), ("plan", model, x)):
            env = gym.make(env_id(task), entropy=entropy)
            rec = record_episode(env, lambda config: Replay(config, inst, m, plan, marks), options={"episode": n})  # noqa: B023
            out[f"J_replay_{name}"] = rec["meta"]["J_cents"] / 100
    return out


def table(header: list[str], rows: list[list]) -> str:
    def cell(v) -> str:
        return f"{v:,.2f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]) + "\n"


def report(task: str, entropy: int, z: dict, refs: list[dict], replay: bool, skipped: list[int]) -> str:
    from package_baselines import rss
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    N, E, K = inst.nodes, inst.edges, [c.id for c in inst.commodities]
    n_ep, T = z["plan_lots"].shape[:2]
    out = [f"# The best plan an agent could follow: {task}, {n_ep} episodes of root {entropy}\n"]
    out.append(
        "The board's clairvoyant plan may power a fab while its grid sheds base load; the simulator serves the base "
        "load first. `plan` below is the clairvoyant plan with that rule (a mixed-integer program), `relaxed` the "
        "board's. Written by `lab/anastasiia/stats_lab/plan_stats.py`.\n"
    )
    stopped = int((z["status"] != 0).sum())
    out.append(
        f"Solves: {z['seconds'].mean():.0f} s on average; {stopped} of {n_ep} stopped by the time limit (largest gap "
        f"{np.nanmax(z['mip_gap']):.3f}), their plan is feasible and its cost an upper bound.\n"
    )
    if skipped:
        out.append(f"Left out, no plan within four times the time limit: episodes {skipped}.\n")

    # 1. the ceiling
    out.append("## 1. The score such a plan reaches\n")
    drift = max(abs(a - r["J_oracle_cents"] / 100) for a, r in zip(z["J_relaxed"], refs))
    out.append(
        f"Check against the cached references: the largest difference of the relaxed plan's cost is ${drift:,.2f}.\n"
    )
    names = [("plan", "J_plan", "the plan with base load first (its own cost)")]
    if replay:
        names += [("plan, replayed", "J_replay_plan", "its orders played blindly in the simulator")]
        names += [("relaxed, replayed", "J_replay_relaxed", "the board's plan's orders played blindly")]
    rows = []
    for label, key, what in names:
        score, by = rss(refs, [int(round(100 * j)) for j in z[key]])
        rows.append([label, score] + [by.get(s, float("nan")) for s in (1, 2, 3, 4)] + [what])
    out.append(table(["", "score", "level 1", "level 2", "level 3", "level 4", "what it is"], rows))
    level = np.array([r["stratum"] for r in refs])
    out.append("Episodes per harm level: " + ", ".join(f"{s}: {int((level == s).sum())}" for s in (1, 2, 3, 4)) + ".\n")

    # 2. grids
    out.append("## 2. Grids: when the fabs get power\n")
    out.append(
        "A fab runs only on what its grid delivers above the base load. `powered` is the share of weeks in which the "
        "plan gives the grid's fabs any power; `shed` the base load not served, GWh per week.\n"
    )
    rows = []
    for go, g in enumerate(inst.grids):
        fabs = inst.grid_fabs[go]
        row = [N[g].id, 100 * N[g].grid.base_load / N[g].grid.deliverable]
        for who in ("plan", "relaxed"):
            on = z[f"{who}_fab_energy"][:, :, fabs].sum(axis=2) > 1e-6
            shed = z[f"{who}_shed"][:, :, go]
            row += [100 * float(on.mean()), float(shed.mean()), 100 * float((on & (shed > 1e-6)).mean())]
        runs = []  # lengths of the plan's powered stretches
        on = z["plan_fab_energy"][:, :, fabs].sum(axis=2) > 1e-6
        for ep in on:
            edges = np.flatnonzero(np.diff(np.concatenate([[0], ep.astype(int), [0]])))
            runs += (edges[1::2] - edges[::2]).tolist()
        rows.append(row + [float(np.median(runs)) if runs else 0.0])
    head = ["grid", "base load % of output", "plan: powered %", "plan: shed", "plan: powered while shedding %"]
    head += [
        "relaxed: powered %",
        "relaxed: shed",
        "relaxed: powered while shedding %",
        "plan: weeks in a powered stretch (median)",
    ]
    out.append(table(head, rows))

    # 3. fabs
    out.append("## 3. Fabs: lots started, % of nominal capacity\n")
    last = max(T - 12, 1)  # later lots cannot reach a market
    rows = []
    for fo, f in enumerate(inst.fabs):
        cap = N[f].fab.cap0
        rows.append(
            [N[f].id, K[N[f].fab.product]]
            + [100 * float(z[f"{who}_lots"][:, :last, fo].mean()) / cap for who in ("plan", "relaxed")]
            + [100 * float((z["plan_lots"][:, :last, fo] > 0.01 * cap).mean())]
        )
    out.append(table(["fab", "product", "plan", "relaxed", "plan: weeks with a start %"], rows))
    out.append(f"Weeks 1 to {last}: a lot started later cannot be sold before the end.\n")

    # 4. where each stock goes
    out.append("## 4. Where the plan sends each stock\n")
    out.append(
        "Per origin and commodity: the plan's mean weekly flow to each destination and its share, beside the share "
        "'send the maximum' would give (the slots' nominal capacities). `open %` is the share of weeks the slot's "
        "route is not prohibited; `use %` the plan's flow over the first edge's capacity in those weeks. Tanker cargo "
        "(lng, crude) sent through a strait is left out: the program pools it at the strait and decides its "
        "destination there, so the lane it was dispatched on says nothing; the next table has what reaches each grid.\n"
    )
    dest = [E[e].head if lane is None else E[inst.lanes[lane].edges[-1]].head for e, _k, lane in inst.action_slots]
    sent, cap, ok = z["plan_sent"], z["capacity"], z["banned"] < 0.5
    groups: dict[tuple[int, int], list[int]] = {}
    for s, (e, k, _lane) in enumerate(inst.action_slots):
        groups.setdefault((E[e].tail, k), []).append(s)
    rows = []
    for (tail, k), members in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if len({dest[s] for s in members}) < 2:
            continue
        if inst.commodities[k].override and any(inst.action_slots[s][2] is not None for s in members):
            continue  # pooled at the strait: the lane's destination is not the cargo's
        total = sent[:, :, members].sum(axis=2).mean()
        nominal = sum(E[inst.action_slots[s][0]].u0 for s in members)
        by: dict[int, list[int]] = {}
        for s in members:
            by.setdefault(dest[s], []).append(s)
        for d, ss in sorted(by.items(), key=lambda kv: -sent[:, :, kv[1]].sum()):
            flow = sent[:, :, ss].sum(axis=2).mean()
            live = ok[:, :, ss]
            use = (sent[:, :, ss] / np.maximum(cap[:, :, ss], 1e-9))[live].mean() if live.any() else 0.0
            rows.append(
                [N[tail].id, K[k], N[d].id, len(ss), float(flow), 100 * float(flow / max(total, 1e-9))]
                + [
                    100 * sum(E[inst.action_slots[s][0]].u0 for s in ss) / nominal,
                    100 * float(live.mean()),
                    100 * float(use),
                ]
            )
    head = [
        "origin",
        "commodity",
        "destination",
        "slots",
        "plan: flow / week",
        "plan: share %",
        "send-the-maximum share %",
    ]
    out.append(table(head + ["open %", "use %"], rows))
    out.append("Fuel moved into each grid (its terminal-to-grid and pipeline slots), units per week:\n")
    rows = []
    for g in inst.grids:
        grid = N[g].grid
        for k in grid.fuels:
            into = [
                s for s, (e, kk, lane) in enumerate(inst.action_slots) if kk == k and lane is None and E[e].head == g
            ]
            burn = grid.shares[k] * grid.deliverable
            flow = [float(z[f"{who}_sent"][:, :, into].sum(axis=2).mean()) for who in ("plan", "relaxed")]
            rows.append([N[g].id, K[k], float(burn), flow[0], 100 * flow[0] / burn, flow[1], 100 * flow[1] / burn])
    head = ["grid", "fuel", "burn at full output", "plan", "plan % of burn", "relaxed", "relaxed % of burn"]
    out.append(table(head, rows))

    # 5. straits
    out.append("## 5. Lanes through straits: how much the plan sends against the strait's state\n")
    chk = {c: i for i, c in enumerate(inst.chokepoints)}
    rows = []
    for label, test in (
        ("every strait of the lane fully open", lambda o: o >= 0.999),
        ("a strait partly open (0.25 to 1)", lambda o: (o >= 0.25) & (o < 0.999)),
        ("a strait under 0.25", lambda o: o < 0.25),
    ):
        use, weeks = [], 0
        for s, (e, k, lane) in enumerate(inst.action_slots):
            if lane is None:
                continue
            worst = z["open"][:, :, [chk[c] for c in inst.lanes[lane].chokepoints]].min(axis=2)
            m = test(worst) & ok[:, :, s]
            weeks += int(m.sum())
            use.append((sent[:, :, s][m] / np.maximum(cap[:, :, s][m], 1e-9)))
        use = np.concatenate(use) if weeks else np.zeros(1)
        rows.append([label, weeks, 100 * float(use.mean()), 100 * float((use > 0.01).mean())])
    out.append(
        table(
            ["state of the lane's straits", "slot-weeks", "plan: use of the first edge %", "slot-weeks used at all %"],
            rows,
        )
    )

    # 6. markets
    out.append("## 6. Markets: demand served by the plan, units per week\n")
    rows = []
    for do, d in enumerate(inst.demands):
        if d.dbar <= 0:
            continue
        a, b = float(z["plan_served"][:, :, do].mean()), float(z["relaxed_served"][:, :, do].mean())
        rows.append([N[d.node].id, K[d.k], float(d.dbar), a, 100 * a / d.dbar, b, 100 * b / d.dbar])
    out.append(table(["market", "commodity", "demand", "plan", "plan %", "relaxed", "relaxed %"], rows))
    return "\n".join(out)


def main(
    task: str = "tiny",
    episodes: int = 8,
    entropy: int = 444,
    time_limit: float = 60.0,
    gap: float = 2e-3,
    replay: bool = True,
    n_jobs: int = 3,
    out: str | None = None,
) -> None:
    """Solve every episode's plan with base load first and write what it does.

    Args:
        task: tiny, small or full.
        episodes: episodes 0 .. episodes - 1 of the root.
        entropy: the scenarios' root (444, the team's root for plans; no agent is tuned on it).
        time_limit: seconds a mixed-integer solve may take; its best plan so far is kept.
        gap: the relative gap at which a solve stops.
        replay: also play each plan's orders in the simulator (--noreplay skips it).
        n_jobs: workers.
        out: the folder to write to (default: outputs/plan_stats/<date_time>/).

    """
    from sbf_starter import ROOT, scoring

    folder = Path(out) if out else ROOT / "outputs" / "plan_stats" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)  # fills the cache too
    start = time.perf_counter()
    eps = Parallel(n_jobs=n_jobs)(delayed(episode)(task, entropy, n, time_limit, gap, replay) for n in range(episodes))
    print(f"{episodes} episodes of {task} solved in {time.perf_counter() - start:.0f} s")
    skipped = [n for n, e in enumerate(eps) if e is None]  # no plan within four times the limit
    refs = [r for r, e in zip(refs, eps) if e is not None]
    eps = [e for e in eps if e is not None]
    z = {k: np.stack([np.asarray(e[k]) for e in eps]) for k in eps[0]}
    (folder / "summary.md").write_text(report(task, entropy, z, refs, replay, skipped))
    np.savez_compressed(folder / "episodes.npz", **z)
    print(f"written {folder}")


if __name__ == "__main__":
    fire.Fire(main)
