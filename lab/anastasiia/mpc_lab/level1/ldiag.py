"""Per-episode decomposition of an agent's loss against the clairvoyant plan and the base-first plan.

    uv run python lab/anastasiia/mpc_lab/level1/ldiag.py run --episodes=0,3,5 --n_jobs=2 --out=outputs/level1/diag
    uv run python lab/anastasiia/mpc_lab/level1/ldiag.py report --out=outputs/level1/diag --rows=outputs/level1/.../rows.json

``run`` plays the agent once per episode (gym, as hub/eval/diag.py) and solves the board's clairvoyant LP and the
plan with base load first (the MILP of lab/anastasiia/stats_lab/plan_stats.py, a feasible plan; cost an upper bound when
the time limit stops it). Each episode is saved as ``<out>/ep<n>.npz``. Everything is weekly: the eight cost components,
the salvage credit, shed cost per grid, shortage cost per demand, lots per fab.

``report`` groups the saved episodes by harm level (from a rows.json of arith.py) and prints the decompositions.
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "stats_lab"))

COMP = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def _plan_arrays(inst, model, x):
    from shockbench_flow.oracle.lp import lp_costs

    T, nc, tmpl = model.T, model.meta["nc"], model.meta["template"]
    w = np.asarray(x)[: T * nc].reshape(T, nc)

    def col(*key):
        return w[:, tmpl[key]] if key in tmpl else np.zeros(T)

    weekly, credit = lp_costs(model, np.asarray(x)[: T * nc])
    G, F = len(inst.grids), len(inst.fabs)
    short = np.zeros((T, len(inst.demands)))
    for do, d in enumerate(inst.demands):
        short[:, do] = d.pi * (col("B", do) if d.backlog else col("U", do))
    voll = np.array([inst.nodes[g].grid.voll for g in inst.grids])
    shed = np.stack([col("ysh", go) for go in range(G)], axis=1)
    return {
        "costs": np.array([[w_.as_dict()[c] for c in COMP] for w_ in weekly]),  # (T, 8)
        "salvage": float(credit),
        "shed_qty": shed,
        "shed_usd": shed * voll,
        "short_usd": short,
        "lots": np.stack([col("p", fo) for fo in range(F)], axis=1),
        "fab_energy": np.stack([col("E", fo) for fo in range(F)], axis=1),
    }


def episode(agent: str, entropy: int, n: int, time_limit: float, out: str) -> str:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from plan_stats import base_first_plan
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp, lp_cents, solve_oracle
    from shockbench_flow_gym.dashboard import record_episode

    from sbf_starter import env_id
    from sbf_starter.agents import load

    path = Path(out) / f"ep{n}.npz"
    if path.exists():
        return f"ep{n}: kept"
    t0 = time.perf_counter()
    task = "small"
    inst, params = task_generator(task)
    T, F, G, D = inst.T, len(inst.fabs), len(inst.grids), len(inst.demands)
    env = gym.make(env_id(task), entropy=entropy)
    rec = record_episode(env, load(agent), options={"episode": n})
    obs = rec["obs"]
    voll = np.array([inst.nodes[g].grid.voll for g in inst.grids])
    pi = np.array([d.pi for d in inst.demands])
    costs = np.asarray(rec["costs"])
    J = rec["meta"]["J_cents"] / 100
    shed = np.asarray(obs["last_week.shed.qty"][1:])
    lost = np.asarray(obs["last_week.sinks.lost"][1:])
    backlog = np.asarray(obs["backlog.qty"][1:])
    tau = {f: inst.nodes[f].fab.tau for f in inst.fabs}
    starts = np.zeros((T, F))
    for t in range(1, T + 1):
        live = np.asarray(obs["wip.qty.observed"][t]) == 1
        for node, q, ow in zip(obs["wip.node"][t][live], obs["wip.qty"][t][live], obs["wip.out_week"][t][live]):
            node = int(node)
            if node in inst.fab_ordinal and int(ow) == t + tau[node]:
                starts[t - 1, inst.fab_ordinal[node]] += q
    a = {
        "a_costs": costs,
        "a_salvage": costs.sum() - J,
        "a_J": J,
        "a_shed_qty": shed,
        "a_shed_usd": shed * voll,
        "a_short_usd": (lost + backlog) * pi,
        "a_lots": starts,
    }
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train"))
    relaxed_model = build_lp(inst, marks)
    relaxed = solve_oracle(relaxed_model)
    c = {f"c_{k}": v for k, v in _plan_arrays(inst, relaxed_model, relaxed.x).items()}
    c["c_J"] = relaxed.J_cents / 100
    model = build_lp(inst, marks, planning_rules=True)
    x, status, gap = base_first_plan(inst, marks, model, time_limit, 2e-3)
    if x is None:
        x, status, gap = base_first_plan(inst, marks, model, 3 * time_limit, 2e-3)
    b = {}
    if x is not None:
        b = {f"b_{k}": v for k, v in _plan_arrays(inst, model, x).items()}
        b["b_J"] = lp_cents(model, x) / 100
        b["b_status"], b["b_gap"] = status, gap
    np.savez(path, **a, **c, **b)
    msg = f"ep{n}: agent {J / 1e9:.1f}  clairv {c['c_J'] / 1e9:.1f}"
    if b:
        msg += f"  base-first {b['b_J'] / 1e9:.1f} (status {status}, gap {gap:.4f})"
    return msg + f"  [{time.perf_counter() - t0:.0f} s]"


def run(episodes, agent: str = "agents/anastasiia_hybrid_hub", entropy: int = 111, time_limit: float = 120.0,
        n_jobs: int = 2, out: str = "outputs/level1/diag") -> None:
    eps = [int(e) for e in (episodes if isinstance(episodes, (list, tuple)) else str(episodes).split(","))]
    Path(ROOT / out).mkdir(parents=True, exist_ok=True)
    for msg in Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(episode)(str(ROOT / agent), entropy, n, time_limit, str(ROOT / out)) for n in eps
    ):
        print(msg, flush=True)


def _refine_one(entropy: int, n: int, time_limit: float, out: str) -> str:
    from plan_stats import base_first_plan
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.oracle.lp import build_lp, lp_cents

    path = Path(out) / f"ep{n}.npz"
    z = dict(np.load(path))
    inst, params = task_generator("small")
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "train"))
    model = build_lp(inst, marks, planning_rules=True)
    x, status, gap = base_first_plan(inst, marks, model, time_limit, 2e-3)
    if x is None:
        return f"ep{n}: no plan"
    J = lp_cents(model, x) / 100
    old = (float(z["b_J"]), float(z["b_gap"]))
    lb_old, lb_new = old[0] * (1 - old[1]), J * (1 - gap)
    if J < old[0]:
        z |= {f"b_{k}": v for k, v in _plan_arrays(inst, model, x).items()}
        z["b_J"], z["b_status"] = J, status
    z["b_gap"] = (z["b_J"] - max(lb_old, lb_new)) / abs(z["b_J"])  # the best incumbent against the best bound
    np.savez(path, **z)
    return f"ep{n}: bf {old[0] / 1e9:.1f} (gap {old[1]:.4f}) -> {z['b_J'] / 1e9:.1f} (gap {float(z['b_gap']):.4f})"


def refine(episodes, entropy: int = 111, time_limit: float = 600.0, n_jobs: int = 2,
           out: str = "outputs/level1/diag") -> None:
    """Solve the base-first MILP of saved episodes again with a longer limit; keep the better plan and bound."""
    eps = [int(e) for e in (episodes if isinstance(episodes, (list, tuple)) else str(episodes).split(","))]
    for msg in Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_refine_one)(entropy, n, time_limit, str(ROOT / out)) for n in eps
    ):
        print(msg, flush=True)


BANDS = ((0, 13), (13, 26), (26, 39), (39, 52))


def report(out: str = "outputs/level1/diag", rows: str = "", groups: str = "1|2,3") -> None:
    """Decomposition per group of harm levels (``groups``: '|'-separated, comma lists of levels)."""
    import json

    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator("small")
    N, K = inst.nodes, [c.id for c in inst.commodities]
    level = {r["episode"]: r["stratum"] for r in json.loads(Path(ROOT / rows).read_text())}
    eps = {int(p.stem[2:]): dict(np.load(p)) for p in sorted(Path(ROOT / out).glob("ep*.npz"))}
    eps = {n: z for n, z in eps.items() if "b_J" in z}
    bn = 1e9
    for spec in groups.split("|"):
        lv = {int(s) for s in spec.split(",")}
        zs = [eps[n] for n in sorted(eps) if level.get(n) in lv]
        names = [n for n in sorted(eps) if level.get(n) in lv]
        if not zs:
            continue

        def m(f):
            return float(np.mean([f(z) for z in zs])) / bn

        print(f"\n===== harm level(s) {sorted(lv)}: {len(zs)} episodes {names}")
        aJ, cJ, bJ = m(lambda z: z["a_J"]), m(lambda z: z["c_J"]), m(lambda z: z["b_J"])
        print(f"J bn/ep: agent {aJ:.1f}  clairv {cJ:.1f}  base-first plan {bJ:.1f}")
        print(f"  loss to clairv {aJ - cJ:.1f} = relaxation (bf - clairv) {bJ - cJ:.1f} + loss to bf {aJ - bJ:.1f}")
        lo = m(lambda z: z["b_J"] * (1 - z["b_gap"]))
        print(f"  with the MILP's dual bound: relaxation >= {lo - cJ:.1f}, loss to the base-first optimum <= {aJ - lo:.1f}")
        gaps = [z["b_gap"] for z in zs]
        print(f"  MILP: {sum(z['b_status'] != 0 for z in zs)} stopped by time; largest gap {max(gaps):.4f}")
        print(f"\n{'component':14s} {'agent':>8} {'clairv':>8} {'bf':>8} {'a-c':>7} {'a-bf':>7} {'bf-c':>7}")
        for i, c in enumerate(COMP):
            a, cc, b = (m(lambda z, p=p: z[f"{p}_costs"][:, i].sum()) for p in "acb")
            print(f"{c:14s} {a:8.1f} {cc:8.1f} {b:8.1f} {a - cc:7.1f} {a - b:7.1f} {b - cc:7.1f}")
        a, cc, b = (m(lambda z, p=p: -z[f"{p}_salvage"]) for p in "acb")
        print(f"{'-salvage':14s} {a:8.1f} {cc:8.1f} {b:8.1f} {a - cc:7.1f} {a - b:7.1f} {b - cc:7.1f}")

        print("\nweek bands, bn/ep:   shed a-c  a-bf  bf-c | shortage a-c  a-bf  bf-c | other a-c  a-bf")
        for lo, hi in BANDS:
            def d(p, q, i, lo=lo, hi=hi):
                return m(lambda z: z[f"{p}_costs"][lo:hi, i].sum() - z[f"{q}_costs"][lo:hi, i].sum())

            def o(p, q, lo=lo, hi=hi):
                return m(lambda z: np.delete(z[f"{p}_costs"][lo:hi], [5, 7], axis=1).sum()
                         - np.delete(z[f"{q}_costs"][lo:hi], [5, 7], axis=1).sum())

            print(f"weeks {lo + 1:2d}-{hi:2d}          {d('a', 'c', 7):7.1f} {d('a', 'b', 7):6.1f} {d('b', 'c', 7):6.1f} |"
                  f"     {d('a', 'c', 5):7.1f} {d('a', 'b', 5):6.1f} {d('b', 'c', 5):6.1f} |"
                  f"  {o('a', 'c'):7.1f} {o('a', 'b'):6.1f}")

        print("\ngrids: shed bn/ep a-c  a-bf  bf-c | GWh/wk agent clairv bf | weeks with shed % a / c / bf"
              " | shed bn a-bf by band")
        for go, g in enumerate(inst.grids):
            def s(p, go=go):
                return m(lambda z: z[f"{p}_shed_usd"][:, go].sum())

            q = [m(lambda z, p=p: z[f"{p}_shed_qty"][:, go].mean() * bn) for p in "acb"]
            w = [m(lambda z, p=p: 100 * (z[f"{p}_shed_qty"][:, go] > 1e-6).mean() * bn) for p in "acb"]
            bands = " ".join(
                f"{m(lambda z, lo=lo, hi=hi: z['a_shed_usd'][lo:hi, go].sum() - z['b_shed_usd'][lo:hi, go].sum()):6.1f}"
                for lo, hi in BANDS
            )
            print(f"{N[g].id:10s} {s('a') - s('c'):7.1f} {s('a') - s('b'):6.1f} {s('b') - s('c'):6.1f} | "
                  f"{q[0]:7.0f} {q[1]:7.0f} {q[2]:7.0f} | {w[0]:3.0f} / {w[1]:3.0f} / {w[2]:3.0f} | {bands}")

        print("\nmarkets: shortage bn/ep a-c  a-bf  bf-c | a-bf by band")
        for do, dm in enumerate(inst.demands):
            def s(p, do=do):
                return m(lambda z: z[f"{p}_short_usd"][:, do].sum())

            if s("a") == 0 and s("c") == 0:
                continue
            bands = " ".join(
                f"{m(lambda z, lo=lo, hi=hi: z['a_short_usd'][lo:hi, do].sum() - z['b_short_usd'][lo:hi, do].sum()):6.1f}"
                for lo, hi in BANDS
            )
            print(f"{N[dm.node].id:8s} {K[dm.k]:9s} {s('a') - s('c'):7.1f} {s('a') - s('b'):6.1f} "
                  f"{s('b') - s('c'):6.1f} | {bands}")

        print("\nfabs (weeks 1-40): lots per week, % of nominal capacity: agent / clairv / bf;"
              " clairv lots started while its grid sheds %")
        for fo, f in enumerate(inst.fabs):
            cap = N[f].fab.cap0
            go = inst.grid_ordinal[N[f].fab.grid]
            r = [m(lambda z, p=p: 100 * z[f"{p}_lots"][:40, fo].mean() / cap * bn) for p in "acb"]
            dry = sum(z["c_lots"][:40, fo][z["c_shed_qty"][:40, go] > 1e-6].sum() for z in zs)
            tot = sum(z["c_lots"][:40, fo].sum() for z in zs)
            print(f"{N[f].id:18s} {r[0]:5.1f} / {r[1]:5.1f} / {r[2]:5.1f}   {100 * dry / max(tot, 1e-9):4.0f}%")

        print("\nper episode, bn: ep  a-c  bf-c  a-bf  (a-bf: shed  shortage  other)  MILP gap")
        for n, z in zip(names, zs):
            sh = (z["a_costs"][:, 7].sum() - z["b_costs"][:, 7].sum()) / bn
            st = (z["a_costs"][:, 5].sum() - z["b_costs"][:, 5].sum()) / bn
            print(f"  {n:3d} L{level[n]} {(z['a_J'] - z['c_J']) / bn:7.1f} {(z['b_J'] - z['c_J']) / bn:6.1f} "
                  f"{(z['a_J'] - z['b_J']) / bn:7.1f}   ({sh:6.1f} {st:6.1f} {(z['a_J'] - z['b_J']) / bn - sh - st:6.1f})"
                  f"  {float(z['b_gap']):.4f}")
        good = [z for z in zs if z["b_gap"] < 0.005]
        if good:
            g = np.array([[z["a_J"] - z["c_J"], z["b_J"] - z["c_J"], z["a_J"] - z["b_J"]] for z in good]).mean(axis=0) / bn
            print(f"  MILP gap < 0.5% only ({len(good)} ep): a-c {g[0]:.1f} = relaxation {g[1]:.1f} + a-bf {g[2]:.1f}")


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report, "refine": refine})
