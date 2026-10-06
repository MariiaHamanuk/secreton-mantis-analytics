"""Planner G: simulate-and-fix. Play the window plan through the simulator, read what the automatic steps (lot starts,
fab energy, shed) really did, bound the program by those values and re-solve the flows; keep the plan with the lowest
simulated cost.

    uv run python lab/anastasiia/mpc_lab/planners/planner_G.py first --episodes=3 --mode=fixp
    uv run python lab/anastasiia/mpc_lab/planners/planner_G.py run --variant=fixp:3 --episodes=16 --n_jobs=4
"""

import sys
import time

import fire
import numpy as np

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
import planner_B as B  # noqa: E402
import mpc_foresight as mf  # noqa: E402
from package_baselines import rss  # noqa: E402


def sim_records(h, env, model, x, week0, first_action=None):
    sc = B.scratch(env)
    B.playback(h, sc, model, x, week0, 1, first_action)
    return sc.trajectory.J_cents, sc.trajectory.records


def bound_from_sim(h, model, recs, mode, lb, ub, weeks_max=None, slack=0.0):
    """New (lb, ub): lots p (and energy E) of window week r tied to the simulated values.
    mode letters: p = fix lots, e = fix energy, u = only upper bounds (<= sim), s = shed fixed."""
    ix = model.index
    lb, ub = lb.copy(), ub.copy()
    n = min(len(recs), B.model_weeks(model))
    if weeks_max:
        n = min(n, weeks_max)
    F, G = len(h.inst.fabs), len(h.inst.grids)
    inst = h.inst
    zs = []  # slots whose plan stock is forced to 0 where the simulator leaves none
    if "i" in mode or "o" in mode:
        for f in inst.fabs if "i" in mode else ():
            zs.append(inst.slot_index[(f, inst.nodes[f].fab.input)])
        for o in inst.osats if "o" in mode else ():
            for kr in inst.nodes[o].osat.packages:
                zs.append(inst.slot_index[(o, kr)])
    for i in range(n):
        r = i + 1
        for sl in zs:
            if recs[i].stock[sl] <= 1e-6 * (1 + slack):
                for key in (("I", r, sl), ("O", r, sl)):
                    if key in ix:
                        ub[ix[key]] = 0.0
        for f in range(F):
            if "p" in mode:
                v = float(recs[i].lots_started[f])
                ub[ix[("p", r, f)]] = v * (1 + slack) + 1e-6
                if "u" not in mode:
                    lb[ix[("p", r, f)]] = v * (1 - slack)
            if "e" in mode and ("E", r, f) in ix:
                v = float(recs[i].energy[f])
                ub[ix[("E", r, f)]] = v * (1 + slack) + 1e-6
                if "u" not in mode:
                    lb[ix[("E", r, f)]] = v * (1 - slack)
        if "s" in mode:
            for g in range(G):
                v = float(recs[i].shed[g])
                ub[ix[("ysh", r, g)]] = v * (1 + slack) + 1e-6
                if "u" not in mode:
                    lb[ix[("ysh", r, g)]] = v * (1 - slack)
    return lb, ub


def milp_base(h, pol, obs, gap=0.02, tlimit=20.0):
    """(model, x, None, (lb, ub)): the base-load MILP plan and the bounds that fix its z."""
    base = pol.solve(obs)
    if base is None:
        return None
    model = base[0]
    r = B.milp_plan(h, model, gap=gap, tlimit=tlimit)
    if r is None or r[0] is None:
        return base[0], base[1], None, (model.lb, model.ub)
    zmap = {(int(obs["week"]) + rr - 1, g): v for (rr, g), v in r[2].items()}
    bd = B.z_bounds(h, model, zmap, int(obs["week"]))
    return model, r[0], None, (bd if bd is not None else (model.lb, model.ub))


def loop(h, pol, env, obs, mode="p", rounds=3, weeks_max=None, slack=0.0, base=None, verbose=False):
    """Returns (model, best x, history of simulated costs [round 0 = plan as is])."""
    week = int(obs["week"])
    base = base if base is not None else pol.solve(obs)
    model, x = base[0], base[1]
    blb, bub = base[3] if len(base) > 3 and base[3] is not None else (model.lb, model.ub)
    c, recs = sim_records(h, env, model, x, week)
    best, bx = c, x
    hist = [c]
    for _ in range(rounds):
        lb2, ub2 = bound_from_sim(h, model, recs, mode, blb, bub, weeks_max, slack)
        x2 = pol.lp(model, lb2, ub2)
        if x2 is None:
            hist.append(None)
            break
        c, recs = sim_records(h, env, model, x2, week)
        hist.append(c)
        if c < best:
            best, bx = c, x2
    return model, bx, hist


def first(episodes=3, milp=False, task="small", entropy=111, horizon=52, mode="p", rounds=3, weeks_max=0, slack=0.0, uk=False):
    from shockbench_flow.oracle.lp import lp_costs

    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, max(episodes, 2), entropy=entropy, n_jobs=2).references)
    for n in range(episodes):
        nv, orc = refs[n]["J_naive_cents"], refs[n]["J_oracle_cents"]
        sc = lambda c: (nv - c) / (nv - orc)
        h = make_h(task, entropy, n, horizon)
        env, pol, obs = B.start(h)
        t0 = time.process_time()
        base = milp_base(h, pol, obs) if milp else pol.solve(obs)
        model, x = base[0], base[1]
        weekly, credit = lp_costs(model, x)
        claim = sum(w.total() for w in weekly) - credit
        for mode_ in str(mode).split(";"):
            _, bx, hist = loop(h, pol, env, obs, mode_, rounds, weeks_max or None, slack, base=base)
            print(f"ep {n} mode {mode_:6s} claim {sc(claim*100):.3f}  sim score by round "
                  + " ".join("None" if c is None else f"{sc(c):.3f}" for c in hist), flush=True)


def make_h(task, entropy, n, horizon, known="everything"):
    return B.make(task, entropy, n, horizon, known)


def zfix_base(h, pol, obs, every, stats):
    week = int(obs["week"])
    out = B.zfix_plan(h, pol, obs, every, stats)
    if out is None:
        return None
    model, x = out
    bd = B.z_bounds(h, model, pol.zmap, week) if getattr(pol, "zmap", None) else None
    return model, x, None, (bd if bd is not None else (model.lb, model.ub))


def episode_cost(variant, task, entropy, n, horizon, known="everything"):
    h = make_h(task, entropy, n, horizon, known)
    env, pol, obs = B.start(h)
    kind, _, arg = variant.partition(":")
    every, mode, rounds = (arg.split(",") + ["4", "i", "3"])[:3] if arg else ("4", "i", "3")
    every, rounds = int(every), int(rounds)
    stats = {"trials": 0, "accepted": 0}
    L = h.L
    times, done = [], False
    while not done:
        t0 = time.process_time()
        if kind == "mpc":
            a = pol.act(obs)
        else:
            base = zfix_base(h, pol, obs, every, stats) if kind.startswith("z") else pol.solve(obs)
            if base is None:
                a = pol._fallback.act(obs)
            else:
                model, x, _ = loop(h, pol, env, obs, mode, rounds, base=base)
                a = L.week1_action(h.inst, model, x, obs, L.prohibited_now(pol._memory, int(obs["week"])))
        times.append(time.process_time() - t0)
        obs, _, done, _, _ = env.step(a)
    return int(env.trajectory.J_cents), times


def run(variant="mpc", episodes=16, task="small", entropy=111, horizon=52, n_jobs=4, known="everything"):
    from joblib import Parallel, delayed
    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    t0 = time.perf_counter()
    out = Parallel(n_jobs=n_jobs)(delayed(episode_cost)(variant, task, entropy, n, horizon, known) for n in range(episodes))
    score, by_level = rss(refs, [o[0] for o in out])
    ts = np.concatenate([o[1] for o in out])
    print(f"{variant} {task} {episodes} ep h{horizon} known={known}: score {score:.4f} levels "
          + " ".join(f"{v:.3f}" for v in by_level.values())
          + f" | cpu/week mean {ts.mean():.2f} max {ts.max():.2f} | {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    fire.Fire({"first": first, "run": run})
