"""Planner B: judge the window LP's plan by the simulator itself (open-loop playback), and correct week 1 from that.

    uv run python lab/anastasiia/mpc_lab/planners/planner_B.py diag --episodes=2
    uv run python lab/anastasiia/mpc_lab/planners/planner_B.py run --variant=cand --episodes=16 --n_jobs=2
"""

import sys
import time

import fire
import numpy as np

sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
import mpc_foresight as mf  # noqa: E402
from package_baselines import rss  # noqa: E402

COMP = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def plan_action(inst, model, x, r, Z, week):
    """Wire action of window-relative week r (1-based) from the LP solution ``x``; Z = prohibited (E, K) that week."""
    nc = len(model.columns)
    off = (r - 1) * nc
    tmpl = model.meta["template"]
    fs, fq = [], []
    for s, (e, k, lane) in enumerate(inst.action_slots):
        if Z[e, k]:
            continue
        fs.append(s)
        fq.append(max(float(x[off + tmpl[("x", e, k, lane)]]), 0.0))
    first = {}
    for o, (c, k, e, _l) in enumerate(inst.override_slots):
        first.setdefault((c, k, e), o)
    sendable, ov = {}, []
    for (c, k, e), o in first.items():
        j = tmpl.get(("x", e, k, None))
        if j is None:
            continue
        ok = not bool(Z[e, k])
        sendable[(c, k)] = sendable.get((c, k), False) or ok
        if ok:
            ov.append((o, max(float(x[off + j]), 0.0)))
    ov.sort()
    holds = [ck for ck, ok in sendable.items() if not ok]
    return {
        "week": int(week),
        "flows": {"slot": fs, "qty": fq},
        "overrides": {"slot": [o for o, _ in ov], "qty": [q for _, q in ov]} if ov else None,
        "hold": {"chokepoint": [c for c, _ in holds], "k": [k for _, k in holds]} if holds else None,
    }


def make(task, entropy, n, horizon, known_name="everything"):
    """Build the world and an MpcDet-based planner. Returns a namespace of handles."""
    from scipy.optimize import linprog
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    known = mf.GROUPS[known_name]
    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    q = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    ctx = PolicyContext(fq_quantile=q, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {w: i for i, w in L.NOW_FIELDS}

    class P(MpcDet):
        def _horizon(self, inst, plan):
            return horizon

        def solve(self, obs):
            """LP of the window of this week: (model, x, weeks) or None."""
            inst, week = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            arrays = {k: a.copy() for k, a in L.persistence_arrays(inst, obs, self._memory, weeks).items()}
            f = week - 1
            for name in known:
                arrays[name] = np.array(getattr(marks, name)[f : f + weeks])
                if name in now:
                    arrays[now[name]] = np.array(getattr(marks, now[name])[f : f + weeks])
            model = L.rolled_lp(inst, obs, L.read_only(arrays), weeks, planning_rules=True)
            x = self.lp(model)
            return None if x is None else (model, x, weeks)

        @staticmethod
        def lp(model, lb=None, ub=None):
            kw = {}
            if model.A_ub.shape[0]:
                kw.update(A_ub=model.A_ub, b_ub=model.b_ub)
            if model.A_eq.shape[0]:
                kw.update(A_eq=model.A_eq, b_eq=model.b_eq)
            lb = model.lb if lb is None else lb
            ub = model.ub if ub is None else ub
            res = linprog(model.objective(), bounds=np.column_stack([lb, ub]), method="highs-ds", **kw)
            if res.status != 0 or res.x is None:
                return None
            return np.asarray(res.x)

        def act(self, obs):
            out = self.solve(obs)
            if out is None:
                return self._fallback.act(obs)
            model, x, _ = out
            return L.week1_action(self._inst, model, x, obs, L.prohibited_now(self._memory, int(obs["week"])))

    class H:
        pass

    h = H()
    h.inst, h.omega, h.marks, h.fallback = inst, omega, marks, fallback
    h.ctx, h.L, h.P, h.Env = ctx, L, P, Env
    h.seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    return h


def start(h):
    from shockbench_flow.policies.base import reset_policy

    env = h.Env(fallback=h.fallback)
    pol = h.P(None, h.ctx)
    obs, info = env.reset(h.inst, "standard", h.omega, h.seed, marks=h.marks, policy_name=pol.name)
    reset_policy(pol, info["static"], obs, h.seed, info.get("omega"))
    return env, pol, obs


def scratch(env):
    """An independent Env on a copy of env's current episode (no snapshot/restore, so no naive re-solve)."""
    s = type(env)(fallback=None)
    s._ep = env._ep.copy()
    s.fallback = None
    return s


def playback(h, sc, model, x, week0, r0, first_action=None):
    """Play the plan's weeks r0.. (window-relative) open-loop on the scratch env; returns (cost USD incl. salvage, comps)."""
    from shockbench_flow.dynamics.sim import terminal_salvage

    comps = np.zeros((h.inst.T - week0 + 1, len(COMP)))
    total = 0.0
    w = week0 + r0 - 1
    r = r0
    done = False
    while not done:
        if first_action is not None and r == r0:
            a = first_action
        else:
            a = plan_action(h.inst, model, x, r, h.marks.prohibited[w - 1], w)
        _, rew, done, _, info = sc.step(a)
        total -= rew * 100  # reward is cents/100 -> USD? see below
        comps[w - week0] = [info["cost_components"][c] for c in COMP]
        w += 1
        r += 1
    return total, comps


def plan_vs_sim(h, model, x, sc_env, t_real, r0=1):
    """Per week: plan and simulated lots by fab, energy by fab, shed by grid (arrays (weeks, n))."""
    inst = h.inst
    recs = sc_env.trajectory.records
    n = len(recs)
    F, G = len(inst.fabs), len(inst.grids)
    ix = model.index
    pl = {"p": np.zeros((n, F)), "E": np.zeros((n, F)), "ysh": np.zeros((n, G))}
    sm = {"p": np.zeros((n, F)), "E": np.zeros((n, F)), "ysh": np.zeros((n, G))}
    for i in range(n):
        r = r0 + i
        for f in range(F):
            pl["p"][i, f] = x[ix[("p", r, f)]]
            if ("E", r, f) in ix:
                pl["E"][i, f] = x[ix[("E", r, f)]]
        for g in range(G):
            pl["ysh"][i, g] = x[ix[("ysh", r, g)]]
        sm["p"][i] = recs[i].lots_started
        sm["E"][i] = recs[i].energy
        sm["ysh"][i] = recs[i].shed
    return pl, sm


def sim_cost(h, env, model, x, week0, first_action=None):
    """Simulated total cost (cents) of playing the plan ``x`` open-loop from the env's current state; and the Env."""
    sc = scratch(env)
    playback(h, sc, model, x, week0, 1, first_action)
    return sc.trajectory.J_cents, sc


def flagged(h, model, x, week0):
    """Boolean (weeks, G): base_first grid-weeks (window weeks >= 2) where the plan sheds base load AND powers fabs."""
    inst = h.inst
    ix = model.index
    G = len(inst.grids)
    n = model_weeks(model)
    out = np.zeros((n, G), bool)
    for r in range(2, n + 1):
        for g in range(G):
            if inst.nodes[inst.grids[g]].grid.priority != "base_first":
                continue
            fo = inst.grid_fabs[g]
            E = sum(x[ix[("E", r, f)]] for f in fo if ("E", r, f) in ix)
            if x[ix[("ysh", r, g)]] > 1e-6 and E > 1e-6:
                out[r - 1, g] = True
    return out


def model_weeks(model):
    return len(model.lb) // len(model.columns)


def apply_fix(h, model, flags, mode, lb, ub):
    """Copies of lb, ub with flagged grid-weeks fixed: 'A' fabs of the grid get no energy or lots, 'B' no shed."""
    ix = model.index
    lb, ub = lb.copy(), ub.copy()
    for r0, g in zip(*np.nonzero(flags)):
        r = r0 + 1
        if mode == "A":
            for f in h.inst.grid_fabs[g]:
                ub[ix[("E", r, f)]] = 0.0
                ub[ix[("p", r, f)]] = 0.0
        else:
            ub[ix[("ysh", r, g)]] = 0.0
    return lb, ub


def refine(h, pol, env, obs, rounds=2, stats=None):
    """Greedy: per grid with flagged weeks try A and B (cumulative), keep the lower simulated cost."""
    week = int(obs["week"])
    model, x, weeks = pol.solve(obs)
    best, _ = sim_cost(h, env, model, x, week)
    lb, ub = model.lb, model.ub
    for _ in range(rounds):
        fl = flagged(h, model, x, week)
        if not fl.any():
            break
        improved = False
        for g in np.nonzero(fl.any(axis=0))[0]:
            sub = np.zeros_like(fl)
            sub[:, g] = fl[:, g]
            for mode in "AB":
                lb2, ub2 = apply_fix(h, model, sub, mode, lb, ub)
                x2 = pol.lp(model, lb2, ub2)
                if x2 is None:
                    continue
                c, _ = sim_cost(h, env, model, x2, week)
                if stats is not None:
                    stats["trials"] += 1
                if c < best:
                    best, x, lb, ub, improved = c, x2, lb2, ub2, True
                    if stats is not None:
                        stats["accepted"] += 1
        if not improved:
            break
    return model, x


def milp_plan(h, model, gap=0.01, tlimit=20.0, weeks_from=2):
    """The window LP plus, per base_first grid-week (window week >= weeks_from), a binary z: fabs powered (z=1) only
    when no base load is shed:  E_f <= Gbar z,  ysh <= ybar (1 - z).  Solved with HiGHS MIP. Returns x or None."""
    import scipy.sparse as sp
    from scipy.optimize import LinearConstraint, Bounds, milp

    inst = h.inst
    ix = model.index
    nc = len(model.lb)
    n = model_weeks(model)
    zs, rows_ub, rhs = [], [], []  # entries (row, col, val)
    ent, b = [], []
    nz = 0
    zkeys = []
    for r in range(weeks_from, n + 1):
        for g in range(len(inst.grids)):
            ga = inst.nodes[inst.grids[g]].grid
            fabs = [f for f in inst.grid_fabs[g] if ("E", r, f) in ix and inst.nodes[inst.fabs[f]].fab.e > 0]
            if ga.priority != "base_first" or not fabs:
                continue
            Gbar = float(model.ub[ix[("G", r, g, None)]]) if False else None
            zc = nc + nz
            nz += 1
            zkeys.append((r, g, [ix[('E', r, f)] for f in fabs], ix[('ysh', r, g)]))
            # ybar from the base row's upper bound of ysh/y: ub of ysh is inf; use y's rhs through ga.base_load
            ybar = ga.base_load
            # E_f - M z <= 0, M = deliverable * max-ish
            M = ga.deliverable * 1.5
            for f in fabs:
                ent += [(len(b), ix[("E", r, f)], 1.0), (len(b), zc, -M)]
                b.append(0.0)
            ent += [(len(b), ix[("ysh", r, g)], 1.0), (len(b), zc, ybar)]
            b.append(ybar)
    if nz == 0:
        return None
    rr, cc, vv = zip(*ent)
    Az = sp.csr_matrix((vv, (rr, cc)), shape=(len(b), nc + nz))
    A_ub = sp.vstack([sp.hstack([model.A_ub, sp.csr_matrix((model.A_ub.shape[0], nz))]), Az]).tocsr()
    b_ub = np.concatenate([model.b_ub, b])
    A_eq = sp.hstack([model.A_eq, sp.csr_matrix((model.A_eq.shape[0], nz))]).tocsr()
    c = np.concatenate([model.objective(), np.zeros(nz)])
    lo = np.concatenate([model.lb, np.zeros(nz)])
    hi = np.concatenate([model.ub, np.ones(nz)])
    integ = np.concatenate([np.zeros(nc), np.ones(nz)])
    cons = [LinearConstraint(A_ub, -np.inf, b_ub), LinearConstraint(A_eq, model.b_eq, model.b_eq)]
    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lo, hi),
               options={"mip_rel_gap": gap, "time_limit": tlimit})
    if res.x is None:
        return None, nz, {}
    return res.x[:nc], nz, {(k[0], k[1]): int(round(v)) for k, v in zip(zkeys, res.x[nc:])}


def z_bounds(h, model, zmap, week):
    """lb, ub from a map {(absolute week, grid): z}: z=0 -> fabs of the grid get no energy or lots; z=1 -> no shed.
    Returns None when a base_first grid-week of the window is missing from zmap."""
    inst = h.inst
    ix = model.index
    lb, ub = model.lb.copy(), model.ub.copy()
    for r in range(2, model_weeks(model) + 1):
        for g in range(len(inst.grids)):
            ga = inst.nodes[inst.grids[g]].grid
            fabs = [f for f in inst.grid_fabs[g] if ("E", r, f) in ix and inst.nodes[inst.fabs[f]].fab.e > 0]
            if ga.priority != "base_first" or not fabs:
                continue
            z = zmap.get((week + r - 1, g))
            if z is None:
                return None
            if z == 0:
                for f in fabs:
                    ub[ix[("E", r, f)]] = 0.0
                    ub[ix[("p", r, f)]] = 0.0
            else:
                ub[ix[("ysh", r, g)]] = 0.0
    return lb, ub


def zfix_plan(h, pol, obs, every, stats):
    """MILP every ``every`` weeks (and when stale/infeasible); in between the LP with z fixed from the last MILP."""
    week = int(obs["week"])
    z = getattr(pol, "zmap", {})
    base = pol.solve(obs)
    if base is None:
        return None
    model = base[0]
    if (week - 1) % every != 0 or not z:
        bd = z_bounds(h, model, z, week) if z else None
        if bd is not None and (week - 1) % every != 0:
            x = pol.lp(model, *bd)
            if x is not None:
                stats["trials"] += 1
                return model, x
    r = milp_plan(h, model)
    stats["accepted"] += 1
    if r is None or r[0] is None:
        return base[0], base[1]
    for (rr, g), v in r[2].items():
        z[(week + rr - 1, g)] = v
    pol.zmap = z
    return model, r[0]


def diag(episodes=2, task="small", entropy=111, horizon=52):
    from shockbench_flow.oracle.lp import lp_costs

    for n in range(episodes):
        h = make(task, entropy, n, horizon)
        env, pol, obs = start(h)
        t0 = time.process_time()
        model, x, weeks = pol.solve(obs)
        print("solve s", round(time.process_time() - t0, 2))
        weekly, credit = lp_costs(model, x)
        claim = sum(w.total() for w in weekly) - credit
        lpcomp = np.array([[w.as_dict()[c] for c in COMP] for w in weekly])
        c0, _ = sim_cost(h, env, model, x, 1)
        t0 = time.time()
        r = milp_plan(h, model)
        print("MILP s", round(time.time() - t0, 1), "nz", r[1] if r else None)
        if r and r[0] is not None:
            c1, _ = sim_cost(h, env, model, r[0], 1)
            print(f"  LP-plan sim {c0/1e11:.1f}B  MILP-plan sim {c1/1e11:.1f}B")
        t0 = time.time()
        rf = refine(h, pol, env, obs)
        print("refine s", round(time.time() - t0, 1))
        c2, _ = sim_cost(h, env, rf[0], rf[1], 1)
        print(f"  refine-plan sim {c2/1e11:.1f}B")
        sc = scratch(env)
        t0 = time.process_time()
        total, comps = playback(h, sc, model, x, 1, 1)
        print("playback s", round(time.process_time() - t0, 2))
        pl, sm = plan_vs_sim(h, model, x, sc, 1)
        for g in range(len(h.inst.grids)):
            fo = h.inst.grid_fabs[g]
            ga = h.inst.nodes[h.inst.grids[g]].grid
            print(f"  grid {g} {ga.priority:16s} fabs {list(fo)}: shed plan {pl['ysh'][:, g].sum():9.0f} sim {sm['ysh'][:, g].sum():9.0f}"
                  f" | fab energy plan {pl['E'][:, fo].sum():9.0f} sim {sm['E'][:, fo].sum():9.0f}"
                  f" | lots plan {pl['p'][:, fo].sum():9.0f} sim {sm['p'][:, fo].sum():9.0f}")
        nog = [f for f in range(len(h.inst.fabs)) if h.inst.nodes[h.inst.fabs[f]].fab.grid is None]
        print("  fabs w/o grid", nog, "lots plan", pl["p"][:, nog].sum(), "sim", sm["p"][:, nog].sum())
        done = False
        while not done:
            obs, r, done, _, inf = env.step(pol.act(obs))
        J = env.trajectory.J_cents / 100
        B = 1e9
        print(f"ep {n}: LP claim {claim/B:.1f}  open-loop sim {total/B:.1f}  closed-loop MPC {J/B:.1f}  (sim J {sc.trajectory.J_cents/100/B:.1f})")
        real = np.array([[rec.costs.as_dict()[c] for c in COMP] for rec in env.trajectory.records])
        print("  component  LP-claim  open-loop  closed-loop ($B)")
        for i, c in enumerate(COMP):
            print(f"  {c:14s} {lpcomp[:, i].sum()/B:9.1f} {comps[:, i].sum()/B:9.1f} {real[:, i].sum()/B:9.1f}")
        q = np.linspace(0, len(lpcomp), 5).astype(int)
        print("  by quarter LP-claim:", [round(lpcomp[a:b].sum() / B, 1) for a, b in zip(q[:-1], q[1:])])
        print("  by quarter open-loop:", [round(comps[a:b].sum() / B, 1) for a, b in zip(q[:-1], q[1:])])
        print("  by quarter closed   :", [round(real[a:b].sum() / B, 1) for a, b in zip(q[:-1], q[1:])])


def episode_cost(variant, task, entropy, n, horizon, known="everything"):
    """(J cents, per-week CPU seconds list, stats) of one closed-loop episode of the variant, everything known."""
    h = make(task, entropy, n, horizon, known)
    env, pol, obs = start(h)
    times, done = [], False
    stats = {"trials": 0, "accepted": 0}
    while not done:
        t0 = time.process_time()
        if variant == "mpc":
            a = pol.act(obs)
        else:
            if variant.startswith("milp"):
                gap = float(variant.split(":")[1]) if ":" in variant else 0.02
                base = pol.solve(obs)
                x = None
                if base is not None:
                    r = milp_plan(h, base[0], gap=gap)
                    x = r[0] if r else None
                out = None if base is None else (base[0], x if x is not None else base[1])
            elif variant.startswith("zfix"):
                every = int(variant.split(":")[1]) if ":" in variant else 4
                out = zfix_plan(h, pol, obs, every, stats)
            else:
                out = refine(h, pol, env, obs, stats=stats)
            L = h.L
            if out is None:
                a = pol._fallback.act(obs)
            else:
                a = L.week1_action(h.inst, out[0], out[1], obs, L.prohibited_now(pol._memory, int(obs["week"])))
        times.append(time.process_time() - t0)
        obs, _, done, _, _ = env.step(a)
    return int(env.trajectory.J_cents), times, stats


def run(variant="mpc", episodes=16, task="small", entropy=111, horizon=52, n_jobs=2, known="everything"):
    from joblib import Parallel, delayed
    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    start_t = time.perf_counter()
    out = Parallel(n_jobs=n_jobs)(delayed(episode_cost)(variant, task, entropy, n, horizon, known) for n in range(episodes))
    score, by_level = rss(refs, [o[0] for o in out])
    ts = np.concatenate([o[1] for o in out])
    st = {k: sum(o[2][k] for o in out) for k in ("trials", "accepted")}
    print(f"{variant} {task} {episodes} ep horizon {horizon} known={known}: score {score:.4f} levels "
          + " ".join(f"{v:.3f}" for v in by_level.values())
          + f" | cpu/week mean {ts.mean():.2f} max {ts.max():.2f} | {st} | {time.perf_counter()-start_t:.0f}s", flush=True)


if __name__ == "__main__":
    fire.Fire({"diag": diag, "run": run})
