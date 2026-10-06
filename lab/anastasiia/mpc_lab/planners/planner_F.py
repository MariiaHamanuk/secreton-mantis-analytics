"""Planner F: regimes of base_first grids fixed cheaply, so the window LP stops counting on fab energy that base load will take.

    uv run python lab/anastasiia/mpc_lab/planners/planner_F.py --variants="base;carry:K=8" --episodes=16 --horizon=52 --known=everything

Regime of a (base_first grid with fabs, week t >= 2): short = shed base load (fabs get nothing: E <= 0), served = no shed.
Variants (``name:key=value,...``):
  base                   the unchanged planner
  carry:K,srv,conf,res   the regime of each future grid-week is carried from the previous week's plan (one LP per week).
                         K: weeks beyond week 1 on which carried regimes are imposed (default 52); srv=1: also impose
                         "served" (ysh <= 0), with a fallback to short-only when infeasible; conf: how a grid-week of the
                         new plan that both sheds and powers a fab is classified (1 short, 0 served, -1 left free);
                         res=R: R extra rounds that fix such conflicts for this week and re-solve.
  seed:K,srv,res         as carry, but the first LP of every week is solved relaxed and its week-1 regime persists K
                         weeks (grids short in week 1 are short in weeks 2..K+1), then one more LP (two solves a week).
Prints also the share of grid-weeks where week 2 of the plan has the regime that week 1 of the next week's plan has.
"""

import time

import fire
import numpy as np
from joblib import Parallel, delayed
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))  # the bench is one folder up
from package_baselines import rss

import mpc_foresight as MF
from planner_C import cols_of, parse, solve

EPS = 1e-6


class Regimes:
    def __init__(self, cfg, name):
        self.cfg, self.name = cfg, name
        self.mem = {}  # (go, absolute week) -> +1 served / -1 short
        self.prev = None  # (week, {go: regime of week 2 of the plan})
        self.agree = [0, 0]
        self.nsolve = 0

    def rf(self, model, inst, week, arrays):
        """Relax-and-fix: D's rule rows with continuous s per (grid, week), rounded in ``rounds`` LP solves."""
        from scipy import sparse

        cfg = self.cfg
        T = model.T
        N = model.cost.shape[0]
        ysh, E, lam = cols_of(model, "ysh"), cols_of(model, "E"), cols_of(model, "lam")
        gs = [go for go, g in enumerate(inst.grids) if inst.nodes[g].grid.priority == "base_first" and inst.grid_fabs[go]]
        B = int(cfg.get("B", 52))
        rounds = int(cfg.get("rounds", 2))
        tol = cfg.get("tol", 0.05)
        thr = cfg.get("thr", 0.5)
        ybar, alpha = np.asarray(arrays["y_bar"]), np.asarray(arrays["alpha_bar"])
        cells = [(go, t) for go in gs for t in range(2, min(T, B + 1) + 1)]
        m = len(cells)
        if m == 0:
            res = solve(model, model.objective(), model.lb, model.ub, model.A_ub, model.b_ub)
            return np.asarray(res.x)[:N] if res.status == 0 else None
        rr, cc, vv, rhs = [], [], [], []
        for i, (go, t) in enumerate(cells):
            sj = N + i
            Em = sum(inst.nodes[inst.fabs[fo]].fab.e * alpha[t - 1, fo] * inst.nodes[inst.fabs[fo]].fab.cap0 for fo in inst.grid_fabs[go])
            yb = float(ybar[t - 1, go])
            r = 3 * i
            rr += [r, r]; cc += [ysh[(go,)][t], sj]; vv += [1.0, -yb]; rhs.append(0.0)
            for fo in inst.grid_fabs[go]:
                rr.append(r + 1); cc.append(E[(fo,)][t]); vv.append(1.0)
            rr.append(r + 1); cc.append(sj); vv.append(Em); rhs.append(Em)
            rr += [r + 2, r + 2]; cc += [sj, lam[(go,)][t]]; vv += [1.0, -1.0]; rhs.append(0.0)
        add = sparse.csr_matrix((vv, (rr, cc)), shape=(3 * m, N + m))
        A = sparse.vstack([sparse.hstack([model.A_ub, sparse.csr_matrix((model.A_ub.shape[0], m))]), add]).tocsr()
        b = np.concatenate([model.b_ub, rhs])
        c = np.concatenate([model.objective(), np.zeros(m)])
        lb = np.concatenate([model.lb, np.zeros(m)])
        ub = np.concatenate([model.ub, np.ones(m)])
        for k in range(rounds):
            self.nsolve += 1
            res = solve(model, c, lb, ub, A, b)
            if res.status != 0:
                break
            x = np.asarray(res.x)
            sv = x[N:]
            self.frac = getattr(self, "frac", [0, 0]); self.frac[0] += int(np.sum((sv > 1e-6) & (sv < 1 - 1e-6))); self.frac[1] += m
            last = k == rounds - 1
            free = lb[N:] != ub[N:]
            if last:
                tgt = np.where(sv >= thr, 1.0, 0.0)
                lb[N:] = np.where(free, tgt, lb[N:]); ub[N:] = np.where(free, tgt, ub[N:])
            else:
                hi, lo = free & (sv >= 1 - tol), free & (sv <= tol)
                lb[N:][hi] = 1.0
                ub[N:][lo] = 0.0
                if not (hi.any() or lo.any()):
                    last = True
                    tgt = np.where(sv >= thr, 1.0, 0.0)
                    lb[N:] = np.where(free, tgt, lb[N:]); ub[N:] = np.where(free, tgt, ub[N:])
            if last:
                self.nsolve += 1
                res = solve(model, c, lb, ub, A, b)
                if res.status != 0:  # served cells infeasible: keep only the "short" fixings
                    lb2, ub2 = lb.copy(), ub.copy()
                    lb2[N:] = np.where(ub2[N:] == 0.0, 0.0, lb2[N:])
                    ub2[N:] = np.where(lb2[N:] == 1.0, 1.0, 1.0 - 0.0)
                    ub2[N:] = np.where(lb[N:] == 1.0, 1.0, 1.0)
                    lb2[N:] = np.where(lb[N:] == 1.0, 1.0, 0.0)
                    res = solve(model, c, lb2, ub2, A, b)
                    self.fallbacks = getattr(self, "fallbacks", 0) + 1
                break
        if res.status != 0:
            self.nsolve += 1
            res = solve(model, model.objective(), model.lb, model.ub, model.A_ub, model.b_ub)
            self.fallbacks = getattr(self, "fallbacks", 0) + 1
            if res.status != 0:
                return None
        x = np.asarray(res.x)
        gs1 = {go: x[ysh[(go,)][1]] > EPS for go in gs}
        if self.prev is not None and self.prev[0] == week - 1:
            for go in gs:
                self.agree[0] += int(self.prev[1][go] == gs1[go]); self.agree[1] += 1
        if T >= 2:
            self.prev = (week, {go: x[ysh[(go,)][2]] > EPS for go in gs})
        return x[:N]

    def step(self, model, inst, week):
        cfg = self.cfg
        nc, T = model.meta["nc"], model.T
        N = model.cost.shape[0]
        c = model.objective().copy()
        if int(cfg.get("pr", 1)) == 0:  # prices of the planning rules only in week 1, except on a fuel shortfall
            pr = model.meta["priority"]
            isshort = np.tile([k[0] == "short" for k in model.columns], T)
            keep = (np.arange(N) < nc) | isshort
            c = c - pr + np.where(keep, pr, 0.0)
        lb, ub = model.lb.copy(), model.ub.copy()
        A_ub, b_ub = model.A_ub, model.b_ub
        ysh, E = cols_of(model, "ysh"), cols_of(model, "E")
        gs = [go for go, g in enumerate(inst.grids) if inst.nodes[g].grid.priority == "base_first" and inst.grid_fabs[go]]
        K = int(cfg.get("K", 52))
        srv = int(cfg.get("srv", 0))

        def run(ubv):
            self.nsolve += 1
            return solve(model, c, lb, ubv, A_ub, b_ub)

        def regime_of(x, go, t):
            s = x[ysh[(go,)][t]] > EPS
            e = sum(x[E[(fo,)][t]] for fo in inst.grid_fabs[go]) > EPS
            return s, e

        def apply(ubv, cells, with_served):
            for (go, t), r in cells.items():
                if r < 0:
                    for fo in inst.grid_fabs[go]:
                        ubv[E[(fo,)][t]] = 0.0
                elif with_served:
                    ubv[ysh[(go,)][t]] = 0.0

        if self.name == "seed":
            first = run(ub)
            if first.status != 0:
                return None
            x0 = np.asarray(first.x)
            cells = {}
            for go in gs:
                s1, _ = regime_of(x0, go, 1)
                for t in range(2, min(T, K + 1) + 1):
                    cells[(go, t)] = -1 if s1 else +1
        elif self.name == "orc":
            cells = {(go, t): (-1 if self.oracle[(go, week + t - 1)] else +1) for go in gs for t in range(2, min(T, K + 1) + 1)
                     if (go, week + t - 1) in self.oracle}
        else:
            cells = {}
            for go in gs:
                for t in range(2, min(T, K + 1) + 1):
                    r = self.mem.get((go, week + t - 1))
                    if r is not None:
                        cells[(go, t)] = r
        ub2 = ub.copy()
        apply(ub2, cells, srv)
        res = run(ub2)
        if res.status != 0 and srv:
            ub2 = ub.copy()
            apply(ub2, cells, False)
            res = run(ub2)
        if res.status != 0:
            cells = {}
            res = run(ub)
            if res.status != 0:
                return None
        x = np.asarray(res.x)
        for _ in range(int(cfg.get("res", 0))):
            conf = {}
            for go in gs:
                for t in range(2, T + 1):
                    if (go, t) in cells:
                        continue
                    s, e = regime_of(x, go, t)
                    if s and e:
                        conf[(go, t)] = -1 if cfg.get("conf", 1) >= 1 else +1
            if not conf:
                break
            cells.update(conf)
            ub2 = ub.copy()
            apply(ub2, cells, srv)
            r2 = run(ub2)
            if r2.status != 0:
                ub2 = ub.copy()
                apply(ub2, cells, False)
                r2 = run(ub2)
            if r2.status != 0:
                break
            x = np.asarray(r2.x)
        # diagnostics: week 1 of this plan vs week 2 of the previous
        wk1 = {go: regime_of(x, go, 1)[0] for go in gs}
        if self.prev is not None and self.prev[0] == week - 1:
            for go in gs:
                self.agree[0] += int(self.prev[1][go] == wk1[go])
                self.agree[1] += 1
        if T >= 2:
            self.prev = (week, {go: regime_of(x, go, 2)[0] for go in gs})
        # carry
        if self.name == "carry":
            conf = int(cfg.get("conf", 1))
            self.mem = {k: v for k, v in self.mem.items() if k[1] > week}
            for go in gs:
                for t in range(2, T + 1):
                    if (go, t) in cells:
                        self.mem[(go, week + t - 1)] = cells[(go, t)]
                        continue
                    s, e = regime_of(x, go, t)
                    if s and e:
                        r = conf if conf != -1 else None
                        r = None if r is None else (-1 if r >= 1 else +1)
                    elif s:
                        r = -1
                    elif e:
                        r = +1
                    else:
                        r = None
                    if r is None:
                        self.mem.pop((go, week + t - 1), None)
                    else:
                        self.mem[(go, week + t - 1)] = r
        return x[:N]


def episode(known, task, entropy, n, horizon, spec):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world

    name, cfg = parse(spec)
    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    quantiles = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    context = PolicyContext(fq_quantile=quantiles, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {week: instant for instant, week in L.NOW_FIELDS}
    times = []
    reg = Regimes(cfg, name)
    if name == "orc":
        import pickle

        reg.oracle = pickle.load(open(f"oracle_regimes_{n}.pkl", "rb"))

    class Fixed(MpcDet):
        def _horizon(self, inst, plan):
            return horizon

        def _window_arrays(self, inst, obs, weeks):
            arrays = {k: a.copy() for k, a in super()._window_arrays(inst, obs, weeks).items()}
            first = int(obs["week"]) - 1
            for fld in known:
                arrays[fld] = np.array(getattr(marks, fld)[first : first + weeks])
                if fld in now:
                    arrays[now[fld]] = np.array(getattr(marks, now[fld])[first : first + weeks])
            return L.read_only(arrays)

        def act(self, obs):
            inst, week = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            start = time.process_time()
            model = L.rolled_lp(inst, obs, self._window_arrays(inst, obs, weeks), weeks, planning_rules=True)
            if name == "base":
                res = solve(model, model.objective(), model.lb, model.ub, model.A_ub, model.b_ub)
                x = np.asarray(res.x) if res.status == 0 else None
            elif name == "rf":
                x = reg.rf(model, inst, week, self._window_arrays(inst, obs, weeks))
            else:
                x = reg.step(model, inst, week)
            times.append(time.process_time() - start)
            if x is None:
                return self._fallback.act(obs)
            return L.week1_action(inst, model, x, obs, L.prohibited_now(self._memory, week))

    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    J = int(rollout(inst, Fixed(None, context), omega, "standard", seed, marks=marks, fallback=fallback).J_cents)
    return J, float(np.mean(times)), float(np.max(times)), reg.agree, reg.nsolve / max(1, len(times))


def main(task="small", episodes=16, entropy=111, horizon=52, known="everything", variants="base", n_jobs=3):
    """Score each variant (';'-separated); known: a key of mpc_foresight.GROUPS."""
    from sbf_starter import scoring

    refs = list(scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs).references)
    kn = MF.GROUPS[known if known != "nothing" else "nothing (the MPC as it is)"]
    for spec in variants.split(";"):
        t0 = time.perf_counter()
        out = Parallel(n_jobs=n_jobs)(delayed(episode)(kn, task, entropy, n, horizon, spec) for n in range(episodes))
        score, by_level = rss(refs, [o[0] for o in out])
        lv = " ".join(f"{v:.3f}" for v in by_level.values())
        ag = sum(o[3][0] for o in out) / max(1, sum(o[3][1] for o in out))
        print(
            f"{spec:30s} known={known:10s} n={episodes} h={horizon} score={score:.4f}  {lv}  "
            f"cpu/wk mean={np.mean([o[1] for o in out]):.2f} max={np.max([o[2] for o in out]):.2f} "
            f"solves/wk={np.mean([o[4] for o in out]):.2f} agree={ag:.3f}  ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
