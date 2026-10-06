import sys, numpy as np
sys.path[:0] = [__import__("os").path.dirname(__import__("os").path.abspath(__file__)), __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))]  # this folder and the bench one folder up
from joblib import Parallel, delayed
import mpc_foresight as mf
COMP = ("freight","war_risk","tariff","holding","queue_holding","shortage","disposal","shed")
def one(n, known_name, horizon):
    import sbf_starter
    from scipy.optimize import linprog
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.evaluation.cache import default_cache_dir, fq_quantiles
    from shockbench_flow.hosting.tasks import TASKS, task_generator
    from shockbench_flow.oracle.lp import lp_costs
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow.policies.base import reset_policy
    from shockbench_flow.policies.mpc_det import MpcDet
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow.policies.registry import GeneratorRef, PolicyContext
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world
    task, entropy = "small", 111
    known = mf.GROUPS[known_name]
    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
    _, params = task_generator(task)
    q = fq_quantiles(inst, params, REPLICATIONS, cache_dir=cache).quantiles
    ctx = PolicyContext(fq_quantile=q, generator=GeneratorRef(task, TASKS[task].gamma))
    now = {w: i for i, w in L.NOW_FIELDS}
    log = {"plan_week": [], "plan_total": []}
    class P(MpcDet):
        def _horizon(self, inst, plan): return horizon
        def act(self, obs):
            inst, week = self._inst, int(obs["week"])
            self._memory.update(inst, obs)
            weeks = L.window_length(self._H, week, inst.T)
            arrays = {k: a.copy() for k, a in L.persistence_arrays(inst, obs, self._memory, weeks).items()}
            f = week - 1
            for name in known:
                arrays[name] = np.array(getattr(marks, name)[f:f + weeks])
                if name in now: arrays[now[name]] = np.array(getattr(marks, now[name])[f:f + weeks])
            model = L.rolled_lp(inst, obs, L.read_only(arrays), weeks, planning_rules=True)
            kw = {}
            if model.A_ub.shape[0]: kw.update(A_ub=model.A_ub, b_ub=model.b_ub)
            if model.A_eq.shape[0]: kw.update(A_eq=model.A_eq, b_eq=model.b_eq)
            res = linprog(model.objective(), bounds=np.column_stack([model.lb, model.ub]), method="highs-ds", **kw)
            weekly, credit = lp_costs(model, res.x)
            log["plan_week"].append([weekly[0].as_dict()[c] for c in COMP])
            log["plan_total"].append(sum(w.total() for w in weekly) - credit)
            return L.week1_action(inst, model, np.asarray(res.x), obs, L.prohibited_now(self._memory, week))
    env = Env(fallback=fallback); pol = P(None, ctx)
    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    obs, info = env.reset(inst, "standard", omega, seed, marks=marks, policy_name=pol.name)
    reset_policy(pol, info["static"], obs, seed, info.get("omega"))
    done = False
    while not done:
        obs, r, done, tr, inf = env.step(pol.act(obs))
    real = np.array([[rec.costs.as_dict()[c] for c in COMP] for rec in env.trajectory.records])
    return np.array(log["plan_week"]), real, np.array(log["plan_total"]), env.trajectory.J_cents / 100
if __name__ == "__main__":
    name, H, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out = Parallel(n_jobs=8)(delayed(one)(n, name, H) for n in range(N))
    plan = np.stack([o[0] for o in out]); real = np.stack([o[1] for o in out]); tot = np.stack([o[2] for o in out]); J = np.array([o[3] for o in out])
    B = 1e9
    print(f"known: {name}, horizon {H}, {N} episodes; $B per episode (means)")
    print("planned total at week 1:", round(tot[:, 0].mean() / B, 1), "| realised total:", round(J.mean() / B, 1))
    print("week's cost: realised minus planned for that week, summed over the episode, by component:")
    d = (real - plan).sum(axis=1).mean(axis=0) / B
    for c, v, p_, r_ in zip(COMP, d, plan.sum(axis=1).mean(axis=0) / B, real.sum(axis=1).mean(axis=0) / B): print(f"  {c:14s} planned {p_:9.1f} realised {r_:9.1f} diff {v:+8.1f}")
    q = np.linspace(0, real.shape[1], 5).astype(int)
    print("by weeks (realised - planned, all components):", [round(float((real - plan)[:, a:b].sum(axis=(1, 2)).mean() / B), 1) for a, b in zip(q[:-1], q[1:])])
    # drift of the planned total: cost so far + plan to go
    cum = np.concatenate([np.zeros((N, 1)), real.sum(axis=2).cumsum(axis=1)[:, :-1]], axis=1)
    S = cum + tot
    print("cost so far + plan to go, at weeks 1,5,9,..:", [round(float(S[:, t].mean() / B)) for t in range(0, real.shape[1], 4)])
