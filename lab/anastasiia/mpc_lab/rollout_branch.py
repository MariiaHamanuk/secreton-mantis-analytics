"""Does the rollout's ranking of fuel candidates predict the realized episode cost? (diagnostics of hyb_rollout)

For episode n and week t: play agents/anastasiia_hybrid_chiplp's actions (hyb_rollout with every other week as-is),
at week t play candidate c, finish the episode as-is; realized dJ(c) = J(c) - J(as_is) against the rollout's predicted
score difference at week t.

    uv run python lab/anastasiia/mpc_lab/rollout_branch.py --task=small --entropy=111 --eps=0,1 --weeks=2,6,10,... --n_jobs=3
"""
import os
import time

import fire
import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
os.environ.setdefault("SBF_CACHE_DIR", os.path.join(ROOT, "hub", "refcache"))
AGENT = os.path.join(HERE, "agents", "hyb_rollout")


def branch(task, entropy, n, t, cand):
    """(J, predicted scores {name: score} at week t, candidates available)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    env = gym.make(env_id(task), entropy=entropy)
    u = env.unwrapped
    obs, info = env.reset(options={"episode": n})
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    ag = load(AGENT)(config)
    roll = ag.roll
    orig = roll.choose
    seen = {}

    def forced(observation, flows, extra, t0):
        week = int(observation["week"][0])
        if week != t:
            return "as_is", flows
        roll.p["abort_share"] = 1e9  # evaluate every candidate here (diagnostics, no budget)
        cands = roll.candidates(flows, extra, np.asarray(observation["stock.qty"], dtype=float))
        orig(observation, flows, extra, -1e9)
        seen["names"] = [c for c, _ in cands]
        seen["scores"] = list(roll._last_scores)
        pick = dict(cands)
        return cand, pick.get(cand, flows)

    roll.choose = forced
    done = False
    while not done:
        obs, _r, term, trunc, _inf = env.step(ag.act(obs))
        done = term or trunc
    comp = {}
    for r in u.core.trajectory.records:
        for c, v in r.costs.as_dict().items():
            comp[c] = comp.get(c, 0.0) + v
    weekly = [r.cost_cents / 100 for r in u.core.trajectory.records]
    return dict(n=n, t=t, cand=cand, J=u.core.trajectory.J_cents / 100, comp=comp, weekly=weekly, **seen)


def main(task="small", entropy=111, eps="0,1", weeks="2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47", n_jobs=3, h=16,
         out=os.path.join(ROOT, "outputs", "mpc_research", "rollout_branch.txt")):
    def ints(x):
        return [int(x)] if isinstance(x, int) else [int(y) for y in (x.split(",") if isinstance(x, str) else x)]

    eps, weeks = ints(eps), ints(weeks)
    jobs = [(n, t, c) for n in eps for t in weeks for c in ("as_is", "valve_boost", "valve_hold", "order_boost")]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(branch)(task, entropy, n, t, c) for n, t, c in jobs)
    rows = []
    with open(out, "w") as fh:
        for n in eps:
            for t in weeks:
                grp = {r["cand"]: r for r in res if r["n"] == n and r["t"] == t}
                base = grp["as_is"]
                names, scores = base.get("names", []), base.get("scores", [])
                pred = dict(zip(names, scores))
                for c in names[1:]:
                    if c not in grp or c not in pred:
                        continue
                    real = (grp[c]["J"] - base["J"]) / 1e9
                    p = (pred[c] - pred["as_is"]) / 1e9
                    wd = np.array(grp[c]["weekly"]) - np.array(base["weekly"])
                    real_h = wd[t - 1:t - 1 + h].sum() / 1e9  # realized within the rollout's own window
                    rows.append((n, t, c, p, real, real_h))
                    line = (f"ep {n} week {t:2d} {c:12s} predicted dJ {p:+8.3f} bn  realized dJ {real:+8.3f} bn"
                            f"  realized in weeks t..t+{h - 1} {real_h:+8.3f} bn")
                    print(line, flush=True)
                    fh.write(line + "\n")
        if rows:
            P = np.array([r[3] for r in rows])
            R = np.array([r[4] for r in rows])
            RH = np.array([r[5] for r in rows])
            agree = np.mean(np.sign(P) == np.sign(R))
            corr = np.corrcoef(P, R)[0, 1] if len(rows) > 2 else float("nan")
            summ = [f"pairs {len(rows)}; sign agreement {agree:.2f}; corr(pred, real) {corr:+.3f}; "
                    f"in-window: sign agreement {np.mean(np.sign(P) == np.sign(RH)):.2f}, corr {np.corrcoef(P, RH)[0, 1]:+.3f}; "
                    f"std of realized dJ {R.std():.2f} bn, of in-window {RH.std():.2f} bn"]
            for c in ("valve_boost", "valve_hold", "order_boost"):
                m = np.array([r[2] == c for r in rows])
                if m.any():
                    summ.append(f"  {c:12s} n={m.sum():2d} mean pred {P[m].mean():+.3f} mean real {R[m].mean():+.3f} bn;"
                                f" mean real in-window {RH[m].mean():+.3f};"
                                f" pred<0: {np.sum(P[m] < 0)} of which real<0: {np.sum((P[m] < 0) & (R[m] < 0))};"
                                f" corr {np.corrcoef(P[m], R[m])[0, 1] if m.sum() > 2 else float('nan'):+.3f}")
            summ.append(f"({time.time() - t0:.0f}s)")
            for s in summ:
                print(s)
                fh.write(s + "\n")


if __name__ == "__main__":
    fire.Fire(main)
