"""Commands around ``regime_core`` (the simulator's rules as rows of a plan under a trajectory's yes/no answers).

    uv run python lab/anastasiia/plan_lab/regime.py check --agent=agents/anastasiia_hybrid_hub --entropy=444 --n=0
    uv run python lab/anastasiia/plan_lab/regime.py run --entropy=444 --episodes=8 --rounds=4

``check``: an agent's played episode against the rows (the trajectory must satisfy the rules under its own answers).
``improve`` and ``run``: the plan under the answers of the agent's trajectory, with the whole future known, played
open loop; then again under the answers of that play. The functions of ``regime_core`` are re-exported here.
"""

import os
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("SBF_CACHE_DIR", str(ROOT / "hub" / "refcache"))
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / "lab" / "anastasiia" / "search_lab")]

from regime_core import *  # noqa: E402, F401, F403
from regime_core import Regimes, Rows, regimes, residual, rows, solve, summary  # noqa: E402, F401


# ----- episodes, played trajectories ----------------------------------------------------------------------------------
def world(task: str, entropy: int, n: int):
    """(instance, omega, marks) of an episode: the true network of every week."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow_agent.scoring import _label

    inst, params = task_generator(task)
    omega = sample_omega(inst, params, entropy, n, _label(entropy))
    marks = compute_marks(inst, omega)
    return inst.at_digest(marks.instance_digest), omega, marks


def replay(inst, omega, marks, actions, entropy: int, n: int):
    """The trajectory of weekly wire actions in the package's own environment (records of every week, J in cents)."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    env = Env(fallback=None)
    env.reset(inst, "standard", omega, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, policy_name="plan_lab")
    for a in actions:
        _o, _r, done, _tr, _i = env.step(a)
        if done:
            break
    return env.trajectory


def plan_actions(inst, omega, marks, model, x, entropy: int, n: int):
    """A plan's columns played open loop: each week's flows and tanker releases as requests; the trajectory."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    nc = model.meta["nc"]
    env = Env(fallback=None)
    obs, _ = env.reset(inst, "standard", omega, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks,
                       policy_name="plan_lab")
    done, t, sent = False, 0, []
    while not done:
        xt = np.zeros_like(model.lb)
        xt[:nc] = x[t * nc : (t + 1) * nc]
        a = L.week1_action(inst, model, xt, obs, np.asarray(marks.prohibited[t]))
        sent.append(a)
        obs, _r, done, _tr, _i = env.step(a)
        t += 1
    return env.trajectory, sent


COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def weekly_arrays(inst, marks, traj) -> dict:
    """A trajectory's weekly records as arrays, in the format of ``harness.py --save`` (``gap.py``, ``ledger.py``)."""
    recs = traj.records
    T, G = inst.T, len(inst.grids)
    seg = np.zeros((T, G, len(inst.commodities) + 1))  # last column: the no-fuel segment
    sent = np.zeros((T, len(inst.action_slots)))
    for t, r in enumerate(recs):
        for (gi, k), q in r.segment.items():
            seg[t, gi, -1 if k is None else k] = q
        for s, q in r.executed.items():
            sent[t, s] = q
    return {
        "J": traj.J_cents / 100,
        "cost": np.array([[getattr(r.costs, c) for c in COMPONENTS] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]),
        "energy": np.array([r.energy for r in recs]),
        "disposal": np.array([r.disposal for r in recs]),
        "stock": np.array([r.stock for r in recs]),
        "shed": np.array([r.shed for r in recs]),
        "served": np.array([r.served for r in recs]),
        "demand": np.array([r.demand for r in recs]),
        "segment": seg,
        "sent": sent,
        "G_bar": np.asarray(marks.G_bar)[:T],
        "y_bar": np.asarray(marks.y_bar)[:T],
    }


# ----- commands -------------------------------------------------------------------------------------------------------
def check(agent: str = "agents/anastasiia_hybrid_hub", task: str = "small", entropy: int = 444, n: int = 0,
          pack: bool = True, stores: bool = True, release: bool = True) -> None:
    """A played episode against the rows: the trajectory must satisfy the rules under its own answers."""
    import search
    from shockbench_flow.dynamics.sim import initial_stock
    from shockbench_flow.marks import osat_throughput
    from shockbench_flow.oracle.lp import build_lp, lp_cents
    from shockbench_flow.oracle.replay import trajectory_vector

    inst, omega, marks = world(task, entropy, n)
    d = search.played(agent, task, entropy, n)
    traj = replay(inst, omega, marks, d["actions"], entropy, n)
    assert int(traj.J_cents) == d["J"], (traj.J_cents, d["J"])
    model = build_lp(inst, marks)
    z = trajectory_vector(model, traj)
    reg = regimes(inst, marks, traj.records, initial_stock(inst), osat_throughput(inst, marks.R_osat))
    R = rows(inst, marks, model, reg, initial_stock(inst), pack=pack, stores=stores, release=release)
    worst, name = residual(model, R, z)
    print(f"{task} {entropy} ep {n}: J {d['J'] / 1e11:.1f} bn; LP cost of the trajectory {lp_cents(model, z) / 1e11:.1f}")
    print("  " + summary(inst, reg))
    print(f"  columns {len(z)}, rows added {len(R.lo)}; worst residual {worst:.2e} at {name}")


def improve(agent: str = "agents/anastasiia_hybrid_hub", task: str = "small", entropy: int = 444, n: int = 0,
            pack: bool = True, rounds: int = 3, stores: bool = True, release: bool = True,
            method: str = "highs", arrays: bool = False) -> dict:
    """The plan under a played trajectory's answers, played open loop; then again under the answers of that play."""
    import search
    from shockbench_flow.dynamics.sim import initial_stock
    from shockbench_flow.marks import osat_throughput
    from shockbench_flow.oracle.lp import build_lp, lp_cents
    from shockbench_flow.oracle.replay import trajectory_vector

    inst, omega, marks = world(task, entropy, n)
    d = search.played(agent, task, entropy, n)
    traj = replay(inst, omega, marks, d["actions"], entropy, n)
    model = build_lp(inst, marks)
    i0 = initial_stock(inst)
    def chain(t) -> tuple[float, float, float]:
        """Lots started, chips sold (millions) and energy to the fabs (TWh) of a trajectory."""
        return (sum(float(np.sum(r.lots_started)) for r in t.records) / 1e6,
                sum(float(np.sum(r.served)) for r in t.records) / 1e6,
                sum(float(np.sum(r.energy)) for r in t.records) / 1e3)

    out = {"n": n, "J_agent": int(traj.J_cents), "claimed": [], "played": [], "secs": [], "chain": [chain(traj)]}
    print(f"{task} {entropy} ep {n}: the agent {traj.J_cents / 1e11:.1f} bn", flush=True)
    for it in range(rounds):
        reg = regimes(inst, marks, traj.records, i0, osat_throughput(inst, marks.R_osat))
        z = trajectory_vector(model, traj)
        R = rows(inst, marks, model, reg, i0, pack=pack, stores=stores, release=release)
        worst, name = residual(model, R, z)
        res, secs = solve(model, R, method=method)
        if res.x is None:
            print(f"  round {it}: no plan ({res.message}); the trajectory's residual {worst:.1e} at {name}")
            break
        claimed = lp_cents(model, res.x)
        traj, _sent = plan_actions(inst, omega, marks, model, res.x, entropy, n)
        out["claimed"].append(int(claimed)), out["played"].append(int(traj.J_cents)), out["secs"].append(secs)
        out["chain"].append(chain(traj))
        if arrays and int(traj.J_cents) <= min(out["played"]):  # the cheapest plan so far, as played
            out["arrays"] = weekly_arrays(inst, marks, traj)
        print(f"  round {it}: residual of the start {worst:.1e}; claimed {claimed / 1e11:.1f}, played "
              f"{traj.J_cents / 1e11:.1f} bn ({secs:.1f} s); {summary(inst, reg)}", flush=True)
    return out


def run(agent: str = "agents/anastasiia_hybrid_hub", task: str = "small", entropy: int = 444, episodes: int = 8,
        first: int = 0, pack: bool = True, rounds: int = 3, n_jobs: int = 4, stores: bool = True,
        release: bool = True, method: str = "highs", save: str = "") -> None:
    """``improve`` on several episodes: the board's score of the agent, of each round's plan on paper and as played."""
    from joblib import Parallel, delayed

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import harness

    out = Parallel(n_jobs=n_jobs)(
        delayed(improve)(agent, task, entropy, n, pack, rounds, stores, release, method, bool(save))
        for n in range(first, first + episodes)
    )
    if save:  # the cheapest played plan of every episode
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, **{k: np.stack([o["arrays"][k] for o in out]) for k in out[0]["arrays"]})
    refs = harness.references(task, entropy, first + episodes)[first:]
    level = np.array([r["stratum"] for r in refs])
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    room = naive - np.array([r["J_oracle_cents"] for r in refs], dtype=float)

    def score(costs):
        return harness.pooled(level, naive - np.array(costs, dtype=float), room)

    def old(costs):  # the convention of hub/tried/mpc.md's numbers on sets that lack a harm level
        return harness.by_level_mean(level, naive - np.array(costs, dtype=float), room)

    agent_j = [o["J_agent"] for o in out]
    print(f"\n{task}, root {entropy}, episodes {first}..{first + episodes - 1}: the agent {score(agent_j):.4f} "
          f"(levels weighing the same: {old(agent_j):.4f})")
    done = min(len(o["played"]) for o in out)
    for it in range(done):
        cl, pl = [o["claimed"][it] for o in out], [o["played"][it] for o in out]
        print(f"  after round {it}: on paper {score(cl):.4f}, played {score(pl):.4f} [{old(pl):.4f}] "
              f"({(np.mean(pl) - np.mean(agent_j)) / 1e11:+.1f} bn an episode to the agent; "
              f"paper to played {(np.mean(pl) - np.mean(cl)) / 1e11:+.1f}); "
              f"solve {np.mean([o['secs'][it] for o in out]):.1f} s")
    best = [min([o["J_agent"], *o["played"]]) for o in out]
    print(f"  best played per episode: {score(best):.4f} [{old(best):.4f}]")
    for o in out:
        print(f"    ep {o['n']}: agent {o['J_agent'] / 1e11:.1f}; played " + " ".join(f"{j / 1e11:.1f}" for j in o["played"]))
    first, last = np.array([o["chain"][0] for o in out]), np.array([o["chain"][-1] for o in out])
    print(f"  per episode, the agent | the last plan: lots {first[:, 0].mean():.2f} | {last[:, 0].mean():.2f} M, chips sold "
          f"{first[:, 1].mean():.2f} | {last[:, 1].mean():.2f} M, energy to the fabs {first[:, 2].mean():.2f} | "
          f"{last[:, 2].mean():.2f} TWh")


if __name__ == "__main__":
    fire.Fire({"check": check, "improve": improve, "run": run})
