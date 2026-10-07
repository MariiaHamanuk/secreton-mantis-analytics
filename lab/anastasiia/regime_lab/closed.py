"""The cell program week by week in the real environment: a plan carried from week to week (closed loop).

    uv run python lab/anastasiia/regime_lab/closed.py run --episodes=8 --known=everything --start=offline --passes=2
    uv run python lab/anastasiia/regime_lab/closed.py run --episodes=8 --known=nothing --start=oracle --passes=3

Every week: the observation gives the simulator's state; the forecast gives the network of the weeks left (the
package's persistence, ``lp_common.persistence_arrays``, with the fields of ``--known`` replaced by the scenario's
own, as ``../mpc_lab/mpc_foresight.py`` does; ``--look=N``: only the next N weeks of them, the rest as week N); last week's plan, minus its first week, is played on that model, read
for its regimes and improved by ``plan.descend`` for ``--passes`` passes; the first week of the best plan is sent.
The plan of the first week comes from ``--start``: ``oracle`` (the window's relaxed program), or ``offline`` (the
plan ``plan.py starts`` kept for the episode; it knows the scenario, so only with ``--known=everything``).

The cost printed is the environment's. CPU seconds per week are this machine's process time, not the container's.
"""

import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import core  # noqa: E402
import plan  # noqa: E402 - also points the package at the team's reference cache


GROUPS = {
    "nothing": (),
    "everything": (
        *("prohibited", "u", "tariff", "o", "kappa", "wr_class", "h_queue", "c_wr", "c", "G_bar", "y_bar"),
        *("R", "alpha_bar", "R_osat", "sigma_scr", "supply", "demand"),
    ),
    "straits": ("o", "kappa", "wr_class", "h_queue", "c_wr", "c"),
    "edges": ("u",),
    "grids": ("G_bar", "y_bar"),
    "prohibitions": ("prohibited",),
    "tariffs": ("tariff",),
    "plants": ("R", "alpha_bar", "R_osat", "sigma_scr", "supply"),
    "demand": ("demand",),
    # the future of edge capacities in halves: only what comes back (a capacity never forecast below today's), and
    # only what is cut (never above today's)
    "edges_up": ("u",),
    "edges_down": ("u",),
}


RECOVER = {"u": "u", "kappa": "kappa", "G_bar": "grid_G"}  # forecast field -> its nominal value in the observed graph


def recovering(arrays: dict, nominal: dict, rates: dict, now: dict) -> None:
    """Let what is cut come back, in expectation: a field below its nominal value closes the gap by 1 - s^h after h
    weeks, s the chance per week that a cut lasts (``rates``, per field); the first week stays as observed."""
    for name, s in rates.items():
        cur = arrays[name]
        target = np.broadcast_to(nominal[RECOVER[name]], cur.shape[1:])
        gap = np.where(np.isfinite(target), np.maximum(target - cur, 0.0), 0.0)
        w = (1.0 - s ** np.arange(cur.shape[0])).reshape(-1, *([1] * (cur.ndim - 1)))
        arrays[name] = cur + gap * w
        if name in now:
            arrays[now[name]] = arrays[name].copy()


def episode(task: str, entropy: int, n: int, known: str, start: str, passes: int, first_passes: int, hints: bool, recover: str = "", look: int = 0, warm: bool = True) -> dict:
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _label, _policy_seed

    inst, params = task_generator(task)
    omega = sample_omega(inst, params, entropy, n, _label(entropy))
    marks = compute_marks(inst, omega)
    inst = inst.at_digest(marks.instance_digest)
    env = Env(fallback=None)  # the plan's actions are always well formed: the naive fallback never plays
    obs, _info = env.reset(inst, "standard", omega, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, policy_name="regime")
    memory = L.ObservedGraph.nominal(inst)
    now = {week: instant for instant, week in L.NOW_FIELDS}
    fields = tuple(g for name in known.split("+") for g in GROUPS[name])
    half = np.maximum if "edges_up" in known.split("+") else np.minimum if "edges_down" in known.split("+") else None
    rates = {a.split(":")[0]: float(a.split(":")[1]) for a in recover.split(",") if a}
    nominal = L.ObservedGraph.nominal(inst).values
    acts, cpu, gains, done, basis, iters = None, [], [], False, None, []
    while not done:
        t0 = time.process_time()
        t = int(obs["week"])
        memory.update(inst, obs)
        H = inst.T - t + 1
        arrays = {k: a.copy() for k, a in L.persistence_arrays(inst, obs, memory, H).items()}
        recovering(arrays, nominal, rates, now)
        ahead = np.minimum(np.arange(H), look - 1) if look else np.arange(H)  # known ``look`` weeks ahead, then as then
        for name in fields:
            seen = arrays[name]
            arrays[name] = np.array(getattr(marks, name)[t - 1 : t - 1 + H])[ahead]
            if name in now:
                arrays[now[name]] = np.array(getattr(marks, now[name])[t - 1 : t - 1 + H])[ahead]
            if half is not None and name == "u":
                arrays["u"] = half(seen, arrays["u"])
                arrays["u_now"] = half(seen, arrays["u_now"])
        inst_r, backlog = L.rolled_window(inst, obs, H)
        ep = core.Episode(inst_r, L.window_marks(inst_r, L.read_only(arrays), backlog))
        if acts is None:
            if start == "offline":
                acts = pickle.loads((plan.OUT / "descend" / f"{task}_{entropy}_{n}_hybrid.pkl").read_bytes())["acts"]
            elif start == "basefirst":  # the window's plan with base load first: a minute, once an episode
                from plan_stats import base_first_plan

                path = plan.OUT / "basefirst" / f"closed_{task}_{entropy}_{n}_{known}_{recover}_{look}.pkl"
                if path.is_file():
                    x = pickle.loads(path.read_bytes())
                else:
                    x, _status, _gap = base_first_plan(ep.inst, ep.marks, ep.m, 60.0, 1e-3)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(pickle.dumps(x))
                acts = ep.actions(x)
            else:
                acts = plan.STARTS[start](ep)
            k = first_passes
        else:
            acts = ep.clean(acts[1:])
            k = passes
        d = plan.descend(ep, acts, iters=k, hints=hints, basis=ep.shifted(basis) if warm else None) if k > 0 else {"acts": acts, "J0": 0, "J": 0}
        acts, basis = d["acts"], d.get("basis")
        iters += [h[3] for h in d.get("hist", []) if h[1] is not None]
        gains.append((d["J0"] - d["J"]) / 1e11)
        cpu.append(time.process_time() - t0)
        obs, _r, done, _tr, _inf = env.step(ep.as_wire(acts[0], t))
    recs = env.trajectory.records
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    return {
        "n": n, "J": int(env.trajectory.J_cents), "cpu": cpu, "gains": gains, "iters": iters,
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "shed": np.array([r.shed for r in recs]), "lots": np.array([r.lots_started for r in recs]),
        "served": np.array([r.served for r in recs]),
    }


def run(task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0, known: str = "everything",
        start: str = "offline", passes: int = 2, first_passes: int = 0, hints: bool = True, n_jobs: int = 3, tag: str = "", recover: str = "", look: int = 0, warm: bool = True) -> None:
    """The closed loop on episodes ``first .. first + episodes - 1``: the environment's cost and RSS, CPU per week."""
    refs = plan.references(task, entropy, first + episodes)[first:]
    first_passes = first_passes or (0 if start == "offline" else max(passes, 8))
    res = Parallel(n_jobs=n_jobs)(delayed(episode)(task, entropy, n, known, start, passes, first_passes, hints, recover, look, warm) for n in range(first, first + episodes))
    for r, ref in zip(res, refs):
        room = ref["J_naive_cents"] - ref["J_oracle_cents"]
        cpu = np.array(r["cpu"])
        print(f"ep {r['n']} lvl {ref['stratum']}: J {r['J'] / 1e11:8.1f}  rss {(ref['J_naive_cents'] - r['J']) / room:.4f}  cpu/week median {np.median(cpu):.2f} "
              f"p95 {np.percentile(cpu, 95):.2f} max {cpu.max():.2f}  model gain per week {np.mean(r['gains']):.2f} bn  simplex iterations a pass {np.mean(r['iters']) if r['iters'] else 0:.0f}", flush=True)
    print(f"known={known} start={start} passes={passes} first_passes={first_passes} hints={hints} recover={recover!r} look={look}: RSS {plan.both(refs, [r['J'] for r in res])}")
    out = plan.OUT / "closed" / f"{task}_{entropy}_{first}_{episodes}_{known}_{start}_p{passes}{tag}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(res))


if __name__ == "__main__":
    fire.Fire({"run": run})
