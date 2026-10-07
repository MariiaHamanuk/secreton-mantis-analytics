"""Dispersion-margin grid for hub_delta: per-episode J, queue metrics and LP timing, paired against k = 0.

    uv run python lab/anastasiia/mpc_lab/delta/grid.py check --episodes=0,1          # k=0 vs stock hybrid_hub, cents
    uv run python lab/anastasiia/mpc_lab/delta/grid.py run --ks=0,1 --episodes=0,1,2,3 --tag=smoke
    uv run python lab/anastasiia/mpc_lab/delta/grid.py run --ks=0,0.5,1,2 --episodes=32 --n_jobs=2 --tag=grid
    uv run python lab/anastasiia/mpc_lab/delta/grid.py report --tag=grid

Every (variant, episode) is played once in the gymnasium env (policy seed of the env, as teacher_gen.record) and its
row appended to outputs/delta_margin/<tag>/rows.jsonl (resumable: done rows are skipped). Row: J cents, cost
components, queue totals by pool (fuel = lng/crude/nucfuel, chip = the rest) at the final state and averaged over
weeks, the LP's CPU seconds per week (both passes) and how many weeks ran pass 2 / bound edges.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))
import sbf_starter  # noqa: E402,F401  (sets the reference cache dir)

DELTA = HERE.parent / "agents" / "hub_delta"
STOCK = ROOT / "agents" / "anastasiia_hybrid_hub"
OUT = ROOT / "outputs" / "delta_margin"
LOG = HERE.parent / "delta_margin.log"


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_module(folder, tag):
    folder = Path(folder).resolve()
    spec = importlib.util.spec_from_file_location(f"{folder.name}_{tag}_agent", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def play(task, entropy, n, variant, lp_params):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow.dynamics.state import COST_COMPONENTS
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id

    folder = STOCK if variant == "stock" else DELTA
    module = load_module(folder, f"{variant}_{os.getpid()}")
    module.PARAMS["lp"] = dict(lp_params or {})
    env = gym.make(env_id(task), entropy=entropy)
    u = env.unwrapped
    obs, info = env.reset(options={"episode": n})
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    agent = module.Agent(config)
    pools = list(config["static"]["commodities"]["id"])
    fuel_k = {pools.index(c) for c in ("lng", "crude", "nucfuel") if c in pools}
    lot_k = np.array([int(key[1]) for key in config["layout"]["lot_keys"]])
    is_fuel_lot = np.isin(lot_k, list(fuel_k))
    lng_lot = lot_k == pools.index("lng")
    planner = agent.planner
    orig = planner.flows
    lp_sec, pass2, nbind = [], 0, []

    def timed(o):
        t0 = time.process_time()
        try:
            return orig(o)
        finally:
            lp_sec.append(time.process_time() - t0)

    planner.flows = timed
    q_fuel, q_chip, q_lng, q_by_k = [], [], [], []
    is_lp_lot = np.isin(lot_k, [i for i, c in enumerate(pools) if c not in ("lng", "crude", "nucfuel", "wafer")])
    done, act_sec = False, []
    while not done:
        t0 = time.process_time()
        act = agent.act(obs)
        act_sec.append(time.process_time() - t0)
        last = getattr(planner, "last", None)
        if last is not None and last.get("week") == int(obs["week"][0]):
            pass2 += int(last["pass2"])
            nbind.append(last["n_bind"])
        obs, _r, term, trunc, _inf = env.step(act)
        q = np.asarray(obs["queue_lots.qty"], dtype=float).sum(axis=1)
        q_fuel.append(float(q[is_fuel_lot].sum()))
        q_lng.append(float(q[lng_lot].sum()))
        q_chip.append(float(q[is_lp_lot].sum()))  # raw and packaged chips: what the LP ships
        q_by_k.append([float(q[lot_k == k].sum()) for k in range(len(pools))])
        done = term or trunc
    traj = u.core.trajectory
    comp = {c: 0.0 for c in COST_COMPONENTS}
    for rec in traj.records:
        for c in COST_COMPONENTS:
            comp[c] += float(getattr(rec.costs, c))
    T = len(q_fuel)
    return {
        "variant": variant,
        "lp": lp_params,
        "ep": n,
        "J": int(traj.J_cents),
        "costs": comp,
        "q_fuel_end": q_fuel[-1],
        "q_lng_end": q_lng[-1],
        "q_chip_end": q_chip[-1],
        "q_fuel_last4": float(np.mean(q_fuel[-4:])),
        "q_chip_last4": float(np.mean(q_chip[-4:])),
        "q_fuel_mean": float(np.mean(q_fuel)),
        "q_lng_mean": float(np.mean(q_lng)),
        "q_chip_mean": float(np.mean(q_chip)),
        "q_by_k_end": dict(zip(pools, q_by_k[-1])),
        "q_by_k_mean": dict(zip(pools, np.mean(q_by_k, axis=0).tolist())),
        "lp_sec": lp_sec,
        "act_sec": act_sec,
        "pass2_weeks": pass2,
        "nbind_mean": float(np.mean(nbind)) if nbind else 0.0,
        "planned_weeks": int(agent.planned_weeks),
        "T": T,
    }


def _eps(episodes):
    if isinstance(episodes, int):
        return list(range(episodes))
    if isinstance(episodes, str):
        return [int(x) for x in episodes.split(",") if x != ""]
    return [int(x) for x in episodes]


def _ks(ks):
    if isinstance(ks, (int, float)):
        return [float(ks)]
    if isinstance(ks, str):
        return [float(x) for x in ks.split(",")]
    return [float(x) for x in ks]


def check(episodes="0,1", task="small", entropy=111):
    """k = 0 of hub_delta against the stock hybrid_hub: the same J to the cent."""
    for n in _eps(episodes):
        a = play(task, entropy, n, "stock", {})
        b = play(task, entropy, n, "delta", {"delta_k": 0.0})
        same = a["J"] == b["J"]
        log(f"[check] {task} root {entropy} ep {n}: stock J {a['J']} k=0 J {b['J']} -> {'SAME' if same else 'DIFFERENT'}"
            f"  (planned weeks {a['planned_weeks']}/{b['planned_weeks']} of {a['T']})")


def run(ks="0,1", episodes="0,1,2,3", task="small", entropy=111, n_jobs=1, tag="smoke", bind_frac=0.9, cap=0.5,
        sigma_fuel=0.062, sigma_chip=0.059):
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rows.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["lp"]["delta_k"], r["lp"]["bind_frac"], r["ep"]))
    jobs = []
    for n in _eps(episodes):  # episode-major: a partial run still pairs
        for k in _ks(ks):
            lp = {"delta_k": k, "bind_frac": bind_frac, "delta_cap": cap, "sigma_fuel": sigma_fuel,
                  "sigma_chip": sigma_chip}
            if (k, bind_frac, n) not in done:
                jobs.append((n, lp))
    log(f"[run {tag}] {task} root {entropy}: {len(jobs)} plays to go (ks {ks}, bind_frac {bind_frac}), n_jobs {n_jobs}")

    def one(n, lp):
        t0 = time.time()
        r = play(task, entropy, n, "delta", lp)
        r["wall"] = time.time() - t0
        return r

    t0 = time.time()
    for r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(delayed(one)(n, lp) for n, lp in jobs):
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        lp = np.array(r["lp_sec"])
        log(f"[run {tag}] ep {r['ep']} k {r['lp']['delta_k']}: J {r['J'] / 100:.4e}  q_fuel_end {r['q_fuel_end']:.0f} "
            f"q_lng_mean {r['q_lng_mean']:.0f} q_chip_end {r['q_chip_end']:.0f}  pass2 {r['pass2_weeks']}/{r['T']} "
            f"bind {r['nbind_mean']:.1f}  lp s/wk med {np.median(lp):.2f} p95 {np.quantile(lp, 0.95):.2f} "
            f"max {lp.max():.2f}  ({r['wall']:.0f} s wall, {time.time() - t0:.0f} s total)")


def _rss_tools(task, entropy, eps):
    from sbf_starter import scoring
    from package_baselines import rss

    es = scoring.episode_set(task, max(eps) + 1, entropy=entropy, n_jobs=1, verbose=False)
    refs = list(es.references)
    return refs, rss


def report(tag="grid", task="small", entropy=111, base_k=0.0, bind_frac=None):
    rows = [json.loads(x) for x in (OUT / tag / "rows.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        if bind_frac is not None and r["lp"]["bind_frac"] != bind_frac:
            continue
        by.setdefault((r["lp"]["delta_k"], r["lp"]["bind_frac"]), {})[r["ep"]] = r
    base_key = next(k for k in by if k[0] == base_k)
    base = by[base_key]
    variants = sorted(by)
    eps = sorted(set.intersection(*(set(v) for v in by.values())))
    refs, rss = _rss_tools(task, entropy, eps)
    room = np.array([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in eps], dtype=float)
    lines = [f"{task} root {entropy}, {len(eps)} paired episodes ({tag}); diff = variant - k={base_k}, "
             f"per-episode RSS points = (J_base - J) / (J_naive - J_oracle)"]
    lines.append(f"{'k':>5} {'bind':>5} {'RSS':>7} {'dRSS':>8} {'dJ bn/ep':>9} {'SE':>6} {'d per-ep rss':>13} "
                 f"{'SE':>7} {'win':>5} {'q_fuel_end':>10} {'q_lng_mean':>10} {'q_chip_end':>10} {'q_chip_mean':>11} {'q_waf_mean':>10} "
                 f"{'pass2':>6} {'bind#':>5} {'lp med':>6} {'p95':>5} {'max':>5}")
    for key in variants:
        v = by[key]
        J = np.array([v[n]["J"] for n in eps], dtype=float)
        Jb = np.array([base[n]["J"] for n in eps], dtype=float)
        score = rss([refs[n] for n in eps], [v[n]["J"] for n in eps])[0]
        score_b = rss([refs[n] for n in eps], [base[n]["J"] for n in eps])[0]
        d = (Jb - J) / 100 / 1e9  # bn USD per episode saved
        pe = (Jb - J) / room
        lp = np.concatenate([v[n]["lp_sec"] for n in eps])
        m = lambda f: np.mean([v[n][f] for n in eps])  # noqa: E731
        lines.append(
            f"{key[0]:5.2f} {key[1]:5.2f} {score:7.4f} {score - score_b:+8.4f} {d.mean():+9.2f} "
            f"{d.std(ddof=1) / np.sqrt(len(d)):6.2f} {pe.mean():+13.4f} {pe.std(ddof=1) / np.sqrt(len(pe)):7.4f} "
            f"{int((d > 0).sum()):2d}/{int((d < 0).sum()):<2d} {m('q_fuel_end'):10.0f} {m('q_lng_mean'):10.0f} "
            f"{m('q_chip_end'):10.0f} {m('q_chip_mean'):11.0f} "
            f"{np.mean([v[n].get('q_by_k_mean', {}).get('wafer', np.nan) for n in eps]):10.0f} {m('pass2_weeks'):6.1f} {m('nbind_mean'):5.1f} "
            f"{np.median(lp):6.2f} {np.quantile(lp, 0.95):5.2f} {lp.max():5.2f}"
        )
    print("\n".join(lines))
    with open(LOG, "a") as f:
        f.write("\n".join(lines) + "\n")


def drift(task, entropy, n):
    """(down, up): mean over finite edges and weeks 2..T of max(0, u1 - ut) / u0 and max(0, ut - u1) / u0 (true marks)."""
    sys.path.insert(0, str(HERE))
    from measure_sigma import marks_of

    inst, m = marks_of(task, entropy, n)
    u = np.asarray(m.u, dtype=float)
    cols = [e for e, ed in enumerate(inst.edges) if ed.u0 is not None and np.all(np.isfinite(u[:, e]))]
    u0 = np.array([inst.edges[e].u0 for e in cols])
    d = (u[1:, cols] - u[0, cols]) / u0
    return float(np.maximum(-d, 0).mean()), float(np.maximum(d, 0).mean())


def split(tag="grid", task="small", entropy=111, base_k=0.0):
    """The paired difference of each k by episode subtype: queue-heavy vs not (k=0's mean fuel queue, median split),
    and capacity drift mostly down vs mostly up after week 1 (true marks: the pessimistic case is 'up')."""
    rows = [json.loads(x) for x in (OUT / tag / "rows.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["lp"]["delta_k"], {})[r["ep"]] = r
    eps = sorted(set.intersection(*(set(v) for v in by.values())))
    refs, _ = _rss_tools(task, entropy, eps)
    room = {n: refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in eps}
    base = by[base_k]
    qf = np.array([base[n]["q_fuel_mean"] for n in eps])
    dr = np.array([drift(task, entropy, n) for n in eps])
    groups = {
        "queue-heavy (q_fuel_mean >= median)": qf >= np.median(qf),
        "queue-light": qf < np.median(qf),
        "drift mostly DOWN (down > up)": dr[:, 0] > dr[:, 1],
        "drift mostly UP (recoveries)": dr[:, 0] <= dr[:, 1],
    }
    lines = [f"split {tag}: per-episode RSS points (J_base - J)/(J_naive - J_oracle), mean +- SE (n)"]
    for k in sorted(by):
        if k == base_k:
            continue
        pe = np.array([(base[n]["J"] - by[k][n]["J"]) / room[n] for n in eps])
        dq = np.array([by[k][n]["q_fuel_mean"] - base[n]["q_fuel_mean"] for n in eps])
        parts = []
        for name, mask in groups.items():
            x = pe[mask]
            se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
            parts.append(f"{name}: {x.mean():+.4f} +- {se:.4f} (n={len(x)}, dq_fuel {dq[mask].mean():+.0f})")
        lines.append(f"k={k}: " + " | ".join(parts))
    lines.append("drift per ep (down, up): " + ", ".join(f"{n}:{a:.3f}/{b:.3f}" for n, (a, b) in zip(eps, dr)))
    print("\n".join(lines))
    with open(LOG, "a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fire.Fire({"check": check, "run": run, "report": report, "split": split})
