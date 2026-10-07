"""Joint static search (cross-entropy method) over the numbers of ``agents/anastasiia_hybrid_hub``.

``knob_search.py --mode=static`` for the hub agent (``tuned_rules_hub.py``): one set of numbers for Small and Full
together, each network's cost divided by naive's mean week there, every candidate played on the same episodes
(paired). Resumable: every generation writes the search state, and a run started with the same ``--out`` picks up
from the last finished generation. One line per generation goes to ``hub_cem.log`` beside this file.

    uv run python lab/anastasiia/rl_lab/hub_cem.py check --task=small --episodes=4
    uv run python lab/anastasiia/rl_lab/hub_cem.py search --out=outputs/rl_lab/hub_cem/run1 --workers=2

``check`` plays the folder's own ``agent.py`` and the tuned agent on the shipped numbers on the same episodes and
prints both costs to the cent (and the seconds each took).
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import tuned_rules_hub as T  # noqa: E402


LOG = HERE / "hub_cem.log"
REWARD_SCALE = {"small": 67_024_547_307.78, "full": 89_880_564_005.63}  # naive's mean week (knob_search.py)
_STATE = {}


def _env(task, entropy, n_scenarios):
    import gymnasium as gym
    from shockbench_flow_gym.wrappers import ScenarioPool

    from sbf_starter import env_id

    key = (task, entropy, n_scenarios)
    if key not in _STATE:
        _STATE[key] = ScenarioPool(gym.make(env_id(task), regime="standard"), n_scenarios, entropy)
    if "parts" not in _STATE:
        _STATE["parts"] = T.Parts()
    return _STATE[key], _STATE["parts"]


def _play(env, obs, agent):
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        total += -float(reward)
        if terminated or truncated:
            return total


def cost_of(task, entropy, n_scenarios, index, values, lp=True):
    """What one episode costs in USD when the hub agent runs on ``values`` (None: the shipped numbers)."""
    from shockbench_flow_gym import agent_config_from_reset

    env, parts = _env(task, entropy, n_scenarios)
    obs, info = env.reset(options={"pool_index": int(index)})
    config = agent_config_from_reset(env, obs, info)
    return _play(env, obs, parts.agent(config, values, lp=lp))


def cost_of_folder(task, entropy, n_scenarios, index, folder, solve_share=None):
    """The same episode played by a folder's own ``agent.py`` (imported once per process).

    ``solve_share`` replaces the program's CPU share: the shipped 0.4 of the week's budget is a CPU-time limit, and a
    week that reaches it keeps the rules' entries, so on a machine where the program runs close to it (Full here) the
    folder's cost depends on timing. A large share makes it deterministic.
    """
    import importlib.util

    from shockbench_flow_gym import agent_config_from_reset

    key = ("folder", str(folder))
    if key not in _STATE:
        spec = importlib.util.spec_from_file_location(f"folder_{Path(folder).name}", Path(folder) / "agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _STATE[key] = module
    if solve_share is not None:
        _STATE[key]._lp.LP["solve_share"] = float(solve_share)
    env, _ = _env(task, entropy, n_scenarios)
    obs, info = env.reset(options={"pool_index": int(index)})
    config = agent_config_from_reset(env, obs, info)
    return _play(env, obs, _STATE[key].Agent(config))


def unit_to_values(x):
    lo, hi = np.array(T.LOW), np.array(T.HIGH)
    return T.clip(lo + np.clip(np.asarray(x, dtype=float), 0.0, 1.0) * (hi - lo))


def values_to_unit(values):
    lo, hi = np.array(T.LOW), np.array(T.HIGH)
    return (np.array([values[n] for n in T.NAMES]) - lo) / (hi - lo)


def job_episode(payload):
    """One candidate on one episode: (scaled cost, CPU seconds)."""
    task, entropy, n_scenarios, index, x, lp = payload
    t0 = time.process_time()
    c = cost_of(task, entropy, n_scenarios, index, None if x is None else unit_to_values(np.asarray(x)), lp=lp)
    return c / REWARD_SCALE[task], time.process_time() - t0


def _check_job(payload):
    task, entropy, n, i, what, share = payload
    t0 = time.process_time()
    if what == "folder":
        c = cost_of_folder(task, entropy, n, i, T.HUB, share)
    elif what == "tuned":
        values = dict(zip(T.NAMES, T.DEFAULTS))
        c = cost_of(task, entropy, n, i, values if share is None else dict(values, solve_share=float(share)))
    else:
        c = cost_of(task, entropy, n, i, None, lp=False)
    return task, i, what, c, time.process_time() - t0


def check(task="small", episodes=4, entropy=111, workers=2, nolp=False, solve_share=None, first=0):
    """The tuned agent on the shipped numbers against the folder's agent.py, to the cent (episodes first..)."""
    whats = ["folder", "tuned"] + (["nolp"] if nolp else [])
    jobs = [(task, entropy, first + episodes, i, w, solve_share) for i in range(first, first + episodes) for w in whats]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_check_job, jobs))
    got = {(i, w): (c, s) for _, i, w, c, s in rows}
    ok = True
    for i in range(first, first + episodes):
        f, t = got[(i, "folder")], got[(i, "tuned")]
        same = round(f[0] * 100) == round(t[0] * 100)
        ok &= same
        extra = f"  no-LP {got[(i, 'nolp')][0]:.2f} ({got[(i, 'nolp')][1]:.1f} s)" if nolp else ""
        print(
            f"{task} ep {i}: folder {f[0]:.2f} ({f[1]:.1f} s)  tuned {t[0]:.2f} ({t[1]:.1f} s)  "
            f"{'SAME' if same else 'DIFFERENT'}{extra}",
            flush=True,
        )
    print(f"{task}: {'all to the cent' if ok else 'MISMATCH'}", flush=True)
    return ok


def _log(line):
    with LOG.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def search(
    out="outputs/rl_lab/hub_cem/run1",
    entropy=111,
    small=32,
    full=16,
    pop=24,
    iters=14,
    elite=6,
    sigma0=0.25,
    workers=2,
    seed=0,
    patience=4,
    hours=4.0,
    lp=True,
):
    """CEM in the unit cube, Small x ``small`` plus Full x ``full`` per candidate, resumable from ``out``."""
    out = Path(out) if Path(out).is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    plan = [(t, n) for t, n in (("small", small), ("full", full)) if n > 0]
    state_file = out / "state.json"
    settings = dict(entropy=entropy, small=small, full=full, pop=pop, iters=iters, elite=elite, sigma0=sigma0,
                    seed=seed, lp=lp, knobs=T.KNOBS)
    # without the program its own numbers change nothing: they stay at the shipped values
    pinned = [j for j, k in enumerate(T.KNOBS) if k[1] == "lp"] if not lp else []
    home = values_to_unit(T.clip(T.DEFAULTS))
    t_start = time.time()

    # the jobs: one per (candidate, episode); Full first, the long ones, so the pool stays busy at the end
    def judge(xs, pool):
        jobs, where = [], []
        for c, x in enumerate(xs):
            for t, n in plan:
                for i in range(n):
                    jobs.append((t, entropy, n, i, None if x is None else list(map(float, x)), lp))
                    where.append((c, t, i))
        order = sorted(range(len(jobs)), key=lambda j: jobs[j][0] != "full")
        res = list(pool.map(job_episode, [jobs[j] for j in order], chunksize=1))
        per = np.zeros((len(xs), len(plan)))
        raw = [dict() for _ in xs]
        cpu = 0.0
        for j, (scaled, sec) in zip(order, res):
            c, t, i = where[j]
            raw[c][f"{t}{i}"] = scaled
            cpu += sec
        for c in range(len(xs)):
            for p, (t, n) in enumerate(plan):
                per[c, p] = np.mean([raw[c][f"{t}{i}"] for i in range(n)])
        return per.mean(axis=1), per, raw, cpu

    with ProcessPoolExecutor(max_workers=workers) as pool:
        if state_file.is_file():
            st = json.loads(state_file.read_text())
            if st["settings"]["knobs"] != json.loads(json.dumps(T.KNOBS)) or st["settings"].get("lp") != lp:
                raise SystemExit("state.json was written by a different search")
            rng = np.random.default_rng(seed)
            rng.bit_generator.state = st["rng"]
            mean, sigma = np.array(st["mean"]), np.array(st["sigma"])
            base, base_per = st["base"], st["base_per"]
            best_x, best_cost, history = np.array(st["best_x"]), st["best_cost"], st["history"]
            _log(f"[{time.strftime('%H:%M:%S')}] resume {out} after generation {len(history)}")
        else:
            (out / "settings.json").write_text(json.dumps(settings, indent=2))
            _log(f"[{time.strftime('%H:%M:%S')}] start {out}: {len(T.NAMES)} knobs, pop {pop}, elite {elite}, "
                 f"{' + '.join(f'{t} x{n}' for t, n in plan)}, root {entropy}, program {'on' if lp else 'off'}")
            rng = np.random.default_rng(seed)
            mean = values_to_unit(T.clip(T.DEFAULTS))
            sigma = np.full(len(T.NAMES), float(sigma0))
            b, bp, _, cpu = judge([mean], pool)
            base, base_per = float(b[0]), bp[0].tolist()
            best_x, best_cost, history = mean.copy(), base, []
            _log(f"  base {base:.6f} (small {base_per[0]:.6f}, full {base_per[-1]:.6f}), {cpu:.0f} CPU s")

        stale = 0
        for it in range(len(history), iters):
            if (time.time() - t_start) / 3600 > hours:
                _log("  time budget reached: stop")
                break
            g0 = time.time()
            xs = np.clip(rng.normal(mean, sigma, size=(pop, len(mean))), 0.0, 1.0)
            xs[0] = mean
            xs[:, pinned] = home[pinned]
            costs, per, _, cpu = judge(xs, pool)
            order = np.argsort(costs)
            improved = costs[order[0]] < best_cost
            if improved:
                best_cost, best_x = float(costs[order[0]]), xs[order[0]].copy()
            el = xs[order[:elite]]
            mean = el.mean(axis=0)
            sigma = np.maximum(el.std(axis=0), 0.02) * (1.0 - 0.5 * it / max(iters - 1, 1))
            bi = order[0]
            row = {
                "iter": it + 1,
                "gen_best": float(costs[bi]),
                "gen_best_gain": (base - float(costs[bi])) / base,
                "gen_best_small_gain": (base_per[0] - per[bi, 0]) / base_per[0],
                "gen_best_full_gain": (base_per[-1] - per[bi, -1]) / base_per[-1],
                "centre_gain": (base - float(costs[0])) / base,
                "best": best_cost,
                "gain": (base - best_cost) / base,
                "sigma_mean": float(sigma.mean()),
                "wall_s": time.time() - g0,
                "cpu_s": cpu,
            }
            history.append(row)
            _log(
                f"  gen {it + 1:2d}: best-so-far {row['gain']:+.4%} | gen best {row['gen_best_gain']:+.4%} "
                f"(small {row['gen_best_small_gain']:+.4%}, full {row['gen_best_full_gain']:+.4%}) | "
                f"centre {row['centre_gain']:+.4%} | sigma {row['sigma_mean']:.3f} | "
                f"{row['wall_s'] / 60:.1f} min, {cpu:.0f} CPU s"
            )
            st = dict(
                settings=json.loads(json.dumps(settings)),
                rng=rng.bit_generator.state,
                mean=mean.tolist(),
                sigma=sigma.tolist(),
                base=base,
                base_per=list(base_per),
                best_x=best_x.tolist(),
                best_cost=best_cost,
                history=history,
            )
            tmp = out / "state.json.tmp"
            tmp.write_text(json.dumps(st))
            tmp.replace(state_file)
            (out / "params.json").write_text(json.dumps(unit_to_values(best_x), indent=2))
            (out / "centre.json").write_text(json.dumps(unit_to_values(mean), indent=2))
            stale = 0 if improved else stale + 1
            if stale >= patience:
                _log(f"  no new best for {patience} generations: stop")
                break
    _log(f"[{time.strftime('%H:%M:%S')}] done: best {(base - best_cost) / base:+.4%} -> {out / 'params.json'}")
    return str(out)


if __name__ == "__main__":
    import fire

    fire.Fire({"check": check, "search": search})
