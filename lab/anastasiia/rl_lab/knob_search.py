"""Search the rules' own numbers against the true cost in the simulator.

Two modes, and they answer different questions.

``--mode=hindsight`` searches every episode on its own, knowing that episode. Nothing can be submitted from it: it
measures **how much is left in the family of policies the rules can express** if their numbers were right. That is
a ceiling, and unlike the planner's it is constructive and feasible by definition — every number it reports was
paid in the simulator. Its by-product is a set of labels: the best numbers for each episode, which is what a
state-conditioned policy would have to predict.

``--mode=static`` searches one set of numbers for all the episodes at once. That one can be submitted, and it is
the joint version of the one-at-a-time hand search in ``hub/tried/heuristics.md``.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/knob_search.py \
        --mode=hindsight --task=small --entropy=111 --episodes=32 --workers=10

The search is a cross-entropy method in the unit cube: a candidate is a point per knob between its low and its
high. Every candidate is played on the same episodes, so the comparison is paired and the noise of the episode
draw cancels.
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

import tuned_rules as T  # noqa: E402


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


def cost_of(task, entropy, n_scenarios, index, values, segments=1):
    """What one episode costs in USD when the rules run on ``values``.

    With ``segments`` above one, ``values`` is a flat vector of that many sets laid end to end, and each governs
    its share of the weeks — the cheapest way to ask whether the numbers want to change over the episode at all.
    """
    from shockbench_flow_gym import agent_config_from_reset

    env, parts = _env(task, entropy, n_scenarios)
    obs, info = env.reset(options={"pool_index": int(index)})
    config = agent_config_from_reset(env, obs, info)
    agent = parts.agent(config, values) if segments < 2 else parts.agent(config, schedule=values)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        total += -float(reward)
        if terminated or truncated:
            break
    return total


def unit_to_values(x, segments=1):
    """A point of the unit cube as the numbers the rules want: one set, or ``segments`` sets laid end to end."""
    lo, hi = np.array(T.LOW), np.array(T.HIGH)
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    if segments < 2:
        return T.clip(lo + x * (hi - lo))
    return [T.clip(lo + part * (hi - lo)) for part in x.reshape(segments, len(T.NAMES))]


def values_to_unit(values):
    lo, hi = np.array(T.LOW), np.array(T.HIGH)
    return (np.array([values[n] for n in T.NAMES]) - lo) / (hi - lo)


# ---- the two jobs a worker can be given ----------------------------------------------------------------------
REWARD_SCALE = {"small": 67_024_547_307.78, "full": 89_880_564_005.63}  # naive's mean week, from the package


def job_static(payload):
    """One candidate on the episodes of one network. The cost comes back divided by naive's average week there.

    That division is what lets a candidate be judged on Small and Full together: the two networks' dollars differ
    by a factor, and an unscaled mean would be a search on Full with Small as rounding error. One submission plays
    both boards, so the numbers have to suit both.
    """
    task, entropy, n_scenarios, indices, x = payload
    values = unit_to_values(np.asarray(x))
    scale = REWARD_SCALE[task]
    return float(np.mean([cost_of(task, entropy, n_scenarios, i, values) / scale for i in indices]))


def job_hindsight(payload):
    """A whole search on one episode, run inside the worker: episodes are independent, so this is free parallelism."""
    task, entropy, n_scenarios, index, pop, iters, elite_frac, sigma0, seed, segments = payload
    rng = np.random.default_rng(seed)
    base = cost_of(task, entropy, n_scenarios, index, None)
    mean = np.tile(values_to_unit(T.clip(T.DEFAULTS)), segments)
    sigma = np.full(mean.shape[0], float(sigma0))
    best_x, best_cost = mean.copy(), base
    for it in range(iters):
        xs = np.clip(rng.normal(mean, sigma, size=(pop, mean.shape[0])), 0.0, 1.0)
        xs[0] = mean  # always keep the current centre, so the search can never go backwards
        costs = np.array(
            [cost_of(task, entropy, n_scenarios, index, unit_to_values(x, segments), segments) for x in xs]
        )
        order = np.argsort(costs)
        if costs[order[0]] < best_cost:
            best_cost, best_x = float(costs[order[0]]), xs[order[0]].copy()
        elite = xs[order[: max(2, int(pop * elite_frac))]]
        mean = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.02) * (1.0 - 0.5 * it / max(iters - 1, 1))
    return {
        "episode": int(index),
        "base_cost": base,
        "best_cost": best_cost,
        "gain": (base - best_cost) / base,
        "values": unit_to_values(best_x, segments),
    }


def main(
    mode="hindsight",
    task="small",
    entropy=111,
    episodes=32,
    pop=48,
    iters=15,
    elite_frac=0.25,
    sigma0=0.25,
    workers=10,
    seed=0,
    segments=1,
    out=None,
):
    # `--episodes=32,16` names a count per network for the static mode; the hindsight mode takes the first
    first_count = int(episodes[0]) if isinstance(episodes, (list, tuple)) else int(str(episodes).split(",")[0])
    indices = list(range(first_count))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out) if out else ROOT / "outputs" / "rl_lab" / f"knobs_{mode}_{task}_{entropy}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(
        mode=mode,
        task=task,
        entropy=entropy,
        episodes=episodes,
        pop=pop,
        iters=iters,
        elite_frac=elite_frac,
        sigma0=sigma0,
        seed=seed,
        segments=segments,
        knobs=T.NAMES,
    )
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(
        f"{mode} search on {task}, root {entropy}, {episodes} episode(s), {len(T.NAMES)} knobs "
        f"x {segments} segment(s) = {len(T.NAMES) * segments} dimensions, "
        f"{pop * iters} evaluations each -> {out}",
        flush=True,
    )

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if mode == "hindsight":
            payloads = [
                (task, entropy, episodes, i, pop, iters, elite_frac, sigma0, seed * 1000 + i, segments) for i in indices
            ]
            rows = []
            for row in pool.map(job_hindsight, payloads):
                rows.append(row)
                print(
                    f"  episode {row['episode']:3d}: {row['base_cost']:.4e} -> {row['best_cost']:.4e} "
                    f"({row['gain']:+.2%})",
                    flush=True,
                )
            base = np.array([r["base_cost"] for r in rows])
            best = np.array([r["best_cost"] for r in rows])
            rel = (base - best) / base
            (out / "result.json").write_text(json.dumps(rows, indent=2))
            print(
                f"\nhindsight gain over the shipped numbers: {rel.mean():+.4%} "
                f"± {rel.std(ddof=1) / np.sqrt(len(rel)):.4%}, better in {int((rel > 0).sum())}/{len(rel)}"
            )
            print(f"total cost {base.sum():.6e} -> {best.sum():.6e}")
        else:
            # one candidate is judged on every network asked for, each cost divided by naive's week there, so
            # that Small and Full weigh the same: the same zip plays both boards
            # fire turns `--task=small,full` into a tuple and `--task=small` into a string: take both
            def listed(v):
                items = v if isinstance(v, (list, tuple)) else str(v).split(",")
                return [str(x).strip() for x in items if str(x).strip()]

            nets_asked = listed(task)
            counts = listed(episodes)
            eps_of = (
                {t: int(n) for t, n in zip(nets_asked, counts)}
                if len(counts) == len(nets_asked)
                else {t: int(counts[0]) for t in nets_asked}
            )
            plan = [(t, list(range(eps_of[t]))) for t in nets_asked]
            print(f"  judged on: {', '.join(f'{t} x{len(ix)}' for t, ix in plan)}", flush=True)

            def judge(xs):
                """The mean scaled cost of each candidate over all the networks, one pool map for the lot."""
                jobs = [(t, entropy, len(ix), ix, x) for x in xs for t, ix in plan]
                got = np.array(list(pool.map(job_static, jobs))).reshape(len(xs), len(plan))
                return got.mean(axis=1)

            rng = np.random.default_rng(seed)
            mean = values_to_unit(T.clip(T.DEFAULTS))
            sigma = np.full(len(T.NAMES), float(sigma0))
            base = float(judge([mean])[0])
            best_x, best_cost, history = mean.copy(), base, []
            for it in range(iters):
                xs = np.clip(rng.normal(mean, sigma, size=(pop, len(mean))), 0.0, 1.0)
                xs[0] = mean
                costs = judge(xs)
                order = np.argsort(costs)
                if costs[order[0]] < best_cost:
                    best_cost, best_x = float(costs[order[0]]), xs[order[0]].copy()
                elite = xs[order[: max(2, int(pop * elite_frac))]]
                mean = elite.mean(axis=0)
                sigma = np.maximum(elite.std(axis=0), 0.02) * (1.0 - 0.5 * it / max(iters - 1, 1))
                history.append({"iter": it + 1, "best": best_cost, "gain": (base - best_cost) / base})
                print(
                    f"  iter {it + 1:3d}: best {best_cost:.6e} ({(base - best_cost) / base:+.4%}), "
                    f"{time.perf_counter() - t0:.0f} s",
                    flush=True,
                )
                # written every iteration, not only at the end: a run that has to be cut short is still usable
                (out / "params.json").write_text(json.dumps(unit_to_values(best_x), indent=2))
                (out / "result.json").write_text(
                    json.dumps(
                        {
                            "base_cost": base,
                            "best_cost": best_cost,
                            "gain": (base - best_cost) / base,
                            "values": unit_to_values(best_x),
                            "history": history,
                            "iters_done": it + 1,
                        },
                        indent=2,
                    )
                )
            values = unit_to_values(best_x)
            (out / "params.json").write_text(json.dumps(values, indent=2))
            (out / "result.json").write_text(
                json.dumps(
                    {
                        "base_cost": base,
                        "best_cost": best_cost,
                        "gain": (base - best_cost) / base,
                        "values": values,
                        "history": history,
                    },
                    indent=2,
                )
            )
            print(f"\nstatic gain over the shipped numbers: {(base - best_cost) / base:+.4%}")
            print(f"params written to {out / 'params.json'}")
    print(f"done in {time.perf_counter() - t0:.0f} s: {out}", flush=True)
    return str(out)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
