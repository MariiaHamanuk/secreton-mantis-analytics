"""Exact advantages: what a week's action is really worth, measured instead of estimated.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/counterfactual.py \
        --task=small --entropy=555 --episodes=8 --weeks=6 --out=outputs/rl_lab/cf_small

This is the piece that makes a learner possible here. PPO had to infer the value of an action from the cost of a
whole episode, and the measurement was hopeless: an action moves 0.04 to 0.5 % of the cost while the scenario moves
tens of percent, so the critic explained almost nothing and the policy found a constant bias instead of a policy.

The simulator makes the measurement unnecessary. Its state restores to the cent and an episode plays in 0.04 s, so
for a state and two actions the difference in finished cost can be **computed**: restore, play A to the end, restore,
play B to the end, subtract. Under the same scenario and the same tail policy, that difference is the advantage —
no critic, no discount, no variance.

What this writes is a dataset: for sampled (episode, week) states, every candidate action with the exact cost of
finishing the episode after it, plus the week's feature tables. The labels are what a policy can be fitted to by
ranking or regression — a supervised problem on an exact signal, which is the opposite of what PPO was given.

Costs: one record is (candidates) rollouts of the remaining weeks. On Small a tail averages ~0.1 s, so a state with
21 candidates costs ~2 s and a dataset of 8 episodes x 6 weeks is minutes, not hours.
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

from action_search import FACTORS, _env, _finish, _restore, _snapshot, groups_of  # noqa: E402


def record(payload):
    """One episode: at ``weeks`` sampled weeks, the exact finished cost of every candidate action."""
    from shockbench_flow_gym import agent_config_from_reset

    task, entropy, n_scenarios, index, weeks, grouping, seed, rules_folder, base_folder, with_tables = payload
    env, parts = _env(task, entropy, n_scenarios)
    if base_folder:  # measure against a full submission agent (e.g. the hub) instead of the composed rules
        from sbf_starter.agents import load as load_agent

        cls = load_agent(base_folder)

        class _BaseParts:
            @staticmethod
            def agent(config, values=None, schedule=None):
                return cls(config)

        parts = _BaseParts()
    rng = np.random.default_rng(seed)

    obs, info = env.reset(options={"pool_index": int(index)})
    config = agent_config_from_reset(env, obs, info)
    groups = groups_of(config, grouping)
    masks = [m for _, m in groups]
    horizon = int(config["T"])
    # the weeks to probe: spread over the episode, and never the last, where there is no tail to measure
    wanted = sorted(rng.choice(np.arange(1, horizon - 1), size=min(weeks, horizon - 2), replace=False).tolist())

    feat = None
    if with_tables:  # the feature tables need torch; the advantages themselves do not
        import residual as R

        rules_module = R.load_rules(rules_folder)
        feat = R.Residual(config, rules_module, None)
    agent = parts.agent(config, None)
    rows, w = [], 0
    while True:
        proposal = agent.act(obs)
        flows = np.asarray(proposal["flows"], dtype=np.float64)
        if feat is not None:
            feat.history.update(feat.layout, obs)  # every week: the counters of how long a cut or a shed has lasted
        if w in wanted:
            tables = feat.tables(obs, flows) if feat is not None else None  # the same features the networks read
            snap = _snapshot(env)
            cands = []
            for j, mask in enumerate(masks):
                if not np.any(flows[mask] > 0.0):
                    continue  # a coordinate the rules leave empty cannot be scaled
                for f in FACTORS:
                    scale = np.ones(flows.shape[0])
                    scale[mask] = f
                    _restore(env, snap)
                    cost = _finish(env, parts, config, obs, {**proposal, "flows": flows * scale}, w, None, masks)
                    cands.append({"coordinate": j, "factor": float(f), "cost_to_go": float(cost)})
            _restore(env, snap)
            # the reference: the rules' own action, finished by the rules — the zero of every advantage below
            plain = _finish(env, parts, config, obs, proposal, w, None, masks)
            _restore(env, snap)
            best = min(cands, key=lambda c: c["cost_to_go"]) if cands else None
            rows.append(
                {
                    "episode": int(index),
                    "week": w,
                    "plain_cost_to_go": float(plain),
                    "candidates": cands,
                    "best": best,
                    "exact_advantage_of_best": float(plain - best["cost_to_go"]) if best else 0.0,
                    # the tables stay as arrays: as JSON a few hundred states would be a hundred megabytes of text
                    "_tables": tables,
                }
            )
        obs, reward, terminated, truncated, _ = env.step(proposal)
        w += 1
        if terminated or truncated:
            break
    return rows


def main(
    task="small",
    entropy=555,
    episodes=8,
    weeks=6,
    grouping="destination",
    workers=10,
    seed=0,
    rules="agents/anastasiia_rules_v2",
    base=None,
    out=None,
    keep_features=False,
):
    folder = Path(rules) if Path(rules).is_absolute() else ROOT / rules
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out) if out else ROOT / "outputs" / "rl_lab" / f"cf_{task}_{entropy}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(
        json.dumps(
            {
                "task": task,
                "entropy": entropy,
                "episodes": episodes,
                "weeks": weeks,
                "grouping": grouping,
                "factors": FACTORS,
                "rules": str(folder.relative_to(ROOT)),
                "seed": seed,
            },
            indent=2,
        )
    )
    print(
        f"exact advantages on {task}, root {entropy}, {episodes} episode(s) x {weeks} week(s), "
        f"grouping {grouping} -> {out}",
        flush=True,
    )

    t0 = time.perf_counter()
    base_folder = str(Path(base) if Path(str(base)).is_absolute() else ROOT / base) if base else None
    jobs = [
        (task, entropy, episodes, i, weeks, grouping, seed * 100 + i, str(folder), base_folder, keep_features)
        for i in range(episodes)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(record, jobs):
            for r in got:
                adv = r["exact_advantage_of_best"] / max(r["plain_cost_to_go"], 1.0)
                print(
                    f"  episode {r['episode']:3d} week {r['week']:3d}: {len(r['candidates']):3d} candidates, "
                    f"best beats the rules by {adv:+.3%} of the cost to go",
                    flush=True,
                )
            rows.extend(got)

    gains = np.array([r["exact_advantage_of_best"] / max(r["plain_cost_to_go"], 1.0) for r in rows])
    tables = [r.pop("_tables") for r in rows]
    if keep_features and rows:  # one array per table, states stacked: the shapes are fixed within a network
        np.savez_compressed(
            out / "features.npz",
            **{name: np.stack([t[i] for t in tables]) for i, name in enumerate(("node", "edge", "slot", "global"))},
        )
        for i, r in enumerate(rows):
            r["features_row"] = i
    (out / "dataset.json").write_text(json.dumps(rows))
    print(
        f"\n{len(rows)} state(s); one week's best candidate is worth "
        f"{gains.mean():+.4%} ± {gains.std(ddof=1) / np.sqrt(len(gains)):.4%} of the cost to go"
    )
    print(f"  states where some candidate beats the rules: {int((gains > 0).sum())}/{len(gains)}")
    print(f"done in {time.perf_counter() - t0:.0f} s: {out}", flush=True)
    return str(out)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
