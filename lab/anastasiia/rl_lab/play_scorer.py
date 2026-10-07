"""Play whole episodes with the learnt scorer, causally, and compare with the rules on the same scenarios.

    PYTHONPATH=src .venv/bin/python lab/anastasiia/rl_lab/play_scorer.py \
        --scorer=outputs/rl_lab/cf_small/scorer --task=small --entropy=666 --episodes=24

Every number in this session's searches used the episode's true future in the tail of its rollouts, so every one of
them is an upper bound. This is the first that is not: the scorer sees only the observation, scores each candidate
from it, and the best is applied when its predicted advantage is positive. No rollouts, no foresight, nothing the
submission could not do.

One week costs one pass of the slot encoder over the action slots plus one head over the candidates — milliseconds,
against a budget of 2 s on Small and 4 s on Full.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


_STATE = {}


def _ready(task, entropy, n_scenarios, scorer_dir, rules_folder):
    import gymnasium as gym
    import residual as R
    import torch
    import tuned_rules as T
    from learn_advantage import Scorer
    from shockbench_flow_gym.wrappers import ScenarioPool

    from sbf_starter import env_id

    key = (task, entropy, n_scenarios)
    if key not in _STATE:
        _STATE[key] = ScenarioPool(gym.make(env_id(task), regime="standard"), n_scenarios, entropy)
    if "parts" not in _STATE:
        _STATE["parts"] = T.Parts()
        _STATE["rules"] = R.load_rules(rules_folder)
    if "scorer" not in _STATE:
        meta = json.loads((Path(scorer_dir) / "meta.json").read_text())
        model = Scorer(meta["sizes"], meta["width"])
        model.load_state_dict(torch.load(Path(scorer_dir) / "scorer.pt", map_location="cpu", weights_only=True))
        model.eval()
        torch.set_num_threads(1)
        _STATE["scorer"] = (model, meta)
    return _STATE[key], _STATE["parts"], _STATE["rules"], _STATE["scorer"]


def play(payload):
    """One episode by the scorer and the same episode by the plain rules. Costs in USD."""
    import residual as R
    import torch
    from action_search import groups_of
    from learn_advantage import coordinate_destinations
    from shockbench_flow_gym import agent_config_from_reset

    task, entropy, n_scenarios, index, scorer_dir, rules_folder, threshold, mode = payload
    env, parts, rules_module, (model, meta) = _ready(task, entropy, n_scenarios, scorer_dir, rules_folder)
    factors = tuple(meta["factors"])

    # the plain rules on this scenario
    obs, info = env.reset(options={"pool_index": int(index)})
    config = agent_config_from_reset(env, obs, info)
    agent = parts.agent(config, None)
    base = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs))
        base += -float(reward)
        if terminated or truncated:
            break

    groups = groups_of(config, meta["grouping"])
    masks = [np.flatnonzero(m) for _, m in groups]
    coord_dest = coordinate_destinations(config, meta["grouping"])  # the node each coordinate sends to

    obs, info = env.reset(options={"pool_index": int(index)})
    agent = parts.agent(config, None)
    feat = R.Residual(config, rules_module, None)
    total, moved, weeks = 0.0, 0, 0
    while True:
        proposal = agent.act(obs)
        flows = np.asarray(proposal["flows"], dtype=np.float64)
        feat.history.update(feat.layout, obs)
        node, _, slot, glob = feat.feat.week(obs, flows, feat.history)
        live = [j for j, m in enumerate(masks) if m.size and np.any(flows[m] > 0.0)]
        action = proposal
        if live:
            idx, owner, fs, pairs, dest = [], [], [], [], []
            for j in live:
                for f in factors:
                    idx.extend(masks[j].tolist())
                    owner.extend([len(fs)] * masks[j].size)
                    fs.append(float(f))
                    pairs.append((j, float(f)))
                    dest.append(coord_dest[j])
            with torch.inference_mode():
                pred = model(
                    torch.from_numpy(slot),
                    torch.from_numpy(glob).unsqueeze(0),
                    torch.tensor(idx, dtype=torch.long),
                    torch.tensor(owner, dtype=torch.long),
                    torch.tensor(fs, dtype=torch.float32),
                    torch.from_numpy(node),
                    torch.tensor(dest, dtype=torch.long),
                )
            scale = np.ones(flows.shape[0])
            touched = 0
            if mode == "best":  # the single most promising coordinate, as the first version did
                k = int(torch.argmax(pred))
                if float(pred[k]) > threshold:
                    j, f = pairs[k]
                    scale[masks[j]] = f
                    touched = 1
            else:  # every coordinate the model expects to gain on, which is what the search itself did
                by_coord = {}
                for k, (j, f) in enumerate(pairs):
                    v = float(pred[k])
                    if j not in by_coord or v > by_coord[j][0]:
                        by_coord[j] = (v, f)
                for j, (v, f) in by_coord.items():
                    if v > threshold and f != 1.0:
                        scale[masks[j]] = f
                        touched += 1
            if touched:
                out = flows * scale
                if np.all(np.isfinite(out)) and np.all(out >= 0.0):
                    action = {**proposal, "flows": out}
                    moved += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        total += -float(reward)
        weeks += 1
        if terminated or truncated:
            break
    return {
        "episode": int(index),
        "base_cost": base,
        "cost": total,
        "gain": (base - total) / base,
        "weeks_moved": moved,
        "weeks": weeks,
    }


def main(
    scorer,
    task="small",
    entropy=666,
    episodes=24,
    workers=10,
    threshold=0.0,
    mode="all",
    rules="agents/anastasiia_rules_v2",
):
    """``mode='all'`` moves every coordinate the model expects to gain on, ``'best'`` only the single top one."""
    scorer_dir = Path(scorer) if Path(scorer).is_absolute() else ROOT / scorer
    folder = Path(rules) if Path(rules).is_absolute() else ROOT / rules
    jobs = [
        (task, entropy, max(episodes, 1), i, str(scorer_dir), str(folder), threshold, mode) for i in range(episodes)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(play, jobs):
            rows.append(r)
            print(
                f"  episode {r['episode']:3d}: {r['base_cost']:.4e} -> {r['cost']:.4e} "
                f"({r['gain']:+.3%}; moved {r['weeks_moved']}/{r['weeks']} weeks)",
                flush=True,
            )
    g = np.array([r["gain"] for r in rows])
    se = g.std(ddof=1) / np.sqrt(len(g))
    print(f"\ncausal gain over the rules on {task}, root {entropy}, {episodes} episode(s): {g.mean():+.4%} ± {se:.4%}")
    print(
        f"  90 % interval {g.mean() - 1.645 * se:+.4%} to {g.mean() + 1.645 * se:+.4%}; "
        f"better in {int((g > 0).sum())}/{len(g)}"
    )
    print(f"  weeks the model moved: {sum(r['weeks_moved'] for r in rows)}/{sum(r['weeks'] for r in rows)}")
    return float(g.mean())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
