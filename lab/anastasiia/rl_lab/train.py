"""PPO over the correction to the rules. One run answers one question of the brief.

    .venv/bin/python lab/anastasiia/rl_lab/train.py --kind=slot --tasks=small --iters=40

What a run does: plays whole episodes in worker processes with the current weights, turns the weeks into one batch
whatever networks they came from, and updates. Every few iterations it replays a held-out set of scenarios with the
mean correction and reports the cost against the very same scenarios played by the plain rules — a paired number,
which is the only kind this repository trusts.

Written here rather than with Stable-Baselines3 because the brief asks for Small and Full mixed from the start, and
a vectorised SB3 run fixes one observation and one action shape for the whole run: Small has 108 action slots and
Full has 395.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import batching  # noqa: E402
import nets  # noqa: E402
import rollout  # noqa: E402


DEFAULT_RULES = ROOT / "agents" / "anastasiia_rules_v2"
TRAIN_ENTROPY = 555  # training scenarios; 666 is the held-out set. Neither is 111/222/333/444.
EVAL_ENTROPY = 666


def gae(costs, values, gamma, lam, centre=0.0):
    """Advantages and returns of one episode.

    The reward of a week is minus what it cost, less what the plain rules spend. ``centre`` is either a constant —
    the rules' average week on this network, which changes no policy gradient but leaves the returns near zero
    instead of near −30 — or the rules' own week-by-week costs on this very scenario (``--control``), which also
    takes out the part of the week that the scenario decided and the agent could not.
    """
    centre = np.asarray(centre, dtype=np.float64)
    if centre.ndim and centre.shape[0] != len(costs):  # the rules' episode may differ in length by a week
        centre = np.resize(centre, len(costs))
    rewards = -(costs - centre)
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    carry = 0.0
    for t in range(n - 1, -1, -1):
        nxt = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        carry = delta + gamma * lam * carry
        adv[t] = carry
    return adv, adv + values[:n]


def main(
    kind="slot",
    tasks="small",
    iters=40,
    episodes_per_iter=16,
    n_scenarios=512,
    entropy=TRAIN_ENTROPY,
    eval_entropy=EVAL_ENTROPY,
    eval_episodes=24,
    eval_every=4,
    width=64,
    rounds=2,
    lr=3e-4,
    gamma=0.997,
    lam=0.95,
    clip=0.2,
    epochs=3,
    minibatch=256,
    value_coef=0.5,
    entropy_coef=0.0,
    log_std=-1.0,
    lo=0.5,
    hi=2.0,
    control=False,
    families=None,
    groups=None,
    rules=str(DEFAULT_RULES),
    workers=9,
    seed=0,
    out=None,
    note="",
):
    global FAMILIES

    # fire turns `--tasks=small,full` into a tuple and `--tasks=small` into a string: take both
    def names(v):
        items = v if isinstance(v, (list, tuple)) else str(v).split(",")
        return tuple(str(x).strip() for x in items if str(x).strip())

    tasks = list(names(tasks))
    groups = names(groups) if groups else None
    families = names(families) if families else None
    FAMILIES = families
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(out) if out else ROOT / "outputs" / "rl_lab" / f"{kind}_{'-'.join(tasks)}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)  # the minibatch order of an update is drawn from numpy's global generator
    torch.set_num_threads(1)

    sizes, indices = None, {}
    for task in tasks:
        s, ix, _ = rollout.sizes_for(task, rules, groups)
        indices[task] = ix
        if sizes is None:
            sizes = s
        elif sizes != s:
            raise SystemExit(f"feature widths differ between networks: {sizes} vs {s} on {task}")
    print(f"feature widths {sizes}", flush=True)

    policy = nets.build(kind, sizes, width, rounds, log_std)
    n_params = sum(p.numel() for p in policy.parameters())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    weights = out / "policy.pt"
    torch.save(policy.state_dict(), weights)

    settings = dict(
        kind=kind,
        tasks=tasks,
        iters=iters,
        episodes_per_iter=episodes_per_iter,
        n_scenarios=n_scenarios,
        entropy=entropy,
        eval_entropy=eval_entropy,
        eval_episodes=eval_episodes,
        width=width,
        rounds=rounds,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip=clip,
        epochs=epochs,
        minibatch=minibatch,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        log_std=log_std,
        lo=lo,
        hi=hi,
        control=control,
        families=families,
        groups=groups,
        # relative to the repository when it lies inside it, so that a run folder reads on another machine
        rules=str(Path(rules).resolve().relative_to(ROOT))
        if Path(rules).resolve().is_relative_to(ROOT)
        else str(rules),
        seed=seed,
        n_params=n_params,
        sizes=sizes,
        note=note,
    )
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"{n_params} parameters, writing to {out}", flush=True)

    # the scenarios, drawn once here rather than nine times over inside the workers
    from shockbench_flow_gym.wrappers import draw_scenarios

    for task in tasks:
        t0 = time.perf_counter()
        draw_scenarios(task, n_scenarios, entropy, n_jobs=workers)
        draw_scenarios(task, max(eval_episodes, 1), eval_entropy, n_jobs=workers)
        print(f"{task}: scenarios ready in {time.perf_counter() - t0:.0f} s", flush=True)

    rng = np.random.default_rng(seed)
    pool = ProcessPoolExecutor(max_workers=workers, initializer=rollout.setup, initargs=(rules,))
    log = []
    baseline = {}
    centre = {}
    rules_costs = {}
    try:
        # what a week of the plain rules costs on this root: the constant the returns are measured from
        for task in tasks:
            probe = [
                _job(
                    task,
                    int(rng.integers(n_scenarios)),
                    entropy,
                    n_scenarios,
                    "centre",
                    weights,
                    kind,
                    sizes,
                    width,
                    rounds,
                    groups,
                    lo,
                    hi,
                    False,
                    0,
                )
                for _ in range(min(workers * 2, 18))
            ]
            played = list(pool.map(rollout.play, probe))
            centre[task] = float(np.mean([ep["costs"].mean() for ep in played]))
        print(f"rules cost per week: { {k: round(v, 4) for k, v in centre.items()} }", flush=True)
        settings["centre"] = centre
        (out / "settings.json").write_text(json.dumps(settings, indent=2))
        for it in range(1, iters + 1):
            version = f"{it}"
            torch.save(policy.state_dict(), weights)
            jobs = []
            for i in range(episodes_per_iter):
                task = tasks[i % len(tasks)]
                jobs.append(
                    _job(
                        task,
                        int(rng.integers(n_scenarios)),
                        entropy,
                        n_scenarios,
                        version,
                        weights,
                        kind,
                        sizes,
                        width,
                        rounds,
                        groups,
                        lo,
                        hi,
                        True,
                        int(rng.integers(2**31)),
                    )
                )
            t0 = time.perf_counter()
            episodes = list(pool.map(rollout.play, jobs))
            if control:  # the plain rules on the very same scenarios; deterministic, so computed once and kept
                missing = sorted({(ep["task"], ep["pool_index"]) for ep in episodes} - set(rules_costs))
                if missing:
                    probes = [
                        dict(
                            _job(
                                task,
                                idx,
                                entropy,
                                n_scenarios,
                                "rules",
                                weights,
                                kind,
                                sizes,
                                width,
                                rounds,
                                groups,
                                lo,
                                hi,
                                False,
                                0,
                            ),
                            rules_only=True,
                        )
                        for task, idx in missing
                    ]
                    for key, played in zip(missing, pool.map(rollout.play, probes)):
                        rules_costs[key] = played["costs"]
                for ep in episodes:
                    ep["control"] = rules_costs[(ep["task"], ep["pool_index"])]
            t_play = time.perf_counter() - t0

            t0 = time.perf_counter()
            stats = update(
                policy, opt, episodes, indices, gamma, lam, clip, epochs, minibatch, value_coef, entropy_coef, centre
            )
            t_update = time.perf_counter() - t0

            cost = float(np.mean([ep["costs"].sum() for ep in episodes]))
            row = {"iter": it, "cost": cost, "play_s": round(t_play, 1), "update_s": round(t_update, 1), **stats}
            if control:  # the training scenarios, paired: a curve that is readable iteration by iteration
                paired = [
                    (ep["control"].sum() - ep["costs"].sum()) / ep["control"].sum()
                    for ep in episodes
                    if "control" in ep
                ]
                row["train_gain"] = round(float(np.mean(paired)), 5)
                row["train_gain_se"] = round(float(np.std(paired, ddof=1) / np.sqrt(len(paired))), 5)
            if it % eval_every == 0 or it == iters:
                torch.save(policy.state_dict(), weights)
                # a version of its own: the workers cache by it, and these are the weights after this iteration's
                # update, not the ones the rollouts were played with
                row.update(
                    evaluate(
                        pool,
                        policy,
                        weights,
                        f"{it}-eval",
                        tasks,
                        indices,
                        eval_entropy,
                        eval_episodes,
                        kind,
                        sizes,
                        width,
                        rounds,
                        groups,
                        lo,
                        hi,
                        baseline,
                    )
                )
                torch.save(policy.state_dict(), out / f"policy_{it:04d}.pt")
            log.append(row)
            (out / "log.json").write_text(json.dumps(log, indent=2))
            print(json.dumps(row), flush=True)
    finally:
        pool.shutdown(wait=True)
    torch.save(policy.state_dict(), out / "policy.pt")
    print(f"done: {out}", flush=True)
    return str(out)


FAMILIES = None  # set by main(); which commodities the correction may touch


def _job(
    task, pool_index, entropy, n_scenarios, version, weights, kind, sizes, width, rounds, groups, lo, hi, sample, seed
):
    return {
        "task": task,
        "pool_index": pool_index,
        "entropy": entropy,
        "n_scenarios": n_scenarios,
        "version": version,
        "policy_path": str(weights),
        "kind": kind,
        "sizes": sizes,
        "width": width,
        "rounds": rounds,
        "groups": groups,
        "lo": lo,
        "hi": hi,
        "sample": sample,
        "seed": seed,
        "families": FAMILIES,
    }


def update(policy, opt, episodes, indices, gamma, lam, clip, epochs, minibatch, value_coef, entropy_coef, centre=None):
    centre = centre or {}
    weeks, samples, lives, logps, advs, rets = [], [], [], [], [], []
    for ep in episodes:
        values = np.array([w["value"] for w in ep["weeks"]], dtype=np.float64)
        base = ep.get("control")
        if base is None:
            base = centre.get(ep["task"], 0.0)
        adv, ret = gae(ep["costs"][: len(values)], values, gamma, lam, base)
        for i, w in enumerate(ep["weeks"]):
            weeks.append((ep["task"], w["node"], w["edge"], w["slot"], w["global"]))
            samples.append(w["sample"])
            lives.append(w["live"])
            logps.append(w["logp"])
        advs.append(adv)
        rets.append(ret)
    advs = np.concatenate(advs)
    rets = np.concatenate(rets)
    old_logp = torch.tensor(logps, dtype=torch.float32)
    advs_t = torch.tensor((advs - advs.mean()) / (advs.std() + 1e-8), dtype=torch.float32)
    rets_t = torch.tensor(rets, dtype=torch.float32)

    n = len(weeks)
    order = np.arange(n)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0, "clipfrac": 0.0, "batches": 0}
    for _ in range(epochs):
        np.random.shuffle(order)
        for start in range(0, n, minibatch):
            rows = order[start : start + minibatch]
            if len(rows) < 2:
                continue
            batch = batching.collate([weeks[i] for i in rows], indices)
            sample = torch.from_numpy(np.concatenate([samples[i] for i in rows]))
            live = torch.from_numpy(np.concatenate([lives[i] for i in rows]))
            raw, value = policy(batch)
            logp_slot = policy.log_prob(sample, raw) * live
            logp = torch.zeros(len(rows)).index_add_(0, batch["slot_graph"], logp_slot)
            ratio = (logp - old_logp[rows]).exp()
            a = advs_t[rows]
            loss_p = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            loss_v = torch.nn.functional.mse_loss(value, rets_t[rows])
            loss = loss_p + value_coef * loss_v - entropy_coef * policy.entropy().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                totals["policy_loss"] += float(loss_p)
                totals["value_loss"] += float(loss_v)
                totals["kl"] += float((old_logp[rows] - logp).mean())
                totals["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean())
                totals["batches"] += 1
    b = max(totals.pop("batches"), 1)
    stats = {k: round(v / b, 5) for k, v in totals.items()}
    stats["log_std"] = round(float(policy.log_std.detach()), 4)
    stats["weeks"] = n
    # how much of the return the critic explains: near 0 means the advantages are mostly noise and
    # nothing will be learnt however long the run is
    values_all = np.array([w["value"] for ep in episodes for w in ep["weeks"]], dtype=np.float64)
    stats["explained"] = round(float(1.0 - np.var(rets - values_all) / max(np.var(rets), 1e-12)), 4)
    stats["adv_std"] = round(float(advs.std()), 4)
    return stats


def evaluate(
    pool,
    policy,
    weights,
    version,
    tasks,
    indices,
    entropy,
    episodes,
    kind,
    sizes,
    width,
    rounds,
    groups,
    lo,
    hi,
    baseline,
):
    """The held-out scenarios with the mean correction, against the plain rules on the very same scenarios."""
    out = {}
    for task in tasks:
        jobs = [
            _job(
                task,
                i,
                entropy,
                max(episodes, 1),
                version,
                weights,
                kind,
                sizes,
                width,
                rounds,
                groups,
                lo,
                hi,
                False,
                0,
            )
            for i in range(episodes)
        ]
        played = list(pool.map(rollout.play, jobs))
        cost = np.array([ep["costs"].sum() for ep in played])
        if task not in baseline:  # the rules alone: a head that is still zero gives exactly their action
            zero = nets.build(kind, sizes, width, rounds, -20.0)
            path = Path(weights).with_name(f"zero_{task}.pt")
            torch.save(zero.state_dict(), path)
            base_jobs = [
                _job(
                    task,
                    i,
                    entropy,
                    max(episodes, 1),
                    f"zero-{task}",
                    path,
                    kind,
                    sizes,
                    width,
                    rounds,
                    groups,
                    lo,
                    hi,
                    False,
                    0,
                )
                for i in range(episodes)
            ]
            baseline[task] = np.array([ep["costs"].sum() for ep in pool.map(rollout.play, base_jobs)])
        rel = (baseline[task] - cost) / baseline[task]
        out[f"eval_{task}"] = round(float(cost.mean()), 4)
        out[f"gain_{task}"] = round(float(rel.mean()), 5)
        out[f"gain_{task}_se"] = round(float(rel.std(ddof=1) / np.sqrt(len(rel))), 5)
        out[f"fallbacks_{task}"] = int(sum(ep["fallbacks"] for ep in played))
    return out


if __name__ == "__main__":
    import fire

    fire.Fire(main)
