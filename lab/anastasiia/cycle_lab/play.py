"""Play agent folders under gymnasium and keep every episode's cost, so that variants are compared without replaying.

    uv run python lab/anastasiia/regime_lab/play.py run agents/anastasiia_hybrid_hub --tag=hub --entropy=111 --episodes=16
    uv run python lab/anastasiia/regime_lab/play.py run outputs/regime_lab/agents/regime_x --tag=x --entropy=111 --episodes=16
    uv run python lab/anastasiia/regime_lab/play.py show hub x --entropy=111 --episodes=16

``run`` keeps, per episode, the environment's cost, each week's costs by item, lots, shed load, lost sales, stocks and
disposal, and the agent's own ``log`` when it has one (the lab agent's: CPU seconds and the model's costs per week) in ``outputs/regime_lab/play/<tag>_<task>_<entropy>.pkl``; episodes already
there are not played again. ``show`` prints each tag's score and its paired difference with the first tag, by
episode when asked. No CPU limit is applied, and the seconds are this machine's process time, not the container's.
The official paired interval is ``hub/eval/compare.py``'s; the one here is a bootstrap over episodes. ``arrays``
writes a tag's weeks as the ``.npz`` that ``plan_lab/ledger.py`` reads (generation, energy to the fabs, sales).
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
import plan  # noqa: E402


OUT = plan.OUT / "play"


def episode(agent: str, task: str, entropy: int, n: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    t0 = time.process_time()
    ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    if hasattr(ag, "tell_truth"):  # a lab agent may ask for the scenario's own network (its ``truth`` setting)
        ag.tell_truth({"marks": u.core._ep.marks})
    cpu, done = [], False
    while not done:
        action = ag.act(obs)
        cpu.append(time.process_time() - t0)
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
        t0 = time.process_time()
    recs = u.core._ep.traj.records
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    inst = u.core._ep.inst
    seg = np.zeros((len(recs), len(inst.grids), len(inst.commodities) + 1))  # last column: the no-fuel segment
    sent = np.zeros((len(recs), len(inst.action_slots)))
    for t, r in enumerate(recs):
        for (gi, k), q in r.segment.items():
            seg[t, gi, -1 if k is None else k] = q
        for slot, q in r.executed.items():
            sent[t, slot] = q
    return {
        "n": n, "J": int(u.core._ep.traj.J_cents), "cpu": cpu, "log": list(getattr(ag, "log", [])),
        "detail": list(getattr(ag, "detail", [])),  # the lab agent's: every run of the solver, by week
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
        "lost": np.array([r.lost for r in recs]), "demand": np.array([r.demand for r in recs]),
        "stock": np.array([r.stock for r in recs]), "disposal": np.array([r.disposal for r in recs]),
        "served": np.array([r.served for r in recs]), "energy": np.array([r.energy for r in recs]),
        "scrap": np.array([r.scrapped for r in recs]), "segment": seg, "sent": sent,
    }


def _path(tag: str, task: str, entropy: int) -> Path:
    return OUT / f"{tag}_{task}_{entropy}.pkl"


def run(agent: str, tag: str, task: str = "small", entropy: int = 111, episodes: int = 16, first: int = 0, n_jobs: int = 3,
        only: str | tuple = "") -> None:
    """``only``: episodes to play instead of ``first``..``first + episodes - 1`` ("2,11,16")."""
    path = _path(tag, task, entropy)
    kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
    wanted = [int(n) for n in (only.split(",") if isinstance(only, str) else only)] if only else range(first, first + episodes)
    todo = [n for n in wanted if n not in kept]
    for r in Parallel(n_jobs=n_jobs)(delayed(episode)(agent, task, entropy, n) for n in todo):
        kept[r["n"]] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(kept))
    if not only:
        show(tag, task=task, entropy=entropy, episodes=episodes, first=first)


def show(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 16, first: int = 0, by_episode: bool = False) -> None:
    refs = plan.references(task, entropy, first + episodes)
    ns = list(range(first, first + episodes))
    data = {tag: pickle.loads(_path(tag, task, entropy).read_bytes()) for tag in tags}
    ns = [n for n in ns if all(n in d for d in data.values()) and refs[n]["J_oracle_cents"] is not None]
    rs = [refs[n] for n in ns]
    room = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in rs], dtype=float)
    base = np.array([data[tags[0]][n]["J"] for n in ns], dtype=float)
    rng = np.random.default_rng(0)
    print(f"{task}, root {entropy}, {len(ns)} episodes ({ns[0]}..{ns[-1]}); score as `sbf evaluate` [levels weighing the same]")
    for tag in tags:
        J = np.array([data[tag][n]["J"] for n in ns], dtype=float)
        cpu = np.concatenate([data[tag][n]["cpu"] for n in ns])
        line = f"{tag:14s} {plan.both(rs, J.tolist())}  cpu/week median {np.median(cpu):.2f} p95 {np.percentile(cpu, 95):.2f} max {cpu.max():.2f}"
        if tag != tags[0]:
            d = base - J  # cents saved against the first tag
            boot = [d[i].sum() / room[i].sum() for i in (rng.integers(0, len(ns), len(ns)) for _ in range(2000))]
            lo, hi = np.percentile(boot, [5, 95])
            line += f"  vs {tags[0]}: {d.sum() / room.sum():+.4f} ({lo:+.4f} to {hi:+.4f}), better in {int((d > 0).sum())} of {len(ns)}, {d.mean() / 1e11:+.1f} bn an episode"
        print(line)
    if by_episode:
        for i, n in enumerate(ns):
            print(f"  ep {n:3d} lvl {rs[i]['stratum']}: " + "  ".join(f"{tag} {(rs[i]['J_naive_cents'] - data[tag][n]['J']) / room[i]:.4f}" for tag in tags))


def arrays(tag: str, out: str, task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0) -> None:
    """Write a tag's weekly arrays in the layout ``plan_lab/ledger.py`` reads, so both labs keep one ledger."""
    kept = pickle.loads(_path(tag, task, entropy).read_bytes())
    ns = range(first, first + episodes)
    names = {"cost": "costs", "lots": "lots", "scrap": "scrap", "energy": "energy", "disposal": "disposal", "stock": "stock",
             "shed": "shed", "served": "served", "demand": "demand", "segment": "segment", "sent": "sent"}  # fmt: skip
    data = {theirs: np.stack([kept[n][ours] for n in ns]) for theirs, ours in names.items()}
    data["J"] = np.array([kept[n]["J"] / 100 for n in ns])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show, "arrays": arrays})
