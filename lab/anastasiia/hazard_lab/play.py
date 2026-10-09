"""Play agent folders under gymnasium and keep every episode's cost; compare the kept costs in pairs.

    uv run python lab/anastasiia/hazard_lab/play.py run outputs/hazard_lab/agents/base_s --tag=base_s --episodes=64
    uv run python lab/anastasiia/hazard_lab/play.py show base_s ends_all_s --episodes=64

``run`` keeps, per episode, the environment's cost, the week's costs by item, lots, shed load, lost sales and what
was sent, and the agent's own ``log`` (CPU seconds and the model's costs per week), in
``outputs/hazard_lab/play/<tag>_<task>_<entropy>.pkl``, written after every episode; episodes already there are not
played again. No CPU limit is applied; the seconds are this machine's process time.

A lab agent with ``tell_truth`` is handed the scenario: its marks and its events, each event with the factors it
alone puts on the edge capacities, on the straits' openness and on the grids' output (``shockbench_flow.marks.graph_marks`` of that one
event), by week. An agent uses them only if its own settings say so.

``show`` prints each tag's score as ``sbf evaluate`` gives it, by harm level, and its paired difference with the
first tag: the episodes are drawn again with replacement inside each harm level, 2,000 times, both tags on the same
draw; the interval runs from the 5th to the 95th percentile of the differences. A set with one or two episodes of a
harm level gives that level's weight to those episodes and no spread to them, so the line also has the difference as
all that was saved over all that could be, the episodes drawn again without regard to the level.
"""

import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "hazard_lab" / "play"
WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}  # the board's weight of each harm level


def events_of(inst, omega) -> list[dict]:
    """The scenario's events, each with what it alone does to the edge capacities and the straits' openness."""
    from shockbench_flow import marks as M
    from shockbench_flow.omega import codes

    params = M.mark_params_from_json(str(omega.arrays["meta_mark_params"]))
    inst = inst.at_digest(str(omega.arrays["meta_instance_digest"]))
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
    G0 = np.array([inst.nodes[g].grid.deliverable for g in inst.grids], dtype=float)
    out = []
    for q in M.read_events(inst, omega.arrays):
        g = M.graph_marks(inst, (q,), params)
        ev = {"type": codes.EVENT_TYPES[q.type], "onset": q.onset, "duration": q.duration, "severity": q.severity,
              "region": q.region, "counterpart": q.counterpart, "target_kind": codes.TARGET_KINDS[q.target_kind],
              "target": q.target, "commodity": q.commodity}  # fmt: skip
        finite = np.isfinite(u0)
        ratio = {"u": (np.where(finite, g["u"] / np.where(finite, u0, 1.0), 1.0),
                       np.where(finite, g["u_now"] / np.where(finite, u0, 1.0), 1.0)),
                 "o": (g["o"], g["o_now"]),
                 "G_bar": (g["G_bar"] / G0, g["G_bar_now"] / G0)}  # fmt: skip
        for name, (avg, now) in ratio.items():
            idx = np.flatnonzero((avg < 1.0 - 1e-12).any(axis=0) | (now < 1.0 - 1e-12).any(axis=0))
            ev[name] = (idx, np.ascontiguousarray(avg[:, idx]), np.ascontiguousarray(now[:, idx]))
        out.append(ev)
    return out


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
    if hasattr(ag, "tell_truth"):  # a lab agent may ask for the scenario's own network (its settings say which part)
        wants = getattr(ag, "p", {}).get("truth_events") or getattr(ag, "p", {}).get("events_seen")
        ag.tell_truth({"marks": u.core._ep.marks, "events": events_of(u.core._ep.inst, u._omega(n)) if wants else None})
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
    sent = np.zeros((len(recs), len(inst.action_slots)))
    seg = np.zeros((len(recs), len(inst.grids), len(inst.commodities) + 1))  # last column: the no-fuel segment
    for t, r in enumerate(recs):
        for slot, q in r.executed.items():
            sent[t, slot] = q
        for (gi, k), q in r.segment.items():
            seg[t, gi, -1 if k is None else k] = q
    return {
        "n": n, "J": int(u.core._ep.traj.J_cents), "cpu": cpu, "log": list(getattr(ag, "log", [])),
        "notes": getattr(ag, "notes", None),  # whatever a lab agent counted over the episode
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
        "lost": np.array([r.lost for r in recs]), "served": np.array([r.served for r in recs]),
        "energy": np.array([r.energy for r in recs]), "stock": np.array([r.stock for r in recs]),
        "segment": seg, "sent": sent,
    }  # fmt: skip


def _path(tag: str, task: str, entropy: int) -> Path:
    return OUT / f"{tag}_{task}_{entropy}.pkl"


def _kept(path: Path) -> dict:
    return pickle.loads(path.read_bytes()) if path.is_file() else {}


def run(agent: str, tag: str, task: str = "small", entropy: int = 111, episodes: int = 16, first: int = 0, n_jobs: int = 4,
        only: str | tuple = "") -> None:
    """``only``: episodes to play instead of ``first``..``first + episodes - 1`` ("2,11,16")."""
    path = _path(tag, task, entropy)
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = _kept(path)
    wanted = [int(n) for n in (only.split(",") if isinstance(only, str) else only)] if only else range(first, first + episodes)
    todo = [n for n in wanted if n not in kept]
    for r in Parallel(n_jobs=n_jobs, return_as="generator_unordered")(delayed(episode)(agent, task, entropy, n) for n in todo):
        kept = _kept(path) | {r["n"]: r}  # another run of the same tag may have added episodes meanwhile
        path.write_bytes(pickle.dumps(kept))
    print(f"{tag} {task} {entropy}: {len(kept)} episodes kept")


def references(task: str, entropy: int, episodes: int) -> list[dict]:
    import sbf_starter  # noqa: F401 - points the package at the team's reference cache
    from sbf_starter import scoring

    return list(scoring.episode_set(task, episodes, entropy=entropy, verbose=False).references)


def rss(rows: list[dict], costs) -> tuple[float, dict[int, float]]:
    """The board's score of per-episode costs on the references ``rows``, and the score of each harm level. A set
    without one of the four levels: all that was saved over all that could be, the levels present weighing the same."""
    saved, room = {}, {}
    for r, j in zip(rows, costs):
        saved.setdefault(r["stratum"], []).append(r["J_naive_cents"] - j)
        room.setdefault(r["stratum"], []).append(r["J_naive_cents"] - r["J_oracle_cents"])
    by_level = {s: float(np.sum(saved[s]) / np.sum(room[s])) for s in sorted(saved)}
    weights = {s: WEIGHTS[s] for s in saved} if set(saved) == set(WEIGHTS) else dict.fromkeys(saved, 1.0)
    total = sum(w * np.mean(saved[s]) for s, w in weights.items()) / sum(w * np.mean(room[s]) for s, w in weights.items())
    return float(total), by_level


def show(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 16, first: int = 0, by_episode: bool = False,
         draws: int = 2000) -> None:
    refs_all = references(task, entropy, first + episodes)
    data = {tag: _kept(_path(tag, task, entropy)) for tag in tags}
    ns = [n for n in range(first, first + episodes)
          if all(n in d for d in data.values()) and refs_all[n]["J_oracle_cents"] is not None]  # fmt: skip
    if not ns:
        print("no episode is kept for every tag")
        return
    refs = [refs_all[n] for n in ns]
    level = np.array([r["stratum"] for r in refs])
    room = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float)
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    costs = {tag: np.array([data[tag][n]["J"] for n in ns], dtype=float) for tag in tags}
    groups = [np.flatnonzero(level == s) for s in sorted(set(level))]
    rng = np.random.default_rng(0)
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]

    def drawn(tag: str) -> np.ndarray:
        return np.array([rss([refs[i] for i in p], costs[tag][p])[0] for p in picks])

    print(f"{task}, root {entropy}, {len(ns)} episodes ({ns[0]}..{ns[-1]}), by harm level {[int((level == s).sum()) for s in (1, 2, 3, 4)]}")
    base, drawn0 = tags[0], drawn(tags[0])
    for tag in tags:
        total, by_level = rss(refs, costs[tag])
        cpu = np.concatenate([data[tag][n]["cpu"] for n in ns])
        line = f"{tag:18s} {total:.4f}  levels " + " / ".join(f"{by_level.get(s, float('nan')):.3f}" for s in (1, 2, 3, 4))
        line += f"  cpu/week median {np.median(cpu):.2f} p95 {np.percentile(cpu, 95):.2f} max {cpu.max():.2f}"
        if tag != base:
            lo, hi = np.percentile(drawn(tag) - drawn0, [5, 95])
            d = costs[base] - costs[tag]  # cents saved against the first tag
            pooled = [d[i].sum() / room[i].sum() for i in (rng.integers(0, len(ns), len(ns)) for _ in range(draws))]
            plo, phi = np.percentile(pooled, [5, 95])
            line += (f"\n{'':18s} to {base}: {total - rss(refs, costs[base])[0]:+.4f} ({lo:+.4f} to {hi:+.4f}), cheaper in "
                     f"{int((d > 0).sum())} of {len(ns)}, the same in {int((d == 0).sum())}, {d.mean() / 1e11:+.1f} bn an episode, "
                     f"worst {d.min() / 1e11:+.1f} best {d.max() / 1e11:+.1f}; all saved over all room "
                     f"{d.sum() / room.sum():+.4f} ({plo:+.4f} to {phi:+.4f})")  # fmt: skip
        print(line)
    if by_episode:
        for i, n in enumerate(ns):
            print(f"  ep {n:3d} lvl {level[i]}: " + "  ".join(f"{tag} {(naive[i] - costs[tag][i]) / room[i]:.4f}" for tag in tags))


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
