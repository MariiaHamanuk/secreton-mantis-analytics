"""Play agent folders on cached episodes, keep the cost of every episode, compare paired; agents may be told the truth.

    uv run python lab/anastasiia/plan_lab/harness.py run agents/anastasiia_hybrid_hub --tag=hub --task=small \
        --entropy=444 --episodes=8 --n_jobs=4
    uv run python lab/anastasiia/plan_lab/harness.py run lab/anastasiia/plan_lab/agents/<x> --tag=x --base=hub ...
    uv run python lab/anastasiia/plan_lab/harness.py show --task=small --entropy=444 --episodes=8 --base=hub

``hub/eval/compare.py`` is the tool for agents that play by the rules. This one is for experiments it cannot run:

- an agent that defines ``tell_truth(truth)`` gets, before its first week, the episode's instance, its true weekly
  marks (the network of every week), the task, the root and the episode's number: experiments "with the true future";
- the cost of every episode is kept in ``outputs/plan_lab/costs/<task>_<entropy>.json`` under ``--tag``, so a later
  run compares paired with any tag already there without playing it again (``--base``);
- ``--params`` (a JSON object) is handed to the agent's module as ``PLAN_LAB_PARAMS`` before the agent is built, so
  one folder serves several variants; ``--save`` keeps the weekly arrays ``gap.py`` and ``plan_gap.py`` read.

Scores are the board's (levels weighted 50 / 30 / 15 / 5%), from the cached references of ``hub/refcache``; the
interval of a difference resamples episodes inside their harm level, as ``hub/eval/formal_eval.py`` does. No CPU budget
is applied; ``cpu`` is the agent's own CPU seconds per week on this machine (not the container's).
"""

import json
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
COSTS = ROOT / "outputs" / "plan_lab" / "costs"
LEVEL_WEIGHTS = (0.50, 0.30, 0.15, 0.05)
COMPONENTS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def pooled(level: np.ndarray, saved: np.ndarray, room: np.ndarray) -> float:
    """The score as ``sbf evaluate`` prints it: weighted by harm level when all four are there, else over all episodes.

    (``EpisodeSet.rss`` of the package: the board's pooled score needs every level; a set that lacks one is scored as
    the summed savings over the summed attainable savings.)
    """
    if all((level == s).any() for s in (1, 2, 3, 4)):
        w = LEVEL_WEIGHTS
        return float(
            sum(w[s - 1] * saved[level == s].mean() for s in (1, 2, 3, 4))
            / sum(w[s - 1] * room[level == s].mean() for s in (1, 2, 3, 4))
        )
    return float(saved.sum() / room.sum())


def by_level_mean(level: np.ndarray, saved: np.ndarray, room: np.ndarray) -> float:
    """The convention of ``lab/anastasiia/mpc_lab/package_baselines.py::rss`` (the numbers of ``hub/tried/mpc.md``):
    the levels present weigh the same when one is missing."""
    present = [s for s in (1, 2, 3, 4) if (level == s).any()]
    w = dict(zip((1, 2, 3, 4), LEVEL_WEIGHTS)) if len(present) == 4 else dict.fromkeys(present, 1.0)
    return float(
        sum(w[s] * saved[level == s].mean() for s in present) / sum(w[s] * room[level == s].mean() for s in present)
    )


def interval(level, saved_a, saved_b, room, draws: int = 2000) -> tuple[float, float, float]:
    """a's score minus b's and its 90% interval (episodes resampled inside their harm level)."""
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(level == s) for s in (1, 2, 3, 4) if (level == s).any()]
    diffs = np.empty(draws)
    for i in range(draws):
        pick = np.concatenate([rng.choice(g, len(g)) for g in groups])
        diffs[i] = pooled(level[pick], saved_a[pick], room[pick]) - pooled(level[pick], saved_b[pick], room[pick])
    diff = pooled(level, saved_a, room) - pooled(level, saved_b, room)
    return diff, float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))


def references(task: str, entropy: int, upto: int) -> list[dict]:
    """The cached reference rows (naive and clairvoyant cost, harm level) of episodes 0..upto-1."""
    from sbf_starter import scoring

    return list(scoring.episode_set(task, upto, entropy=entropy, verbose=False).references)


def play(agent: str, task: str, entropy: int, n: int, params: dict | None = None, arrays: bool = False) -> dict:
    """One episode played by the agent under gymnasium; its cost in cents, CPU seconds, and the weekly arrays."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    import sbf_starter.agents as agents_mod
    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    inst = u.instance
    import builtins

    builtins.PLAN_LAB_PARAMS = dict(params or {})  # read by the agent's module when it is built
    try:
        ag = agents_mod.load(agent)(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    finally:
        builtins.PLAN_LAB_PARAMS = {}
    ep = u.core._ep
    if hasattr(ag, "tell_truth"):
        ag.tell_truth({"inst": inst, "marks": ep.marks, "task": task, "entropy": entropy, "episode": n, "env": u})
    cpu, done = [], False
    while not done:
        t0 = time.process_time()
        action = ag.act(obs)
        cpu.append(time.process_time() - t0)
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    recs, marks = ep.traj.records, ep.marks
    out = {
        "episode": n,
        "J_cents": int(ep.traj.J_cents),
        "cpu_mean": float(np.mean(cpu)),
        "cpu_p95": float(np.quantile(cpu, 0.95)),
        "cpu_max": float(np.max(cpu)),
        "invalid": int(sum(len(r.invalid) for r in recs)),
        "notes": getattr(ag, "notes", None),
    }
    if arrays:
        T, G = inst.T, len(inst.grids)
        seg = np.zeros((T, G, len(inst.commodities) + 1))  # last column: the no-fuel segment
        sent, asked = np.zeros((T, len(inst.action_slots))), np.zeros((T, len(inst.action_slots)))
        for t, r in enumerate(recs):
            for (gi, k), q in r.segment.items():
                seg[t, gi, -1 if k is None else k] = q
            for s, q in r.executed.items():
                sent[t, s] = q
            for s, q in r.requested.items():
                asked[t, s] = q
        out["arrays"] = {
            "J": ep.traj.J_cents / 100,
            "cost": np.array([[getattr(r.costs, c) for c in COMPONENTS] for r in recs]),
            "lots": np.array([r.lots_started for r in recs]),
            "scrap": np.array([r.scrapped for r in recs]),
            "energy": np.array([r.energy for r in recs]),
            "disposal": np.array([r.disposal for r in recs]),
            "stock": np.array([r.stock for r in recs]),
            "shed": np.array([r.shed for r in recs]),
            "served": np.array([r.served for r in recs]),
            "demand": np.array([r.demand for r in recs]),
            "segment": seg,
            "sent": sent,
            "asked": asked,
            "alpha": np.asarray(marks.alpha_bar)[:T],
            "R": np.asarray(marks.R)[:T],
            "G_bar": np.asarray(marks.G_bar)[:T],
            "y_bar": np.asarray(marks.y_bar)[:T],
        }
    return out


def _json_words(value):
    """``value`` with the strings "true", "false" and "null" (JSON's words, at any depth) as Python's values."""
    if isinstance(value, dict):
        return {k: _json_words(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_words(v) for v in value]
    return {"true": True, "false": False, "null": None}.get(value, value) if isinstance(value, str) else value


def _book(task: str, entropy: int) -> Path:
    return COSTS / f"{task}_{entropy}.json"


def _load(task: str, entropy: int) -> dict:
    path = _book(task, entropy)
    return json.loads(path.read_text()) if path.is_file() else {}


def _table(task: str, entropy: int, tags: list[str], base: str | None, first: int, episodes: int) -> None:
    book = _load(task, entropy)
    idx = [str(n) for n in range(first, first + episodes)]
    refs = references(task, entropy, first + episodes)[first:]
    ok = np.array([r["J_oracle_cents"] is not None for r in refs])
    level = np.array([r["stratum"] for r in refs])[ok]
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)[ok]
    room = naive - np.array([r["J_oracle_cents"] or 0 for r in refs], dtype=float)[ok]
    counts = [int((level == s).sum()) for s in (1, 2, 3, 4)]
    print(f"{task}, root {entropy}, episodes {first}..{first + episodes - 1}; by harm level {counts}; 0.01 = "
          f"{0.01 * room.mean() / 1e11:.1f} bn USD an episode")
    print(f"{'score':>7} {'vs base':>8} {'90% interval':>20} {'bn/ep':>7} {'better':>7} {'levels 1..4':>26} "
          f"{'cpu mean/p95/max':>17}  tag")

    def costs(tag):
        rows = book.get(tag, {})
        if any(i not in rows for i in idx):
            return None
        return np.array([rows[i]["J_cents"] for i in idx], dtype=float)[ok], [rows[i] for i in idx]

    b = costs(base) if base else None
    for tag in tags:
        c = costs(tag)
        if c is None:
            print(f"{'':>7} {'':>8} {'':>20} {'':>7} {'':>7} {'':>26} {'':>17}  {tag}: episodes missing")
            continue
        j, rows = c
        saved = naive - j
        lv = " ".join(f"{saved[level == s].sum() / room[level == s].sum():.3f}" if (level == s).any() else "-"
                      for s in (1, 2, 3, 4))
        cpu = f"{np.mean([r['cpu_mean'] for r in rows]):.2f}/{np.max([r['cpu_p95'] for r in rows]):.2f}/" \
              f"{np.max([r['cpu_max'] for r in rows]):.2f}"
        if b is not None and tag != base:
            d, lo, hi = interval(level, saved, naive - b[0], room)
            extra = f"{d:+8.4f} {f'{lo:+.4f}..{hi:+.4f}':>20} {(j - b[0]).mean() / 1e11:+7.1f} " \
                    f"{int((j < b[0]).sum()):>3}/{len(j):<3}"
        else:
            extra = f"{'':>8} {'':>20} {'':>7} {'':>7}"
        print(f"{pooled(level, saved, room):7.4f} {extra} {lv:>26} {cpu:>17}  {tag}", flush=True)


def run(agent: str, tag: str, task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0,
        n_jobs: int = 4, base: str | None = None, params: str | dict | None = None, save: str = "",
        again: bool = False) -> None:
    """Play ``agent`` on the episodes, keep the costs under ``tag`` and print it beside ``base``."""
    from sbf_starter.agents import resolve

    agent = str(resolve(agent))
    if isinstance(params, str):
        params = json.loads(params)
    params = _json_words(params)  # fire reads a JSON object as a Python literal and leaves true / false as strings
    book = _load(task, entropy)
    have = {} if again else book.get(tag, {})
    todo = [n for n in range(first, first + episodes) if str(n) not in have or save]
    out = Parallel(n_jobs=n_jobs)(delayed(play)(agent, task, entropy, n, params, bool(save)) for n in todo)
    if save:
        keys = out[0]["arrays"].keys()
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save, **{k: np.stack([o["arrays"][k] for o in out]) for k in keys})
    book = _load(task, entropy)  # another run may have written meanwhile
    rows = book.setdefault(tag, {}) if not again else book.__setitem__(tag, {}) or book[tag]
    for o in out:
        rows[str(o["episode"])] = {k: v for k, v in o.items() if k != "arrays"}
    COSTS.mkdir(parents=True, exist_ok=True)
    _book(task, entropy).write_text(json.dumps(book))
    _table(task, entropy, [base, tag] if base and base != tag else [tag], base, first, episodes)


def show(*tags: str, task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0,
         base: str | None = None) -> None:
    """Print the tags kept for the task and root (all of them when none is named), paired with ``base``."""
    names = list(tags) or sorted(_load(task, entropy))
    if base and base not in names:
        names = [base, *names]
    _table(task, entropy, names, base, first, episodes)


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
