"""The offline kill test of the "opening" (notes/r_blank.md, lever A, step 1): is the first weeks' pattern of the
integer plan (shed where the grid could be closed, start no lots, spend the fuel on whole weeks later) executable?

    uv run python lab/anastasiia/frontier_lab/opening.py record outputs/hazard_lab/agents/h3_s --tag=h3_s --only=0,2,4
    uv run python lab/anastasiia/frontier_lab/opening.py run --only=0,2,4 --ks=5
    uv run python lab/anastasiia/frontier_lab/opening.py run --only=0,2,4 --ks=5 --gs=all,each --offset=18   # the control
    uv run python lab/anastasiia/frontier_lab/opening.py plan --only=0,2,4      # the integer plan's own pattern as a start
    uv run python lab/anastasiia/frontier_lab/opening.py show --episodes=24
    uv run python lab/anastasiia/frontier_lab/opening.py early --labels=base,all_K5_vw

Everything is offline with the whole future known (Small, root 444). ``record`` plays an agent folder once per
episode and keeps its weekly actions (the reference start). ``run`` descends from the reference start and from every
start ``opening(G, K)`` of it with the lab's own descent (``regime_lab/core.py``'s ``descend``, then rounds of whole
weeks asked for by the hull with the search, exactly as ``regime_lab/plan.py``'s ``starts --hull --search``); every
cost is the simulator's. ``show`` scores the starts, the best start per episode (the family's ceiling) and prints who
wins where.

``opening(G, K)``: the reference start with, for every grid of ``G``, (valves) no fuel sent from its terminal into
the grid in weeks 2..K and (wafers) no wafers dispatched to its fabs in weeks 1..K-1. ``G``: one grid by name, "all"
(every grid with fabs), or "full": the grids with a fab whose every packaging plant holds a full store of its packaged
chip (at least 98 % of the store) in at least half of weeks 4-11 of the reference start as played.

Results: ``outputs/frontier_lab/opening/starts/`` (the recorded actions) and ``runs/`` (one file per episode, written
after every start, so a stopped run resumes).
"""

import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REGIME = HERE.parent / "regime_lab"
sys.path[:0] = [str(REGIME), str(HERE.parent / "mpc_lab"), str(HERE.parent / "stats_lab")]

OUT = ROOT / "outputs" / "frontier_lab" / "opening"
WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}  # the board's weight of each harm level
FULL_SHARE, FULL_WEEKS = 0.98, (4, 11)  # a plant's store counts as full; the weeks of the reference that are read


def _episodes(only: str | tuple | int, first: int, episodes: int) -> list[int]:
    if only != "":
        return [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)]
    return list(range(first, first + episodes))


def _start_path(tag: str, task: str, entropy: int, n: int) -> Path:
    return OUT / "starts" / f"{tag}_{task}_{entropy}_{n}.pkl"


def _run_path(tag: str, task: str, entropy: int, n: int) -> Path:
    return OUT / "runs" / f"{tag}_{task}_{entropy}_{n}.pkl"


# ----- the reference start: an agent's own played actions ----------------------------------------------------------------
def record(agent: str, tag: str, task: str = "small", entropy: int = 444, only: str | tuple | int = "", first: int = 0,
           episodes: int = 24) -> None:
    """Play ``agent`` once per episode in the package's environment and keep every week's action as it took it."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import action_to_wire, agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    cls = load(str(resolve(agent).resolve()))
    for n in _episodes(only, first, episodes):
        path = _start_path(tag, task, entropy, n)
        if path.is_file():
            continue
        t0 = time.process_time()
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": n})
        u = env.unwrapped
        ag = cls(agent_config(info["static"], info["policy_seed"], u.layout, obs))
        actions = []
        while True:
            action = ag.act(obs)
            week = int(np.asarray(obs["week"]).ravel()[0])
            actions.append(action_to_wire(u.layout, week, action))
            obs, _r, term, trunc, _i = env.step(action)
            if term or trunc:
                break
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"actions": actions, "J": int(u.core._ep.traj.J_cents)}))
        print(f"ep {n}: {len(actions)} weeks, {u.core._ep.traj.J_cents / 1e11:.1f} bn, {time.process_time() - t0:.0f} s CPU", flush=True)


# ----- the openings ---------------------------------------------------------------------------------------------------
def _tables(ep) -> dict:
    """By grid ordinal: the valve slots (terminal into the grid, lag 0) and the wafer slots bound for its fabs."""
    inst = ep.inst
    grid_of_fab = {inst.fabs[fi]: gi for gi, fabs in enumerate(inst.grid_fabs) for fi in fabs}
    valves = {gi: [] for gi in range(len(inst.grids))}
    wafers = {gi: [] for gi in range(len(inst.grids))}
    dest = []
    for s, (e, k, lane) in enumerate(inst.action_slots):
        edge = inst.edges[e]
        head = edge.head if lane is None else inst.edges[inst.lanes[lane].edges[-1]].head
        dest.append(head)
        if lane is None and edge.head in inst.grid_ordinal and inst.nodes[edge.tail].type == "terminal":
            valves[inst.grid_ordinal[edge.head]].append(s)
        if inst.commodities[k].id == "wafer" and head in grid_of_fab:
            wafers[grid_of_fab[head]].append(s)
    return {"valves": valves, "wafers": wafers, "dest": dest,
            "with_fabs": [gi for gi, fabs in enumerate(inst.grid_fabs) if fabs],
            "names": [inst.nodes[g].id for g in inst.grids]}


def full_grids(ep, recs: list, tb: dict) -> list[int]:
    """The grids with a fab whose every packaging plant holds a full store of its packaged chip in at least half of
    the weeks ``FULL_WEEKS`` of the trajectory ``recs``."""
    inst = ep.inst
    lo, hi = FULL_WEEKS
    weeks = [t for t in range(lo, hi + 1) if t <= len(recs)]
    out = []
    for gi, fabs in enumerate(inst.grid_fabs):
        blocked = False
        for fi in fabs:
            f = inst.fabs[fi]
            raw = inst.nodes[f].fab.product
            plants = {tb["dest"][s] for s, (e, k, _lane) in enumerate(inst.action_slots)
                      if k == raw and inst.edges[e].tail == f and inst.nodes[tb["dest"][s]].osat is not None}
            if not plants:
                continue
            full = []
            for o in plants:
                slot = inst.slot_index[(o, inst.nodes[o].osat.packages[raw])]
                cap = inst.stock_slots[slot].storage
                full.append(sum(1 for t in weeks if recs[t - 1].stock[slot] >= FULL_SHARE * cap) >= len(weeks) / 2)
            blocked = blocked or all(full)
        if blocked:
            out.append(gi)
    return out


def opening(acts: list, tb: dict, grids: list[int], K: int, valves: bool = True, wafers: bool = True,
            offset: int = 0) -> list:
    """``acts`` with the opening of ``grids``: no valve flow in weeks 2..K, no wafers dispatched in weeks 1..K-1.
    ``offset``: the same cut so many weeks later (a control: the same disturbance of the start, not an opening)."""
    shut_v = {s for gi in grids for s in tb["valves"][gi]} if valves else set()
    shut_w = {s for gi in grids for s in tb["wafers"][gi]} if wafers else set()
    out = []
    for t, (fl, ov, ho) in enumerate(acts, start=1):
        drop = (shut_v if 2 <= t - offset <= K else set()) | (shut_w if 1 <= t - offset <= K - 1 else set())
        out.append(({s: q for s, q in fl.items() if s not in drop} if drop else dict(fl), ov, ho))
    return out


def pattern(acts: list, tb: dict, cells: set, valves: bool = True, wafers: bool = True) -> list:
    """``acts`` with the grid-weeks ``cells`` ((week, grid ordinal)) cut: no valve flow into the grid that week and no
    wafers dispatched to its fabs the week before."""
    shut_v, shut_w = {}, {}
    for t, gi in cells:
        if valves:
            shut_v.setdefault(t, set()).update(tb["valves"][gi])
        if wafers and t > 1:
            shut_w.setdefault(t - 1, set()).update(tb["wafers"][gi])
    out = []
    for t, (fl, ov, ho) in enumerate(acts, start=1):
        drop = shut_v.get(t, set()) | shut_w.get(t, set())
        out.append(({s: q for s, q in fl.items() if s not in drop} if drop else dict(fl), ov, ho))
    return out


def _descent(core, ep, acts: list, iters: int, hull: int, search: int) -> dict:
    """``regime_lab/plan.py``'s ``_one`` for one start: the descent, then rounds of whole weeks with the search."""
    method = "ipm" if ep.N > 4 * core.BIG else "simplex"
    d = core.descend(ep, acts, iters)
    for _ in range(hull):
        d2 = core.descend(ep, d["acts"], iters=iters, hull="round", close_until=ep.T - 12, method=method, search=search)
        if d2["J"] > d["J"] - 1e8:
            break
        d = {**d2, "J0": d["J0"], "hist": d["hist"] + d2["hist"]}
    return d


def _weeks(recs: list) -> dict:
    return {"lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
            "served": np.array([r.served for r in recs]), "energy": np.array([r.energy for r in recs])}


def run(tag: str = "h3_s", task: str = "small", entropy: int = 444, only: str | tuple | int = "", first: int = 0,
        episodes: int = 24, ks: str | tuple | int = "5", gs: str | tuple = "each,all,full", parts: str | tuple = "vw",
        iters: int = 60, hull: int = 3, search: int = 8, offset: int = 0) -> None:
    """Descents from the reference start ("base") and from every opening of it, for the episodes asked.

    ``ks``: the K's ("3,5,7"). ``gs``: "each" (every grid with fabs alone), "all", "full". ``parts``: "vw" (valves and
    wafers), "v", "w" - several of them as "vw,v,w". ``offset``: the cut so many weeks later (``opening``; the label
    ends in "_o<offset>"). A start already in the episode's file is not run again."""
    import core  # noqa: E402 - regime_lab's

    ks = [int(k) for k in (str(ks).split(",") if not isinstance(ks, tuple) else ks)]
    gs = list(gs.split(",") if isinstance(gs, str) else gs)
    parts = list(parts.split(",") if isinstance(parts, str) else parts)
    for n in _episodes(only, first, episodes):
        start = pickle.loads(_start_path(tag, task, entropy, n).read_bytes())
        ep = core.Episode.of(task, entropy, n)
        base = ep.validated(start["actions"])
        tb = _tables(ep)
        recs0, J0 = ep.simulate(base)
        if J0 != start["J"]:
            print(f"ep {n}: the start replays to {J0 / 1e11:.2f} bn, the environment had {start['J'] / 1e11:.2f}", flush=True)
        full = full_grids(ep, recs0, tb)
        sets = {}
        for g in gs:
            if g == "each":
                sets |= {tb["names"][gi]: [gi] for gi in tb["with_fabs"]}
            elif g == "all":
                sets["all"] = list(tb["with_fabs"])
            elif g == "full":
                sets["full"] = full
        todo = [("base", None, 0, "")]
        late = f"_o{offset}" if offset else ""
        todo += [(f"{name}_K{K}_{p}{late}", grids, K, p) for p in parts for K in ks for name, grids in sets.items()]
        path = _run_path(tag, task, entropy, n)
        path.parent.mkdir(parents=True, exist_ok=True)
        kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
        kept["_meta"] = {"full": [tb["names"][gi] for gi in full], "names": tb["names"], "J_start": J0,
                         "start": _weeks(recs0)}
        for label, grids, K, p in todo:
            if label in kept:
                continue
            if grids is not None and not grids:  # no grid in the set: the opening is the reference start
                kept[label] = {"same_as": "base"}
                path.write_bytes(pickle.dumps(kept))
                continue
            t0 = time.process_time()
            acts = base if grids is None else opening(base, tb, grids, K, "v" in p, "w" in p, offset)
            d = _descent(core, ep, acts, iters, hull, search)
            kept = (pickle.loads(path.read_bytes()) if path.is_file() else {}) | kept
            kept[label] = {"J": int(d["J"]), "J0": int(d["J0"]), "passes": len(d["hist"]), "seconds": time.process_time() - t0,
                           "grids": None if grids is None else [tb["names"][gi] for gi in grids], "K": K, "parts": p,
                           "offset": offset if grids is not None else 0,
                           **_weeks(d["recs"])}
            path.write_bytes(pickle.dumps(kept))
            print(f"ep {n} {label:22s} start {d['J0'] / 1e11:8.1f} -> {d['J'] / 1e11:8.1f} bn  ({len(d['hist'])} passes, {time.process_time() - t0:.0f} s)", flush=True)


def plan(tag: str = "h3_s", task: str = "small", entropy: int = 444, only: str | tuple | int = "", first: int = 0,
         episodes: int = 24, until: int = 8, iters: int = 60, hull: int = 3, search: int = 8) -> None:
    """The opening the integer plan itself names (label "plan_w<until>_vw"): every grid-week of weeks 2..``until``
    that the plan with base load first leaves short (it sheds over 0.5 % of the base load) and the reference start
    closes (it sheds under 0.05 %) is cut in the start (``pattern``). The plan's weeks are those of
    ``outputs/plan_stats/20261006_040041/episodes.npz`` (Small 444, episodes 0-39)."""
    import core  # noqa: E402 - regime_lab's

    z = np.load(ROOT / "outputs" / "plan_stats" / "20261006_040041" / "episodes.npz")
    label = f"plan_w{until}_vw"
    for n in _episodes(only, first, episodes):
        path = _run_path(tag, task, entropy, n)
        kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
        if label in kept:
            continue
        start = pickle.loads(_start_path(tag, task, entropy, n).read_bytes())
        ep = core.Episode.of(task, entropy, n)
        base = ep.validated(start["actions"])
        tb = _tables(ep)
        recs0, _J0 = ep.simulate(base)
        ybar = np.asarray(ep.marks.y_bar, dtype=float)
        cells = {(t, gi) for t in range(2, until + 1) for gi in tb["with_fabs"]
                 if z["plan_shed"][n][t - 1, gi] > 5e-3 * ybar[t - 1, gi] and recs0[t - 1].shed[gi] < 5e-4 * ybar[t - 1, gi]}
        if not cells:
            kept[label] = {"same_as": "base"}
            path.write_bytes(pickle.dumps(kept))
            print(f"ep {n} {label}: no grid-week to cut", flush=True)
            continue
        t0 = time.process_time()
        d = _descent(core, ep, pattern(base, tb, cells), iters, hull, search)
        kept = (pickle.loads(path.read_bytes()) if path.is_file() else {}) | kept
        kept[label] = {"J": int(d["J"]), "J0": int(d["J0"]), "passes": len(d["hist"]), "seconds": time.process_time() - t0,
                       "cells": sorted((t, tb["names"][gi]) for t, gi in cells), "K": until, "parts": "vw", "offset": 0,
                       **_weeks(d["recs"])}
        path.write_bytes(pickle.dumps(kept))
        print(f"ep {n} {label:22s} {len(cells)} grid-weeks cut, start {d['J0'] / 1e11:8.1f} -> {d['J'] / 1e11:8.1f} bn  ({len(d['hist'])} passes, {time.process_time() - t0:.0f} s)", flush=True)


# ----- the score ------------------------------------------------------------------------------------------------------
def _refs(task: str, entropy: int, upto: int) -> list[dict]:
    import sbf_starter  # noqa: F401 - points the package at the team's reference cache
    from sbf_starter import scoring

    return list(scoring.episode_set(task, upto, entropy=entropy, n_jobs=1, verbose=False).references)


def _rss(level: np.ndarray, naive: np.ndarray, oracle: np.ndarray, J: np.ndarray) -> float:
    """The board's score; a set without one of the four levels: all that was saved over all that could be."""
    if set(level.tolist()) == set(WEIGHTS):
        num = sum(w * (naive - J)[level == s].mean() for s, w in WEIGHTS.items())
        return float(num / sum(w * (naive - oracle)[level == s].mean() for s, w in WEIGHTS.items()))
    return float((naive - J).sum() / (naive - oracle).sum())


def show(tag: str = "h3_s", task: str = "small", entropy: int = 444, only: str | tuple | int = "", first: int = 0,
         episodes: int = 24, draws: int = 2000, weeks: int = 13) -> None:
    """Scores by start; the best start per episode of each family of starts (its ceiling) with the paired interval to
    the base descent; the winner of every episode among the openings; the winners' lots and shed load in the first
    ``weeks`` weeks against the base. Families: the openings (valves and wafers, every K and set of grids), the same
    at each K, the late cuts (the control), valves or wafers alone, the integer plan's own pattern."""
    ns = [n for n in _episodes(only, first, episodes) if _run_path(tag, task, entropy, n).is_file()]
    data = {n: pickle.loads(_run_path(tag, task, entropy, n).read_bytes()) for n in ns}
    ns = [n for n in ns if "base" in data[n]]
    refs = _refs(task, entropy, max(64, max(ns) + 1))  # the first 64 of root 444 are in the team's reference cache
    level = np.array([refs[n]["stratum"] for n in ns])
    naive = np.array([refs[n]["J_naive_cents"] for n in ns], dtype=float)
    oracle = np.array([refs[n]["J_oracle_cents"] for n in ns], dtype=float)
    labels = sorted({k for n in ns for k in data[n] if not k.startswith("_")}, key=lambda k: (k != "base", k))

    def cost(n: int, label: str) -> float:
        r = data[n].get(label)
        if r is None:
            return np.nan
        return float(data[n]["base"]["J"] if "same_as" in r else r["J"])

    def rss(J: np.ndarray, pick=slice(None)) -> float:
        return _rss(level[pick], naive[pick], oracle[pick], J[pick])

    J = {label: np.array([cost(n, label) for n in ns]) for label in labels}
    whole = [label for label in labels if not np.isnan(J[label]).any()]
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(level == s) for s in sorted(set(level.tolist()))]
    picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]

    def interval(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        return tuple(np.percentile([rss(a, p) - rss(b, p) for p in picks], [5, 95]))

    start = np.array([float(data[n]["_meta"]["J_start"]) for n in ns])
    print(f"{task} root {entropy}, {len(ns)} episodes {ns}, by harm level {[int((level == s).sum()) for s in WEIGHTS]}")
    print(f"reference start as played {rss(start):.4f}")
    base = J["base"]
    for label in whole:
        line = f"{label:22s} {rss(J[label]):.4f}"
        if label != "base":
            lo, hi = interval(J[label], base)
            d = (base - J[label]) / 1e11
            line += (f"  to base {rss(J[label]) - rss(base):+.4f} ({lo:+.4f} to {hi:+.4f})"
                     f"  cheaper in {int((d > 0.05).sum()):2d}, dearer in {int((d < -0.05).sum()):2d} of {len(ns)}, mean {d.mean():+6.1f} bn"
                     f"  | best of it and base {rss(np.minimum(J[label], base)):.4f}")
        print(line)
    part = [label for label in labels if label not in whole]
    if part:
        print("not on every episode (left out):", ", ".join(f"{k} ({int((~np.isnan(J[k])).sum())})" for k in part))

    def family(name: str, members: list) -> np.ndarray | None:
        members = [label for label in members if label in whole]
        if not members:
            return None
        best = np.min([base] + [J[label] for label in members], axis=0)
        lo, hi = interval(best, base)
        print(f"best of base and {name} ({len(members)} starts): {rss(best):.4f}  to base {rss(best) - rss(base):+.4f} "
              f"({lo:+.4f} to {hi:+.4f}), mean {(base - best).mean() / 1e11:+.1f} bn an episode")
        return best

    def is_opening(label: str) -> bool:
        return label.endswith("_vw") and "_K" in label and not label.startswith("plan_")

    sets5 = ["all"] + [k[: -len("_K5_vw")] for k in whole if k.startswith("grid_") and k.endswith("_K5_vw")]
    openings = [k for k in whole if is_opening(k)]
    print("--- ceilings of the families of starts")
    best = family("every opening (valves and wafers)", openings)
    for K in (3, 5, 7):
        family(f"the openings of K = {K}", [k for k in openings if f"_K{K}_" in k])
    family("the openings of K = 5 on 'all' and each grid (to set beside the control)", [f"{g}_K5_vw" for g in sets5])
    family("the control: the same cuts 18 weeks later", [k for k in whole if k.endswith("_vw_o18")])
    family("valves alone, K = 5", [k for k in whole if k.endswith("_K5_v")])
    family("wafers alone, K = 5", [k for k in whole if k.endswith("_K5_w")])
    family("the integer plan's own pattern", [k for k in whole if k.startswith("plan_")])
    family("everything above", [k for k in whole if k != "base"])
    if best is None:
        return
    who = [(["base"] + openings)[int(np.argmin([base[i]] + [J[label][i] for label in openings]))] for i in range(len(ns))]
    plan_cost = None  # the integer plan with base load first (``stats_lab/plan_stats.py``), where its episodes are kept
    stats = ROOT / "outputs" / "plan_stats" / "20261006_040041" / "episodes.npz"
    if task == "small" and entropy == 444 and stats.is_file() and max(ns) < 40:
        z = np.load(stats)
        if np.allclose(z["J_relaxed"][ns] * 100.0, oracle, rtol=1e-6):
            plan_cost = z["J_plan"][ns] * 100.0
            print(f"{'integer plan, its cost':22s} {rss(plan_cost):.4f}  (base is {(base - plan_cost).mean() / 1e11:+.1f} bn an episode "
                  f"above it, the best opening per episode {(best - plan_cost).mean() / 1e11:+.1f})")
    print("--- by episode, among the openings: level, base bn, winner, its gain bn, base minus the integer plan bn, the grids with full plants")
    for i, n in enumerate(ns):
        room = "" if plan_cost is None else f"{(base[i] - plan_cost[i]) / 1e11:+7.1f}"
        print(f"  ep {n:2d} lvl {level[i]}  {base[i] / 1e11:8.1f}  {who[i]:22s} {(base[i] - best[i]) / 1e11:+7.1f}  {room}  full: {','.join(data[n]['_meta']['full']) or '-'}")
    won = [i for i in range(len(ns)) if who[i] != "base" and base[i] - best[i] > 5e9]
    print(f"episodes won by an opening by more than 0.05 bn: {len(won)} of {len(ns)}; winners by K: "
          + ", ".join(f"K={K}: {sum(1 for i in won if f'_K{K}_' in who[i])}" for K in (3, 5, 7)) + "; by set: "
          + ", ".join(f"{g}: {sum(1 for i in won if who[i].startswith(g + '_K'))}" for g in ["all", "full"] + sets5[1:]))
    if won:
        for key, unit, scale in (("lots", "thousand lots, all fabs", 1e3), ("shed", "GWh shed, all grids", 1.0)):
            s0 = np.mean([data[ns[i]]["_meta"]["start"][key][:weeks].sum(axis=1) for i in won], axis=0) / scale
            b = np.mean([data[ns[i]]["base"][key][:weeks].sum(axis=1) for i in won], axis=0) / scale
            w = np.mean([data[ns[i]][who[i]][key][:weeks].sum(axis=1) for i in won], axis=0) / scale
            print(f"  weeks 1-{weeks}, {unit}, mean over those episodes\n    start  " + " ".join(f"{x:7.0f}" for x in s0)
                  + "\n    base   " + " ".join(f"{x:7.0f}" for x in b) + "\n    winner " + " ".join(f"{x:7.0f}" for x in w))



def early(labels: str | tuple = "base,all_K5_vw", tag: str = "h3_s", task: str = "small", entropy: int = 444,
          only: str | tuple | int = "", first: int = 0, episodes: int = 24, weeks: int = 13) -> None:
    """Mean lots and shed load of the first ``weeks`` weeks after the descent from each of ``labels`` (and of the
    reference start as played), over the episodes that have them all: does an opening survive its own descent?"""
    labels = list(labels.split(",") if isinstance(labels, str) else labels)
    ns = [n for n in _episodes(only, first, episodes) if _run_path(tag, task, entropy, n).is_file()]
    data = {n: pickle.loads(_run_path(tag, task, entropy, n).read_bytes()) for n in ns}
    ns = [n for n in ns if all(k in data[n] and "J" in data[n][k] for k in labels)]
    print(f"{len(ns)} episodes {ns}")
    for key, unit, scale in (("lots", "thousand lots, all fabs", 1e3), ("shed", "GWh shed, all grids", 1.0)):
        print(f"weeks 1-{weeks}, {unit}")
        rows = [("start as played", [data[n]["_meta"]["start"][key] for n in ns])]
        rows += [(k, [data[n][k][key] for n in ns]) for k in labels]
        for name, arrs in rows:
            print(f"  {name:22s} " + " ".join(f"{x:7.0f}" for x in np.mean([a[:weeks].sum(axis=1) for a in arrs], axis=0) / scale))
    stats = ROOT / "outputs" / "plan_stats" / "20261006_040041" / "episodes.npz"
    if task == "small" and entropy == 444 and stats.is_file() and ns and max(ns) < 40:
        z = np.load(stats)
        print(f"  {'integer plan (lots)':22s} " + " ".join(f"{x:7.0f}" for x in z["plan_lots"][ns][:, :weeks].sum(axis=2).mean(axis=0) / 1e3))
        print(f"  {'integer plan (shed)':22s} " + " ".join(f"{x:7.0f}" for x in z["plan_shed"][ns][:, :weeks].sum(axis=2).mean(axis=0)))


def tables(task: str = "small", entropy: int = 444, n: int = 0) -> None:
    """The slots an opening shuts, by grid (a check of ``_tables``)."""
    import core  # noqa: E402 - regime_lab's

    ep = core.Episode.of(task, entropy, n)
    inst, tb = ep.inst, _tables(ep)

    def name(s: int) -> str:
        e, k, lane = inst.action_slots[s]
        return f"{s}:{inst.edges[e].id}/{inst.commodities[k].id}" + ("" if lane is None else f"/{inst.lanes[lane].id}")

    for gi in tb["with_fabs"]:
        print(tb["names"][gi], "\n  valves:", ", ".join(name(s) for s in tb["valves"][gi]), "\n  wafers:", ", ".join(name(s) for s in tb["wafers"][gi]))


if __name__ == "__main__":
    fire.Fire({"record": record, "run": run, "plan": plan, "show": show, "early": early, "tables": tables})
