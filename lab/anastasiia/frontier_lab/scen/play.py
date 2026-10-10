"""Play the scenario planner (``planner.Scen``) and compare it in pairs with the kept plays of hazard_lab.

    uv run python lab/anastasiia/frontier_lab/scen/play.py run saa8 --task=small --entropy=444 --episodes=0-23
    uv run python lab/anastasiia/frontier_lab/scen/play.py show h3_s saa8 truthall_s --episodes=24
    uv run python lab/anastasiia/frontier_lab/scen/play.py blind --episodes=0-5
    uv run python lab/anastasiia/frontier_lab/scen/play.py blind --episodes=0-5 --new=conditional --ages=mixture
    uv run python lab/anastasiia/frontier_lab/scen/play.py cover --episodes=0-23
    uv run python lab/anastasiia/frontier_lab/scen/play.py cover4 --task=small --episodes=0-23

``run`` plays the variants named (``VARIANTS``), one file an episode in
``outputs/frontier_lab/scen/play/<tag>_<task>_<entropy>/<n>.pkl``; an episode is claimed with a lock file, so several
processes started with the same command share the work and none plays an episode twice. ``show`` is hazard_lab's
(scores, paired differences with their 90 % intervals); a tag is looked up here first, then among hazard_lab's kept
plays. ``blind`` runs ``world.blind_check``. ``cover`` counts the episode's new cuts that a scenario had; ``cover4``
counts, element by element, the new cuts of the coming four weeks that the scenarios hold, for the independent and
for the conditional draw of the new events.
"""

import os
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HAZARD = ROOT / "lab" / "anastasiia" / "hazard_lab"
OUT = ROOT / "outputs" / "frontier_lab" / "scen" / "play"
sys.path[:0] = [str(HERE), str(HAZARD)]
BASE = {"small": "outputs/hazard_lab/agents/h3_s", "full": "outputs/hazard_lab/agents/h3_f"}  # the model without its clock

# tag -> the options of ``planner.Scen`` and of ``world.World`` ("new", "ends", "ages")
VARIANTS = {
    "point": {"K": 0, "choose": "point"},  # the identity gate: must be hazard_lab's h3 play to the cent
    "saa8": {"K": 8, "choose": "saa"},
    "saa4": {"K": 4, "choose": "saa"},
    "saa16": {"K": 16, "choose": "saa"},
    "saa4r": {"K": 4, "choose": "saa", "recourse": True},  # the judge lets a scenario's later weeks answer the first
    "saa8r": {"K": 8, "choose": "saa", "recourse": True},
    "two4": {"K": 4, "choose": "joint"},  # the two-stage program: one first week for all the scenarios
    "two8": {"K": 8, "choose": "joint"},
    "two4i": {"K": 4, "choose": "joint", "solver": "ipm"},
    "two8e": {"K": 8, "choose": "joint", "new": False},  # scenarios over the ends of running events only
    "two8n": {"K": 8, "choose": "joint", "ends": False},  # over new events only
    "two8m": {"K": 8, "choose": "joint", "margin": 0.3},  # the point plan stands unless the program claims 0.3 bn
    "two16": {"K": 16, "choose": "joint"},
    # the control: two "scenarios" that are both the model's own forecast (no sampled end, no new event), so the
    # two-stage program is only one more pass of the model's program from its own plan
    "two2p": {"K": 2, "choose": "joint", "new": False, "ends": False},
    # the point forecast as a block of the two-stage program with half the weight, every planner carries the solution
    "two8b": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 0.1},
    "two4b": {"K": 4, "choose": "joint", "point_weight": 0.5, "margin": 0.1},
    "two2pb": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "new": False, "ends": False},
                # its control
    "two8be": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "new": False},
    "two8bn": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "ends": False},
    "two8b1": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 1.0},
    "two1b": {"K": 1, "choose": "joint", "point_weight": 0.5, "margin": 0.1},
    "two2b": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1},
    # the lean form: the scenarios' planners solve their exact cell only and no program is solved again for the judge
    "two2l": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True},
    "two1l": {"K": 1, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True},
    "two4l": {"K": 4, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True},
    "two1lc": {"K": 1, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True, "ages": "mixture",
               "new": "conditional"},
    "two2lc": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True, "ages": "mixture",
               "new": "conditional"},
    # Full only: the lean form over the model with a window of 20 and of 16 weeks (the hull asks the same whole weeks)
    "two2l20": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True,
                "base": "outputs/hazard_lab/agents/h3w20_f"},
    "two1l20": {"K": 1, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True,
                "base": "outputs/hazard_lab/agents/h3w20_f"},
    "two2l16": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True,
                "base": "outputs/hazard_lab/agents/h3w16_f"},
    "two1l16": {"K": 1, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True,
                "base": "outputs/hazard_lab/agents/h3w16_f"},
    "two16b": {"K": 16, "choose": "joint", "point_weight": 0.5, "margin": 0.1},
    # the same program over the model with three passes of its weekly program
    "two8bp": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 0.1,
               "base": "outputs/hazard_lab/agents/h3p3_s"},
    # two8b without what an agent cannot see: the age of an event carried in is drawn, not the true one ("a"); and
    # the new events are drawn given the threads, the events seen and the warning, not independently ("c")
    "two8a": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "ages": "mixture"},
    "two8c": {"K": 8, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "ages": "mixture",
              "new": "conditional"},
    "two4c": {"K": 4, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "ages": "mixture",
              "new": "conditional"},
    "two2c": {"K": 2, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "ages": "mixture", "new": "conditional"},
    # frontier_lab/mech: the control of the lean form, its one "scenario" is the model's own forecast (no sampled end,
    # no new event): what is left is the program solved once more, in two cells of the same network
    "two1pl": {"K": 1, "choose": "joint", "point_weight": 0.5, "margin": 0.1, "lean": True, "new": False,
               "ends": False},
    # a measuring tool, not an agent: the candidates' judge is a planner told the episode's own network
    "ora8": {"K": 8, "choose": "oracle"},
    "ora4": {"K": 4, "choose": "oracle"},
    "ora8m": {"K": 8, "choose": "oracle", "margin": 0.3},
    "med8": {"K": 8, "choose": "medoid"},
    "mean8": {"K": 8, "choose": "mean"},
    "exp64": {"K": 0, "expect": 64, "choose": "expect"},
    "saa8e": {"K": 8, "choose": "saa", "new": False},  # scenarios over the ends of running events only
    "saa8n": {"K": 8, "choose": "saa", "ends": False},  # over new events only
    "saa8m": {"K": 8, "choose": "saa", "margin": 0.5},
    "exp64e": {"K": 0, "expect": 64, "choose": "expect", "new": False},
}


def _world(task: str, u, n: int, K: int, seed: int, **kw):
    from shockbench_flow import marks as M

    import world as W

    omega = u._omega(n)
    inst = u.core._ep.inst.at_digest(str(omega.arrays["meta_instance_digest"]))
    params = M.mark_params_from_json(str(omega.arrays["meta_mark_params"]))
    return W.World(task, inst, M.read_events(inst, omega.arrays), K, seed, n, mark_params=params, feed=_feed(u), **kw)


def _feed(u) -> dict:
    """What the episode's observations show of the announcements and of the warning (the environment's own view):
    the messages' public columns, how many of them each week shows, and the weekly scores."""
    view = u.core._ep.view
    return {"columns": dict(view.message_columns), "counts": tuple(view.message_counts),
            "scores": np.asarray(view.scores)}


def episode(tag: str, task: str, entropy: int, n: int, weeks: int = 0) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    import world as W
    from planner import Scen
    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    opts = dict(VARIANTS[tag])
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    t0 = time.process_time()
    config = agent_config(info["static"], info["policy_seed"], u.layout, obs)
    kw = {key: opts.pop(key) for key in ("new", "ends", "ages") if key in opts}
    world = _world(task, u, n, max(opts.get("K", 0), opts.get("expect", 0)), info["policy_seed"], **kw)
    base = opts.pop("base", BASE[task])
    Agent = load(str(resolve(str(ROOT / base)).resolve()))
    ag = Scen(Agent, config, world, groups=W.GROUPS, truth=u.core._ep.marks, **opts)
    cpu, done = [], False
    while not done:
        action = ag.act(obs)
        cpu.append(time.process_time() - t0)
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc or (weeks and len(cpu) >= weeks)
        t0 = time.process_time()
    recs = u.core._ep.traj.records
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    ag.notes["redrawn"] = world.redrawn
    return {
        "n": n, "J": int(sum(r.cost_cents for r in recs) if weeks else u.core._ep.traj.J_cents), "cpu": cpu, "log": list(ag.log), "notes": ag.notes,
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
        "lost": np.array([r.lost for r in recs]), "weeks": len(recs),
        "point_log": list(ag.point.log),
    }  # fmt: skip


def _numbers(spec) -> list:
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (tuple, list)):
        return [int(x) for x in spec]
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run(*tags: str, task: str = "small", entropy: int = 444, episodes="0-23", weeks: int = 0) -> None:
    """Plays every (tag, episode) nobody has played or claimed yet, the tags in the order given."""
    for tag in tags:
        folder = OUT / f"{tag}_{task}_{entropy}"
        folder.mkdir(parents=True, exist_ok=True)
        for n in _numbers(episodes):
            done, lock = folder / f"{n}.pkl", folder / f"{n}.lock"
            if not weeks:
                if done.is_file():
                    continue
                try:
                    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    continue
            started = time.time()
            r = episode(tag, task, entropy, n, weeks)
            if weeks:  # a short trial: nothing is kept
                print(tag, n, "J", r["J"], "cpu", round(sum(r["cpu"]), 1), r["notes"]["chosen"], r["log"][:weeks], flush=True)
                continue
            tmp = folder / f"{n}.tmp"
            tmp.write_bytes(pickle.dumps(r))
            tmp.replace(done)
            lock.unlink(missing_ok=True)
            print(f"{tag} {task} {entropy} episode {n}: {time.time() - started:.0f} s wall, {sum(r['cpu']):.0f} s CPU, "
                  f"{r['notes']['chosen']}", flush=True)  # fmt: skip


def kept(tag: str, task: str, entropy: int) -> dict:
    folder = OUT / f"{tag}_{task}_{entropy}"
    if folder.is_dir():
        return {int(p.stem): pickle.loads(p.read_bytes()) for p in folder.glob("*.pkl")}
    path = ROOT / "outputs" / "hazard_lab" / "play" / f"{tag}_{task}_{entropy}.pkl"
    return pickle.loads(path.read_bytes()) if path.is_file() else {}


def show(*tags: str, task: str = "small", entropy: int = 444, episodes: int = 24, first: int = 0, by_episode: bool = False) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("hazard_play", HAZARD / "play.py")
    hazard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hazard)
    hazard._kept = lambda path: kept(path.stem[: -len(f"_{task}_{entropy}")], task, entropy)
    hazard.show(*tags, task=task, entropy=entropy, episodes=episodes, first=first, by_episode=by_episode)


def blind(task: str = "small", entropy: int = 444, episodes="0-5", K: int = 4, new=True, ages: str = "true") -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow import marks as M

    import world as W
    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    for n in _numbers(episodes):
        _obs, info = env.reset(options={"episode": n})
        u = env.unwrapped
        omega = u._omega(n)
        inst = u.core._ep.inst.at_digest(str(omega.arrays["meta_instance_digest"]))
        real = M.read_events(inst, omega.arrays)
        compared = W.blind_check(task, inst, real, K, info["policy_seed"], n, feed=_feed(u), new=new, ages=ages,
                                 weeks=(2, 9, 20, 33, 47, 70, 100),  # the last two are beyond a Small episode
                                 mark_params=M.mark_params_from_json(str(omega.arrays["meta_mark_params"])))  # fmt: skip
        world = _world(task, u, n, K, info["policy_seed"], new=new, ages=ages)
        bad = sum(bool(W.present_bad(world.marks(j, week), u.core._ep.marks, week))
                  for week in range(1, inst.T + 1) for j in range(K))  # fmt: skip
        print(f"{task} {entropy} episode {n} (new {new}, ages {ages}): {len(real)} events, {compared} (week, scenario) "
              "pairs the same on the "
              f"tampered truth; scenarios whose present is not the observed one: {bad} of {inst.T * K}; "
              f"durations drawn again {world.redrawn}", flush=True)  # fmt: skip


def cover(task: str = "small", entropy: int = 444, episodes="0-23", K: int = 8, window: int = 26) -> None:
    """Of the episode's new cuts (events that start inside it and cut an edge, a strait or a grid), the share that
    some scenario had: at the week before the cut's onset, a sampled new event of the same type that cuts one of
    the same elements within ``window`` weeks (and within 4 weeks)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow import marks as M
    from shockbench_flow.omega import codes

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    tally: dict = {}

    def cuts(inst, params, q) -> set:
        g = M.graph_marks(inst, (q,), params)
        base = M.graph_marks(inst, (), params)
        out = set()
        for name in ("u", "o", "G_bar"):
            change = (np.abs(g[name] - base[name]) > 1e-12) & np.isfinite(base[name])
            out |= {(name, int(i)) for i in np.flatnonzero(change.any(axis=0))}
        return out

    for n in _numbers(episodes):
        _obs, info = env.reset(options={"episode": n})
        u = env.unwrapped
        world = _world(task, u, n, K, info["policy_seed"])
        inst, params = world.inst, world.mp
        late = [replace_onset(q, inst.T) for q in world._real if 0 < q.onset < inst.T]
        drawn = [[(q, cuts(inst, params, replace_onset(q, inst.T))) for q in d if 0 < q.onset < inst.T] for d in world.draws]
        for q in late:
            mine = cuts(inst, params, q)
            if not mine:
                continue
            s = float(np.floor(q.onset))  # the last instant observed before the onset
            near = far = 0
            for d in drawn:
                same = [p for p, theirs in d if p.type == q.type and p.onset > s and theirs & mine]
                far += any(p.onset <= s + window for p in same)
                near += any(p.onset <= s + 4 for p in same)
            row = tally.setdefault(codes.EVENT_TYPES[q.type], [0, 0, 0, 0.0, 0.0])
            row[0] += 1
            row[1] += far > 0
            row[2] += near > 0
            row[3] += far / K
            row[4] += near / K
    print(f"{task} {entropy} episodes {episodes}, K = {K}: new cuts of the episode, the share some scenario had within "
          f"{window} weeks / within 4 weeks, and the mean share of scenarios that had it")
    for kind, (count, far, near, pf, pn) in sorted(tally.items()):
        print(f"  {kind:22s} {count:4d}  {far / count:.2f} / {near / count:.2f}   per scenario {pf / count:.3f} / {pn / count:.3f}")
    total = np.array(list(tally.values()), dtype=float).sum(axis=0)
    print(f"  {'all':22s} {int(total[0]):4d}  {total[1] / total[0]:.2f} / {total[2] / total[0]:.2f}   per scenario "
          f"{total[3] / total[0]:.3f} / {total[4] / total[0]:.3f}")


def cover4(task: str = "small", entropy: int = 444, episodes="0-23", K: int = 8, ahead: int = 4,
           ages: str = "mixture") -> None:
    """Of the new cuts of the coming ``ahead`` weeks, element by element, the share the scenarios hold.

    A case is an element and an observed instant ``s`` with a real new event that cuts the element in (s, s + ahead]:
    a militarised closure of a strait, an energy shock of a grid, a sanction on an edge (the prohibition), a cut of
    an edge's capacity (a sanction on another edge of its dyad, or a conflict), the fuel edges among them. A scenario
    holds the case when one of its own new events cuts the same element in the same weeks. Printed for the
    independent draw of the new events and for the conditional one: the mean share of scenarios that hold a case, and
    the share of cases that at least one of the ``K`` holds, each with the 90 % interval of a resampling of the
    episodes (an event makes many cases, so the cases are not independent)."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    import world as W

    from sbf_starter import env_id

    if str(W.INFO) not in sys.path:
        sys.path.insert(0, str(W.INFO))
    tab = __import__("filter").Tables.of(task)
    items = ("edge", "edge_fuel", "strait", "proh", "grid")
    env = gym.make(env_id(task), entropy=entropy)
    modes = ("independent", "conditional")
    numbers = _numbers(episodes)
    tally = {(mode, it): np.zeros((len(numbers), 3)) for mode in modes for it in items}  # by episode: cases, one, any
    cpu = {"independent": 0.0, "conditional": 0.0}
    calls = 0

    def cut(events, lo: float, hi: float) -> dict:
        """{item: the elements a list of events cuts with an onset in (lo, hi]}."""
        out = {it: set() for it in items}
        for q in events:
            if lo < q.onset <= hi:
                for it, elems in tab.cuts(q).items():
                    out[it] |= elems
        out["edge_fuel"] = {e for e in out["edge"] if tab.fuel_edge[e]}
        return out

    for i, n in enumerate(numbers):
        _obs, info = env.reset(options={"episode": n})
        u = env.unwrapped
        worlds = {"independent": _world(task, u, n, K, info["policy_seed"], ages=ages),
                  "conditional": _world(task, u, n, K, info["policy_seed"], ages=ages, new="conditional")}  # fmt: skip
        T = worlds["independent"].inst.T
        real = [q for q in worlds["independent"]._real if q.onset > 0]
        for week in range(1, T - ahead + 2):  # the instants s with s + ahead <= T
            s = week - 1.0
            truth = cut(real, s, s + ahead)
            for mode, world in worlds.items():
                t0 = time.process_time()
                fresh = [world.fresh(j, week) for j in range(K)]
                cpu[mode] += time.process_time() - t0
                held = [cut(f, s, s + ahead) for f in fresh]
                for it in items:
                    for e in truth[it]:
                        hits = sum(e in h[it] for h in held)
                        tally[(mode, it)][i] += (1, hits / K, hits > 0)
            calls += K
    print(f"{task} {entropy} episodes {episodes}, K = {K}, the coming {ahead} weeks: cases; the share of scenarios "
          f"that hold a case, independent -> conditional; at least one of {K}, independent -> conditional")
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(numbers), size=(2000, len(numbers)))

    def share(rows: np.ndarray, col: int) -> str:
        """The share over all cases with the 90 % interval of a resampling of the episodes."""
        boot = rows[picks].sum(axis=1)
        lo, hi = np.nanpercentile(boot[:, col] / np.maximum(boot[:, 0], 1e-9), [5, 95])
        return f"{rows[:, col].sum() / rows[:, 0].sum():.3f} ({lo:.3f}..{hi:.3f})"

    for it in items:
        a, b = tally[("independent", it)], tally[("conditional", it)]
        if a[:, 0].sum():
            print(f"  {it:10s} {int(a[:, 0].sum()):5d}   one {share(a, 1)} -> {share(b, 1)}   "
                  f"any of {K} {share(a, 2)} -> {share(b, 2)}")
    print(f"  CPU of one scenario's events a week: independent {1e3 * cpu['independent'] / calls:.2f} ms, "
          f"conditional {1e3 * cpu['conditional'] / calls:.2f} ms; "
          f"conditional draws by source {worlds['conditional'].futures.count}")


def diag(tag: str, base: str = "h3_s", task: str = "small", entropy: int = 444) -> None:
    """Where the week's action left the point plan, what the judge expected of it and what the episode paid.

    Per episode: the weeks whose action is not the point plan's (and how far from it, as a share of the point
    plan's flows), the judge's expected gain summed over those weeks (bn USD: the point plan's score minus the
    pick's), and the episode's cost against ``base`` (bn USD, positive: cheaper)."""
    mine, ref = kept(tag, task, entropy), kept(base, task, entropy)
    rows, names = [], {}
    for n in sorted(mine):
        r = mine[n]
        weeks = r["notes"]["rows"]
        away = [w for w in weeks if w["pick"] != "point" and w["dist"] > 1e-9]
        hoped = sum(w.get("score_point", 0.0) - w.get("score_best", 0.0) for w in away if "score_point" in w)
        hoped += sum(w["joint_gain"] for w in away if "joint_gain" in w)  # the two-stage program's own claim
        hoped += sum(w["oracle_gain"] for w in away if "oracle_gain" in w)  # the told planner's judgement
        for w in weeks:
            names[w["pick"]] = names.get(w["pick"], 0) + 1
        gain = (ref[n]["J"] - r["J"]) / 1e11 if n in ref else float("nan")
        rows.append((n, len(weeks), len(away), float(np.mean([w["dist"] for w in away])) if away else 0.0, hoped, gain,
                     float(np.mean([w.get("distinct", 1) for w in weeks])), r["notes"]["judge_errors"],
                     r["notes"]["present_bad"], sum(r["cpu"])))
    print(f"{tag} against {base}, {task} {entropy}: episode, weeks, weeks away from the point plan, mean distance there, "
          "the judge's expected gain (bn), the episode's gain (bn), distinct candidates a week, judge errors, bad presents, CPU s")
    for row in rows:
        print("  {:3d} {:3d} {:3d}  {:.3f}  {:+8.2f}  {:+8.2f}  {:.1f}  {}  {}  {:.0f}".format(*row))
    a = np.array([(r[2], r[4], r[5]) for r in rows], dtype=float)
    total = sum(names.values())
    print(f"  all: away in {a[:, 0].sum() / sum(r[1] for r in rows):.2f} of weeks; expected {a[:, 1].mean():+.2f} bn an episode, "
          f"paid {np.nanmean(a[:, 2]):+.2f} bn an episode; correlation of the two over episodes "
          f"{np.corrcoef(a[:, 1], a[:, 2])[0, 1] if len(a) > 2 else float('nan'):+.2f}")
    print("  picks: " + ", ".join(f"{k} {v / total:.2f}" for k, v in sorted(names.items(), key=lambda kv: -kv[1])))
    # the two-stage program: of what it claims in a week, the part that one scenario alone gives, and what the
    # other scenarios pay for it (bn USD a week, over the weeks its first week was played)
    top, paid, claimed, weeks_n = [], [], [], 0
    for r in mine.values():
        for w in r["notes"]["rows"]:
            if w["pick"] == "joint" and "start" in w:
                by = np.array(w["start"]) - np.array(w["played"])
                top.append(by.max() / max(by[by > 0].sum(), 1e-9))
                paid.append(-by[by < 0].sum() / len(by))
                claimed.append(by.mean())
                weeks_n += 1
    if weeks_n:
        print(f"  two-stage: {weeks_n} weeks played; it claims {np.mean(claimed):.3f} bn a week (median {np.median(claimed):.3f}); "
              f"one scenario gives {np.mean(top):.2f} of the gains of a week on average; the other scenarios pay "
              f"{np.mean(paid):.3f} bn a week for it")


def replace_onset(q, T: int):
    """The event as it is (a hook kept apart so that ``cover`` reads one rule for both lists)."""
    return q


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show, "blind": blind, "cover": cover, "cover4": cover4, "diag": diag})
