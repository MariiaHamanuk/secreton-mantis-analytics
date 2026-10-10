"""The first weeks of one episode of a model's folder, in this process: CPU seconds by kind of work, and the actions.

    uv run python lab/anastasiia/frontier_lab/speed/weeks.py play outputs/speed_lab/t_ref --tag=ref26 --episode=3
    uv run python lab/anastasiia/frontier_lab/speed/weeks.py play outputs/speed_lab/t_ref --tag=ref20 --episode=3 \
        --numbers='{"horizon": 20, "hull_until": 6}'
    uv run python lab/anastasiia/frontier_lab/speed/weeks.py play outputs/speed_lab/t_ref --tag=prof26 --profile
    uv run python lab/anastasiia/frontier_lab/speed/weeks.py show ref26 new26 --episode=3

``play`` is ``hazard_lab/same.py``'s play (root 444, the folder copied to a temporary one, no solver limits) with
the folder's own ``regime.json`` and ``numbers`` over it, so a setting can be timed; a window of 20 weeks without
the clock is ``horizon`` 20 with ``hull_until`` 6, which is what the model's short window is. Every week it keeps the
action arrays, the agent's note, the week's CPU seconds (``time.process_time``) and the same seconds by kind of
work: the functions of ``KINDS`` are wrapped by a timer that books a call's own time (its callees' taken off) under
its kind, and the solver's runs are booked by the cell they solve. A week is tens to a few thousand such calls, so
the timer's own cost is under a millisecond. ``--profile`` adds cProfile over ``act`` (the seconds by kind are then
inflated: read that run for the functions only) and prints the functions by own and by cumulative time.

``show`` compares the kept weeks of tags of one episode with the first: whether the actions and notes are the same,
the weeks' CPU (median, 95th percentile, sum) and the seconds by kind per week.
"""

import cProfile
import io
import json
import pickle
import pstats
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs" / "speed_lab" / "weeks"
NO_LIMITS = {"warm_share": 0, "solve_share": 0, "solve_seconds": 600}  # as ``same.py``: no run is cut short


class Clock:
    """Own CPU seconds of the wrapped calls, by kind: a call's time without the wrapped calls made inside it."""

    def __init__(self) -> None:
        self.seconds: dict = {}
        self.calls: dict = {}
        self.stack: list = []

    def book(self, kind: str, seconds: float, calls: int = 1) -> None:
        self.seconds[kind] = self.seconds.get(kind, 0.0) + seconds
        self.calls[kind] = self.calls.get(kind, 0) + calls

    def wrap(self, fn, kind, after=None):
        """``kind``: a name, or a function of the names on the stack that gives one. ``after(result, args)`` may move
        part of the call's own time to other kinds: it returns {kind: seconds}."""
        clock = self

        def timed(*args, **kwargs):
            name = kind(clock.stack) if callable(kind) else kind
            clock.stack.append([name, 0.0])
            t0 = time.process_time()
            try:
                result = fn(*args, **kwargs)
            finally:
                took = time.process_time() - t0
                _name, inside = clock.stack.pop()
                if clock.stack:
                    clock.stack[-1][1] += took
            own = took - inside
            if after is not None:
                for other, seconds in after(result, args).items():
                    clock.book(other, seconds)
                    own -= seconds
            clock.book(name, own)
            return result

        return timed

    def week(self) -> dict:
        out = (dict(self.seconds), dict(self.calls))
        self.seconds, self.calls = {}, {}
        return out


def _within(stack: list, inner: str) -> str:
    """The simulator's step and the rules' calls are booked by who plays them: the rules' sweep or the judge."""
    names = [n for n, _ in stack]
    if "sweep" in names:
        return f"sweep: {inner}"
    return inner


def _wrap_all(clock: Clock, cls) -> None:
    """Timers on the folder's modules, reached from its ``Agent`` class."""
    mod = sys.modules[cls.__module__]
    core, model, hybrid, sim = mod._core, mod._model, mod._hybrid, mod.sim
    Ep = core.Episode

    booked = {"list": None, "n": 0}  # the week's list of runs and how many of them are booked

    def solver(result, args) -> dict:
        """The solver's own runs of this call, by the cell: ``Episode.solve`` notes each in ``solves``. The count is
        kept by the list, not by the episode: with ``hull_weeks`` the hull's episode notes its run in the week's
        list, and a count by episode booked that run twice (the kept ``hw20`` and ``hw16`` weeks have it so: read
        their hull from the solver's runs, not from the seconds by kind)."""
        ep = args[0]
        if ep.solves is not booked["list"]:
            booked["list"], booked["n"] = ep.solves, 0
        out: dict = {}
        for s in ep.solves[booked["n"] :]:
            cell = "hull" if s["what"].startswith("hull") else "search" if s["what"].startswith("search") else "exact"
            out[f"highs: {cell}"] = out.get(f"highs: {cell}", 0.0) + s["cpu"]
        booked["n"] = len(ep.solves)
        return out

    def put(owner, name: str, kind, after=None) -> None:
        if hasattr(owner, name):
            setattr(owner, name, clock.wrap(getattr(owner, name), kind, after))

    put(cls, "act", "act: the rest")
    put(cls, "_plan", "plan: the rest")
    put(cls, "_roll", "sweep")
    for name in ("_arrays", "_ask_early", "_ask_scaled", "_wasted", "_wishes", "_seen"):
        put(cls, name, "act: requests and watch")
    put(Ep, "__init__", "episode: build_lp and columns")
    put(Ep, "simulate", "judge: play")
    put(Ep, "cost", lambda st: _within(st, "judge: cost"))
    put(Ep, "regimes", "regimes")
    put(Ep, "cell", "cell")
    put(Ep, "rows", "rows")
    put(Ep, "solve", "solve: model in and out", solver)
    put(Ep, "actions", "actions")
    put(Ep, "hints", "hints")
    put(Ep, "clean", "plan: the rest")
    put(Ep, "shifted", "plan: the rest")
    put(Ep, "vector", "plan: the rest")
    for name in ("_rounded", "_beside", "_tilt", "_whole_weeks"):
        put(core, name, f"core: {name}")
    if getattr(mod, "_terms", None) is not None:
        put(mod._terms, "add", "cell: terms")
    put(model.Model, "window", "window")
    put(model.Model, "flat", lambda st: _within(st, "flat"))
    put(model.Model, "wire", lambda st: _within(st, "wire"))
    put(sim, "step", lambda st: _within(st, "judge: step") if "sweep" not in [n for n, _ in st] else "sweep: step")
    put(hybrid._chip.Agent, "fill_chip_flows", lambda st: _within(st, "rules: chips"))
    put(hybrid._fuel.FuelRules, "fill", lambda st: _within(st, "rules: fuel"))
    put(hybrid._strait.StraitRules, "fill", lambda st: _within(st, "rules: straits"))
    put(hybrid._lp.Planner, "_remember", "act: requests and watch")
    real = mod.copy.deepcopy

    class _Copy:  # ``copy.deepcopy`` as the agent's module calls it; its own recursion stays untimed
        deepcopy = staticmethod(clock.wrap(real, "copies of the rules"))

    mod.copy = _Copy


def _play(folder: str, task: str, entropy: int, episode: int, weeks: int, numbers: dict, profile: bool) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    source = Path(folder).resolve()
    clock, prof = Clock(), cProfile.Profile() if profile else None
    with tempfile.TemporaryDirectory(prefix="weeks_") as tmp:
        target = Path(tmp) / f"{source.name}_{task}"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((target / "regime.json").read_text()) | NO_LIMITS | numbers
        (target / "regime.json").write_text(json.dumps(regime))
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        cls = load(str(target))
        _wrap_all(clock, cls)
        t0 = time.process_time()
        agent = cls(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        rows, done = [], False
        while not done and len(rows) < weeks:
            if prof is not None:
                prof.enable()
            action = agent.act(obs)
            if prof is not None:
                prof.disable()
            cpu = time.process_time() - t0
            kinds, calls = clock.week()
            keys = ("what", "attempt", "method", "status", "cpu", "simplex", "ipm", "rows", "cols")
            runs = agent.detail[-1]["solves"] if getattr(agent, "detail", None) else []
            solves = [{k: s[k] for k in keys} for s in runs]
            rows.append({"action": {k: np.array(v) for k, v in action.items()}, "note": tuple(agent.log[-1][1:]),
                         "cpu": cpu, "kinds": kinds, "calls": calls, "solves": solves})  # fmt: skip
            obs, _r, term, trunc, _i = env.step(action)
            done = term or trunc
            t0 = time.process_time()
    text = ""
    if prof is not None:
        for order in ("tottime", "cumulative"):
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats(order).print_stats(70)
            lines = [line for line in s.getvalue().splitlines() if line.strip()]
            text += f"--- by {order}\n" + "\n".join(line.replace(str(target), "")[:200] for line in lines) + "\n"
    return {"rows": rows, "regime": regime, "profile": text, "folder": str(source)}


def _path(tag: str, task: str, entropy: int, episode: int) -> Path:
    return OUT / f"{tag}_{task}_{entropy}_{episode}.pkl"


def play(folder: str, tag: str, task: str = "full", episode: int = 3, weeks: int = 30, entropy: int = 444,
         numbers: dict | None = None, profile: bool = False) -> None:
    words = {"true": True, "false": False, "null": None}
    numbers = {k: words.get(v, v) if isinstance(v, str) else v for k, v in (numbers or {}).items()}
    played = _play(folder, task, entropy, episode, weeks, numbers, profile)
    OUT.mkdir(parents=True, exist_ok=True)
    _path(tag, task, entropy, episode).write_bytes(pickle.dumps(played))
    cpu = np.array([r["cpu"] for r in played["rows"]])
    print(f"{tag}: {folder} {numbers or ''} {len(cpu)} weeks of {task} {entropy} episode {episode}: CPU a week "
          f"median {np.median(cpu):.3f} p95 {np.percentile(cpu, 95):.3f} max {cpu.max():.3f} "
          f"sum {cpu.sum():.1f} s")  # fmt: skip
    if profile:
        print(played["profile"])


def _kinds(rows: list) -> dict:
    """Seconds by kind, one array of weeks each."""
    names = sorted({k for r in rows for k in r["kinds"]})
    return {k: np.array([r["kinds"].get(k, 0.0) for r in rows]) for k in names}


def show(*tags: str, task: str = "full", episode: int = 3, entropy: int = 444, skip: int = 0) -> None:
    """``skip``: weeks left out at the start (week 1 holds ``Agent(config)`` and has no carried plan)."""
    data = {tag: pickle.loads(_path(tag, task, entropy, episode).read_bytes()) for tag in tags}
    base = data[tags[0]]["rows"]
    for tag in tags:
        rows = data[tag]["rows"]
        first = None
        for week, (a, b) in enumerate(zip(base, rows), 1):
            keys = sorted(set(a["action"]) | set(b["action"]))
            moved = [k for k in keys if k not in a["action"] or k not in b["action"]
                     or not np.array_equal(a["action"][k], b["action"][k])]  # fmt: skip
            if moved or a["note"] != b["note"]:
                first = week
                break
        cpu = np.array([r["cpu"] for r in rows])[skip:]
        same = "the same actions and notes" if first is None else f"DIFFERS from week {first}"
        line = (f"{tag:10s} {len(rows)} weeks, {same if tag != tags[0] else 'the base'}; CPU a week median "
                f"{np.median(cpu):.3f} p95 {np.percentile(cpu, 95):.3f} max {cpu.max():.3f} mean {cpu.mean():.3f}")
        if tag != tags[0] and len(rows) == len(base):
            d = np.array([r["cpu"] for r in base])[skip:] - cpu
            line += f"; saved a week median {np.median(d):+.3f} p95 {np.percentile(d, 95):+.3f} mean {d.mean():+.3f}"
        print(line)
    kinds = {tag: _kinds(data[tag]["rows"][skip:]) for tag in tags}
    names = sorted({k for v in kinds.values() for k in v}, key=lambda k: -float(np.mean(kinds[tags[0]].get(k, [0.0]))))
    print(f"{'seconds a week by kind: mean (median)':38s}" + "".join(f"{tag:>18s}" for tag in tags))
    for name in names:
        cells = ""
        for tag in tags:
            v = kinds[tag].get(name)
            cells += f"{'':18s}" if v is None else f"{v.mean():10.3f} ({np.median(v):.3f})"
        print(f"{name:38s}{cells}")
    print(f"{'all':38s}" + "".join(f"{sum(v.mean() for v in kinds[tag].values()):10.3f}{'':8s}" for tag in tags))
    for tag in tags:
        runs = [s for r in data[tag]["rows"][skip:] for s in r["solves"]]
        for what in sorted({s["what"] for s in runs}):
            mine = [s for s in runs if s["what"] == what]
            cpu, it = np.array([s["cpu"] for s in mine]), np.array([s["simplex"] + s["ipm"] for s in mine])
            failed = sum(1 for s in mine if s["status"] != "Optimal")
            print(f"{tag:10s} solver runs of {what:8s}: {len(mine):3d}, CPU median {np.median(cpu):.3f} p95 "
                  f"{np.percentile(cpu, 95):.3f} mean {cpu.mean():.3f}, iterations median {int(np.median(it))}, "
                  f"rows x columns {mine[0]['rows']} x {mine[0]['cols']}, not optimal {failed}")  # fmt: skip


if __name__ == "__main__":
    fire.Fire({"play": play, "show": show})
