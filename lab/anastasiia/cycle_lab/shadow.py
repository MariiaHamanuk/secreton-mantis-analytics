"""In one state, what each way of choosing whole weeks is worth: the solve with the hull against a fitted model.

    uv run python lab/anastasiia/regime_lab/shadow.py run outputs/regime_lab/agents/hull_model --tag=s --episodes=24
    uv run python lab/anastasiia/regime_lab/shadow.py show s --episodes=24

``run`` plays an agent folder built with ``hull: "model"`` but with the solve with the hull deciding every week, so
the states are those the model with two solves meets. In each of them the week's plan is also made from the same
start with other choices of the weeks asked to be whole, and nothing of that is played:

- ``none``: no week asked to be whole;
- ``table``: the hull's own shares rounded from a table (its weeks, without its solution as a start for the next);
- ``h_total``: the hull's grids and whole weeks asked, spread over the weeks as the start's scarce fuel is burned;
- ``h_first``: only the first week the hull asks of each grid;
- ``h_later``, ``h_earlier``, ``h_earlier3``: each of the hull's weeks one short week of its grid later, one earlier,
  three earlier (how exact a week has to be);
- ``h_last``: for every grid the hull asks a week of, the last short week it may ask for instead;

and for the fitted model (``--model``: an ``.npz`` of ``shares.py fit``, the folder's own when not given) at each
probability to act on (``--thresholds``, the file's own when not given), ``<kind>@<threshold>``:

- ``model``: the model's shares as the agent uses them (``plan_core.told_shares``);
- ``flagged``: the hull's weeks, only for the grids the model flags (what its misses cost);
- ``last``: for every grid the model flags, its last short week the hull may ask for;
- ``plus``: the hull's weeks, and that last week for the grids the model flags and the hull asks nothing of (what
  its false alarms cost).

Kept per state: each plan's cost as the simulator plays it on the forecast, the weeks it asked to be whole and which
of them came out whole, and the model's probability for every grid. ``show`` sums the plans' gains over ``none`` as a
share of the episodes' room between the naive rule and the clairvoyant plan, so the numbers read as RSS; a gain is
that of one week's plan, and the sum over an episode's weeks is not the episode's gain.
"""

import math
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


OUT = plan.OUT / "shadow"
HULL = ("table", "h_total", "h_first", "h_later", "h_earlier", "h_earlier3", "h_last")
MODEL = ("model", "flagged", "last", "plus")


def _likely(core, ep, recs, mode, ref, until, model, after) -> dict:
    """{grid: the model's probability that the hull asks a whole week of it} (as ``plan_core.told_shares`` reads it)."""
    inst = ep.inst
    keys = [key for key, gm in mode["grid"].items() if gm == "OFF" and (until is None or key[0] <= until) and inst.grid_fabs[key[1]]]
    feats = core.share_features(ep, recs, mode, ref, keys)
    i_rmax, rows = core.SHARE_FEATURES.index("rmax"), {}
    for (t, gi), f in feats.items():
        if f[i_rmax] > 0.0:
            rows.setdefault(gi, []).append((t, f))
    out = {}
    for gi, own in rows.items():
        own.sort(key=lambda r: r[0])
        v = core.plan_features(np.array([r[0] for r in own], dtype=float), np.array([r[1] for r in own], dtype=float), float(ep.T), after)
        out[gi] = 1.0 / (1.0 + math.exp(-model[0].raw(v)))
    return out


def _rounded(core, ep, table: dict, domain: set, mode: dict) -> frozenset:
    trial = {**mode, "grid": dict(mode["grid"])}
    core._rounded_from(ep, table, domain, trial, write="SOFT")
    return frozenset(key for key, gm in trial["grid"].items() if gm == "SOFT" and mode["grid"][key] != "SOFT")


def _asked(core, ep, recs, mode, ref, until, model, after, hull: dict, thresholds: tuple) -> tuple:
    """({kind: the weeks asked to be whole}, {grid: the model's probability}) for the state the plan starts from."""
    mark, total = model[:2]
    every, domain = core.told_shares(ep, recs, mode, ref, until, (mark, total, -1.0), after)  # every grid: its spread
    by_grid = lambda table: {gi: sum(v for (_t, g), v in table.items() if g == gi) for gi in {g for _t, g in table}}  # noqa: E731
    h_sum, e_sum = by_grid(hull), by_grid(every)
    weeks = {}
    for t, gi in sorted(domain):
        weeks.setdefault(gi, []).append(t)
    last = lambda grids: frozenset((weeks[gi][-1], gi) for gi in grids if gi in weeks)  # noqa: E731
    asked = {"table": _rounded(core, ep, hull, set(hull), mode)}
    asked["h_total"] = _rounded(core, ep, {key: v * h_sum.get(key[1], 0.0) / e_sum[key[1]] for key, v in every.items() if e_sum[key[1]] > 0}, domain, mode)
    own = {}
    for t, gi in sorted(asked["table"]):
        own.setdefault(gi, []).append(t)

    def shift(by: int) -> frozenset:
        out = set()
        for gi, ts in own.items():
            short = weeks.get(gi, [])
            for t in ts:
                if t in short and 0 <= short.index(t) + by < len(short):
                    out.add((short[short.index(t) + by], gi))
        return frozenset(out)

    asked.update(h_first=frozenset((ts[0], gi) for gi, ts in own.items()), h_later=shift(1), h_earlier=shift(-1),
                 h_earlier3=shift(-3), h_last=last(own))
    likely = _likely(core, ep, recs, mode, ref, until, model, after)
    for thr in thresholds:
        told, _d = core.told_shares(ep, recs, mode, ref, until, (mark, total, thr), after)
        flagged = {gi for _t, gi in told}
        asked[f"model@{thr}"] = _rounded(core, ep, told, domain, mode)
        asked[f"flagged@{thr}"] = frozenset(key for key in asked["table"] if key[1] in flagged)
        asked[f"last@{thr}"] = last(flagged)
        asked[f"plus@{thr}"] = asked["table"] | last(flagged - set(own))
    return asked, likely


def _summary(ep, d: dict, marks) -> dict:
    recs, y = d["recs"], ep.marks.y_bar
    fabs = [gi for gi in range(len(ep.inst.grids)) if ep.inst.grid_fabs[gi]]
    whole = [(t, gi) for t, gi in sorted(marks) if recs[t - 1].shed[gi] <= 1e-6 * max(1.0, float(y[t - 1][gi]))]
    count = sum(1 for t in range(1, ep.T + 1) for gi in fabs if recs[t - 1].shed[gi] <= 1e-6 * max(1.0, float(y[t - 1][gi])))
    return {"J": int(d["J"]), "J_compared": int(d["J_compared"]), "J0": int(d["J0"]), "marks": sorted(marks), "whole": whole,
            "whole_weeks": count, "lots": float(sum(np.sum(r.lots_started) for r in recs)), "last": d["hist"][-1][0] if d["hist"] else None}


def _wrap(core, rows: list, ctx: dict, model: tuple | None, thresholds: tuple) -> None:
    real, real_told, real_rounded = core.descend, core.told_shares, core._rounded_from

    def write(weeks):  # in place of the rounding: these weeks are the ones asked to be whole
        def rounded(_ep, _shares, _domain, mode, write="SOFT"):
            count = 0
            for key in weeks:
                if mode["grid"].get(key) == "OFF":
                    mode["grid"][key] = write
                    count += 1
            return count
        return rounded

    def descend(ep, acts, **kw):
        if kw.get("hull") != "model":
            return real(ep, acts, **kw)
        own, after = kw["model"]
        fitted = model or own
        played = real(ep, acts, **{**kw, "hull": "round", "record": True, "model": None})
        hull = played.pop("shares", None)
        played.pop("share_features", None)
        if hull is None:  # the solve with the hull gave nothing to compare with
            return played
        recs0 = kw["played"][0] if kw.get("played") is not None else ep.simulate(acts)[0]
        mode, ref = ep.regimes(recs0, kw.get("hint"))
        asked, likely = _asked(core, ep, recs0, mode, ref, kw.get("close_until"), fitted, after, hull, thresholds or (fitted[2],))
        row = {"week": ctx["week"], "T": ep.T, "after": after, "likely": likely, "hull": _summary(ep, played, played.get("marks", ()))}
        free = {**kw, "deadline": None, "chain_deadline": None}  # what is not played is not on the week's clock
        row["none"] = _summary(ep, real(ep, acts, **{**free, "hull": False, "model": None}), ())
        done = {frozenset(): row["none"]}
        for kind, weeks in asked.items():
            if weeks not in done:
                core.told_shares, core._rounded_from = (lambda *_a: (None, None)), write(weeks)
                try:
                    done[weeks] = _summary(ep, real(ep, acts, **free), weeks)
                finally:
                    core.told_shares, core._rounded_from = real_told, real_rounded
            row[kind] = done[weeks]
        rows.append(row)
        return played

    core.descend = descend


def episode(agent: str, task: str, entropy: int, n: int, model: str, thresholds: tuple) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    cls = load(str(resolve(agent).resolve()))
    core = sys.modules[cls.__module__]._core
    rows, ctx = [], {"week": 0}
    _wrap(core, rows, ctx, core.share_model(Path(model)) if model else None, thresholds)
    ag = cls(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    done, t0 = False, time.process_time()
    while not done:
        ctx["week"] += 1
        obs, _r, term, trunc, _i = env.step(ag.act(obs))
        done = term or trunc
    return {"n": n, "J": int(u.core._ep.traj.J_cents), "rows": rows, "cpu": time.process_time() - t0}


def _path(tag: str, task: str, entropy: int) -> Path:
    return OUT / f"{tag}_{task}_{entropy}.pkl"


def run(agent: str, tag: str, task: str = "small", entropy: int = 111, episodes: int = 8, first: int = 0, n_jobs: int = 3,
        model: str = "", thresholds: str | tuple | float = ()) -> None:
    thresholds = tuple(float(x) for x in (thresholds.split(",") if isinstance(thresholds, str) else np.atleast_1d(thresholds)))
    path = _path(tag, task, entropy)
    kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
    todo = [n for n in range(first, first + episodes) if n not in kept]
    for r in Parallel(n_jobs=n_jobs)(delayed(episode)(agent, task, entropy, n, model, thresholds) for n in todo):
        kept[r["n"]] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(kept))
    show(tag, task=task, entropy=entropy, episodes=episodes, first=first)


def show(tag: str, task: str = "small", entropy: int = 111, episodes: int = 8, first: int = 0) -> None:
    kept = pickle.loads(_path(tag, task, entropy).read_bytes())
    refs = plan.references(task, entropy, first + episodes)
    ns = [n for n in range(first, first + episodes) if n in kept]
    room = sum(refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns)
    rows = [r for n in ns for r in kept[n]["rows"]]
    skip = ("week", "T", "after", "likely", "none", "marks", "totals")
    kinds = [k for k in rows[0] if k not in skip]
    hull = sum(r["none"]["J"] - r["table"]["J"] for r in rows)
    asks = [r for r in rows if r["table"]["marks"]]
    print(f"{task}, root {entropy}, episodes {ns[0]}..{ns[-1]} ({len(ns)}): {len(rows)} states, all met by the solve with the hull; it asks a whole week in {len(asks)}")
    print("  the plan's gain over no whole weeks asked, summed over the states, as a share of the episodes' room")
    print(f"  {'':16s} {'gain':>8s} {'of the hull':>12s} {'states':>7s} {'weeks':>6s} {'whole':>6s} {'worse than none':>16s}")
    for kind in kinds:
        gain = [r["none"]["J"] - r[kind]["J"] for r in rows]
        marked = [i for i, r in enumerate(rows) if r[kind]["marks"]]
        weeks = sum(len(rows[i][kind]["marks"]) for i in marked)
        whole = sum(len(rows[i][kind]["whole"]) for i in marked)
        worse = sum(1 for i in marked if gain[i] < -1e6)
        print(f"  {kind:16s} {sum(gain) / room:+8.4f} {sum(gain) / hull:12.0%} {len(marked):7d} {weeks:6d} {whole:6d} {worse:16d}")
    for kind in [k for k in kinds if k.startswith("model@")]:
        thr = float(kind.split("@")[1])
        pairs = [(gi in {g for _t, g in r["table"]["marks"]}, p >= thr) for r in rows for gi, p in r["likely"].items()]
        truth, said = np.array([a for a, _b in pairs]), np.array([b for _a, b in pairs])
        gate = [r for r in rows if not any(p >= thr for p in r["likely"].values())]
        lost = sum(r["none"]["J"] - r["table"]["J"] for r in gate)
        same = [r for r in asks if r[kind]["marks"]]
        dt = np.array([r[kind]["marks"][0][0] - r["table"]["marks"][0][0] for r in same])
        print(f"  at {thr}: of the grids the hull asks a week of the model flags {np.mean(said[truth]):.0%}; of those it flags the hull asks {np.mean(truth[said]):.0%}; "
              f"it flags no grid in {len(gate) / len(rows):.0%} of the states, which hold {lost / hull:.0%} of the hull's gain")
        if len(same):
            print(f"     where both ask: the same weeks in {np.mean([r[kind]['marks'] == r['table']['marks'] for r in same]):.0%}; the model's first week "
                  f"earlier in {np.mean(dt < 0):.0%} (by 3 or more: {np.mean(dt <= -3):.0%}), later in {np.mean(dt > 0):.0%} (by 3 or more: {np.mean(dt >= 3):.0%})")


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
