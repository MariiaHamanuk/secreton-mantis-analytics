"""The anatomy of one more pass of the week's program, on the model's own play.

    uv run python lab/anastasiia/frontier_lab/mech/pass2.py run --episodes=0-7
    uv run python lab/anastasiia/frontier_lab/mech/pass2.py show

``run`` plays a model folder as it is (the play is the folder's own: nothing below changes a decision) and, every
week after the action is made, solves the week's program once more in the regimes of the plan just made, as
``passes`` 2 would (``plan_core.descend``'s next pass: the regimes read off the played plan with the last hint, the
plain cell, the solve from the last basis, the play on the model). Kept per week: the model's cost of the plan and
of the second plan by week of the window, what each asks and what the model executes by kind of slot and by week of
the window, lots and shed load by week of the window, the first week slot by slot, the regimes that differ.

``show`` prints where in the window the second pass's saving sits and what it changes in the first week.
"""

import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(HERE.parent / "scen"), str(ROOT / "lab" / "anastasiia" / "hazard_lab")]
OUT = ROOT / "outputs" / "frontier_lab" / "mech" / "pass2"
KINDS = ("order", "valve", "wafer", "raw", "pack")
COMP = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def _weekly(ep, acts: list, recs: list, kinds: np.ndarray) -> dict:
    T, S = len(recs), len(kinds)
    ask, done = np.zeros((T, S)), np.zeros((T, S))
    for t, ((fl, _ov, _ho), rec) in enumerate(zip(acts, recs)):
        for s, q in fl.items():
            ask[t, s] = q
        for s, q in rec.executed.items():
            done[t, s] = q
    return {"ask": ask, "done": done, "cost": np.array([r.cost_cents for r in recs], dtype=float) / 1e11,
            "lots": np.array([np.sum(r.lots_started) for r in recs], dtype=float),
            "shed": np.array([np.sum(r.shed) for r in recs], dtype=float),
            "lots_fab": np.array([r.lots_started for r in recs], dtype=float),
            "shed_grid": np.array([r.shed for r in recs], dtype=float),
            "items": np.array([[getattr(r.costs, c) for c in COMP] for r in recs], dtype=float) / 1e9,
            "release": np.array([sum(ov.values()) for _fl, ov, _ho in acts], dtype=float)}  # fmt: skip


def _codes(mode: dict, T: int, G: int) -> np.ndarray:
    """The grids' regimes by (week of the window, grid) as short strings."""
    return np.array([[str(mode["grid"].get((t, gi), "")) for gi in range(G)] for t in range(1, T + 1)])


def _record_cells(core) -> None:
    """Every plan made from a cell's solution remembers the regimes of that cell (nothing else changes)."""
    if getattr(core.Episode, "_mech", False):
        return
    cell, actions = core.Episode.cell, core.Episode.actions

    def cell_kept(self, mode, ref, *args, **kwargs):
        self._mech_mode = {family: dict(mode[family]) for family in mode}
        return cell(self, mode, ref, *args, **kwargs)

    def actions_kept(self, x):
        acts = actions(self, x)
        if not hasattr(self, "_mech_by"):
            self._mech_by = {}
        self._mech_by[id(acts)] = getattr(self, "_mech_mode", None)
        return acts

    core.Episode.cell, core.Episode.actions, core.Episode._mech = cell_kept, actions_kept, True


def _flips(a: dict, b: dict, T: int) -> dict:
    """Per family of regimes, the weeks of the window in which the two readings differ (a count per week)."""
    out = {}
    for family in a:
        n = np.zeros(T, dtype=int)
        for key, v in a[family].items():
            w = b[family].get(key)
            same = (abs(v - w) < 1e-6) if isinstance(v, float) and isinstance(w, float) else v == w
            if not same:
                n[key[0] - 1] += 1
        out[family] = n
    return out


def _head(ep, C, recs: list, h: int) -> None:
    """The cell ``C`` with every decision of the weeks after ``h`` held at what the played plan ``recs`` executed (the
    dispatches of the action slots and the tanker releases): only the first ``h`` weeks are decided again. What
    follows from the decisions (stocks, queues, the lanes' flows through the straits, lots, sales) stays free."""
    keys = [(e, k, lane, j) for _s, (e, k, lane), j in ((s, ep.inst.action_slots[s], j) for s, _e, _k, j in ep._slots)]
    keys += [(e, k, None, j) for _c, k, e, _o, j in ep._first if j is not None]
    for t in range(h + 1, ep.T + 1):
        x = recs[t - 1].x
        for e, k, lane, j0 in keys:
            j = (t - 1) * ep.nc + j0
            C.lb[j] = C.ub[j] = min(max(float(x.get((e, k, lane), 0.0)), C.lb[j]), C.ub[j])


def episode(agent: str, task: str, entropy: int, n: int, every: int = 1, heads: tuple = (), warm: bool = False) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    Agent = load(str(resolve(agent).resolve()))
    ag = Agent(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    core = sys.modules[Agent.__module__]._core
    _record_cells(core)
    kinds = np.array(ag.kinds)
    G = len(ag.model.inst.grids)
    if warm:  # a large program's basis is read out of the solver (no search on Full: nothing else changes)
        ag.p["try_warm"] = True
    rows, done, last = [], False, None
    while not done:
        week = int(np.asarray(obs["week"]).ravel()[0])
        action = ag.act(obs)
        made = getattr(ag, "last", None)
        note = ag.log[-1]
        if made is not None and made is not last and ag.acts is not None and (week - 1) % every == 0 and ":new" in str(note[-1]):
            last = made
            ep, d, tweak, bonus, anchor, price, H = made
            t0 = time.process_time()
            row = {"week": week, "H": H, "left": ep.inst.T if False else None, "note": note[-1], "hist": list(d["hist"])}
            row["capped"] = ep.end is not None
            acts1, recs1, J1 = d["acts"], d["recs"], d["J"]
            cell1 = getattr(ep, "_mech_by", {}).get(id(acts1))  # the regimes of the cell the plan was solved in
            mode, ref = ep.regimes(recs1, d.get("hint"))
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None:
                tweak(C)
            # ``warm``: on a large program the solve starts from the plan's own basis, as a try of the search does
            sol = ep.solve(C, method="auto", basis=d.get("basis"), time_limit=30.0, what="search mech" if warm else "mech",
                           big=str(ag.p["big_exact"]))
            row["status"] = sol["status"]
            row["solve"] = dict(ep.solves[-1])
            row["first"] = [dict(x) for x in ep.solves[:-1]]  # the week's own solves
            row["J1"], row["away1"] = J1 / 1e11, core._beside(acts1, recs1, anchor, price, bonus) / 1e11
            row["one"] = _weekly(ep, acts1, recs1, kinds)
            if sol["status"] == "Optimal":
                acts2 = ep.actions(sol["x"])
                recs2, J2 = ep.simulate(acts2)
                row["J2"], row["away2"] = J2 / 1e11, core._beside(acts2, recs2, anchor, price, bonus) / 1e11
                row["claim2"] = sol["J"] / 1e9
                row["two"] = _weekly(ep, acts2, recs2, kinds)
                mode2, _ref2 = ep.regimes(recs2)
                mode1, _ref1 = ep.regimes(recs1)
                row["flips"] = _flips(mode1, mode2, H)
                row["flips_cell"] = _flips(mode, mode1, H)  # what the hint changed in the reading of the plan itself
                row["grid1"], row["grid2"] = _codes(mode1, H, G), _codes(mode2, H, G)
                if cell1 is not None:  # where the plan, as played, has left the cell it was solved in
                    row["left_cell"] = _flips(cell1, mode1, H)
                    row["grid_cell"] = _codes(cell1, H, G)
                for h in heads:  # the same pass with the dispatches of the weeks after ``h`` held
                    if h >= H:
                        continue
                    Ch = ep.cell(mode, ref, anchor, price, bonus)
                    if tweak is not None:
                        tweak(Ch)
                    _head(ep, Ch, recs1, int(h))
                    sh = ep.solve(Ch, method="auto", basis=d.get("basis"), time_limit=30.0,
                                  what="search head" if warm else "head", big=str(ag.p["big_exact"]))
                    out = {"status": sh["status"], "solve": dict(ep.solves[-1])}
                    if sh["status"] == "Optimal":
                        acts_h = ep.actions(sh["x"])
                        recs_h, Jh = ep.simulate(acts_h)
                        out |= {"J": Jh / 1e11, "away": core._beside(acts_h, recs_h, anchor, price, bonus) / 1e11,
                                "week": _weekly(ep, acts_h, recs_h, kinds)}
                    row[f"head{int(h)}"] = out
            row["cpu"] = time.process_time() - t0
            rows.append(row)
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    return {"n": n, "J": int(u.core._ep.traj.J_cents), "rows": rows, "kinds": kinds, "log": list(ag.log)}


def run(agent: str = "outputs/hazard_lab/agents/h3_s", tag: str = "h3_s", task: str = "small", entropy: int = 444,
        episodes="0-7", every: int = 1, heads="", warm: bool = False) -> None:
    import play as P

    folder = OUT / f"{tag}_{task}_{entropy}"
    folder.mkdir(parents=True, exist_ok=True)
    for n in P._numbers(episodes):
        path = folder / f"{n}.pkl"
        if path.is_file():
            continue
        started = time.time()
        hs = tuple(P._numbers(heads)) if heads else ()
        r = episode(str(ROOT / agent), task, entropy, n, every, hs, warm)
        path.write_bytes(pickle.dumps(r))
        gains = [row["J1"] - row["J2"] for row in r["rows"] if "J2" in row]
        print(f"{tag} {task} {entropy} episode {n}: {time.time() - started:.0f} s, J {r['J']}, weeks {len(r['rows'])}, "
              f"second pass saves {np.mean(gains):.2f} bn a week in the model", flush=True)


def show(tag: str = "h3_s", task: str = "small", entropy: int = 444) -> None:
    folder = OUT / f"{tag}_{task}_{entropy}"
    data = [pickle.loads(p.read_bytes()) for p in sorted(folder.glob("*.pkl"))]
    if not data:
        print("nothing kept")
        return
    kinds = data[0]["kinds"]
    rows = [row for r in data for row in r["rows"] if "J2" in row]
    full = [row for row in rows if row["capped"]]
    H = max(row["H"] for row in rows)
    print(f"{tag} {task} {entropy}: {len(data)} episodes, {len(rows)} weeks with a second pass solved, {len(full)} of them in a capped window")
    for name, group in (("capped window", full), ("window to the episode's end", [row for row in rows if not row["capped"]])):
        if not group:
            continue
        g = np.array([row["J1"] - row["J2"] for row in group])
        ga = np.array([row["J1"] + row["away1"] - row["J2"] - row["away2"] for row in group])
        print(f"  {name}: {len(group)} weeks; the second plan is cheaper on the model by {g.mean():+.2f} bn (median {np.median(g):+.2f}); "
              f"with the anchor's price {ga.mean():+.2f}; cheaper in {np.mean(ga > 1e-5):.2f} of weeks, dearer in {np.mean(ga < -1e-5):.2f}")
    # where in the window the saving sits (capped windows of the full length)
    group = [row for row in full if row["H"] == H]
    c = np.array([row["one"]["cost"] - row["two"]["cost"] for row in group])
    end = np.array([(row["J1"] - row["one"]["cost"].sum()) - (row["J2"] - row["two"]["cost"].sum()) for row in group])
    print(f"  capped windows of {H} weeks ({len(group)}): saving by week of the window, bn (then the end credit):")
    print("    " + " ".join(f"{x:+.2f}" for x in c.mean(axis=0)) + f"   end {end.mean():+.2f}")
    for item in ("lots", "shed", "release"):
        a = np.array([row["one"][item] for row in group]).mean(axis=0)
        b = np.array([row["two"][item] for row in group]).mean(axis=0)
        print(f"  {item}: second minus first by week of the window, % of the first's weekly mean ({a.mean():.4g}):")
        print("    " + " ".join(f"{100 * x / max(a.mean(), 1e-9):+.1f}" for x in (b - a)))
    print("  requests by kind: the second plan's minus the first's by week of the window, % of the first's weekly mean; then the turnover |difference| in week 1 and over the window")
    for kind in KINDS:
        m = kinds == kind
        a = np.array([row["one"]["done"][:, m].sum(axis=1) for row in group])
        b = np.array([row["two"]["done"][:, m].sum(axis=1) for row in group])
        turn = np.array([np.abs(row["two"]["done"][:, m] - row["one"]["done"][:, m]).sum(axis=1) for row in group])
        scale = max(a.mean(), 1e-9)
        print(f"    {kind:6s} " + " ".join(f"{100 * x / scale:+.1f}" for x in (b - a).mean(axis=0)) +
              f"   turnover week 1 {100 * turn[:, 0].mean() / scale:.1f} %, window {100 * turn.mean() / scale:.1f} %")
    print("  regimes that differ between the two plans as played, a count by week of the window:")
    for family in group[0]["flips"]:
        f = np.array([row["flips"][family] for row in group]).mean(axis=0)
        if f.sum() > 0:
            print(f"    {family:5s} " + " ".join(f"{x:.2f}" for x in f))


def near(tag: str = "h3_s", task: str = "small", entropy: int = 444, weeks: int = 6) -> None:
    """The first ``weeks`` weeks of the capped windows: what the second pass saves there by item, the grids' regimes
    as the plan's cell had them, as the plan played and as the second plan played, shed load and lots."""
    folder = OUT / f"{tag}_{task}_{entropy}"
    data = [pickle.loads(p.read_bytes()) for p in sorted(folder.glob("*.pkl"))]
    rows = [row for r in data for row in r["rows"] if "J2" in row and row["capped"] and "grid_cell" in row]
    if not rows:
        print("nothing kept")
        return
    kinds = data[0]["kinds"]
    W = min(weeks, min(row["H"] for row in rows))
    print(f"{tag} {task} {entropy}: {len(rows)} capped weeks of {len(data)} episodes; the window's first {W} weeks")
    items = np.array([row["one"]["items"][:W] - row["two"]["items"][:W] for row in rows]).mean(axis=0)  # week, item
    print("  saving of the second plan by item, bn a week of play, by week of the window:")
    for i, name in enumerate(COMP):
        if np.abs(items[:, i]).max() > 5e-3:
            print(f"    {name:14s} " + " ".join(f"{x:+.2f}" for x in items[:, i]) + f"   sum {items[:, i].sum():+.2f}")
    tail = np.array([row["one"]["items"][W:] - row["two"]["items"][W:] for row in rows if row["H"] > W])
    if len(tail):
        t = tail.mean(axis=0).sum(axis=0)
        print("    later weeks of the window, all of them: " + ", ".join(f"{name} {x:+.2f}" for name, x in zip(COMP, t) if abs(x) > 5e-3))
    s1 = np.array([row["one"]["shed_grid"][:W] for row in rows])
    s2 = np.array([row["two"]["shed_grid"][:W] for row in rows])
    print("  shed load a week of play, by week of the window: the plan / the second plan (all grids), then by grid the difference")
    print("    " + " ".join(f"{a:7.1f}/{b:7.1f}" for a, b in zip(s1.sum(axis=2).mean(axis=0), s2.sum(axis=2).mean(axis=0))))
    for gi in range(s1.shape[2]):
        print(f"    grid {gi}: " + " ".join(f"{x:+8.1f}" for x in (s2 - s1)[:, :, gi].mean(axis=0)))
    l1 = np.array([row["one"]["lots_fab"][:W].sum(axis=1) for row in rows])
    l2 = np.array([row["two"]["lots_fab"][:W].sum(axis=1) for row in rows])
    print("  lots a week: " + " ".join(f"{a:8.0f}/{b:8.0f}" for a, b in zip(l1.mean(axis=0), l2.mean(axis=0))))
    for kind in ("valve", "wafer", "raw", "pack"):
        m = kinds == kind
        a = np.array([row["one"]["done"][:W][:, m].sum(axis=1) for row in rows]).mean(axis=0)
        b = np.array([row["two"]["done"][:W][:, m].sum(axis=1) for row in rows]).mean(axis=0)
        print(f"  {kind:6s} executed, second / first: " + " ".join(f"{y / max(x, 1e-9):6.3f}" for x, y in zip(a, b)))
    # the grids' regimes: the cell the plan was solved in -> the plan as played -> the second plan as played
    print("  grid-weeks by regime (the plan's cell -> the plan as played -> the second plan as played), a share of grid-weeks, by week of the window")
    count: dict = {}
    for row in rows:
        c, a, b = row["grid_cell"][:W], row["grid1"][:W], row["grid2"][:W]
        for t in range(W):
            for gi in range(c.shape[1]):
                key = (c[t, gi], a[t, gi], b[t, gi])
                count.setdefault(key, np.zeros(W))[t] += 1
    total = len(rows) * rows[0]["grid_cell"].shape[1]
    for key, v in sorted(count.items(), key=lambda kv: -kv[1].sum()):
        if v.sum() / (total * W) >= 0.002:
            print(f"    {key[0]:>5s} -> {key[1]:>4s} -> {key[2]:>4s}: " + " ".join(f"{x / total:.3f}" for x in v))
    left = {family: np.array([row["left_cell"][family][:W] for row in rows]).mean(axis=0) for family in rows[0]["left_cell"]}
    print("  elements whose regime, as the plan played, is not its cell's (a count a week of play):")
    for family, v in left.items():
        if v.sum() > 0:
            print(f"    {family:5s} " + " ".join(f"{x:.2f}" for x in v))
    # the shed load of the plan's first weeks by what its cell said of the grid-week
    by: dict = {}
    for row in rows:
        c, a = row["grid_cell"][:W], row["grid1"][:W]
        for t in range(W):
            for gi in range(c.shape[1]):
                k = (c[t, gi], a[t, gi])
                e = by.setdefault(k, [0, 0.0, 0.0])
                e[0] += 1
                e[1] += row["one"]["shed_grid"][t, gi]
                e[2] += row["two"]["shed_grid"][t, gi]
    print("  shed load of the first weeks by (the cell's regime, the regime as played): grid-weeks, the plan's shed load, the second plan's")
    for k, (n, a, b) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"    {k[0]:>5s} -> {k[1]:>4s}: {n:6d}  {a:10.0f}  {b:10.0f}")


def heads(tag: str = "h3h_s", task: str = "small", entropy: int = 444, weeks: int = 6) -> None:
    """The pass on the head against the whole pass: what each saves on the model, in the window's first ``weeks``
    weeks and in all, how near its first week is to the whole pass's, and what its solve takes."""
    folder = OUT / f"{tag}_{task}_{entropy}"
    data = [pickle.loads(p.read_bytes()) for p in sorted(folder.glob("*.pkl"))]
    rows = [row for r in data for row in r["rows"] if "J2" in row and row["capped"]]
    if not rows:
        print("nothing kept")
        return
    kinds = data[0]["kinds"]
    names = sorted({k for row in rows for k in row if k.startswith("head")}, key=lambda k: int(k[4:]))
    W = weeks
    exact = [x["cpu"] for row in rows for x in row["first"] if x["what"] == "exact"]
    hull = [x["cpu"] for row in rows for x in row["first"] if x["what"] == "hull"]
    print(f"{tag} {task} {entropy}: {len(rows)} capped weeks of {len(data)} episodes; the week's own solves: hull {np.median(hull):.3f} s, "
          f"exact {np.median(exact):.3f} s (medians)")
    print("  pass: solved; saving on the model, bn a week of play (all / with the anchor's price / the first weeks' costs / the first weeks' shed); "
          "first week's distance to the whole pass's as a share of the plan's distance to it (valves; all chip slots); solve CPU median, p95")

    def line(name: str, get) -> None:
        ok = [row for row in rows if get(row) is not None]
        g = np.array([row["J1"] - get(row)["J"] for row in ok])
        ga = np.array([row["J1"] + row["away1"] - get(row)["J"] - get(row)["away"] for row in ok])
        head = np.array([row["one"]["cost"][:W].sum() - get(row)["week"]["cost"][:W].sum() for row in ok])
        shed = np.array([row["one"]["items"][:W, 7].sum() - get(row)["week"]["items"][:W, 7].sum() for row in ok])
        dist = []
        for m in (kinds == "valve", kinds != "order"):
            far = sum(np.abs(row["one"]["done"][0, m] - row["two"]["done"][0, m]).sum() for row in ok)
            mine = sum(np.abs(get(row)["week"]["done"][0, m] - row["two"]["done"][0, m]).sum() for row in ok)
            dist.append(mine / max(far, 1e-9))
        cpu = np.array([get(row)["solve"]["cpu"] for row in ok])
        print(f"    {name:8s} {len(ok) / len(rows):.2f}  {g.mean():+6.2f} / {ga.mean():+6.2f} / {head.mean():+5.2f} / {shed.mean():+5.2f}   "
              f"{dist[0]:.2f}; {dist[1]:.2f}   {np.median(cpu):.3f}, {np.percentile(cpu, 95):.3f} s")

    line("whole", lambda row: {"J": row["J2"], "away": row["away2"], "week": row["two"], "solve": row["solve"]})
    for name in names:
        line(name, lambda row, name=name: row[name] if row.get(name, {}).get("status") == "Optimal" else None)


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show, "near": near, "heads": heads})
