"""The model with a teacher's prices: a planner told the episode's own future plans beside the model every week, and
what its program is worth past a cut that the model's is not goes into the model's program as prices.

    uv run python lab/anastasiia/frontier_lab/prices/play.py build
    uv run python lab/anastasiia/frontier_lab/prices/play.py run zero --episodes=0-23
    uv run python lab/anastasiia/frontier_lab/prices/play.py run cut1 --episodes=0-23
    uv run python lab/anastasiia/frontier_lab/prices/play.py show h3_s zero cut1 --episodes=24

Three planners see the same observation every week (R2 of ``notes/r_rl.md``):

- the teacher: the model's folder with ``truth`` "everything" and ``horizon`` 0 (the whole future, no end credit). Its
  action is never sent; it carries its own plan from week to week;
- the plain model (``outputs/hazard_lab/agents/h3_s``), beside the player when the player is priced: what the model's
  program is worth without any price;
- the player: the plain model (variant ``zero``: the identity gate and the states of the audit) or the model whose
  program and whose judge (the cost of a played window) carry the prices.

A price. For a program with rows r, duals y and matrix A, and a cut after week h, the cost of the weeks past the cut
as a function of a column j of the weeks up to it has the slope ``g_h[j] = -sum(y[r] * A[r, j] for r past the cut)``
(the rows of later weeks that the column enters: balances with their lags, queues, the gas ration on last week's
stock, the regimes' rows). The price on column j is the teacher's slope less the model's, so that at the model's own
plan the weeks past the cut are worth what the teacher's are, to first order. At the window's end the model has no
rows past the cut: its slope there is its end credit (the objective's end columns and the fuel pools' rows), the
teacher's is that of its own weeks past the model's window. The duals are the interior point's without crossover (the
middle of the dual face, not a vertex of it), read by this file's own solve of the cell of each planner's plan: the
model's files are not changed (``Episode.solve`` keeps the duals of the extra rows only).

``run`` keeps, per episode, what hazard_lab's ``play.py`` keeps and, per week, the audit's numbers: for every stock
slot and the cuts after weeks 1, 4, 13 and the window's last, what a unit held there is worth to the model and to the
teacher, both plans' stock there, and what the model sees of the slot's routes. One file an episode in
``outputs/frontier_lab/prices/play/<tag>_<task>_<entropy>/<n>.pkl``; an episode is claimed with a lock file.
"""

import copy
import json
import os
import pickle
import shutil
import time
import types
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HAZARD = ROOT / "lab" / "anastasiia" / "hazard_lab"
OUT = ROOT / "outputs" / "frontier_lab" / "prices"
BASE = ROOT / "outputs" / "hazard_lab" / "agents" / "h3_s"  # the model without its clock, as its kept play was made
AGENTS = OUT / "agents"
TEACHER = {"truth": ["everything"], "horizon": 0, "watch": None, "watch_ask": 0.0, "ask_scale": 0.0}
AUDIT_CUTS = (1, 4, 13)  # and the window's last week
# the cell whose duals are read: the plan's regimes with every short week of a grid with fabs written as "HULL", as in
# the week's first solve. In the plan's exact cell a unit of fuel is never worth a whole week of lots (Small 444,
# episode 0, week 2: crude at term_eu 4.1 mn USD a unit in the exact cell, 7.1 mn in the cell with the hull)
HULL = os.environ.get("PRICES_EXACT") is None

# tag -> {"cuts": {week: weight}, "end": weight, "scale": every price times this}
VARIANTS = {
    "zero": {},  # no price: hazard_lab's h3 play to the cent, and the audit's states
    # the same with the "teacher" that has no foresight (the plain model, its window to the episode's end): what of
    # the audit's differences is a longer window and another plan, and what is the future
    "zero0": {"teacher": "tea0"},
    "cut1": {"cuts": {1: 1.0}},  # the state after the week of the action, worth what the teacher's is
    "cut1h": {"cuts": {1: 1.0}, "scale": 0.5},
    "cut4": {"cuts": {4: 1.0}},
    "end": {"end": 1.0},  # the window's end at the teacher's prices
    "req": {"cuts": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, "end": 1.0},  # the end and the stock of weeks 1 to 4
    "both": {"cuts": {1: 1.0}, "end": 1.0},
    # the control: the "teacher" is the plain model with its window to the episode's end and no foresight (``tea0``),
    # so the prices carry a longer window's view and whatever a price does by itself, and nothing of the future
    "ctl1": {"cuts": {1: 1.0}, "teacher": "tea0"},
    # which part of the cut's prices does what: the chips' chain alone (wafers, raw and packaged chips), the fuels
    # alone, and the prices read off the plans' exact cells (no "HULL" week: a unit of fuel is not worth a whole week)
    "cut1c": {"cuts": {1: 1.0}, "only": "chips"},
    "cut1f": {"cuts": {1: 1.0}, "only": "fuel"},
    "cut1x": {"cuts": {1: 1.0}, "exact": True},
    # no price larger than the unit's own worth (a chip's penalty, a fuel's price of shed load): without the duals'
    # prices of regimes
    "cut1k": {"cuts": {1: 1.0}, "clip": 1.0},
    "cut1kc": {"cuts": {1: 1.0}, "clip": 1.0, "only": "chips"},
    "cut1kf": {"cuts": {1: 1.0}, "clip": 1.0, "only": "fuel"},
    "cut1kh": {"cuts": {1: 1.0}, "clip": 1.0, "scale": 0.25},
    "ctl1k": {"cuts": {1: 1.0}, "clip": 1.0, "teacher": "tea0"},
    # the future's part alone: the teacher's slope less that of the planner with no foresight and the same window to
    # the episode's end (``tea0``), not less the model's own. What a longer window and its other plan do to the worth
    # of the state (most of the teacher's difference from the model, and no foresight) stays out of the prices
    "info1k": {"cuts": {1: 1.0}, "clip": 1.0, "baseline": "tea0"},
    "info1": {"cuts": {1: 1.0}, "baseline": "tea0"},
    "cut1xc": {"cuts": {1: 1.0}, "exact": True, "only": "chips"},
    "ctlreq": {"cuts": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}, "end": 1.0, "teacher": "tea0"},
}


def build() -> None:
    """The teacher's folder and the priced player's: copies of the model's, the teacher's with its own numbers."""
    for name, numbers in (("tea", TEACHER), ("tea0", {"horizon": 0}), ("pri", {})):
        out = AGENTS / name
        if out.exists():
            continue
        shutil.copytree(BASE, out, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((BASE / "regime.json").read_text()) | numbers
        (out / "regime.json").write_text(json.dumps(regime))
        print(out, numbers)


# ----- the duals of a planner's plan --------------------------------------------------------------------------------
def _module(agent):
    """The namespace of the agent's own ``agent.py`` (its loader keeps no module of that name)."""
    return types.SimpleNamespace(**{k: v for k, v in type(agent)._plan.__globals__.items() if k in ("_core", "_terms")})


def _ipm(core, ep, C, limit: float = 60.0) -> dict | None:
    """The cell's optimum by the interior point without crossover: x and the duals of every row."""
    hs, Highs = core.highs()
    A, lo, hi = ep.rows(C)
    n_row, n_col = A.shape
    inf = hs.kHighsInf
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = n_col, n_row
    lp.col_cost_, lp.col_lower_ = (ep.obj if C.cost is None else ep.obj + C.cost), C.lb
    lp.col_upper_ = np.where(np.isinf(C.ub), inf, C.ub)
    lp.row_lower_, lp.row_upper_ = np.where(np.isinf(lo), -inf, lo), np.where(np.isinf(hi), inf, hi)
    lp.offset_ = float(ep.offset)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n_col, n_row
    lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr.astype(np.int32), A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    h = Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    h.setOptionValue("time_limit", float(limit))
    h.passModel(lp)
    h.run()
    status, info = h.modelStatusToString(h.getModelStatus()), h.getInfo()
    if os.environ.get("PRICES_DEBUG"):
        print(
            f"ipm: {status}, primal {int(info.primal_solution_status)}, dual {int(info.dual_solution_status)}, rows {n_row}, "
            f"cols {n_col}, {h.getRunTime():.2f} s, ipm iterations {int(info.ipm_iteration_count)}, dual infeasibility max "
            f"{float(info.max_dual_infeasibility):.3g} sum {float(info.sum_dual_infeasibilities):.3g}, primal max "
            f"{float(info.max_primal_infeasibility):.3g}, J {float(info.objective_function_value):.6g}",
            flush=True,
        )
    # without crossover HiGHS calls the interior point's end "Unknown" and its duals infeasible by its 1e-7: they
    # are off by a few USD a unit on columns worth thousands to millions, which is what is read here
    if not (status == "Optimal" or (status == "Unknown" and int(info.primal_solution_status) == 2)):
        return None
    if float(info.max_dual_infeasibility) > 5e3:
        return None
    sol = h.getSolution()
    return {"A": A, "x": np.asarray(sol.col_value, dtype=float), "y": np.asarray(sol.row_dual, dtype=float),
            "J": float(info.objective_function_value), "off": float(info.max_dual_infeasibility)}  # fmt: skip


def _row_weeks(ep, C) -> np.ndarray:
    """The week of every row of a cell: a row of the oracle's by its place, an extra row by its latest column (a
    pool's column counts as past the window)."""
    T = ep.T
    weeks = np.arange(1, T + 1)
    col = np.empty(ep.N, dtype=np.int64)
    col[: ep.n0] = np.repeat(weeks, ep.nc)
    col[ep.n0 : ep.n1] = np.repeat(weeks, 2 * ep.G)
    if ep.n2 > ep.n1:
        col[ep.n1 : ep.n2] = np.repeat(weeks, 2 * ep.S)
    col[ep.n2 :] = T + 1
    extra = np.zeros(len(C.lo), dtype=np.int64)
    if len(C.r):
        np.maximum.at(extra, np.asarray(C.r, dtype=np.int64), col[np.asarray(C.c, dtype=np.int64)])
    return np.concatenate([np.repeat(weeks, ep.nub), np.repeat(weeks, ep.neq), extra])


def plan_duals(agent, exact: bool = False) -> dict | None:
    """The cell of the plan the planner made this week, solved here: the program, its x, its duals, its rows' weeks.
    ``exact``: the plan's own cell, no week written as "HULL"."""
    last = getattr(agent, "last", None)
    if last is None:
        return None
    ep, d, tweak, bonus, anchor, price, _H = last
    core = _module(agent)._core
    until = ep.T - int(agent.p["hull_until"])  # the weeks the model's own solve with the hull covers
    # the cell with the hull first, the ties read as the last solve's duals say, then as the regimes ask; where neither
    # solves (the hull's weeks may leave the orders held at the plan's no room), the plan's exact cell the same two ways
    first = HULL and not exact
    for hull, hint in ((first, d.get("hint")), (first, None), (False, d.get("hint")), (False, None)):
        try:
            mode, ref = ep.regimes(d["recs"], hint)
            if hull:  # as the week's first solve has it: a short week of a grid with fabs is a share of a whole week
                for key, gm in list(mode["grid"].items()):
                    if gm == "OFF" and key[0] <= until:
                        mode["grid"][key] = "HULL"
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None:
                tweak(C)
            sol = _ipm(core, ep, C)
        except Exception as error:  # a cell that cannot be written for this plan: no prices this week
            sol = None
            if os.environ.get("PRICES_DEBUG"):
                import traceback

                traceback.print_exc()
                print("plan_duals:", repr(error)[:300], flush=True)
        if sol is not None:
            assert ep.n0 == ep.T * ep.nc and sol["A"].shape[0] == ep.T * (ep.nub + ep.neq) + len(C.lo)
            return sol | {"ep": ep, "week": _row_weeks(ep, C), "T": ep.T, "nc": ep.nc, "hull": bool(hull)}
    return None


def slope(p: dict, h: int) -> np.ndarray:
    """``sum(y[r] * A[r, j])`` over the rows past week ``h``, for the columns j of weeks 1..h: what a unit of column
    j takes off the cost of the later weeks (minus the cost's slope)."""
    n = min(h, p["T"]) * p["nc"]
    return np.asarray(p["A"][:, :n].T @ (p["y"] * (p["week"] > h))).ravel()


def end_credit(ep) -> np.ndarray:
    """The end credit the model's objective pays per column of the oracle's program (USD a unit; 0 without one)."""
    return np.asarray(ep.m.objective(), dtype=float) - ep.obj[: ep.n0]


def worth(inst) -> np.ndarray:
    """USD a unit of each commodity is worth at most: a fuel the price of shed base load, a chip (packaged, raw, and
    the wafer it is made of) the penalty of a lost sale."""
    pi = np.zeros(len(inst.commodities))
    for d in inst.demands:
        pi[d.k] = max(pi[d.k], d.pi)
    for o in inst.osats:
        for raw, packed in inst.nodes[o].osat.packages.items():
            pi[raw] = max(pi[raw], pi[packed])
    W = pi.copy()
    for g in inst.grids:
        for k in inst.nodes[g].grid.fuels:
            W[k] = max(W[k], float(inst.nodes[g].grid.voll))
    W[W == 0] = pi.max()  # wafers
    return W


def families(ep, commodity: bool = False) -> np.ndarray:
    """Per column of a week of the oracle's program: 1 a fuel's, 2 the chips' chain's, 0 neither; with ``commodity``,
    the column's commodity (-1: none)."""
    inst = ep.inst
    fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
    fam = np.zeros(ep.nc, dtype=np.int8)
    com = np.full(ep.nc, -1, dtype=np.int64)
    for key, j in ep.tm.items():
        tag = key[0]
        if tag in ("x", "Q", "xi", "G"):
            k = key[2]
        elif tag in ("I", "O", "lift"):
            k = inst.stock_slots[key[1]].k
        elif tag == "p":
            k = inst.nodes[inst.fabs[key[1]]].fab.product
        elif tag in ("D", "U", "B"):
            k = inst.demands[key[1]].k
        else:
            k = None
        if k is not None:
            fam[j] = 1 if k in fuels else 2
            com[j] = k
    return com if commodity else fam


def prices(
    S: dict,
    T: dict,
    cuts: dict | None = None,
    end: float = 0.0,
    scale: float = 1.0,
    only: str | None = None,
    clip: float | None = None,
) -> np.ndarray | None:
    """USD per unit of the model's columns (its window's weeks, the oracle's columns): the teacher's slope of the
    weeks past each cut less the model's, weighted; at the window's end the model's slope is its end credit."""
    nc, TS = S["nc"], S["T"]
    if S["hull"] != T["hull"]:  # one cell with the hull and one without price a whole week differently: no prices
        return None
    q = np.zeros(TS * nc)
    for h, w in (cuts or {}).items():
        if h < TS and h < T["T"]:
            q[: h * nc] += w * (slope(S, h) - slope(T, h))
    if end and T["T"] > TS:  # the model's window stops before the episode does
        q += end * (slope(S, TS) + end_credit(S["ep"]) - slope(T, TS))
    if only is not None:
        q = q * np.tile(families(S["ep"]) == {"fuel": 1, "chips": 2}[only], TS)
    if clip is not None:  # no price beyond this many times the unit's own worth: the duals of a cell also hold the
        # price of staying in its regimes (0.1 to 0.8 % of the audit's rows are thousands of times a chip's worth)
        com = families(S["ep"], commodity=True)
        limit = np.tile(np.where(com >= 0, clip * worth(S["ep"].inst)[np.maximum(com, 0)], 0.0), TS)
        q = np.clip(q, -limit, limit)
    return scale * q if np.any(q) else None


# ----- the priced player ---------------------------------------------------------------------------------------------
class Context:
    q: np.ndarray | None = None  # this week's prices, set by the harness before the player plans


CTX = Context()  # one a process: the player's module is patched once and reads it in every episode


def patch(module, ctx: Context = CTX) -> None:
    """The player's programs and its judge carry ``ctx.q``: a subclass of its ``Episode`` in its own module."""
    core = module._core
    if getattr(core.Episode, "_priced", False):
        return
    from shockbench_flow.oracle.replay import _vector

    class Priced(core.Episode):
        _priced = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            q = ctx.q
            self.q = q if q is not None and len(q) == self.n0 else None

        def cell(self, mode, ref, anchor=None, price=None, bonus=None):
            C = super().cell(mode, ref, anchor, price, bonus)
            if self.q is not None:
                extra = np.zeros(self.N)
                extra[: self.n0] = self.q
                C.cost = extra if C.cost is None else C.cost + extra
            return C

        def cost(self, recs, state):
            J = super().cost(recs, state)
            if self.q is not None and len(recs) == self.T:
                z, _extra = _vector(self.m, types.SimpleNamespace(records=recs))
                J += int(round(100.0 * float(self.q @ z)))
            return J

    core.Episode = Priced


# ----- the audit's numbers of a week ---------------------------------------------------------------------------------
def _routes(ep, node: int, k: int, w: int) -> tuple[float, float]:
    """The share of the nominal capacity of the edges into and out of ``node`` that carry ``k`` which the model's
    forecast has in week ``w`` of its window (a prohibited edge carries nothing; edges without a capacity left out)."""
    inst, marks = ep.inst, ep.marks
    u, Z = np.asarray(marks.u[w - 1], dtype=float), np.asarray(marks.prohibited[w - 1])
    out = []
    for edges in (inst.in_edges[node], inst.out_edges[node]):
        have = nominal = 0.0
        for e in edges:
            ed = inst.edges[e]
            if k in ed.K and ed.u0 is not None and np.isfinite(ed.u0):
                nominal += float(ed.u0)
                have += 0.0 if Z[e, k] else min(float(u[e]), float(ed.u0))
        out.append(have / nominal if nominal > 0 else 1.0)
    return out[0], out[1]


def _pending(obs, ep, slots: list) -> np.ndarray:
    """Announced prohibitions not yet in force on the edges into and out of each slot's node, for its commodity."""
    live = np.asarray(obs["pending_prohibitions.edge.observed"]) == 1
    edges, ks = np.asarray(obs["pending_prohibitions.edge"])[live], np.asarray(obs["pending_prohibitions.k"])[live]
    inst, out = ep.inst, np.zeros((len(slots), 2))
    for e, k in zip(edges.tolist(), ks.tolist()):
        ed = inst.edges[int(e)]
        for i, s in enumerate(slots):
            sl = inst.stock_slots[s]
            if sl.k == k:
                out[i, 0] += ed.head == sl.node
                out[i, 1] += ed.tail == sl.node
    return out


def audit_week(S: dict, T: dict, obs) -> dict:
    """What a unit held at the end of weeks 1, 4, 13 and the model's last is worth to the model and to the teacher, by
    stock slot (plain slots: no strait), both plans' stock there, and the model's view of the slot's routes."""
    ep, et = S["ep"], T["ep"]
    nc, TS, TT = S["nc"], S["T"], T["T"]
    assert nc == T["nc"] and ep.tm == et.tm
    slots = list(ep.plain)
    jI = np.array([ep.tm[("I", s)] for s in slots])
    cuts = sorted({h for h in AUDIT_CUTS if h < TS} | {TS})
    vS, vT, iS, iT = (np.full((len(cuts), len(slots)), np.nan) for _ in range(4))
    for i, h in enumerate(cuts):
        cols = (h - 1) * nc + jI
        iS[i], iT[i] = S["x"][cols], (T["x"][cols] if h <= TT else np.nan)
        if h < TS:
            vS[i], vT[i] = slope(S, h)[cols], slope(T, h)[cols]
        elif TT > TS:  # the window's end, where the episode goes on: the model's end credit against the teacher's weeks
            vS[i], vT[i] = (slope(S, TS) + end_credit(ep))[cols], slope(T, TS)[cols]
    ctx = np.array([[*_routes(ep, ep.inst.stock_slots[s].node, ep.inst.stock_slots[s].k, 1),
                     *_routes(ep, ep.inst.stock_slots[s].node, ep.inst.stock_slots[s].k, min(4, TS)),
                     *_routes(ep, ep.inst.stock_slots[s].node, ep.inst.stock_slots[s].k, min(13, TS))] for s in slots])  # fmt: skip
    G0 = np.array([ep.inst.nodes[g].grid.deliverable for g in ep.inst.grids], dtype=float)
    first = {}  # the columns of the window's first week: both slopes past it and both plans (what a table is fitted on)
    if 1 < TS and 1 < TT:
        first = {"gS": slope(S, 1), "gT": slope(T, 1), "xS": S["x"][:nc].copy(), "xT": T["x"][:nc].copy()}
    return first | {"TS": TS, "TT": TT, "cuts": cuts, "slots": slots, "vS": vS, "vT": vT, "iS": iS, "iT": iT,
            "i0": np.asarray(ep.i0, dtype=float)[slots], "routes": ctx, "pending": _pending(obs, ep, slots),
            "warning": np.asarray(obs["warning.score"], dtype=float).copy(),
            "grids": np.asarray(ep.marks.G_bar[0], dtype=float) / G0, "off": (S["off"], T["off"]),
            "hull": (S["hull"], T["hull"])}  # fmt: skip


# ----- play ----------------------------------------------------------------------------------------------------------
def episode(tag: str, task: str, entropy: int, n: int, weeks: int = 0) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    opts = dict(VARIANTS[tag])
    teacher_folder, exact = opts.pop("teacher", "tea"), opts.pop("exact", False)
    baseline = opts.pop("baseline", None)  # the planner whose slopes the teacher's are set against (None: the model)
    priced = bool(opts)
    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped

    def made(folder: Path):
        return load(str(folder))(agent_config(info["static"], info["policy_seed"], u.layout, obs))

    ctx = CTX
    ctx.q = None
    teacher = made(AGENTS / teacher_folder)
    teacher.tell_truth({"marks": u.core._ep.marks, "events": None})
    # the model without prices (the player itself when nothing is priced), or the named planner in its place
    plain = made(BASE if baseline is None else AGENTS / baseline)
    player = plain
    if priced:
        player = made(AGENTS / "pri")
        patch(_module(player), ctx)
    cpu, audit, sizes, asked, done = [], [], [], [], False
    missed = {"teacher": 0, "model": 0}
    while not done:
        t_all = time.process_time()
        teacher.last = None
        told = teacher.act(copy.deepcopy(obs))
        T = plan_duals(teacher, exact)
        if priced:
            plain.last = None
            unpriced = plain.act(copy.deepcopy(obs))
            S = plan_duals(plain, exact)
            ctx.q = prices(S, T, **opts) if S is not None and T is not None else None
            if ctx.q is not None and baseline is not None:  # the baseline's window is not the player's: its first weeks
                left = u.core._ep.inst.T - len(cpu)
                width = (min(int(player.p["horizon"]), left) if int(player.p["horizon"]) else left) * S["nc"]
                ctx.q = np.concatenate([ctx.q, np.zeros(max(0, width - len(ctx.q)))])[:width]
            sizes.append(0.0 if ctx.q is None else float(np.abs(ctx.q).sum()))
        t0 = time.process_time()
        player.last = None
        action = player.act(obs)
        took = time.process_time() - t0
        ctx.q = None
        if not priced:
            S = plan_duals(player, exact)
        # the week's requests of the teacher, of the plain model and of the player (the plain model's are the player's
        # own when nothing is priced): how far the prices move the action, and whether towards the teacher's
        asked.append(
            np.array([np.asarray(a["flows"], dtype=float) for a in (told, unpriced if priced else action, action)])
        )
        missed["teacher"] += T is None
        missed["model"] += S is None
        audit.append(audit_week(S, T, obs) if S is not None and T is not None else None)
        cpu.append(took)
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc or (weeks and len(cpu) >= weeks)
        if weeks:
            a = audit[-1]
            print(f"week {len(cpu)}: player {took:.2f} s, all {time.process_time() - t_all:.2f} s, log {player.log[-1][1:3]}"
                  + ("" if a is None else f", cuts {a['cuts']}, model's worth {np.nansum(np.abs(a['vS'])):.3g}, teacher's {np.nansum(np.abs(a['vT'])):.3g}")
                  + (f", prices {sizes[-1]:.3g}" if sizes else ""), flush=True)  # fmt: skip
    recs = u.core._ep.traj.records
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    inst = u.core._ep.inst
    sent = np.zeros((len(recs), len(inst.action_slots)))
    seg = np.zeros((len(recs), len(inst.grids), len(inst.commodities) + 1))
    for t, r in enumerate(recs):
        for slot, q in r.executed.items():
            sent[t, slot] = q
        for (gi, k), q in r.segment.items():
            seg[t, gi, -1 if k is None else k] = q
    return {
        "n": n, "J": int(sum(r.cost_cents for r in recs) if weeks else u.core._ep.traj.J_cents), "cpu": cpu,
        "log": list(player.log),
        "notes": {"missed": missed, "prices": sizes, "errors": sum(len(row) > 1 and row[1] == "error" for row in player.log)},
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
        "lost": np.array([r.lost for r in recs]), "served": np.array([r.served for r in recs]),
        "energy": np.array([r.energy for r in recs]), "stock": np.array([r.stock for r in recs]),
        "segment": seg, "sent": sent, "audit": audit, "teacher_log": list(teacher.log), "asked": np.array(asked),
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
    build()
    for tag in tags:
        folder = OUT / "play" / f"{tag}_{task}_{entropy}"
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
                print(tag, n, "J", r["J"], r["notes"]["missed"], flush=True)
                continue
            tmp = folder / f"{n}.tmp"
            tmp.write_bytes(pickle.dumps(r))
            tmp.replace(done)
            lock.unlink(missing_ok=True)
            print(f"{tag} {task} {entropy} episode {n}: {time.time() - started:.0f} s wall, player {sum(r['cpu']):.0f} s CPU, "
                  f"J {r['J'] / 1e11:.2f} bn, missed {r['notes']['missed']}, weeks with an error {r['notes']['errors']}", flush=True)  # fmt: skip


def kept(tag: str, task: str, entropy: int) -> dict:
    folder = OUT / "play" / f"{tag}_{task}_{entropy}"
    if folder.is_dir():
        return {int(p.stem): pickle.loads(p.read_bytes()) for p in folder.glob("*.pkl")}
    path = ROOT / "outputs" / "hazard_lab" / "play" / f"{tag}_{task}_{entropy}.pkl"
    return pickle.loads(path.read_bytes()) if path.is_file() else {}


def show(
    *tags: str, task: str = "small", entropy: int = 444, episodes: int = 24, first: int = 0, by_episode: bool = False
) -> None:
    """hazard_lab's ``show``; a tag is looked up here first, then among hazard_lab's kept plays."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("hazard_play", HAZARD / "play.py")
    hazard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hazard)
    hazard._kept = lambda path: kept(path.stem[: -len(f"_{task}_{entropy}")], task, entropy)
    hazard.show(*tags, task=task, entropy=entropy, episodes=episodes, first=first, by_episode=by_episode)


if __name__ == "__main__":
    fire.Fire({"build": build, "run": run, "show": show})
