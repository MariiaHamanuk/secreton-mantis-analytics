"""A plan carried from week to week inside the simulator's regimes; the rules order the fuel.

Two labs in one agent. From ``lab/anastasiia/plan_lab`` (``lean_plan``): the fuel rules keep the orders from the
sources, always, and the plan decides the rest (the valves from terminals into grids, the tanker releases at the
straits, wafers, raw and packaged chips); a rollout then needs the fuel rules only; a window shorter than the rest of
the episode ends with a value for the fuel and the chips still in the system. From ``lab/anastasiia/regime_lab``
(``core.py``, here ``plan_core.py``): the program whose rows are the simulator's regimes, read off a played
trajectory; the duals' hints at a border of two regimes; grid-weeks switched on away from a border; a simplex that
starts from last week's basis; SciPy's bundled HiGHS.

Every week:

1. the fuel rules' memory is copied; the simulator's state and the window's network come from the observation
   (``sim_model.py``; the network "as observed");
2. the reference: last week's plan from its second week on, played on the model with the fuel rules ordering every
   week (they read the plan's wafers). Without a plan, and every ``rules_every`` weeks beside it, the rules alone play
   the window; the cheaper of the two is the reference;
3. the regimes are read off the reference and the program is solved inside them with the orders as ``orders`` says
   (held at the reference's, by default); its solution is played open loop on the model and kept when it is cheaper
   (``passes`` such passes, then ``switch`` trials of a grid-week switched on);
4. the first week of the best plan is sent with the rules' orders of this week (computed on the plan's wafers). With
   no plan the hybrid's own action stands (its linear program runs only then).

The planner has a clock: ``share`` of the week's CPU budget, counted from the start of ``act``; a pass that would
start after it does not, and a solve stops at it. What the week has by then is played.

The hybrid's parts (``hybrid_agent.py``, its ``*_part.py`` and ``sbfv/``), ``sim_model.py`` and ``plan_core.py`` sit
beside this file: ``lab/anastasiia/regime_lab/build.py`` copies them there. Numbers of ``PARAMS`` come from
``regime.json`` beside this file when there is one.
"""

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OVERRIDE, HOLD = 1, 2  # release_mode codes
BUDGET_S = {"small": 2.0, "full": 4.0}  # the boards' CPU seconds per week, by the instance's kind
EPISODE_S = {"small": 180.0, "full": 480.0}  # the boards' wall-clock seconds for an episode, from its container's start

PARAMS = {
    "horizon": 26,  # weeks planned; 0: to the end of the episode
    "end_fuel": 3.3e6,  # USD a unit of fuel left in the system at the end of a capped window is worth
    "end_chip": 0.6,  # share of its penalty a chip left in the chain there is worth
    # weeks of burn up to which fuel left in the system is worth ``end_fuel`` (``plan_core.Episode._pools``): at a
    # grid and on the way to it, the grid's own stock level plus this many weeks; at the straits and on the way to
    # them, this many weeks of all the grids' burn. 0: every unit anywhere past its source is worth ``end_fuel``,
    # and a plan that orders lifts fuel to park it at the straits
    "end_weeks": 0.0,
    # the fuel orders from the sources. "rules": the fuel rules order every week, in the rollouts and in the action,
    # and the program holds them at the reference's; "floor": the same, the program may order more; "plan": the program
    # decides them like any entry, and the carried plan is replayed with its own orders
    "orders": "rules",
    # USD per unit a slot sends above or below the plan of the rules alone, by the kind of slot ("wafer", "order",
    # "valve", "raw", "pack"): the program leaves the rules' plan only where that pays. The rules alone are then
    # rolled out every week; meant for ``orders`` "plan"
    "anchor": {},
    "rules_every": 4,  # weeks between rollouts of the rules alone beside the carried plan (0: only without a plan)
    "passes": 1,  # passes of the program a week (regimes, solve, play)
    "hints": True,  # a tie between two regimes is read as the last solution's duals say
    "warm": True,  # the simplex starts from last week's basis
    "switch": 0,  # grid-weeks tried a week for a switch from "sheds" to "runs its fabs" (0: none)
    # in the week's first cell, the weeks a grid nearly closes (it sheds less than this share of its base load, for
    # want of a fuel it gets a little of every week) are gathered into whole weeks: every time the shares of a week's
    # burn add up to one more week, that week is written as closed and the program moves the fuel there. A fab runs
    # only in a closed week. No extra solve unless that cell has no solution (0: off)
    "close": 0.0,
    "close_rationed": False,  # gas that burns under its ration is gathered into whole weeks too
    # the worth of a whole week inside the program (``plan_core.descend``'s ``hull``): a short week of a grid with
    # fabs as a mix of whole weeks and weeks without the scarce fuel, so that the fuel is worth the lots of a whole
    # week. "round": one more solve a week (every short week so, the shares rounded into weeks, those weeks left so
    # in the first cell); "hard": the rounded weeks written as closed; "burn": no extra solve, the weeks rounded
    # from the reference's own burn. Takes the place of ``close``
    "hull": False,
    "carry_hints": False,  # last week's hints read this week's ties (with one pass a week there is no other use of them)
    "lot_bonus": 0.0,  # share of its chip's penalty a lot started is worth beyond what the forecast lets it sell
    # share of its chip's penalty a lot started in the window's first week is worth more than the same lot in its last
    # (falling evenly in between): between two plans the model cannot tell apart, the one whose fabs run sooner. A
    # plan carried from week to week otherwise keeps a grid's weeks of full work some weeks ahead for ever
    "lot_tilt": 0.0,
    "min_gain": 0.0,  # a new plan replaces the reference only if it is this share of the window's cost cheaper
    "pending": False,  # announced prohibitions planned as taking effect
    # a grid in a spell of low output: None (the forecast keeps it low), "blend" (the expected output by the spell's
    # age) or "step" (over from the week it more likely than not is): ``grid_recovery.json``, the rollout lab's table
    "grids": None,
    # a lab's test only: fields of the forecast replaced by the scenario's own, when the harness hands the scenario
    # over (``tell_truth``): names of ``TRUTH``'s groups ("grids", "edges", "straits", "prohibitions", "everything")
    "truth": [],
    # plan_lab, the week's time. ``hull_every``: weeks between solves with the hull (the weeks between keep the whole
    # weeks of the carried plan); ``hull_rough``: that solve without the crossover; ``anchor_every``: weeks between
    # rollouts of the rules alone when there is an ``anchor`` (their plan of the last rollout, moved on, is the anchor
    # in between; 1: every week)
    "hull_every": 1,
    "hull_phase": 0,  # the weeks of the solve with the hull: (week - 1) % hull_every == hull_phase
    "hull_rough": False,
    # the interior point's optimality tolerance in the solve with the hull, which then runs without the crossover too
    # (0: HiGHS's own, 1e-8): only the shares of whole weeks are read from that solution, and they are rounded
    "hull_tol": 0.0,
    # the solver of the exact cell of a large program (Full), where "auto" takes the interior point: "ipm", or "devex",
    # the dual simplex with devex weights from a cold start. A small program (Small) is not touched
    "big_exact": "ipm",
    # the week of the solve with the hull has no other solve: the carried plan stands that week, and the weeks the
    # hull asks to be whole are written into the cells of the weeks after it, until the next such solve
    # "tail": that week's plan is the solve with the hull itself, its first week exact and the later ones a mix
    "hull_only": False,
    "anchor_every": 1,
    # plan_lab's three levers over the hull. ``hull_rounds``: times a week the hull is solved, each from the plan the
    # last one led to. ``end_left``: fuel left at the window's end is worth no more than its grid can burn in the
    # weeks of the episode after the window. ``anchor_free`` "overflow": no price for leaving the rules' wafers at a
    # fab whose plant's store of its packaged chip was full in one of the last four weeks (those lots are thrown away)
    "hull_rounds": 1,
    "end_left": False,
    "anchor_free": None,
    "share": 0.45,  # share of the week's CPU budget the planner may reach, from the start of ``act`` (0: no clock)
    # the clock counts as the server's meter does: ``Agent(config)`` belongs to week 1, and what a week took over the
    # whole budget is taken off the next week's share (the meter charges it to that week, and a week over its budget
    # is played by the naive rule)
    "carry_debt": False,
    # the clock looks ahead and stops the solves where the work after them still fits in its ``share``, so a week is
    # not over the budget on a server of any speed. Before the week's solves it reckons, from this episode's own times
    # (``fit`` times the third quartile of the last eight weeks'), whether the solve with the hull, the exact one and
    # the work around them are likely to fit. If so, the week is the model's: the hull, then the exact cell with its
    # whole weeks. If not, the exact solve goes first, so that the week has a new plan, with the whole weeks the last
    # solve with the hull asked for (``pending``, moved on by a week); the hull is then solved from that plan in what
    # is left of the week, for the next week's exact solve, and it is the one the clock stops. A second round of the
    # hull (``hull_rounds``) starts only where both its solves are likely to fit. 0: no such clock
    "fit": 0.0,
    "exact_first": False,  # a lab's test: with ``fit``, the exact solve goes first every week, likely to fit or not
    # with ``fit``: a week that would have the exact solve first has the hull first instead when last week had no
    # solve with the hull at all (it did not fit after the exact one): its whole weeks then wait for the next week's
    # exact solve if the clock stops this week's. The hull is solved every second week at least
    "fit_turns": False,
    # with ``fit``, a large program (Full): when the clock stops the exact solve of a week that had it first, the
    # exact cells of the next weeks are solved by the interior point as it stops at ``rough_tol``, no crossover
    # (two weeks, then four after the next stop, up to sixteen; a vertex solve that ends resets the count). In an
    # episode whose vertex solve never fits the week the plan is otherwise never renewed and the rules play.
    # "turns": in those weeks the hull has no room after the exact solve, so every second of them has the hull first
    # (its whole weeks wait for the next week's exact solve, and the plan stands if the clock stops the exact one)
    "exact_rough": False,
    "rough_tol": 1e-4,
    # with ``fit`` and an ``anchor``: the rules alone are not rolled out in a week where the rollout and both solves
    # are not likely to fit; their plan of the last rollout, moved on, is the anchor (as between rollouts with
    # ``anchor_every``), for this many weeks in a row at most (0: rolled out always)
    "fit_rules": 0,
    # with ``fit_rules``: in such a week the rules are rolled out for the window's first so many weeks and the rest of
    # their plan is that of the last rollout, moved on and played from there (0: no rollout at all that week)
    "fit_fresh": 0,
    "part_always": False,  # a lab's test: the rollout is cut so every week ``fit_rules`` lets it, likely to fit or not
    # with ``fit``: the window's weeks from the week the clock finds that the week of the whole window (the rules'
    # rollout, both solves and the work around them) is not likely to fit, to the end of the episode; the hull still
    # asks whole weeks up to the same week of the window (``hull_until`` is taken that much shorter). A cell of 20
    # weeks has three quarters of the rows of one of 26 and solves in less. 0: the window stays as it is
    "fit_horizon": 0,
    # with ``fit``: the share of the episode's wall-clock seconds (``EPISODE_S``, counted here from ``Agent(config)``)
    # the weeks may take in all; a week's clock is no longer than an even part of what is left of them. On Full 104
    # weeks at the whole budget are 416 of the 480 seconds, so a server whose wall clock runs ahead of its CPU clock
    # would stop the episode before its last weeks (0: no such limit)
    "episode_share": 0.0,
    # a lab's test: the week's CPU budget as a multiple of the board's, to stand for a slower server under the
    # scorer's meter (0.54 with the meter at 0.54 of the budget: a server 1.85 times slower than this machine)
    "budget_scale": 1.0,
    "solve_seconds": 3.0,  # HiGHS's time limit for one cell, with or without the clock
    # a cell that does not solve at once, in shares of the week's CPU budget so that one rule serves both networks
    # (0: off). ``solve_share``: HiGHS's time limit for one cell, in place of ``solve_seconds``. ``warm_share``: the
    # simplex from last week's basis gives way to a cold solve after this long (it stalls now and then).
    # ``chain_share``: past this point of the week, counted from the start of ``act``, a cell that did not solve is
    # not written another way and solved again: the carried plan stands
    "solve_share": 0.0,
    "warm_share": 0.0,
    "chain_share": 0.0,
    # the primal feasibility tolerance of one more solve of a cell left "Infeasible" or "Unknown" (0: none): such
    # cells fail by a rounding, and without it the week keeps the carried plan
    "tol_retry": 0.0,
    "tol": 0.0,  # the primal feasibility tolerance of every solve instead (0: HiGHS's own, 1e-7): no second solve
    # once a cell of this episode needed ``tol_retry``, every later solve starts with that tolerance: such cells come
    # in runs of weeks, and a week then has no second solve. An episode without such a cell plays as without it
    "tol_sticky": False,
    # a lab's record: every week the solve with the hull ran, the shares of whole weeks it asked for by (week, grid)
    # and what a model in its place may know of them (``plan_core.share_features``), kept in ``detail``
    "record_shares": False,
    # after the week's first plan, up to so many other sets of whole weeks are tried next to the ones the rounding
    # gave (one more grid's week, a week earlier or later, none), each in an exact cell, and the cheapest plan as the
    # simulator plays it stands (``plan_core.descend``'s ``search``); with a clock, none is tried past its deadline
    "search": 0,
    # with a clock, the search starts only when this many tries fit before the week's deadline, each taken to cost
    # what the week's exact cell did (1: any room for one; 3 keeps it out of a network where a try is dear)
    "search_room": 1,
    # the last weeks of the window the solve with the hull leaves alone: it asks whole weeks only up to the window's
    # length minus this many weeks (12: weeks 1 to 14 of a window of 26)
    "hull_until": 12,
    # the rounding of the hull's shares writes a whole week when their sum is this share of a week short of its price
    # (0: only when the price is reached; 0.5: to the nearest whole week)
    "hull_lean": 0.0,
    # the window's instance is named by its own object and not by a hash of all its fields: the simulator and the
    # program only check that the marks, the state and the rows were made for the same instance, and the hash of a
    # window of Full takes 36 ms a week
    "window_name": False,
    # HiGHS's solver: "simplex", "ipm", or "auto": the simplex from last week's basis on a small program (Small),
    # the interior-point method on a large one (Full), where the simplex's time is anywhere between 0.6 and 18 s
    "method": "auto",
}
if (HERE / "regime.json").is_file():
    PARAMS |= json.loads((HERE / "regime.json").read_text())


TRUTH = {
    "grids": ("G_bar", "y_bar"),
    "edges": ("u",),
    "straits": ("o", "kappa", "wr_class", "h_queue", "c_wr", "c"),
    "prohibitions": ("prohibited",),
    "plants": ("R", "alpha_bar", "R_osat", "sigma_scr"),
    "supply": ("supply",),
    "demand": ("demand",),
    "tariffs": ("tariff",),
    "everything": ("prohibited", "u", "tariff", "o", "kappa", "wr_class", "h_queue", "c_wr", "c", "G_bar", "y_bar",
                   "R", "alpha_bar", "R_osat", "sigma_scr", "supply", "demand"),
}
NOW = {"u": "u_now", "o": "o_now", "kappa": "kappa_now", "supply": "supply_now", "G_bar": "G_bar_now",
       "y_bar": "y_bar_now", "R": "R_now", "alpha_bar": "alpha_now"}  # fmt: skip


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_hybrid = _load(f"{HERE.name}_hybrid", HERE / "hybrid_agent.py")  # its lp_part puts sbfv/ on the path
_model = _load(f"{HERE.name}_model", HERE / "sim_model.py")
_core = _load(f"{HERE.name}_core", HERE / "plan_core.py")
if PARAMS["window_name"] and not hasattr(_model.L.rolled_window, "named"):
    _rolled_window, _windows = _model.L.rolled_window, [0]

    def _named_window(inst, obs, weeks):
        inst_r, backlog = _rolled_window(inst, obs, weeks)
        _windows[0] += 1
        inst_r._memo["content_digest"] = f"window {_windows[0]}"
        return inst_r, backlog

    _named_window.named = True
    _model.L.rolled_window = _named_window
_core.PKG = "sbfv"
from sbfv.dynamics import sim  # noqa: E402 - sbfv/ sits beside this file


class Agent(_hybrid.Agent):
    def __init__(self, config=None):
        made, self.born = time.process_time(), time.monotonic()
        super().__init__(config)
        self.p = dict(PARAMS)
        self.made, self.debt = made, 0.0  # for ``carry_debt``
        self.loose = False  # for ``tol_sticky``: a cell of this episode needed the looser tolerance
        self.model = None
        if self.planner is not None:
            if self.p["grids"] in ("blend", "step"):
                table = json.loads((HERE / "grid_recovery.json").read_text())
                self.model = _model.Model(config, self.planner, table, self.p["grids"])
            else:
                self.model = _model.Model(config, self.planner)
        self.acts = None  # last week's plan from its second week on: (flows, overrides, holds) per week
        self.basis = None  # of the last cell solved
        self.hint = None  # of the last cell solved: which side of a tie the duals push to
        self.deadline = None
        self.truth = None  # the scenario's own marks, when a lab's harness hands them over
        self.log = []  # per week: CPU seconds, the model's cost of the reference and of the plan (bn USD), a note
        # ``hull`` "model": fitted trees tell the whole weeks in place of the solve with the hull (``share_model.npz``
        # beside this file, written by the lab's ``shares.py fit``)
        self.share_model = _core.share_model(HERE / "share_model.npz") if self.p["hull"] == "model" else None
        # per week, for a lab's look at the week's time: every run of the solver (``plan_core.Episode.solves``) and
        # the passes (the cell's claim in USD, the played cost in cents, CPU seconds, simplex iterations)
        self.detail = []
        self.week_solves, self.week_passes, self.week_shares = [], [], None
        kind = np.asarray(self.chips.kind)  # 0 fuel, 1 wafer, 2 raw chip, 3 packaged chip
        self.chip_slots = kind != 0
        if self.model is None:
            return
        inst = self.model.inst
        self.budget = BUDGET_S.get(inst.kind, 2.0) * float(self.p["budget_scale"])
        # for ``fit``: CPU seconds of the weeks so far in the solves of each kind, around them (from the clock's look
        # to the end of ``act``, the solves aside) and after the last of them; weeks since a solve with the hull; the
        # whole weeks it asked for that no exact solve has taken yet
        self.took = {"hull": [], "exact": [], "rough": [], "around": [], "after": [], "rules": [], "before": []}
        self.backoff, self.rough_left, self.rough = 1, 0, False  # for ``exact_rough``
        self.short = False  # for ``fit_horizon``: the window is the short one from now on
        self.looked, self.hull_age, self.pending, self.rounds, self.searched = None, 99, set(), 1, 0.0
        self.rules_age, self.rolled = 0, 0.0  # weeks since the rules alone were rolled out; CPU seconds of this week's
        self.fresh = 0  # for ``fit_fresh``: the weeks of this week's rollout that the rules play (0: all of them)
        self.seconds = float(self.p["share"]) * self.budget
        self.wall_end = self.born + float(self.p["episode_share"]) * EPISODE_S.get(inst.kind, 180.0)
        fed = {e.head for e in inst.edges}
        self.valves = np.array([  # a valve: a fuel slot of one edge from a terminal into a grid (it acts this week)
            lane is None and inst.nodes[inst.edges[e].head].grid is not None and inst.edges[e].tail in fed
            for e, _k, lane in inst.action_slots
        ]) & (kind == 0)  # fmt: skip
        self.orders = np.flatnonzero((kind == 0) & ~self.valves)
        names = {1: "wafer", 2: "raw", 3: "pack"}
        self.kinds = [names.get(int(kind[s]), "valve" if self.valves[s] else "order") for s in range(self.n_slots)]
        self.pair_index = {pair: i for i, pair in enumerate(self.model.pairs)}
        # what a unit of each commodity left in the system at the end of a capped window is worth, USD
        value = np.zeros(len(inst.commodities))
        for k in {k for g in inst.grids for k in inst.nodes[g].grid.fuels}:
            value[k] = float(self.p["end_fuel"])
        pi = np.zeros(len(inst.commodities))
        for d in inst.demands:
            pi[d.k] = max(pi[d.k], d.pi)
        for o in inst.osats:
            for raw, packed in inst.nodes[o].osat.packages.items():
                pi[raw] = max(pi[raw], pi[packed])
        self.end_k = value + float(self.p["end_chip"]) * pi
        supply = set(inst.supply_nodes)
        self.end_stock = np.array([0.0 if sl.node in supply else self.end_k[sl.k] for sl in inst.stock_slots])
        self.chip_worth = np.array([pi[inst.nodes[f].fab.product] for f in inst.fabs])  # a lot's chip, by fab

    def _wasted(self, observation) -> np.ndarray:
        """Wafer slots into fabs whose lots end as thrown-away chips: a plant that takes the fab's chips had its
        store of the packaged chip full (the only way chips are thrown away there) in one of the last four weeks."""
        inst = self.model.inst
        if not hasattr(self, "_plant_slots"):
            where = {(int(n), int(k)): i for i, (n, k) in enumerate(self.planner.stock)}
            packed = {raw: pk for o in inst.osats for raw, pk in inst.nodes[o].osat.packages.items()}
            self._plant_slots, self._fab_plants = [], {}
            for o in inst.osats:
                for pk in inst.nodes[o].osat.packages.values():
                    self._plant_slots.append((o, pk, where[(o, pk)], inst.stock_slots[inst.slot_index[(o, pk)]].storage))
            for f in inst.fabs:
                raw = inst.nodes[f].fab.product
                room = {}  # plant -> nominal weekly capacity of the lanes that carry this fab's chips there
                for e, k, lane in inst.action_slots:
                    if k == raw and inst.edges[e].tail == f:
                        head = inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head
                        room[head] = room.get(head, 0.0) + inst.edges[e].u0
                total = sum(room.values()) or 1.0
                self._fab_plants[f] = {(o, packed[raw]) for o, u in room.items() if u / total >= 0.2}
            self._wafer_fab = {}
            for s, (e, _k, lane) in enumerate(inst.action_slots):
                head = inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head
                if self.kinds[s] == "wafer" and inst.nodes[head].fab is not None:
                    self._wafer_fab[s] = head
            self._full = []  # the last four weeks: the (plant, chip) stores that were full
        stock = np.asarray(observation["stock.qty"], dtype=float)
        week = int(np.asarray(observation["week"]).ravel()[0])
        full = {(o, pk) for o, pk, i, cap in self._plant_slots if cap > 0 and stock[i] >= 0.98 * cap}
        if not self._full or self._full[-1][0] != week:
            self._full = (self._full + [(week, full)])[-4:]
        recent = set().union(*(f for _w, f in self._full))
        out = np.zeros(self.n_slots, dtype=bool)
        for s, f in self._wafer_fab.items():
            out[s] = bool(self._fab_plants[f] & recent)
        return out

    def tell_truth(self, truth) -> None:
        """A lab's harness hands the scenario over; used only for the fields ``PARAMS["truth"]`` names."""
        if self.p["truth"]:
            self.truth = truth["marks"]

    def _network(self, observation, week: int, H: int):
        """The window's network: None (as observed), or the forecast with the fields of ``truth`` the scenario's own."""
        if self.truth is None:
            return None
        arrays = {k: np.array(a) for k, a in self.model.forecast(observation, H, bool(self.p["pending"])).items()}
        for group in self.p["truth"]:
            for name in TRUTH[group]:
                arrays[name] = np.array(getattr(self.truth, name)[week - 1 : week - 1 + H])
                if name in NOW:
                    arrays[NOW[name]] = np.array(getattr(self.truth, NOW[name])[week - 1 : week - 1 + H])
        return arrays

    # ----- between the plan's weeks and the environment's arrays -------------------------------------------------------
    def _arrays(self, act) -> dict:
        """One week's (flows, overrides, holds) as the environment's flat arrays."""
        m = self.model
        fl, ov, ho = act
        flows = np.zeros(self.n_slots)
        for s, q in fl.items():
            flows[s] = q
        mode, qty = np.zeros(len(m.pairs), dtype=np.int64), np.zeros(len(m.ov_pair))
        for o, q in ov.items():
            mode[m.ov_pair[o]], qty[o] = OVERRIDE, q
        for pair in ho:
            mode[self.pair_index[pair]] = HOLD
        return {"flows": flows, "override_qty": qty, "release_mode": mode}

    def _with_orders(self, fuel, observation, plan: dict) -> dict:
        """A plan's week with the fuel rules' orders of that week (the rules read the plan's wafers)."""
        flows = np.zeros(self.n_slots)
        flows[self.chip_slots] = plan["flows"][self.chip_slots]
        fuel.fill(observation, flows)
        flows[self.valves] = plan["flows"][self.valves]
        return {"flows": flows, "override_qty": plan["override_qty"], "release_mode": plan["release_mode"]}

    def _roll(self, observation, w, ep, act_of, fresh: int = 0, rest: list | None = None) -> tuple:
        """The window played on the model, ``act_of(week, observation)`` the flat action of each week: the cost, the
        weeks as the simulator took them, the records. With ``rest`` (a plan of the window, one week each),
        ``act_of`` plays the first ``fresh`` weeks only and ``rest`` the weeks after them."""
        m = self.model
        seen, acts, recs = observation, [], []
        for h in range(w.inst.T):
            told = rest is not None and h >= fresh
            wire = rest[h] if told else m.wire(act_of(h, seen), np.asarray(w.marks.prohibited[h]))
            acts.append(wire)
            recs.append(sim.step(w.inst, w.marks, w.state, *wire))
            if h + 1 < w.inst.T and (rest is None or h + 1 < fresh):
                seen = m.flat(observation, w)
        return ep.cost(recs, w.state), acts, recs

    # ----- the week ---------------------------------------------------------------------------------------------------
    def _reckon(self, kind: str) -> float:
        """CPU seconds a week's work of a kind is likely to take: ``fit`` times the third quartile of the last eight
        weeks' (0 before the first)."""
        last = self.took[kind][-8:]
        return float(self.p["fit"]) * float(np.percentile(last, 75)) if last else 0.0

    def act(self, observation):
        t0 = time.process_time()
        if self.p["carry_debt"] and self.made is not None:
            t0, self.made = self.made, None
        week = int(np.asarray(observation["week"]).ravel()[0])
        every = int(self.p["anchor_every"]) if self.p["anchor"] else int(self.p["rules_every"])
        alone = self.acts is None or (every > 0 and (week - 1) % every == 0)
        self.rules_age, self.rolled, self.started, self.fresh = self.rules_age + 1, 0.0, t0, 0
        if (alone and self.acts is not None and self.p["fit"] > 0 and self.rules_age <= int(self.p["fit_rules"])
                and getattr(self, "ruled", None) and self.seconds > 0):
            need = sum(self._reckon(kind) for kind in ("before", "rules", "hull", "exact", "around"))
            # the whole rollout is not likely to fit beside both solves
            if self.seconds - self.debt < need or self.p["part_always"]:
                self.fresh = int(self.p["fit_fresh"])
                alone = self.fresh > 0
        if (self.p["fit"] > 0 and self.p["fit_horizon"] and not self.short and self.seconds > 0
                and len(self.took["exact"]) >= 3):
            need = sum(self._reckon(kind) for kind in ("before", "rules", "hull", "exact", "around"))
            if self.seconds - self.debt < need:  # the week of the whole window is not likely to fit
                self.short, self.hint = True, None
                self.took = {kind: [] for kind in self.took}  # the short window's times are its own
        fuel0 = copy.deepcopy(self.fuel)  # the rules' memory before this week
        rules0 = copy.deepcopy((self.chips, self.strait)) if alone else None
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # the rules' chip entries; keeps their memory
        plan, note = None, ("no model",)
        self.week_solves, self.week_passes, self.week_shares = [], [], None
        if self.model is not None:
            seconds = self.seconds
            if self.p["fit"] > 0 and self.p["episode_share"] > 0:  # an even part of the episode's wall clock at most
                seconds = min(seconds, (self.wall_end - time.monotonic()) / max(1, self.model.inst.T - week + 1))
            self.deadline = t0 + seconds - self.debt if self.seconds > 0 else None
            self.chain_deadline = None
            if self.p["chain_share"] > 0:
                self.chain_deadline = t0 + float(self.p["chain_share"]) * self.budget - self.debt
            try:
                self.planner._remember(observation)  # once a week: it counts the age of disruptions
                plan, note = self._plan(observation, week, fuel0, rules0, alone)
            except Exception as error:  # the hybrid's action stands
                self.acts, self.basis, self.hint = None, None, None
                note = ("error", repr(error)[:300])
        if plan is not None:
            planned = self._arrays(plan)
            flows[self.chip_slots] = planned["flows"][self.chip_slots]
        elif self.planner is not None:  # the hybrid's program for the chips after the fab, when no plan decides them
            remember, self.planner._remember = self.planner._remember, (lambda o: None)
            try:
                own = self._planned(observation)
            finally:
                self.planner._remember = remember
            if own is not None:
                flows[self.take] = own[self.take]
        self.fuel.fill(observation, flows)  # this week's orders (and the rules' valves); keeps the fuel rules' memory
        action = {"flows": flows, **self.strait.fill(observation)}
        if plan is not None:
            fuel_slots = ~self.chip_slots if self.p["orders"] == "plan" else self.valves
            flows[fuel_slots] = planned["flows"][fuel_slots]
            action["override_qty"], action["release_mode"] = planned["override_qty"], planned["release_mode"]
        if self.looked is not None:  # ``fit``: what the week's solves and the work around and after them took
            now = time.process_time()
            exact = ("exact", "plain", "no hint", "no tweak")  # the search over sets of whole weeks is not counted
            solves = {"hull": sum(s["cpu"] for s in self.week_solves if s["what"] == "hull"),
                      "exact": sum(s["cpu"] for s in self.week_solves if s["what"] in exact)}
            for kind, took in solves.items():  # a week of two rounds counts as two
                if took > 0:
                    self.took["rough" if kind == "exact" and self.rough else kind].append(took / self.rounds)
            self.took["before"].append(self.looked - self.started - self.rolled)
            if self.rolled > 0 and self.fresh == 0:  # a whole rollout
                self.took["rules"].append(self.rolled)
            if self.week_solves and plan is not None:  # a week without a plan has the hybrid's program after the solves
                around = now - self.looked - sum(solves.values()) - self.searched
                self.took["around"].append(around / self.rounds)
                self.took["after"].append(now - self.week_solves[-1]["end"])
            self.looked = None
        self.log.append((time.process_time() - t0, *note))
        self.detail.append({"solves": self.week_solves, "passes": self.week_passes, "shares": self.week_shares})
        if self.p["tol_sticky"] and any(s["attempt"] == "tolerance" and s["status"] == "Optimal" for s in self.week_solves):
            self.loose = True
        if self.p["carry_debt"] and self.model is not None:
            self.debt = max(0.0, self.debt + time.process_time() - t0 - self.budget)
        return action

    def _plan(self, observation, week: int, fuel0, rules0, alone: bool):
        """The first week of the best plan as (flows, overrides, holds), or None: the rules alone are the cheapest."""
        m, p = self.model, self.p
        left = m.inst.T - week + 1
        horizon = int(p["fit_horizon"]) if self.short else int(p["horizon"])
        H = left if not horizon else min(horizon, left)
        hull_until = max(0, int(p["hull_until"]) - (int(p["horizon"]) - horizon if self.short else 0))
        w = m.window(observation, H, self._network(observation, week, H), pending=bool(p["pending"]))
        capped = H < left and bool(p["end_fuel"] or p["end_chip"])
        own = p["orders"] == "plan"  # the plan's own orders are played, not the fuel rules'
        end = (self.end_stock, self.end_k, float(p["end_weeks"]), (left - H) if p["end_left"] else None)
        ep = _core.Episode(w.inst, w.marks, end=end if capped else None, anchored=bool(p["anchor"]))
        limit = float(p["solve_share"]) * self.budget if p["solve_share"] > 0 else float(p["solve_seconds"])
        ep.warm_limit = float(p["warm_share"]) * self.budget if p["warm_share"] > 0 else None
        ep.tol_retry = float(p["tol_retry"]) if p["tol_retry"] > 0 else None
        ep.tol = float(p["tol_retry"]) if self.loose else float(p["tol"]) if p["tol"] > 0 else None
        ep.rough_tol = float(p["rough_tol"])
        if p["fit"] > 0:  # no run of the solver starts with less than this left of its limit
            ep.least = max(0.04 * self.budget, 0.5 * min(self.took["hull"][-8:] + self.took["exact"][-8:], default=0.0))
        self.week_solves = ep.solves
        ref, name, ruled = None, "", None
        if self.acts is not None:
            carried = list(self.acts)
            while len(carried) < H:  # a capped window moved on by a week: its last week repeats the one before
                carried.append(carried[-1])
            carried = ep.clean(carried[:H])
            if own:
                recs, J = ep.simulate(carried)
                ref = (J, carried, recs)
            else:
                flat = [self._arrays(a) for a in carried]
                fuel = copy.deepcopy(fuel0)
                ref = self._roll(observation, m.restart(w), ep, lambda h, seen: self._with_orders(fuel, seen, flat[h]))
            name = "carried"
        if alone:
            (chips, strait), fuel = rules0, copy.deepcopy(fuel0)

            def by_rules(_h, seen):
                flows = np.zeros(self.n_slots)
                chips.fill_chip_flows(seen, flows)
                fuel.fill(seen, flows)
                return {"flows": flows, **strait.fill(seen)}

            rolled, rest = time.process_time(), None
            if self.fresh > 0:  # the rules play the first weeks, the last rollout's plan, moved on, the rest
                moved = self.ruled[1:] + [self.ruled[-1]]
                rest = ep.clean((moved + [moved[-1]] * H)[:H])
            cand = self._roll(observation, m.restart(w), ep, by_rules, fresh=self.fresh, rest=rest)
            self.rolled = time.process_time() - rolled
            if rest is None:
                self.rules_age = 0
            ruled = cand[1]
            self.ruled = list(ruled)
            if ref is None or cand[0] < ref[0]:
                ref, name = cand, "rules"
        elif p["anchor"] and getattr(self, "ruled", None):  # the rules' plan of an earlier week, moved on to this window
            self.ruled = self.ruled[1:] + [self.ruled[-1]]
            ruled = ep.clean((self.ruled + [self.ruled[-1]] * H)[:H])
        J_ref, acts_ref, recs_ref = ref

        tweak, price = None, None
        if p["anchor"] and ruled is not None:
            price = np.array([float(p["anchor"].get(kind, 0.0)) for kind in self.kinds])
            if p["anchor_free"] == "overflow":
                price[self._wasted(observation)] = 0.0
        if not own:  # the orders stay at the reference's executed flows, or not below them
            floor = p["orders"] == "floor"
            held = [(ep.col("x", t, *m.inst.action_slots[s]), float(rec.executed.get(int(s), 0.0)))
                    for t, rec in enumerate(recs_ref, start=1) for s in self.orders]

            def tweak(C):
                for j, q in held:
                    C.lb[j] = min(q, C.ub[j])
                    if not floor:
                        C.ub[j] = C.lb[j]

        # plan_lab: the week of the solve with the hull, and the whole weeks an earlier one asked for, moved on by a week.
        # Without a carried plan there is nothing to stand in for the week's plan, so that week solves the usual way
        phase = 1 if p["hull_only"] else int(p["hull_phase"])
        hull_week = bool(p["hull"]) and (week - 1) % max(1, int(p["hull_every"])) == phase % max(1, int(p["hull_every"]))
        if p["hull_only"] and name != "carried":
            hull_week = False
        marks = None
        if p["hull_only"]:
            self.marks = {(t - 1, gi) for (t, gi) in getattr(self, "marks", set()) if t > 1}
            marks = None if hull_week else self.marks
        bonus = None
        if p["lot_bonus"] > 0 or p["lot_tilt"] > 0:
            weeks = np.arange(H)
            bonus = (self.chip_worth, float(p["lot_bonus"]) * (weeks < H - 14) + float(p["lot_tilt"]) * (1.0 - weeks / H))
        fits = ""  # ``fit``: what the clock chose for this week
        deadline, fit = self.deadline, p["fit"] > 0 and self.deadline is not None
        if fit:
            self.looked = time.process_time()
            self.hull_age += 1
            self.pending = {(t - 1, gi) for (t, gi) in self.pending if t > 1}
            room, self.rounds = self.deadline - self.looked, 1
            both = self._reckon("hull") + self._reckon("exact") + self._reckon("around")
            self.rough = bool(p["exact_rough"]) and self.rough_left > 0 and ep.N > _core.BIG
            if self.rough:
                self.rough_left -= 1
            # the exact solve first, the hull after it
            after = hull_week and (room < both or p["exact_first"] or self.rough) and not p["hull_only"]
            if after and self.hull_age >= 2 and (p["fit_turns"] or (self.rough and p["exact_rough"] == "turns")):
                after = False  # the hull's turn to go first
            if after:
                hull_week, fits = False, " exact first" + (" rough" if self.rough else "")
                if self.pending:
                    marks, fits = self.pending, fits + f" pending {len(self.pending)}"
            # before the first week's solves are timed, the work after them is taken as half of the work before them
            deadline -= self._reckon("after") if self.took["after"] else 0.5 * (self.looked - self.started)
        d = _core.descend(
            ep, acts_ref, iters=int(p["passes"]), hints=bool(p["hints"]), tweak=tweak,
            played=(recs_ref, J_ref),
            basis=ep.shifted(self.basis) if p["warm"] and (p["method"] != "auto" or ep.N <= _core.BIG) else None,
            bonus=bonus, deadline=deadline,
            min_gain=max(1e6, p["min_gain"] * abs(J_ref)), anchor=ruled if price is not None else None, price=price,
            time_limit=limit, method=p["method"],
            hint=_core.moved(self.hint) if p["carry_hints"] and name == "carried" else None,
            close=float(p["close"]), close_until=H - hull_until, close_rationed=bool(p["close_rationed"]),
            hull=p["hull"] if hull_week else False, hull_rough=bool(p["hull_rough"]),
            hull_tol=float(p["hull_tol"]) if p["hull_tol"] > 0 else None,
            big_exact="rough" if fit and self.rough else str(p["big_exact"]),
            hull_only=p["hull_only"] if hull_week else False, marks=marks, chain_deadline=self.chain_deadline,
            record=bool(p["record_shares"]),
            model=(self.share_model, left - H) if p["hull"] == "model" else None,
            search=int(p["search"]) if hull_week else 0, hull_lean=float(p["hull_lean"]), search_room=int(p["search_room"]),
        )
        self.week_passes = list(d["hist"])
        self.searched = float(d.get("search_cpu", 0.0))  # CPU seconds of the search over sets of whole weeks
        solved = any(h[1] is not None for h in d["hist"])  # an exact solve ended and its plan was played
        if fit:
            if p["exact_rough"] and after and not self.rough and ep.N > _core.BIG:  # the vertex solve had the week first
                first = next((s for s in ep.solves if s["what"] == "exact"), None)
                if first is not None and first["status"] == "Time limit reached":
                    self.backoff = min(16, 2 * self.backoff)
                    self.rough_left = self.backoff
                elif solved:
                    self.backoff = 1
            if hull_week and d.get("hull") == "Optimal":
                self.hull_age, self.pending = 0, set() if solved else set(d.get("marks", ()))
                fits += "" if solved else " hull alone"
            elif solved:
                self.pending = set()
            # the hull from the week's plan, where the shortest of its last eight solves would fit: a solve the
            # clock stops costs the week nothing
            if after and deadline - time.process_time() >= min(self.took["hull"][-8:], default=0.0):
                d2 = _core.descend(
                    ep, d["acts"], iters=1, hints=bool(p["hints"]), tweak=tweak, played=(d["recs"], d["J"]),
                    bonus=bonus, deadline=deadline, min_gain=max(1e6, p["min_gain"] * abs(J_ref)),
                    anchor=ruled if price is not None else None, price=price, time_limit=limit, method=p["method"],
                    hint=d.get("hint"), close_until=H - hull_until, hull=p["hull"], hull_rough=bool(p["hull_rough"]),
                    hull_tol=float(p["hull_tol"]) if p["hull_tol"] > 0 else None, hull_only=True,
                    chain_deadline=self.chain_deadline, big_exact=str(p["big_exact"]),
                )
                if d2.get("hull") == "Optimal":
                    self.hull_age, self.pending = 0, set(d2.get("marks", ()))
                    fits += f" hull after {len(self.pending)}"
        if "shares" in d:  # the window's length, the episode's weeks past it, the shares, what stood before them
            self.week_shares = (H, left - H, d["shares"], d["share_features"], sorted(d.get("marks", ())))
        for _ in range(int(p["hull_rounds"]) - 1 if hull_week and not p["hull_only"] else 0):  # the hull again from its plan
            if fit and (not solved or deadline - time.process_time() < both):
                break
            self.rounds += 1
            d2 = _core.descend(
                ep, d["acts"], iters=int(p["passes"]), hints=bool(p["hints"]), tweak=tweak, played=(d["recs"], d["J"]),
                basis=d.get("basis") if p["warm"] and (p["method"] != "auto" or ep.N <= _core.BIG) else None,
                bonus=bonus, deadline=deadline, min_gain=max(1e6, p["min_gain"] * abs(J_ref)),
                anchor=ruled if price is not None else None, price=price, time_limit=limit, method=p["method"],
                hint=d.get("hint"), close_until=H - hull_until, hull=p["hull"], hull_rough=bool(p["hull_rough"]),
                hull_tol=float(p["hull_tol"]) if p["hull_tol"] > 0 else None, chain_deadline=self.chain_deadline,
                big_exact=str(p["big_exact"]),
            )
            self.week_passes += list(d2["hist"])
            if d2["J_compared"] >= d["J_compared"] - 1e6:
                break
            d2["J0"], d2["J0_compared"] = d["J0"], d["J0_compared"]  # the week's start stays the reference
            d = d2
        if p["hull_only"] and "marks" in d:
            self.marks = set(d["marks"])
        self.hint = d.get("hint")
        if p["switch"] and d["hist"] and (self.deadline is None or time.process_time() < self.deadline):
            d = _core.switch_step(ep, d, tries=int(p["switch"]), tweak=tweak, bonus=bonus, deadline=self.deadline,
                                  last_week=H - 12, anchor=ruled if price is not None else None, price=price,
                                  time_limit=limit, method=p["method"])
        self.basis = d.get("basis")
        self.last = (ep, d, tweak, bonus, ruled if price is not None else None, price, H)  # for a look from a lab script
        better = d["J_compared"] < d["J0_compared"] - p["min_gain"] * abs(J_ref)  # with the lots' bonus, if any
        # nothing beats the rules alone: no plan this week, and the hybrid's own program decides the chips after the fab.
        # Not so where the clock stopped the week's solve (``fit``): the rules' plan is played and carried, for a week
        # without a carried plan takes longer, and the next solve would be stopped again
        if not better and name == "rules" and (solved or not fit):
            self.acts = None
            return None, (J_ref / 1e11, J_ref / 1e11, "rules alone")
        best = d["acts"] if better else acts_ref
        self.acts = best[1:]
        note = name + (":new" if better else ":kept") + (f" whole {d.get('closed', 0)} of {d.get('rounded', d.get('closed', 0))} {d['hull']}" if "hull" in d else "") + (
            f" search {d['search'][0]} took {d['search'][1]}" if "search" in d else "") + fits
        return best[0], (J_ref / 1e11, min(d["J"], J_ref) / 1e11, note)
