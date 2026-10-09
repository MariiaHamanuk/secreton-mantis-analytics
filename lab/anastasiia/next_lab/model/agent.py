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
    # the price of moving a dated whole week (``plan_core._commit``): a (grid, week) cell that last week's plan made
    # whole is worth this share of what a whole week of that grid's fabs sells for, per unit of its fab ratio, inside
    # the hull cell only. 0: off, and the model is this week's ``anastasiia_plan_hull`` to the cent. The program still
    # moves a week whenever the window says it pays more than the price; what it stops doing is re-choosing the date
    # every week for free
    "commit": 0.0,
    "commit_weeks": 13.0,  # weeks over which the price falls to zero (the cell of this week keeps the full price)
    # the price of a late whole week (``plan_core._commit``, a tilt). ``commit`` prices keeping a date and was measured
    # to do nothing: the date of a whole week is not chosen by the objective at all, it is where ``_rounded``'s scan
    # first finds enough fuel banked, so it follows the fuel's arrival. This prices the arrival instead: inside the
    # hull cell every week the program may make whole is worth this share of a whole week of that grid's fabs, falling
    # to zero over ``commit_weeks``, so the program banks the scarce fuel sooner and ``_rounded`` closes a week sooner.
    # Not ``lot_tilt``, which paid for lots in the exact cell and added them everywhere (measured: -0.005 ... -0.013)
    "hull_tilt": 0.0,
    # of the hull cell's solution only ``x[jrho]`` is used, through ``_rounded``: about 98 numbers on Full out of
    # 50 101 columns. The marks weigh whose chips will sell and share the fuel between the grids, so the lots and all
    # of the fuel stay free; what a chip does after the fab does not move a mark, only prices it. A list of what the
    # hull cell holds at the reference's executed flow (the solver's presolve then takes those columns out): "pack"
    # (slots carrying a packaged chip) is the only one measured to keep the marks - "raw" costs nine marks of
    # twenty-one on Full and dismantles the lever. None or []: off, and the cell is ``anastasiia_plan_hull``'s
    "hull_fix": None,
    # columns above which ``hull_fix`` acts. Measured: on Small (16 186 columns) holding the packaged chip buys
    # no time at all - the hull solve is 0.05 s against 0.06 s, presolve takes those columns out anyway - and
    # costs 0.0037 of score (Small 111 x64, interval -0.0055..-0.0021, better in 0 % of resamples). On Full
    # (50 101) it takes 26 % off the hull cell, which is what lets the week hold two solves. None: ``BIG``
    "hull_fix_big": None,
    # plan_lab's three levers, measured +0.003 Small / +0.007 Full on the tuning sets and left out only for the time
    "end_left": False,      # fuel left at the window's end is worth no more than its grid can burn by the episode's end
    "anchor_free": None,    # "overflow": no price for leaving the rules' wafers at a fab whose plant throws chips away
    "hull_rounds": 1,       # times a week the hull is solved, each from the plan the last one led to
    # columns above which only one round is taken. A round costs a solve, and on Full the week already holds
    # two: with a second round the clock binds every week, and then the score depends on how fast the machine
    # is, which is exactly what must not happen on a board whose processor is unknown. Below the threshold the
    # week has room and the clock never binds. None: ``BIG``
    "hull_rounds_big": None,
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
    "share": 0.45,  # share of the week's CPU budget the planner may reach, from the start of ``act`` (0: no clock)
    "solve_seconds": 3.0,  # HiGHS's time limit for one cell, with or without the clock
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
_core.PKG = "sbfv"
from sbfv.dynamics import sim  # noqa: E402 - sbfv/ sits beside this file


class Agent(_hybrid.Agent):
    def __init__(self, config=None):
        super().__init__(config)
        self.p = dict(PARAMS)
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
        kind = np.asarray(self.chips.kind)  # 0 fuel, 1 wafer, 2 raw chip, 3 packaged chip
        self.chip_slots = kind != 0
        if self.model is None:
            return
        inst = self.model.inst
        self.seconds = float(self.p["share"]) * BUDGET_S.get(inst.kind, 2.0)
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
        # USD a whole week of each grid's fabs sells for, at their nominal capacity: the scale of the commitment price
        self.week_worth = np.array([
            sum(float(inst.nodes[inst.fabs[fi]].fab.cap0) * self.chip_worth[fi] for fi in inst.grid_fabs[gi])
            for gi in range(len(inst.grids))
        ])
        self.whole = None  # (week, grid) cells the last plan made whole, in that plan's weeks

    def tell_truth(self, truth) -> None:
        """A lab's harness hands the scenario over; used only for the fields ``PARAMS["truth"]`` names."""
        if self.p["truth"]:
            self.truth = truth["marks"]

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
            for fi, f in enumerate(inst.fabs):
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

    def _roll(self, observation, w, ep, act_of) -> tuple:
        """The window played on the model, ``act_of(week, observation)`` the flat action of each week: the cost, the
        weeks as the simulator took them, the records."""
        m = self.model
        seen, acts, recs = observation, [], []
        for h in range(w.inst.T):
            wire = m.wire(act_of(h, seen), np.asarray(w.marks.prohibited[h]))
            acts.append(wire)
            recs.append(sim.step(w.inst, w.marks, w.state, *wire))
            if h + 1 < w.inst.T:
                seen = m.flat(observation, w)
        return ep.cost(recs, w.state), acts, recs

    # ----- the week ---------------------------------------------------------------------------------------------------
    def act(self, observation):
        t0 = time.process_time()
        week = int(np.asarray(observation["week"]).ravel()[0])
        every = 1 if self.p["anchor"] else int(self.p["rules_every"])
        alone = self.acts is None or (every > 0 and (week - 1) % every == 0)
        fuel0 = copy.deepcopy(self.fuel)  # the rules' memory before this week
        rules0 = copy.deepcopy((self.chips, self.strait)) if alone else None
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # the rules' chip entries; keeps their memory
        plan, note = None, ("no model",)
        if self.model is not None:
            self.deadline = t0 + self.seconds if self.seconds > 0 else None
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
        self.log.append((time.process_time() - t0, *note))
        return action

    def _plan(self, observation, week: int, fuel0, rules0, alone: bool):
        """The first week of the best plan as (flows, overrides, holds), or None: the rules alone are the cheapest."""
        m, p = self.model, self.p
        left = m.inst.T - week + 1
        H = left if not p["horizon"] else min(int(p["horizon"]), left)
        w = m.window(observation, H, self._network(observation, week, H), pending=bool(p["pending"]))
        capped = H < left and bool(p["end_fuel"] or p["end_chip"])
        own = p["orders"] == "plan"  # the plan's own orders are played, not the fuel rules'
        ep = _core.Episode(
            w.inst, w.marks,
            end=(self.end_stock, self.end_k, float(p["end_weeks"]), (left - H) if p["end_left"] else None) if capped else None,
            anchored=bool(p["anchor"]))
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
        if alone and (self.deadline is None or time.process_time() < self.deadline - 0.5 * self.seconds):
            (chips, strait), fuel = rules0, copy.deepcopy(fuel0)

            def by_rules(_h, seen):
                flows = np.zeros(self.n_slots)
                chips.fill_chip_flows(seen, flows)
                fuel.fill(seen, flows)
                return {"flows": flows, **strait.fill(seen)}

            cand = self._roll(observation, m.restart(w), ep, by_rules)
            ruled = cand[1]
            if ref is None or cand[0] < ref[0]:
                ref, name = cand, "rules"
        J_ref, acts_ref, recs_ref = ref

        tweak, price = None, None
        if p["anchor"] and ruled is not None:
            price = np.array([float(p["anchor"].get(kind, 0.0)) for kind in self.kinds])
            if p["anchor_free"] == "overflow":  # a wafer whose lots end as thrown-away chips costs nothing to drop
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

        hull_tweak = None
        fix = p["hull_fix"] or ()
        fix = [fix] if isinstance(fix, str) else list(fix)
        big = _core.BIG if p["hull_fix_big"] is None else int(p["hull_fix_big"])
        if fix and ep.N > big:  # a small program gains no time and loses score: see ``hull_fix_big``
            kinds = {k for k in ("pack", "raw", "wafer") if k in fix}
            held_hull = [(ep.col("x", t, *m.inst.action_slots[s]), float(rec.executed.get(int(s), 0.0)))
                         for t, rec in enumerate(recs_ref, start=1)
                         for s in range(self.n_slots) if self.kinds[s] in kinds]

            def hull_tweak(C):
                for j, q in held_hull:
                    C.lb[j] = C.ub[j] = min(q, C.ub[j])

        bonus = None
        if p["lot_bonus"] > 0 or p["lot_tilt"] > 0:
            weeks = np.arange(H)
            bonus = (self.chip_worth, float(p["lot_bonus"]) * (weeks < H - 14) + float(p["lot_tilt"]) * (1.0 - weeks / H))
        commit = None
        if p["hull_tilt"] > 0:  # every week the program may make whole: the earlier, the dearer
            commit = (None, float(p["hull_tilt"]) * self.week_worth, float(p["commit_weeks"]))
        elif p["commit"] > 0 and self.whole:  # the dated whole weeks of last week's plan, moved into this window
            span = max(1.0, float(p["commit_weeks"]))
            kept = {(t - 1, gi): 1.0 - (t - 2) / span for (t, gi) in self.whole if 1 < t <= H + 1}
            commit = ({key: w for key, w in kept.items() if w > 0.0},
                      float(p["commit"]) * self.week_worth, 0.0)
        d = _core.descend(
            ep, acts_ref, iters=int(p["passes"]), hints=bool(p["hints"]), tweak=tweak, played=(recs_ref, J_ref),
            commit=commit, hull_tweak=hull_tweak,
            basis=ep.shifted(self.basis) if p["warm"] and (p["method"] != "auto" or ep.N <= _core.BIG) else None,
            bonus=bonus, deadline=self.deadline,
            min_gain=max(1e6, p["min_gain"] * abs(J_ref)), anchor=ruled if price is not None else None, price=price,
            time_limit=float(p["solve_seconds"]), method=p["method"],
            hint=_core.moved(self.hint) if p["carry_hints"] and name == "carried" else None,
            close=float(p["close"]), close_until=H - 12, close_rationed=bool(p["close_rationed"]), hull=p["hull"],
        )
        rounds_big = _core.BIG if p["hull_rounds_big"] is None else int(p["hull_rounds_big"])
        rounds = int(p["hull_rounds"]) if ep.N <= rounds_big else 1
        for _ in range(rounds - 1):  # the hull again from the plan it led to
            if self.deadline is not None and time.process_time() >= self.deadline:
                break
            d2 = _core.descend(
                ep, d["acts"], iters=int(p["passes"]), hints=bool(p["hints"]), tweak=tweak, played=(d["recs"], d["J"]),
                basis=d.get("basis") if p["warm"] and (p["method"] != "auto" or ep.N <= _core.BIG) else None,
                bonus=bonus, deadline=self.deadline, min_gain=max(1e6, p["min_gain"] * abs(J_ref)),
                anchor=ruled if price is not None else None, price=price, time_limit=float(p["solve_seconds"]),
                method=p["method"], hint=d.get("hint"), close_until=H - 12, hull=p["hull"],
                commit=commit, hull_tweak=hull_tweak,
            )
            if d2["J_compared"] >= d["J_compared"] - 1e6:
                break
            d2["J0"], d2["J0_compared"] = d["J0"], d["J0_compared"]  # the week's start stays the reference
            d = d2
        self.hint = d.get("hint")
        if p["commit"] > 0:
            self.whole = d.get("whole") or None  # a week with no hull cell leaves the schedule to the next one
        if p["switch"] and d["hist"] and (self.deadline is None or time.process_time() < self.deadline):
            d = _core.switch_step(ep, d, tries=int(p["switch"]), tweak=tweak, bonus=bonus, deadline=self.deadline,
                                  last_week=H - 12, anchor=ruled if price is not None else None, price=price,
                                  time_limit=float(p["solve_seconds"]), method=p["method"])
        self.basis = d.get("basis")
        self.last = (ep, d, tweak, bonus, ruled if price is not None else None, price, H)  # for a look from a lab script
        better = d["J_compared"] < d["J0_compared"] - p["min_gain"] * abs(J_ref)  # with the lots' bonus, if any
        if not better and name == "rules":  # nothing beats the rules alone: no plan this week
            self.acts = None
            return None, (J_ref / 1e11, J_ref / 1e11, "rules alone")
        best = d["acts"] if better else acts_ref
        self.acts = best[1:]
        note = name + (":new" if better else ":kept") + (f" whole {d.get('closed', 0)} of {d.get('rounded', d.get('closed', 0))} {d['hull']}" if "hull" in d else "")
        return best[0], (J_ref / 1e11, min(d["J"], J_ref) / 1e11, note)
