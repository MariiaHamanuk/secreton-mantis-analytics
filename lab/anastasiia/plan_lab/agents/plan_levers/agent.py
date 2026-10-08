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
    # plan_lab, lever 3. ``end_left``: fuel left at the window's end is worth no more than its grid can burn in the
    # weeks of the episode after the window. ``anchor_free`` "overflow": no price for leaving the rules' wafers at a
    # fab whose plant's store of its packaged chip was full in one of the last four weeks (those lots are thrown away)
    "end_left": False,
    "anchor_free": None,
    "hull_rounds": 1,  # lever 2: times a week the hull is solved, each from the plan the last one led to
    # plan_lab, lever 1: other forecasts of the window beside "as observed". Each is a dict: "fields" (window fields
    # whose elements off their calm value come back to it from week "after" on), "pending" (announced prohibitions
    # take effect), "weight" (its share; "as observed" has the rest). A plan is made under every forecast, each plan
    # is played open loop under every forecast, and the plan with the lowest weighted cost is the week's
    "scenarios": [],
    # how the forecasts are used. "pick": as above. "joint": the plan as observed is made first, then one program
    # for all the forecasts that share the first week's flows (``plan_core.joint``) is solved from it; its plan is
    # the week's when the weighted cost of the forecasts, each played on the model, is lower
    "scenario_mode": "pick",
    "truth": [],
    # plan_lab, which part of that future is handed over. "all": the fields as the scenario has them; "ends": only for
    # the elements that are off their calm value this week (when a running disruption ends); "onsets": only for the
    # elements at their calm value (when a new one starts). The rest stays as observed
    "truth_mode": "all",
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
# the planner's calm value of a window field (``lp_part``'s memory at the start), for ``truth_mode``
CALM = {"u": "u", "o": "open", "kappa": "kappa", "G_bar": "grid_G", "y_bar": "grid_y", "supply": "supply", "R": "fab_R",
        "alpha_bar": "fab_alpha", "R_osat": "osat_R", "prohibited": "prohibited", "tariff": "tariff", "c": "c"}  # fmt: skip
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

    def _ages(self, week: int) -> None:
        """plan_lab: how long every cut edge has been as it is now, how deep the cut is, and whether its start was
        seen (a cut that was there in week 1 has no known age)."""
        v, calm = self.planner.memory.values, self.planner.calm
        if not hasattr(self, "cut_age"):
            n = len(calm["u"])
            self.cut_age, self.cut_seen, self.cut_left = np.zeros(n, dtype=int), np.zeros(n, dtype=bool), np.ones(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            left = np.where(np.isfinite(calm["u"]) & (calm["u"] > 0), v["u"] / calm["u"], 1.0)
        cut = left < 0.999
        same = cut & (np.abs(left - self.cut_left) < 1e-3)
        self.cut_seen = np.where(same, self.cut_seen, cut & (week > 1))
        self.cut_age = np.where(same, self.cut_age + 1, cut.astype(int))
        self.cut_left = left

    def _typed_back(self, H: int, share: float, other) -> np.ndarray:
        """plan_lab: the week of the window from which each cut edge is back at its calm capacity, by what cut it
        (read from the cut's depth, as ``regime_lab``'s ``typed``: a sanction's side effect leaves a quarter, a port's
        stoppage 0.07, a slowdown 0.4 to 0.8) and by its age: the week by which a cut of that kind and age is over
        with chance ``share`` (the generator's laws of duration). ``H`` where it is not back in the window; a cut of
        another kind or one that was there in week 1 is back from week ``other`` (None: it stays)."""
        from scipy.special import ndtr

        laws = {"sanction": lambda w: 1.0 - ndtr((np.log(7.0 * np.maximum(w, 1e-9)) - 4.8125) / 2.0),
                "stoppage": lambda w: 1.0 - ndtr((np.log(np.maximum(w, 1e-9)) - 0.48) / 0.67),
                "slowdown": lambda w: 1.0 - ndtr((np.log(np.maximum(w, 1e-9)) - 1.93) / 0.77)}
        back = np.full(len(self.cut_left), H, dtype=int)
        h = np.arange(1, H + 1, dtype=float)
        for e in np.flatnonzero(self.cut_left < 0.999):
            left = float(self.cut_left[e])
            kind = ("sanction" if abs(left - 0.25) < 0.005 else "stoppage" if abs(left - 0.07) < 0.005
                    else "slowdown" if 0.405 < left < 0.805 else None)
            if kind is None or not self.cut_seen[e]:
                if other is not None:
                    back[e] = int(other)
                continue
            S, age = laws[kind], float(self.cut_age[e])
            over = 1.0 - S(age + h) / max(float(S(age)), 1e-9)  # over by week h of the window
            hit = np.flatnonzero(over >= share)
            if len(hit):
                back[e] = int(hit[0]) + 1
        return back

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
        scen = getattr(self, "_scenario", None)
        if self.truth is None and scen is None:
            return None
        pending = bool(self.p["pending"]) or bool(scen and scen.get("pending"))
        arrays = {k: np.array(a) for k, a in self.model.forecast(observation, H, pending).items()}
        mode = self.p["truth_mode"]
        calm = self.planner.calm
        if scen is not None and scen.get("typed") is not None:  # each cut edge back by its kind and age
            back = self._typed_back(H, float(scen["typed"]), scen.get("other"))
            still = np.asarray(calm["u"])
            rows = np.arange(H)[:, None] >= back[None, :]
            arrays["u"] = np.where(rows, still[None], arrays["u"])
        if scen is not None and scen.get("young") is not None:  # young cuts off the quarter level end soon
            fresh = (self.cut_age > 0) & (self.cut_age <= int(scen["young"])) & (np.abs(self.cut_left - 0.25) > 0.005)
            still = np.asarray(calm["u"])
            rows = (np.arange(H)[:, None] >= int(scen.get("after", 3))) & fresh[None, :]
            arrays["u"] = np.where(rows, still[None], arrays["u"])
        if scen is not None:  # the running disruptions of the named fields are over from week ``after`` on
            after = int(scen.get("after", 4))
            for name in scen.get("fields", ()):
                still = np.asarray(calm[CALM[name]])
                off = ~np.isclose(arrays[name][0], still, rtol=1e-3, atol=1e-9)
                if after < H and off.any():
                    arrays[name][after:] = np.where(off[None], still[None], arrays[name][after:])
            arrays = _model.L.with_now(arrays)
        if self.truth is None:
            return arrays
        for group in self.p["truth"]:
            for name in TRUTH[group]:
                real = np.array(getattr(self.truth, name)[week - 1 : week - 1 + H])
                real_now = np.array(getattr(self.truth, NOW[name])[week - 1 : week - 1 + H]) if name in NOW else None
                if mode != "all":
                    if name not in CALM:  # no calm value to tell a running disruption by: left as observed
                        continue
                    off = ~np.isclose(arrays[name][0], calm[CALM[name]], rtol=1e-3, atol=1e-9)  # disrupted this week
                    take = off if mode == "ends" else ~off
                    real = np.where(take[None], real, arrays[name])
                    if real_now is not None:
                        real_now = np.where(take[None], real_now, arrays[NOW[name]])
                arrays[name] = real
                if real_now is not None:
                    arrays[NOW[name]] = real_now
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
                self._ages(week)
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
        """The week's plan; with ``scenarios``, the plan of the forecast whose plan is cheapest over all of them."""
        specs = list(self.p["scenarios"])
        if not specs:
            return self._plan_one(observation, week, fuel0, rules0, alone)
        if self.p["scenario_mode"] == "joint":
            return self._plan_joint(observation, week, fuel0, rules0, alone, specs)
        acts0, basis0, hint0 = self.acts, self.basis, self.hint
        plans, keep, first_note = [], None, None
        for si, spec in enumerate([None] + specs):
            self.acts, self._scenario = acts0, spec
            self.basis, self.hint = (basis0, hint0) if si == 0 else (None, None)
            rules = copy.deepcopy(rules0) if rules0 is not None else None
            first, note = self._plan_one(observation, week, copy.deepcopy(fuel0), rules, alone)
            if si == 0:
                keep, first_note = (self.basis, self.hint), note
                if first is None:  # nothing beats the rules alone as observed: no plan this week
                    self._scenario = None
                    return None, note
            if first is not None:
                plans.append((si, [first] + list(self.acts), self.last[0]))
        self._scenario = None
        weights = np.array([1.0 - sum(float(s.get("weight", 0.0)) for s in specs)] + [float(s.get("weight", 0.0)) for s in specs])
        eps = {si: ep for si, _plan, ep in plans}
        cost = np.zeros(len(plans))
        for j, (_sj, plan, _ep) in enumerate(plans):
            for si, ep in eps.items():
                cost[j] += weights[si] * ep.simulate(ep.clean(list(plan)))[1]
        j = int(np.argmin(cost))
        self.count = getattr(self, "count", {})
        self.count[plans[j][0]] = self.count.get(plans[j][0], 0) + 1
        self.notes = {"scenario": dict(self.count)}  # weeks each forecast's plan was the week's (0: as observed)
        self.acts = plans[j][1][1:]
        self.basis, self.hint = keep
        return plans[j][1][0], (*first_note[:2], f"{first_note[2]} scenario {plans[j][0]} of {len(plans)}")

    def _plan_joint(self, observation, week: int, fuel0, rules0, alone: bool, specs: list):
        """The plan as observed, then one program over all the forecasts with the first week shared."""
        m, p = self.model, self.p
        self._scenario = None
        first, note = self._plan_one(observation, week, fuel0, rules0, alone)
        if first is None:
            return None, note
        ep0, _d, tweak, bonus, anchor, price, H = self.last
        plan = [first] + list(self.acts)
        left = m.inst.T - week + 1
        capped = H < left and bool(p["end_fuel"] or p["end_chip"])
        eps = [ep0]
        for spec in specs:
            self._scenario = spec
            w = m.window(observation, H, self._network(observation, week, H), pending=bool(p["pending"]))
            eps.append(_core.Episode(
                w.inst, w.marks,
                end=(self.end_stock, self.end_k, float(p["end_weeks"]), (left - H) if p["end_left"] else None) if capped else None,
                anchored=bool(p["anchor"])))
        self._scenario = None
        weights = [1.0 - sum(float(s.get("weight", 0.0)) for s in specs)] + [float(s.get("weight", 0.0)) for s in specs]
        played = [ep.simulate(ep.clean(list(plan))) for ep in eps]  # (records, cost) of the plan under each forecast
        cells = []
        for ep, (recs, _J) in zip(eps, played):
            mode, ref = ep.regimes(recs)
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None and ep is ep0:
                tweak(C)
            cells.append(C)
        self.count = getattr(self, "count", {"joint": 0, "kept": 0, "failed": 0})
        status, xs = _core.joint(eps, cells, weights, method=p["method"], time_limit=float(p["solve_seconds"]))
        if status != "Optimal":
            self.count["failed"] += 1
            self.notes = {"joint": dict(self.count)}
            return first, note
        acts = [ep.actions(x) for ep, x in zip(eps, xs)]
        again = [ep.simulate(a) for ep, a in zip(eps, acts)]
        before = sum(wt * J for wt, (_r, J) in zip(weights, played))
        after = sum(wt * J for wt, (_r, J) in zip(weights, again))
        if after < before - 1e6:
            self.count["joint"] += 1
            self.count["gain_bn"] = self.count.get("gain_bn", 0.0) + (before - after) / 1e11
            self.notes = {"joint": dict(self.count)}
            self.acts = acts[0][1:]
            return acts[0][0], (*note[:2], f"{note[2]} joint")
        self.count["kept"] += 1
        self.notes = {"joint": dict(self.count)}
        return first, note

    def _plan_one(self, observation, week: int, fuel0, rules0, alone: bool):
        """The first week of the best plan as (flows, overrides, holds), or None: the rules alone are the cheapest."""
        m, p = self.model, self.p
        left = m.inst.T - week + 1
        H = left if not p["horizon"] else min(int(p["horizon"]), left)
        w = m.window(observation, H, self._network(observation, week, H), pending=bool(p["pending"]))
        capped = H < left and bool(p["end_fuel"] or p["end_chip"])
        own = p["orders"] == "plan"  # the plan's own orders are played, not the fuel rules'
        ep = _core.Episode(w.inst, w.marks,
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
        if alone:
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

        bonus = None
        if p["lot_bonus"] > 0 or p["lot_tilt"] > 0:
            weeks = np.arange(H)
            bonus = (self.chip_worth, float(p["lot_bonus"]) * (weeks < H - 14) + float(p["lot_tilt"]) * (1.0 - weeks / H))
        d = _core.descend(
            ep, acts_ref, iters=int(p["passes"]), hints=bool(p["hints"]), tweak=tweak, played=(recs_ref, J_ref),
            basis=ep.shifted(self.basis) if p["warm"] and (p["method"] != "auto" or ep.N <= _core.BIG) else None,
            bonus=bonus, deadline=self.deadline,
            min_gain=max(1e6, p["min_gain"] * abs(J_ref)), anchor=ruled if price is not None else None, price=price,
            time_limit=float(p["solve_seconds"]), method=p["method"],
            hint=_core.moved(self.hint) if p["carry_hints"] and name == "carried" else None,
            close=float(p["close"]), close_until=H - 12, close_rationed=bool(p["close_rationed"]), hull=p["hull"],
        )
        for _ in range(int(p["hull_rounds"]) - 1):  # plan_lab, lever 2: the hull again from the plan it led to
            d2 = _core.descend(
                ep, d["acts"], iters=int(p["passes"]), hints=bool(p["hints"]), tweak=tweak, played=(d["recs"], d["J"]),
                basis=d.get("basis") if p["warm"] and (p["method"] != "auto" or ep.N <= _core.BIG) else None,
                bonus=bonus, deadline=self.deadline, min_gain=max(1e6, p["min_gain"] * abs(J_ref)),
                anchor=ruled if price is not None else None, price=price, time_limit=float(p["solve_seconds"]),
                method=p["method"], hint=d.get("hint"), close_until=H - 12, hull=p["hull"],
            )
            if d2["J_compared"] >= d["J_compared"] - 1e6:
                break
            d2["J0"], d2["J0_compared"] = d["J0"], d["J0_compared"]  # the week's start stays the reference
            d = d2
        self.hint = d.get("hint")
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
