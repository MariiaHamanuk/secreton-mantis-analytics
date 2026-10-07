"""Rules order the fuel; a plan carried from week to week decides the valves, the tanker releases and the chips.

``anastasiia_hybrid_hub`` (its files are here unchanged: ``chip_part``, ``fuel_part``, ``strait_part``, ``lp_part``,
``sbfv/``) with a weekly planner on top (``sim_model``: the simulator inside the agent, from
``lab/anastasiia/rollout_lab``; ``regime_core``: the simulator's rules as rows of a linear program, from
``lab/anastasiia/plan_lab``).

The package's plan leaves free what the simulator does by itself (a grid serves its base load first, a fuel segment
burns what it has, a fab starts the lots it has wafers and power for), so the simulator does not execute it. Those
rules are linear once a few yes/no questions per week are answered (does the grid shed; is a fuel at its cap or out;
has the fab wafers for its capacity). Here the answers are read off a trajectory played on the agent's own copy of
the simulator, and the plan is the cheapest set of flows under the same answers: the simulator executes it as
written, and it is never worse than the trajectory the answers came from.

What the plan decides: the valves from terminals into grids, the tanker releases at the straits, wafers, raw and
packaged chips: entries whose effect follows from cargo already on the way. What it does not: the fuel orders from
the sources. On the network "as observed" a plan that orders the fuel too is no better than the rules (it sees no
disruption ahead, so every buffer looks idle); with the orders left to the rules it is.

Every week (``Agent._plan``):

1. the simulator's state is rebuilt from the observation; the network of the next ``horizon`` weeks is the observed
   one, unchanged;
2. the reference trajectory: last week's plan from its second week on, played on the model with the fuel rules
   ordering each week. Without a plan, and every ``rules_every`` weeks beside it, the rules alone play the window; the
   cheaper trajectory is the reference. Carrying the plan matters: the rules' answers change with every state a plan
   puts them in, and a plan re-made under new answers each week never collects what the one before was built for;
3. the plan is solved under the reference's answers with the orders held at the reference's, played open loop on the
   model, and kept if cheaper. A window that ends before the episode credits the fuel and the chips left in the
   system (``end_fuel``, ``end_chip``): without it a 16-week window stops starting lots;
4. the first week of the best plan is sent with this week's orders of the fuel rules.

The planner has a clock (``share`` of the week's CPU budget, counted from the start of ``act``). When it runs out, or
anything fails, the week is the hybrid's: the rules and its own linear program for the chips after the fab, which
runs only then. Every size comes from ``config``; a ``params.json`` beside this file replaces entries of ``PARAMS``.
"""

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
KIND = {"wafer": 1, "raw": 2, "pack": 3}  # chip_part's slot kinds
BUDGET_S = {"small": 2.0, "full": 4.0}  # the boards' CPU seconds per week, by the instance's kind

PARAMS = {
    "take": ["raw", "pack"],  # the entries the hybrid's own program decides in a week without a plan
    "lp": {},  # replaces entries of lp_part.LP
    "plan": True,  # False: the hybrid alone
    "horizon": 26,  # weeks planned (fewer at the end of the episode)
    "rules_every": 4,  # weeks between rollouts of the rules alone beside the carried plan (0: only without a plan)
    "end_fuel": 3.3e6,  # USD a unit of fuel left in the system at the window's end is worth
    "end_chip": 0.6,  # share of its penalty a chip left in the chain at the window's end is worth
    "min_gain": 0.0,  # a new plan replaces the reference only if it is this share of the window's cost cheaper
    "share": 0.45,  # share of the week's CPU budget the planner may reach, from the start of ``act``
}
if (HERE / "params.json").is_file():
    PARAMS |= {k: v for k, v in json.loads((HERE / "params.json").read_text()).items() if k in PARAMS}


def _part(name: str):
    """A module beside this file, under a name of its own (two agents may each ship a ``fuel_part``)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


_chip, _fuel, _strait, _lp = _part("chip_part"), _part("fuel_part"), _part("strait_part"), _part("lp_part")
_model, _regime = _part("sim_model"), _part("regime_core")  # after lp_part, which puts sbfv/ on the path

from sbfv.dynamics import sim  # noqa: E402
from sbfv.dynamics.state import cents  # noqa: E402
from sbfv.marks import osat_throughput  # noqa: E402
from sbfv.oracle.lp import build_lp  # noqa: E402


class _OutOfTime(Exception):
    """The planner's share of the week is spent."""


class Agent:
    def __init__(self, config=None):
        self.chips = _chip.Agent(config)
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL)
        self.strait = _strait.StraitRules(config)
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]
        self.take = np.flatnonzero(np.isin(self.chips.kind, [KIND[name] for name in PARAMS["take"]]))
        try:
            self.planner = _lp.Planner(config, PARAMS["lp"])
        except Exception:  # the rules alone are a complete agent
            self.planner = None
        self.planned_weeks = 0  # weeks whose entries came from a plan (diagnostics)
        self.model = None
        self.prev_x = None  # last week's plan from its second week on, in the window's columns
        self.tables = None
        self.nc = 0  # columns of one week of a plan
        self.end_on = False
        self.deadline = 0.0
        self.notes = {"planned": 0, "carried": 0, "rules": 0, "failed": 0, "late": 0}
        if PARAMS["plan"] and self.planner is not None:
            try:
                self._setup(config)
            except Exception:
                self.model = None

    def _setup(self, config) -> None:
        self.model = _model.Model(config, self.planner)
        inst = self.model.inst
        self.seconds = PARAMS["share"] * BUDGET_S.get(inst.kind, 2.0)
        kind = np.asarray(self.chips.kind)  # 0 fuel, 1 wafer, 2 raw chip, 3 packaged chip
        self.chip_slots = kind != 0
        fed = {e.head for e in inst.edges}
        self.valves = np.array([  # a fuel slot of one edge from a terminal into a grid: it acts this week
            lane is None and inst.nodes[inst.edges[e].head].grid is not None and inst.edges[e].tail in fed
            for e, _k, lane in inst.action_slots
        ]) & (kind == 0)
        self.orders = np.flatnonzero((kind == 0) & ~self.valves)
        self.pair_index = {pair: i for i, pair in enumerate(self.model.pairs)}
        # what a unit of each commodity left in the system at the end of a window is worth, USD
        value = np.zeros(len(inst.commodities))
        for k in {k for g in inst.grids for k in inst.nodes[g].grid.fuels}:
            value[k] = float(PARAMS["end_fuel"])
        pi = np.zeros(len(inst.commodities))
        for d in inst.demands:
            pi[d.k] = max(pi[d.k], d.pi)
        for o in inst.osats:
            for raw, packed in inst.nodes[o].osat.packages.items():
                pi[raw] = max(pi[raw], pi[packed])
        self.end_k = value + float(PARAMS["end_chip"]) * pi
        supply = set(inst.supply_nodes)
        self.end_stock = np.array([0.0 if sl.node in supply else self.end_k[sl.k] for sl in inst.stock_slots])

    # ----- the hybrid's week (no plan) --------------------------------------------------------------------------------
    def _planned(self, observation) -> np.ndarray | None:
        """The hybrid's program's flows for this week, or None: it has no plan, or it failed."""
        if self.planner is None:
            return None
        try:
            return self.planner.flows(observation)
        except Exception:
            return None

    # ----- the model's pieces -----------------------------------------------------------------------------------------
    def _clock(self) -> None:
        if time.process_time() > self.deadline:
            raise _OutOfTime

    def _end(self, w) -> float:
        """USD credited for what the window's last state holds beyond its salvage (the window is not the episode)."""
        if not self.end_on:
            return 0.0
        inst, st = self.model.inst, w.state
        total = float(np.dot(self.end_stock, np.asarray(st.stock, dtype=float)))
        total += sum(self.end_k[s.k] * s.qty for s in st.pipeline)
        for fi, book in st.fab_wip.items():
            total += self.end_k[inst.nodes[inst.fabs[fi]].fab.product] * sum(book.values())
        for book in st.osat_wip.values():
            for lots in book.values():
                total += sum(self.end_k[k] * q for k, q in lots.items())
        return total

    def _cost(self, w, records) -> int:
        return sum(r.cost_cents for r in records) - cents(sim.terminal_salvage(w.inst, w.state) + self._end(w))

    def _end_values(self, lp) -> np.ndarray:
        """What the plan's objective credits for fuel and chips still in the system at the window's end."""
        inst = self.model.inst
        nc, T = lp.meta["nc"], lp.T
        credit = np.zeros(len(lp.lb))
        last = (T - 1) * nc
        for key, j in lp.meta["template"].items():
            if key[0] == "I":
                credit[last + j] = self.end_stock[key[1]]
            elif key[0] == "Q":
                credit[last + j] = self.end_k[key[2]]
            elif key[0] == "x":  # dispatched in the window, arriving after it
                for t in range(max(0, T - inst.edges[key[1]].tau), T):
                    credit[t * nc + j] = self.end_k[key[2]]
            elif key[0] == "p":  # lots still in process
                fab = inst.nodes[inst.fabs[key[1]]].fab
                for t in range(max(0, T - fab.tau), T):
                    credit[t * nc + j] = self.end_k[fab.product]
            elif key[0] == "xi":
                for t in range(max(0, T - inst.nodes[inst.osats[key[1]]].osat.tau), T):
                    credit[t * nc + j] = self.end_k[key[2]]
        return credit

    def _tables(self, lp):
        """Column of every action slot and of the tanker releases of a plan's week, by the package's action mapping."""
        inst, tm = self.model.inst, lp.meta["template"]
        slot_col = np.array([tm[("x", e, k, lane)] for e, k, lane in inst.action_slots])
        first = {}
        for o, (c, k, e, _lane) in enumerate(inst.override_slots):
            first.setdefault((c, k, e), o)
        releases = [(o, c, k, e, tm[("x", e, k, None)]) for (c, k, e), o in first.items() if ("x", e, k, None) in tm]
        return slot_col, releases

    def _week(self, nc: int, x, h: int, prohibited):
        """Week ``h`` of a plan as the simulator takes it: requests, tanker releases, held pairs."""
        slot_col, releases = self.tables
        q = x[h * nc + slot_col]
        banned = prohibited[self.model.slot_edge, self.model.slot_k]
        flows = {int(s): float(q[s]) for s in np.flatnonzero((q > 1e-9) & ~banned)}
        overrides, sendable = {}, {}
        for o, c, k, e, j in releases:
            ok = not bool(prohibited[e, k])
            sendable[(c, k)] = sendable.get((c, k), False) or ok
            if ok:
                overrides[o] = max(float(x[h * nc + j]), 0.0)
        holds = frozenset(ck for ck, ok in sendable.items() if not ok)
        return flows, overrides, holds

    def _flat(self, nc: int, x, h: int, prohibited) -> dict:
        """Week ``h`` of a plan as the environment's flat arrays: flows, override quantities, release modes."""
        flows, overrides, holds = self._week(nc, x, h, prohibited)
        f = np.zeros(self.n_slots)
        for s, q in flows.items():
            f[s] = q
        qty = np.zeros(len(self.model.ov_pair))
        mode = np.zeros(len(self.model.pairs), dtype=np.int64)
        for o, amount in overrides.items():
            qty[o] = amount
            mode[self.model.ov_pair[o]] = _model.OVERRIDE
        for pair in holds:
            mode[self.pair_index[pair]] = _model.HOLD
        return {"flows": f, "override_qty": qty, "release_mode": mode}

    def _with_orders(self, fuel, observation, plan: dict) -> dict:
        """A plan's week with the fuel rules' orders of that week (the rules read the plan's wafers)."""
        flows = np.zeros(self.n_slots)
        flows[self.chip_slots] = plan["flows"][self.chip_slots]
        fuel.fill(observation, flows)
        flows[self.valves] = plan["flows"][self.valves]
        return {"flows": flows, "override_qty": plan["override_qty"], "release_mode": plan["release_mode"]}

    def _carried(self, observation, w, fuel, nc: int, x):
        """The window played by a plan with the fuel rules ordering every week: records and cost."""
        seen, records = observation, []
        weeks = w.inst.T
        for h in range(weeks):
            self._clock()
            plan = self._flat(nc, x, h, np.asarray(w.marks.prohibited[h]))
            records.append(self.model.step(w, self._with_orders(fuel, seen, plan)))
            if h + 1 < weeks:
                seen = self.model.flat(observation, w)
        return records, self._cost(w, records)

    def _rules(self, observation, w, chips, fuel, strait):
        """The window played by the rules alone (no linear program)."""
        seen, records = observation, []
        weeks = w.inst.T
        for h in range(weeks):
            self._clock()
            flows = np.zeros(self.n_slots)
            chips.fill_chip_flows(seen, flows)
            fuel.fill(seen, flows)
            records.append(self.model.step(w, {"flows": flows, **strait.fill(seen)}))
            if h + 1 < weeks:
                seen = self.model.flat(observation, w)
        return records, self._cost(w, records)

    def _play(self, w, nc: int, x):
        """A plan played open loop on a fresh copy of the window, its orders as written: records and cost."""
        w2 = self.model.restart(w)
        records = []
        for h in range(w2.inst.T):
            flows, overrides, holds = self._week(nc, x, h, np.asarray(w2.marks.prohibited[h]))
            records.append(sim.step(w2.inst, w2.marks, w2.state, flows, overrides, holds))
        return records, self._cost(w2, records)

    def _hold_orders(self, R, nc: int, records) -> None:
        """Bounds of ``R``: the fuel orders stay at the reference's executed flows."""
        slot_col = self.tables[0]
        for h, rec in enumerate(records):
            done = np.array([rec.executed.get(int(s), 0.0) for s in self.orders])
            cols = h * nc + slot_col[self.orders]
            R.lb[cols] = R.ub[cols] = done

    # ----- the week ---------------------------------------------------------------------------------------------------
    def act(self, observation):
        start = time.process_time()
        flows = np.zeros(self.n_slots)
        plan = None
        if self.model is not None:
            self.deadline = start + self.seconds
            week = int(observation["week"][0])
            every = int(PARAMS["rules_every"])
            alone = self.prev_x is None or (every > 0 and (week - 1) % every == 0)
            fuel0 = copy.deepcopy(self.fuel)  # the rules' memory before this week
            rules0 = copy.deepcopy((self.chips, self.strait)) if alone else None
            self.chips.fill_chip_flows(observation, flows)  # the rules' chip entries; keeps their memory
            try:
                self.planner._remember(observation)
                plan = self._plan(observation, week, fuel0, rules0)
            except _OutOfTime:
                self.notes["late"] += 1
                if self.prev_x is not None:  # the carried plan moves on a week without having been played
                    self.prev_x = self.prev_x[self.nc :] if self.nc and len(self.prev_x) > self.nc else None
            except Exception:  # the hybrid's week
                self.notes["failed"] += 1
                self.prev_x = None
        else:
            self.chips.fill_chip_flows(observation, flows)
        if plan is not None:
            flows[self.chip_slots] = plan["flows"][self.chip_slots]
            self.planned_weeks += 1
        elif self.planner is not None:  # the hybrid's program for the chips after the fab
            remember = self.planner._remember
            if self.model is not None:  # the planner has taken this week's network into the memory already
                self.planner._remember = lambda o: None
            try:
                planned = self._planned(observation)
            finally:
                self.planner._remember = remember
            if planned is not None:
                flows[self.take] = planned[self.take]
        self.fuel.fill(observation, flows)  # this week's orders (and the rules' valves); keeps the fuel rules' memory
        action = {"flows": flows, **self.strait.fill(observation)}
        if plan is not None:
            flows[self.valves] = plan["flows"][self.valves]
            action["override_qty"], action["release_mode"] = plan["override_qty"], plan["release_mode"]
        return action

    def _plan(self, observation, week: int, fuel0, rules0):
        """The first week of the best plan as flat arrays, or None: the rules alone are the cheapest."""
        left = self.model.inst.T - week + 1
        weeks = min(int(PARAMS["horizon"]), left)
        w = self.model.window(observation, weeks)
        lp = build_lp(w.inst, w.marks)
        nc = self.nc = lp.meta["nc"]
        if self.tables is None:
            self.tables = self._tables(lp)
        self.end_on = weeks < left
        best, best_x, best_records = None, None, None
        if self.prev_x is not None and len(self.prev_x) >= nc:
            x = self.prev_x
            if len(x) < weeks * nc:  # a capped window moved on by a week: its last week repeats the one before
                x = np.concatenate([x, np.tile(x[-nc:], weeks - len(x) // nc)])
            x = x[: weeks * nc]
            best_records, best = self._carried(observation, self.model.restart(w), copy.deepcopy(fuel0), nc, x)
            best_x = x
        if rules0 is not None or best is None:
            chips, strait = rules0 if rules0 is not None else copy.deepcopy((self.chips, self.strait))
            records, j = self._rules(observation, self.model.restart(w), chips, copy.deepcopy(fuel0), strait)
            if best is None or j < best:
                best, best_x, best_records = j, None, records
        reference = best
        self._clock()
        i0 = sim.initial_stock(w.inst)
        reg = _regime.regimes(w.inst, w.marks, best_records, i0, osat_throughput(w.inst, w.marks.R_osat))
        R = _regime.rows(w.inst, w.marks, lp, reg, i0, pack=True, stores=True, release=True)
        self._hold_orders(R, nc, best_records)
        limit = self.deadline - time.process_time()
        if limit > 0.02:
            res, _secs = _regime.solve(lp, R, time_limit=limit, credit=self._end_values(lp) if self.end_on else None)
            if res.x is not None and res.status == 0 and time.process_time() < self.deadline:
                x = np.asarray(res.x)
                records, j = self._play(w, nc, x)
                if j < best - PARAMS["min_gain"] * abs(best):
                    best, best_x, best_records = j, x, records
                    self.notes["planned"] += 1
        if best_x is None:
            self.notes["rules"] += 1
            self.prev_x = None
            return None
        self.notes["carried"] += int(best == reference)
        self.prev_x = best_x[nc:].copy()
        return self._flat(nc, best_x, 0, np.asarray(w.marks.prohibited[0]))
