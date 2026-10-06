"""Plan the next weeks as a linear program each week and play its first week (model predictive control).

Every week: remember the network as observed, assume it stays that way (and that announced prohibitions take effect),
take demand from the forecast, build the linear program of the next ``horizon`` weeks from the observed stocks,
shipments in transit, queues and work in process, solve it with SciPy's HiGHS and send the first week's flows and
strait releases. It is shockbench-flow's own ``mpc_det`` baseline: the program is built by a copy of the package's
modules in ``sbfv/`` (lab/anastasiia/mpc_lab/vendor_mpc.py), since the server has no shockbench-flow to import.

Two of its assumptions can be relaxed, from the generator's statistics (hub/findings/data/event_stats_small.md):

- ``pending``: False ignores announced prohibitions until they are in force (about half are withdrawn);
- ``reopen_after`` / ``grid_back_after``: a strait disrupted, or a grid short of generation, for at most ``young``
  weeks is planned as back to normal after that many weeks instead of staying as it is (most such spells are short;
  one already running in week 1 counts as old).

The package's program is too hopeful about electricity. In the simulator a grid serves its base load first and its fabs
get what is left, so a grid short of generation runs no fab at all; the program knows that in its first week only and
plans later weeks as if it could shed base load to run the fabs. ``power_weeks`` adds the rule to that many later weeks
as yes/no decisions (a fab draws energy only in a week its grid sheds no base load), which makes the program a
mixed-integer one: it is solved every ``power_every`` weeks, until ``power_share`` of the week's CPU budget is used
(counted from the start of ``act``), and the weeks between keep its decisions and solve a linear program.

A week whose program fails to solve sends the maximum instead. A ``params.json`` beside this file replaces ``PARAMS``.

HYBRID (mpc_fuelrules). The program is myopic about fuel beyond its horizon, so after it has planned the week, rules_v2's
``FuelRules`` (``fuel_part.py``) overwrite the fuel dispatch slots (lng, crude, nuclear fuel) of ``flows``: order-up-to
fuel orders, grid priorities, rationing-threshold keeping, early nuclear fuel. Every other slot (chips, wafers) stays the
program's, and FuelRules read its wafer entries to know what the fabs will ask of their grids. Tanker releases of fuel at
the straits are the default release or ``strait_part.StraitRules``'s; container releases stay the program's.
Switches: PARAMS["fuel_rules"], PARAMS["fuel_straits"] (``fuel_part.py`` / ``strait_part.py`` read their own numbers).
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, hstack, vstack


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sbfv.instance import load_instance  # noqa: E402 - sbfv/ sits beside this file
from sbfv.policies import lp_common as L  # noqa: E402


PARAMS = {
    "horizon": 32,  # weeks planned ahead (the package plans 24 on small and full; 32 scores a little higher)
    "method": "highs-ds",  # scipy.optimize.linprog's method
    "planning_rules": True,  # the simulator's sharing rules as rows of the program (the package's default)
    "demand_scale": 1.1,  # the demand planned for, as a multiple of the forecast
    "pending": True,  # plan announced prohibitions as taking effect
    "young": 3,  # a spell of at most this many weeks may still end soon
    "reopen_after": None,  # weeks until a young strait disruption is planned as over; None: it stays
    "grid_back_after": 2,  # the same for a grid's deliverable generation
    "power_weeks": 52,  # later weeks of the plan in which a fab runs only if its grid sheds no base load; 0: off
    "power_every": 1,  # weeks between mixed-integer solves; the weeks between keep the last one's decisions
    "power_share": 0.5,  # share of the week's CPU budget after which no mixed-integer solve runs
    "power_gap": 0.02,  # relative gap at which a mixed-integer solve stops
    "power_rounds": 4,  # 0: a decision for every grid and week; n: only where a plan breaks the rule, in up to n rounds
    "fuel_rules": True,  # overwrite the fuel dispatch slots (lng, crude, nucfuel) with fuel_part.FuelRules; False: pure program
    "fuel_straits": True,  # fuel releases at straits by strait_part.StraitRules; False: the default release for fuel
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())



def _part(name: str):
    """A module beside this file, under a name of its own (another agent may ship a ``fuel_part`` too)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fuel, _strait = _part("fuel_part"), _part("strait_part")

HOLD, OVERRIDE = 2, 1  # release_mode codes
BUDGET_S = {"small": 2.0, "full": 4.0}  # the boards' CPU seconds per week, by the instance's kind
_INSTANCES: dict = {}  # per process: local evaluation plays many episodes in one


def _seen(observation, key: str) -> np.ndarray:
    return observation[f"{key}.observed"] == 1


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        key = static["instance_hash"]
        if key not in _INSTANCES:
            _INSTANCES[key] = load_instance(static["instance"])
        inst = self.inst = _INSTANCES[key]
        self.memory = L.ObservedGraph.nominal(inst)
        self.calm = {name: a.copy() for name, a in self.memory.values.items()}
        self.strait_age = np.zeros(len(inst.chokepoints), dtype=int)  # weeks each strait has been disrupted
        self.grid_age = np.zeros(len(inst.grids), dtype=int)
        self.stock = np.array(layout["stock_slots"])
        self.demands = np.array(layout["demands"])
        self.supply = [inst.slot_index[(int(n), int(k))] for n, k in layout["supply_slots"]]
        self.straits = [inst.chokepoint_ordinal[int(c)] for c in layout["chokepoints"]]
        self.fabs = [inst.fab_ordinal[int(n)] for n in layout["fabs"]]
        self.osats = [inst.osat_ordinal[int(n)] for n in layout["osats"]]
        self.grids = [inst.grid_ordinal[int(n)] for n in layout["grids"]]
        # the grids that serve base load first, each with its fabs that draw energy; and the decisions kept for them
        self.powered = [
            (g, [f for f in inst.grid_fabs[g] if inst.nodes[inst.fabs[f]].fab.e > 0])
            for g, node in enumerate(inst.grids)
            if inst.nodes[node].grid.priority == "base_first"
        ]
        self.powered = [(g, fabs) for g, fabs in self.powered if fabs]
        self.power: dict[tuple[int, int], int] = {}  # (week, grid) -> 1: the fabs run, no base load shed; 0: no fab
        self.seconds = PARAMS["power_share"] * BUDGET_S.get(inst.kind, 2.0)  # of a week, for the integer solves
        self.end = 0.0  # this week's deadline for them, on the process clock
        self.lot_keys = layout.get("lot_keys")  # small and full: the queue is one dense table
        action = config["spaces"]["action"]
        self.n_slots = action["flows"]["shape"][0]
        self.n_override = action["override_qty"]["shape"][0]
        self.pair = {(int(c), int(k)): i for i, (c, k) in enumerate(layout["release_pairs"])}
        slots = static["override_slots"]
        self.override_pair = [self.pair[(c, k)] for c, k in zip(slots["chokepoint"], slots["k"])]
        u0 = static["edges"]["u0"]
        self.capacity = np.array([u0[e] or 0.0 for e in static["action_slots"]["edge"]], dtype=float)
        # the fuel helpers, built as rules_v2's agent builds them (from ``config`` alone)
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL) if PARAMS["fuel_rules"] and _fuel.FUEL["fuel"] else None
        self.strait = _strait.StraitRules(config) if self.fuel is not None and PARAMS["fuel_straits"] else None
        # the (strait, commodity) pairs of fuel, whose release the program does not decide; and their override slots
        fuel_k = self.fuel.fuel_set if self.fuel is not None else set()
        self.fuel_pairs = [i for i, (c, k) in enumerate(layout["release_pairs"]) if int(k) in fuel_k]
        fuel_pair_set = set(self.fuel_pairs)
        self.fuel_override = [i for i, p in enumerate(self.override_pair) if p in fuel_pair_set]

    # ----- the observation as the package's planner reads it -------------------------------------------------------
    def _remember(self, o) -> None:
        """Take this week's ``graph_now`` into the memory; an entry not shown keeps its last value."""
        v = self.memory.values

        def take(target: np.ndarray, key: str, index=None) -> None:
            shown = _seen(o, key)
            rows = np.arange(len(shown)) if index is None else np.asarray(index)
            target[rows[shown]] = o[key][shown]

        take(v["u"], "graph_now.u")  # a grid coupling's capacity is never shown: it stays infinite
        take(v["c"], "graph_now.c")
        take(v["open"], "graph_now.open", self.straits)
        take(v["kappa"][:, 0], "graph_now.kappa.tb", self.straits)
        take(v["kappa"][:, 1], "graph_now.kappa.ct", self.straits)
        take(v["supply"], "graph_now.supply.avail", self.supply)
        take(v["fab_R"], "graph_now.fab.R", self.fabs)
        take(v["fab_alpha"], "graph_now.fab.alpha_bar", self.fabs)
        take(v["osat_R"], "graph_now.osat.R", self.osats)
        take(v["grid_G"], "graph_now.grid.G_bar", self.grids)
        take(v["grid_y"], "graph_now.grid.y_bar", self.grids)
        take(v["war_risk"], "graph_now.war_risk", self.straits)
        if _seen(o, "graph_now.prohibited").any():
            v["prohibited"] = o["graph_now.prohibited"] == 1
        if _seen(o, "graph_now.tariff").any():
            v["tariff"] = np.asarray(o["graph_now.tariff"], dtype=float).copy()
        live = _seen(o, "pending_prohibitions.edge")
        pending = (o[f"pending_prohibitions.{name}"][live].tolist() for name in ("edge", "k", "effective_week"))
        self.memory.pending = list(zip(*pending)) if PARAMS["pending"] else []
        old = 10**6 if int(o["week"][0]) == 1 else 1  # a spell running at the start has no known age
        self.strait_age = np.where(v["open"] < 0.999, self.strait_age + old, 0)
        self.grid_age = np.where(v["grid_G"] < 0.999 * self.calm["grid_G"], self.grid_age + old, 0)

    def _forecast(self, state: dict, weeks: int) -> dict:
        """The planned network of the next ``weeks`` weeks: as observed, less the young spells planned as over."""
        arrays = L.persistence_arrays(self.inst, state, self.memory, weeks)
        after, back = PARAMS["reopen_after"], PARAMS["grid_back_after"]
        arrays = {name: a.copy() for name, a in arrays.items()}
        arrays["demand"] *= PARAMS["demand_scale"]
        if after is not None:
            young = (self.strait_age > 0) & (self.strait_age <= PARAMS["young"])
            arrays["o"][after:, young] = 1.0
            arrays["kappa"][after:, young] = self.calm["kappa"][young]
        if back is not None:
            young = (self.grid_age > 0) & (self.grid_age <= PARAMS["young"])
            arrays["G_bar"][back:, young] = self.calm["grid_G"][young]
        return L.read_only(L.with_now(arrays))

    def _lots(self, o) -> dict:
        """The cargo queued at the straits, one entry per lot (or per lot key and arrival week on small and full)."""
        names = ("chokepoint", "k", "lane", "next_edge")
        if self.lot_keys is None:
            live = _seen(o, "queue_lots.qty")
            names += ("qty", "arrival_week", "dispatch_week", "entry_edge")
            return {name: o[f"queue_lots.{name}"][live].tolist() for name in names}
        rows, weeks = np.nonzero(o["queue_lots.qty"] > 0)
        lots = {name: [self.lot_keys[i][j] for i in rows] for j, name in enumerate(names)}
        lots["qty"] = o["queue_lots.qty"][rows, weeks].tolist()
        lots["arrival_week"] = (weeks + 1).tolist()
        lots["dispatch_week"] = lots["arrival_week"]  # the program reads a lot's arrival week only
        lots["entry_edge"] = [0] * len(rows)
        return lots

    def _state(self, o) -> dict:
        """Stocks, shipments, queues, work in process, backlog and the demand forecast, as lists."""
        moving = _seen(o, "pipeline.qty")
        lane = np.where(_seen(o, "pipeline.lane"), o["pipeline.lane"], -1)[moving]
        making = _seen(o, "wip.qty")
        forecast, shown = o["demand_forecast.qty"], _seen(o, "demand_forecast.qty")
        rows, ahead = np.nonzero(shown)
        return {
            "week": int(o["week"][0]),
            "stock": {
                "node": self.stock[:, 0].tolist(),
                "k": self.stock[:, 1].tolist(),
                "qty": o["stock.qty"].tolist(),
            },
            "pipeline": {
                "edge": o["pipeline.edge"][moving].tolist(),
                "k": o["pipeline.k"][moving].tolist(),
                "lane": [None if x < 0 else int(x) for x in lane],
                "qty": o["pipeline.qty"][moving].tolist(),
                "arrival_week": o["pipeline.arrival_week"][moving].tolist(),
            },
            "queue_lots": self._lots(o),
            "wip": {name: o[f"wip.{name}"][making].tolist() for name in ("node", "k", "qty", "out_week")},
            "backlog": {
                "node": self.demands[:, 0].tolist(),
                "k": self.demands[:, 1].tolist(),
                "qty": o["backlog.qty"].tolist(),
            },
            "demand_forecast": {
                "node": self.demands[rows, 0].tolist(),
                "k": self.demands[rows, 1].tolist(),
                "h": ahead.tolist(),
                "qty": forecast[rows, ahead].tolist(),
            },
        }

    # ----- the program ------------------------------------------------------------------------------------------------
    def _integer(self, model, arrays, cells, week: int) -> np.ndarray | None:
        """The program with a yes/no decision per cell (fabs run, or base load may be shed); None without a solution."""
        n, m = len(model.lb), len(cells)
        entries, limit = [], []
        for i, (r, g, energy, _lots, shed) in enumerate(cells):
            most = 2.0 * float(arrays["G_bar"][r - 1, g])  # no fab draws more than its grid can deliver
            for j in energy:  # energy <= most * decision
                entries += [(len(limit), j, 1.0), (len(limit), n + i, -most)]
                limit.append(0.0)
            base = float(arrays["y_bar"][r - 1, g])  # shed <= base load * (1 - decision)
            entries += [(len(limit), shed, 1.0), (len(limit), n + i, base)]
            limit.append(base)
        row, col, value = zip(*entries)
        rule = csr_matrix((value, (row, col)), shape=(len(limit), n + m))
        wide = csr_matrix((model.A_ub.shape[0], m))
        rows = [LinearConstraint(vstack([hstack([model.A_ub, wide]), rule]).tocsr(), -np.inf, np.r_[model.b_ub, limit])]
        if model.A_eq.shape[0]:
            equal = hstack([model.A_eq, csr_matrix((model.A_eq.shape[0], m))]).tocsr()
            rows.append(LinearConstraint(equal, model.b_eq, model.b_eq))
        plan = milp(
            np.r_[model.objective(), np.zeros(m)],
            constraints=rows,
            integrality=np.r_[np.zeros(n), np.ones(m)],
            bounds=Bounds(np.r_[model.lb, np.zeros(m)], np.r_[model.ub, np.ones(m)]),
            options={"time_limit": max(self.end - time.process_time(), 0.05), "mip_rel_gap": PARAMS["power_gap"]},
        )
        if plan.x is None:
            return None
        for (r, g, *_), decision in zip(cells, plan.x[n:]):
            self.power[(week + r - 1, g)] = int(round(decision))
        return plan.x[:n]

    def _decide(self, model, arrays, cells, week: int, linear) -> np.ndarray | None:
        """Decisions only where a plan breaks the rule: solve, add the cells that shed load and run a fab, repeat."""
        self.power = {}  # the weeks between keep this solve's decisions; a cell never in question stays free
        plan = linear(model.ub)
        chosen: list = []
        for _ in range(PARAMS["power_rounds"]):
            if plan is None or time.process_time() >= self.end:
                break
            taken = {(c[0], c[1]) for c in chosen}
            broken = [
                c
                for c in cells
                if (c[0], c[1]) not in taken
                and plan[c[4]] > 1e-6 * float(arrays["y_bar"][c[0] - 1, c[1]])
                and plan[c[2]].sum() > 1e-9
            ]
            if not broken:
                break
            chosen += broken
            better = self._integer(model, arrays, chosen, week)
            if better is None:
                break
            plan = better
        return plan

    def _solve(self, model, arrays, week: int, weeks: int) -> np.ndarray | None:
        """The plan of the window (the program's columns), or None when nothing solves."""
        rows = {}
        if model.A_ub.shape[0]:
            rows.update(A_ub=model.A_ub, b_ub=model.b_ub)
        if model.A_eq.shape[0]:
            rows.update(A_eq=model.A_eq, b_eq=model.b_eq)

        def linear(upper: np.ndarray) -> np.ndarray | None:
            plan = linprog(
                model.objective(), bounds=np.column_stack([model.lb, upper]), method=PARAMS["method"], **rows
            )
            return np.asarray(plan.x) if plan.status == 0 and plan.x is not None else None

        last = min(weeks, PARAMS["power_weeks"] + 1)
        index = model.index if last >= 2 and self.powered else {}
        cells = [
            (r, g, [index[("E", r, f)] for f in fabs], [index[("p", r, f)] for f in fabs], index[("ysh", r, g)])
            for r in range(2, last + 1)
            for g, fabs in self.powered
            if ("ysh", r, g) in index
        ]
        if not cells:
            return linear(model.ub)
        if (week - 1) % PARAMS["power_every"] == 0 or not self.power:
            plan = self._decide(model, arrays, cells, week, linear) if PARAMS["power_rounds"] else None
            plan = self._integer(model, arrays, cells, week) if not PARAMS["power_rounds"] else plan
            if plan is not None:
                return plan
        upper = model.ub.copy()
        for r, g, energy, lots, shed in cells:
            decision = self.power.get((week + r - 1, g))
            if decision == 0:
                upper[energy + lots] = 0.0
            elif decision == 1:
                upper[shed] = 0.0
        plan = linear(upper)
        return plan if plan is not None else linear(model.ub)

    # ----- the week ---------------------------------------------------------------------------------------------------
    def act(self, observation):
        self.end = time.process_time() + self.seconds
        inst = self.inst
        self._remember(observation)
        state = self._state(observation)
        week = state["week"]
        weeks = L.window_length(PARAMS["horizon"], week, inst.T)
        arrays = self._forecast(state, weeks)
        model = L.rolled_lp(inst, state, arrays, weeks, planning_rules=PARAMS["planning_rules"])
        plan = self._solve(model, arrays, week, weeks)
        if plan is None:
            flows = self.capacity * observation["action_mask"]
            if self.fuel is not None:  # a week the program does not solve still gets the fuel rules
                self.fuel.fill(observation, flows)
            return {"flows": flows}
        wire = L.week1_action(inst, model, plan, state, L.prohibited_now(self.memory, week))
        flows = np.zeros(self.n_slots)
        flows[wire["flows"]["slot"]] = wire["flows"]["qty"]
        override_qty = np.zeros(self.n_override)
        release_mode = np.zeros(len(self.pair), dtype=np.int64)
        for slot, qty in zip(*((wire["overrides"] or {}).get(name, []) for name in ("slot", "qty"))):
            override_qty[slot] = qty
            release_mode[self.override_pair[slot]] = OVERRIDE
        for c, k in zip(*((wire["hold"] or {}).get(name, []) for name in ("chokepoint", "k"))):
            release_mode[self.pair[(c, k)]] = HOLD
        override_qty[[i for i, p in enumerate(self.override_pair) if release_mode[p] != OVERRIDE]] = 0.0
        if self.fuel is not None:
            # the fuel entries of ``flows`` are FuelRules' (it reads the program's wafer entries, which are filled)
            self.fuel.fill(observation, flows)
            # fuel releases at straits are not the program's: default release, or StraitRules' override where it adds flow
            release_mode[self.fuel_pairs] = 0
            override_qty[self.fuel_override] = 0.0
            if self.strait is not None:
                extra = self.strait.fill(observation)
                for i in self.fuel_pairs:
                    release_mode[i] = extra["release_mode"][i]
                override_qty[self.fuel_override] = extra["override_qty"][self.fuel_override]
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
