"""The next weeks as one program, for the chip and wafer entries of the action (the rules keep the fuel).

Every week: remember the network as observed, assume it stays that way (and that announced prohibitions take effect),
take demand from the forecast, build the linear program of the next ``horizon`` weeks from the observed stocks,
shipments in transit, queues and work in process, solve it with SciPy's HiGHS and return its first week's flows. It is
shockbench-flow's own ``mpc_det`` baseline as ``agents/anastasiia_mpc_baseload`` plays it, with its yes/no decisions
on power and ``mpc_fuelval``'s credit for fuel: the program is built by a copy of the package's modules in ``sbfv/``
(lab/anastasiia/mpc_lab/vendor_mpc.py), since the server has no shockbench-flow to import.

The package's program is too hopeful about electricity (it plans later weeks as if a grid could shed base load to run
its fabs) and myopic about fuel (fuel left at the window's end is worth only its salvage). Two fixes, so that its wafer
dispatch can be trusted as well as what it ships of the chips that already exist:

- ``power_*``: the yes/no decisions of ``agents/anastasiia_mpc_baseload`` (a fab draws energy in a later week only if
  its grid sheds no base load there), added only where a plan breaks the rule, in up to ``power_rounds`` rounds of
  mixed-integer solves; a round starts only before ``power_share`` of the week's CPU budget. ``power_weeks`` 0: off.
- ``fuel_value``: the end-of-window credit of ``lab/anastasiia/mpc_lab/agents/mpc_fuelval`` (fuel in stock, queued or
  in transit at the window's end at that share of its value v_k; off when the window reaches the episode's end).

Every solve (the linear program, the integer rounds, the program with the kept decisions) shares one deadline,
``solve_share`` of the week's CPU budget counted from the start of ``flows``: an integer round cut short keeps the last
plan that solved. ``Planner.flows`` returns None when the week has no plan (the first solve failed or ran out of time):
the caller keeps its own entries.
"""

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


LP = {
    "horizon": 32,  # weeks planned ahead
    "method": "highs-ds",  # scipy.optimize.linprog's method
    "planning_rules": True,  # the simulator's sharing rules as rows of the program (the package's default)
    "demand_scale": 1.1,  # the demand planned for, as a multiple of the forecast
    "pending": True,  # plan announced prohibitions as taking effect
    "young": 3,  # a spell of at most this many weeks may still end soon
    "reopen_after": None,  # weeks until a young strait disruption is planned as over; None: it stays
    "grid_back_after": 2,  # the same for a grid's deliverable generation
    "solve_share": 0.4,  # share of the week's CPU budget, counted from the start of ``flows``, every solve may reach
    "power_weeks": 52,  # later weeks of the plan in which a fab runs only if its grid sheds no base load; 0: off
    "power_every": 1,  # weeks between mixed-integer solves; the weeks between keep the last one's decisions
    "power_share": 0.3,  # share of the week's CPU budget after which no mixed-integer round starts (< solve_share)
    "power_gap": 0.02,  # relative gap at which a mixed-integer solve stops
    "power_rounds": 4,  # 0: a decision for every grid and week; n: only where a plan breaks the rule, in up to n rounds
    "fuel_value": 1.0,  # end-of-window credit for fuel, as a share of the commodity value v_k; 0: off
}

BUDGET_S = {"small": 2.0, "full": 4.0}  # the boards' CPU seconds per week, by the instance's kind
_INSTANCES: dict = {}  # per process: local evaluation plays many episodes in one


def _seen(observation, key: str) -> np.ndarray:
    return observation[f"{key}.observed"] == 1


class Planner:
    def __init__(self, config, params: dict | None = None):
        self.p = LP | (params or {})
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
        budget = BUDGET_S.get(inst.kind, 2.0)
        self.seconds = self.p["solve_share"] * budget  # every solve of the week, from the start of ``flows``
        self.power_seconds = min(self.p["power_share"], self.p["solve_share"]) * budget  # integer rounds start before
        self.end = self.power_end = 0.0  # this week's deadlines on the process clock
        self.lot_keys = layout.get("lot_keys")  # small and full: the queue is one dense table
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]
        # the grids that serve base load first, each with its fabs that draw energy; and the decisions kept for them
        self.powered = [
            (g, [f for f in inst.grid_fabs[g] if inst.nodes[inst.fabs[f]].fab.e > 0])
            for g, node in enumerate(inst.grids)
            if inst.nodes[node].grid.priority == "base_first"
        ]
        self.powered = [(g, fabs) for g, fabs in self.powered if fabs]
        self.power: dict[tuple[int, int], int] = {}  # (week, grid) -> 1: the fabs run, no base load shed; 0: no fab
        # fuel commodities and where their end-of-window stock is worth keeping (not at supply nodes)
        fuels = set()
        for g in inst.grids:
            fuels.update(inst.nodes[g].grid.fuels)
        self.supply_nodes = set(inst.supply_nodes)
        self.fuel_value = {k: self.p["fuel_value"] * inst.commodities[k].v for k in fuels}
        self.fuel_slots = [
            (s, self.fuel_value[st.k])
            for s, st in enumerate(inst.stock_slots)
            if st.k in fuels and st.node not in self.supply_nodes
        ]

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
        self.memory.pending = list(zip(*pending)) if self.p["pending"] else []
        old = 10**6 if int(o["week"][0]) == 1 else 1  # a spell running at the start has no known age
        self.strait_age = np.where(v["open"] < 0.999, self.strait_age + old, 0)
        self.grid_age = np.where(v["grid_G"] < 0.999 * self.calm["grid_G"], self.grid_age + old, 0)

    def _forecast(self, state: dict, weeks: int) -> dict:
        """The planned network of the next ``weeks`` weeks: as observed, less the young spells planned as over."""
        arrays = L.persistence_arrays(self.inst, state, self.memory, weeks)
        after, back = self.p["reopen_after"], self.p["grid_back_after"]
        arrays = {name: a.copy() for name, a in arrays.items()}
        arrays["demand"] *= self.p["demand_scale"]
        if after is not None:
            young = (self.strait_age > 0) & (self.strait_age <= self.p["young"])
            arrays["o"][after:, young] = 1.0
            arrays["kappa"][after:, young] = self.calm["kappa"][young]
        if back is not None:
            young = (self.grid_age > 0) & (self.grid_age <= self.p["young"])
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

    # ----- the program (agents/anastasiia_mpc_baseload and mpc_lab/agents/mpc_fuelval) -------------------------------
    def _fuel_horizon_value(self, model, week: int, weeks: int) -> None:
        """Credit fuel left at the window's end (stock, queues, in transit) at ``fuel_value`` of its value v_k.

        The window otherwise values that fuel at its salvage nu (500-4000 USD against v of 41 000-206 000), so it plans
        no fuel for the weeks after it, and so no energy for the lots it starts late in the window. Off in the
        endgame, where the window reaches the true horizon and the real salvage applies.
        """
        if week + weeks - 1 >= self.inst.T or not self.p["fuel_value"]:
            return
        inst, sal = self.inst, model.salvage
        nc = len(model.columns)
        last = (weeks - 1) * nc
        for j, key in enumerate(model.columns):
            tag = key[0]
            if tag == "x":
                e, k = key[1], key[2]
                val = self.fuel_value.get(k)
                if val is not None and inst.edges[e].head not in self.supply_nodes:
                    tau = inst.edges[e].tau
                    for t in range(max(1, weeks - tau + 1), weeks + 1):  # in transit at the window's end
                        i = (t - 1) * nc + j
                        if sal[i] < val:
                            sal[i] = val
            elif tag == "Q":
                val = self.fuel_value.get(key[2])
                if val is not None and sal[last + j] < val:
                    sal[last + j] = val
        slot_cols = {key[1]: j for j, key in enumerate(model.columns) if key[0] == "I"}
        for s, val in self.fuel_slots:
            j = slot_cols.get(s)
            if j is not None and sal[last + j] < val:
                sal[last + j] = val

    def _left(self) -> float:
        """CPU seconds left before this week's deadline for every solve."""
        return self.end - time.process_time()

    def _integer(self, model, arrays, cells, week: int) -> np.ndarray | None:
        """The program with a yes/no decision per cell (fabs run, or base load may be shed); None without a solution."""
        if self._left() <= 0.05:
            return None
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
            options={"time_limit": max(self._left(), 0.05), "mip_rel_gap": self.p["power_gap"]},
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
        for _ in range(self.p["power_rounds"]):
            if plan is None or time.process_time() >= self.power_end:
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
            if better is None:  # cut short without a solution: the last plan that solved stands
                break
            plan = better
        return plan

    def _solve(self, model, arrays, week: int, weeks: int) -> np.ndarray | None:
        """The plan of the window (the program's columns), or None when nothing solves in time."""
        rows = {}
        if model.A_ub.shape[0]:
            rows.update(A_ub=model.A_ub, b_ub=model.b_ub)
        if model.A_eq.shape[0]:
            rows.update(A_eq=model.A_eq, b_eq=model.b_eq)

        def linear(upper: np.ndarray) -> np.ndarray | None:
            left = self._left()
            if left <= 0:
                return None
            plan = linprog(
                model.objective(),
                bounds=np.column_stack([model.lb, upper]),
                method=self.p["method"],
                options={"time_limit": left},
                **rows,
            )
            return np.asarray(plan.x) if plan.status == 0 and plan.x is not None else None

        last = min(weeks, self.p["power_weeks"] + 1)
        index = model.index if last >= 2 and self.powered else {}
        cells = [
            (r, g, [index[("E", r, f)] for f in fabs], [index[("p", r, f)] for f in fabs], index[("ysh", r, g)])
            for r in range(2, last + 1)
            for g, fabs in self.powered
            if ("ysh", r, g) in index
        ]
        if not cells:
            return linear(model.ub)
        if (week - 1) % self.p["power_every"] == 0 or not self.power:
            if self.p["power_rounds"]:
                return self._decide(model, arrays, cells, week, linear)
            plan = self._integer(model, arrays, cells, week) if time.process_time() < self.power_end else None
            if plan is not None:
                return plan
        upper = model.ub.copy()  # a week between integer solves: the last one's decisions as bounds
        for r, g, energy, lots, shed in cells:
            decision = self.power.get((week + r - 1, g))
            if decision == 0:
                upper[energy + lots] = 0.0
            elif decision == 1:
                upper[shed] = 0.0
        plan = linear(upper)
        return plan if plan is not None else linear(model.ub)

    # ----- the week ---------------------------------------------------------------------------------------------------
    def flows(self, observation) -> np.ndarray | None:
        """The first week's flows of this week's plan, one per action slot; None when the week has no plan."""
        start = time.process_time()
        self.end, self.power_end = start + self.seconds, start + self.power_seconds
        inst = self.inst
        self._remember(observation)
        state = self._state(observation)
        week = state["week"]
        weeks = L.window_length(self.p["horizon"], week, inst.T)
        arrays = self._forecast(state, weeks)
        model = L.rolled_lp(inst, state, arrays, weeks, planning_rules=self.p["planning_rules"])
        self._fuel_horizon_value(model, week, weeks)
        plan = self._solve(model, arrays, week, weeks)
        if plan is None:
            return None
        wire = L.week1_action(inst, model, np.asarray(plan), state, L.prohibited_now(self.memory, week))
        flows = np.zeros(self.n_slots)
        flows[wire["flows"]["slot"]] = wire["flows"]["qty"]
        return flows
