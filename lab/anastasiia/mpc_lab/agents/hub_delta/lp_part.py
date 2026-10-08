"""The next weeks as one linear program, for the chip entries of the action (the rules keep the fuel).

Every week: remember the network as observed, assume it stays that way (and that announced prohibitions take effect),
take demand from the forecast, build the linear program of the next ``horizon`` weeks from the observed stocks,
shipments in transit, queues and work in process, solve it with SciPy's HiGHS and return its first week's flows. It is
shockbench-flow's own ``mpc_det`` baseline as ``agents/anastasiia_mpc_baseload`` plays it, without the yes/no
decisions on power: the program is built by a copy of the package's modules in ``sbfv/``
(lab/anastasiia/mpc_lab/vendor_mpc.py), since the server has no shockbench-flow to import.

The program is too hopeful about electricity (it plans later weeks as if a grid could shed base load to run its
fabs), so its lots and its fuel are not to be trusted; what it ships of the chips that already exist is.

``Planner.flows`` returns None when the week has no plan (the solver failed or ran out of ``solve_share`` of the week's
CPU budget): the caller keeps its own entries.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


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
    "solve_share": 0.4,  # share of the week's CPU budget, counted from the start of ``flows``, the solver may reach
    # dispersion margin (hub_delta): a second solve where edges the first plan uses to the limit lose a share
    # delta = min(delta_cap, delta_k * sigma_pool * sqrt(w)) of their capacity in window week w + 1 (w >= 1, the
    # weeks after the observation; week 1 is executed now and stays exact). 0 skips the second solve entirely.
    "delta_k": 0.0,
    "bind_frac": 0.9,  # an edge is near the limit when its planned flow >= bind_frac * u in some window week >= 2
    "delta_cap": 0.5,
    "sigma_fuel": 0.062,  # weekly RMS relative change of u, tanker/bulk pool edges (measured, root 555 x8 Small)
    "sigma_chip": 0.059,  # the same for container pool edges
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
        self.seconds = self.p["solve_share"] * BUDGET_S.get(inst.kind, 2.0)
        self.lot_keys = layout.get("lot_keys")  # small and full: the queue is one dense table
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]

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

    # ----- the dispersion margin -------------------------------------------------------------------------------------
    def _edge_columns(self, model) -> tuple[np.ndarray, np.ndarray]:
        """(column index within a week, edge) of every flow column ("x", e, k, lane) of the window model."""
        key = ("edge_cols", model.meta["nc"])
        if getattr(self, "_ec_key", None) != key:
            pairs = [(j, c[1]) for c, j in model.meta["template"].items() if c[0] == "x"]
            self._ec = np.array(pairs, dtype=np.intp).reshape(-1, 2)
            self._ec_key = key
        return self._ec[:, 0], self._ec[:, 1]

    def _margin(self, model, x: np.ndarray, arrays: dict, rows: dict, end: float) -> np.ndarray | None:
        """Pass 2: the plan again with the near-limit edges' capacities of weeks >= 2 cut by delta; None keeps pass 1."""
        inst, p = self.inst, self.p
        nc, H = int(model.meta["nc"]), int(model.T)
        if H < 2:
            return None
        jcol, ecol = self._edge_columns(model)
        E = len(inst.edges)
        X = x.reshape(H, nc)
        flow = np.zeros((H, E))
        np.add.at(flow.T, ecol, X[:, jcol].T)  # flow[w, e]: the planned flow entering e in window week w + 1
        u = np.asarray(arrays["u"], dtype=float)  # (H, E), the persisted capacities the model was built from
        live = np.isfinite(u) & (u > 0)
        near = live & (flow >= p["bind_frac"] * np.where(live, u, np.inf))
        near[0] = False  # week 1 is executed now: exact
        edges = np.flatnonzero(near.any(axis=0))
        self.last["n_bind"] = int(len(edges))
        if not len(edges):
            return None
        if not hasattr(self, "_sigma"):
            pools = inst.commodity_pool
            self._sigma = np.array(
                [p["sigma_fuel"] if all(pools[k] == 0 for k in ed.K) else p["sigma_chip"] for ed in inst.edges]
            )
        w = np.arange(H, dtype=float)
        delta = np.minimum(p["delta_cap"], p["delta_k"] * np.outer(np.sqrt(w), self._sigma[edges]))  # (H, n)
        delta[0] = 0.0
        scale = 1.0 - delta
        ub = model.ub.copy()
        b_ub = rows.get("b_ub")
        b_ub = None if b_ub is None else np.asarray(b_ub, dtype=float).copy()
        if not hasattr(self, "_cap_row") or self._cap_row[0] is not model.ub_rows:
            self._cap_row = (model.ub_rows, {r[1]: i for i, r in enumerate(model.ub_rows) if r[0] == "edge_cap"})
        cap_row = self._cap_row[1]
        n_ub = len(model.ub_rows)
        for i, e in enumerate(edges):
            r = cap_row.get(int(e))
            if r is not None:  # several columns on e: the (4) row
                if b_ub is not None:
                    b_ub[np.arange(H) * n_ub + r] *= scale[:, i]
            else:  # one column on e: (4) is its bound
                for j in jcol[ecol == e]:
                    ub[np.arange(H) * nc + j] *= scale[:, i]
        left = end - time.process_time()
        if left <= 0:
            return None
        rows2 = dict(rows)
        if b_ub is not None:
            rows2["b_ub"] = b_ub
        plan = linprog(
            model.objective(),
            bounds=np.column_stack([model.lb, np.maximum(ub, model.lb)]),
            method=p["method"],
            options={"time_limit": left},
            **rows2,
        )
        if plan.status != 0 or plan.x is None:
            return None
        self.last["pass2"] = True
        return np.asarray(plan.x)

    # ----- the week ---------------------------------------------------------------------------------------------------
    def flows(self, observation) -> np.ndarray | None:
        """The first week's flows of this week's plan, one per action slot; None when the week has no plan."""
        end = time.process_time() + self.seconds
        inst = self.inst
        self._remember(observation)
        state = self._state(observation)
        week = state["week"]
        weeks = L.window_length(self.p["horizon"], week, inst.T)
        arrays = self._forecast(state, weeks)
        model = L.rolled_lp(inst, state, arrays, weeks, planning_rules=self.p["planning_rules"])
        left = end - time.process_time()
        if left <= 0:
            return None
        rows = {}
        if model.A_ub.shape[0]:
            rows.update(A_ub=model.A_ub, b_ub=model.b_ub)
        if model.A_eq.shape[0]:
            rows.update(A_eq=model.A_eq, b_eq=model.b_eq)
        plan = linprog(
            model.objective(),
            bounds=np.column_stack([model.lb, model.ub]),
            method=self.p["method"],
            options={"time_limit": left},
            **rows,
        )
        if plan.status != 0 or plan.x is None:
            return None
        x = np.asarray(plan.x)
        self.last = {"week": week, "pass2": False, "n_bind": 0}
        if self.p["delta_k"] > 0:
            x2 = self._margin(model, x, arrays, rows, end)
            if x2 is not None:
                x = x2
        wire = L.week1_action(inst, model, x, state, L.prohibited_now(self.memory, week))
        flows = np.zeros(self.n_slots)
        flows[wire["flows"]["slot"]] = wire["flows"]["qty"]
        return flows
