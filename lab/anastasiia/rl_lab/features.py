"""Features of one week, as three tables: nodes, edges and action slots.

Everything is divided by a quantity of its own, so that Small and Full read alike: a flow by the capacity of its
edge, a stock by the weekly need of the node that holds it, time by weeks left rather than by a share of the
episode. The tables are built once per episode from ``config`` (``Layout``) and filled every week from the
observation (``Featurizer.week``).

The groups are named so that an ablation drops one by name: ``time``, ``history``, ``signals`` (the ones the brief
asks about), plus ``rules`` (what the rule agent asked for), ``route`` (the summary along the whole lane) and
``state`` (stock, demand, grid and fab).

Only the standard library, numpy and the config are used: this module ships inside the submission.
"""

import numpy as np


EPS = 1e-9
NODE_TYPES = ("source", "material", "terminal", "grid", "fab", "osat", "sink", "chokepoint")
MODES = ("sea", "air", "pipeline", "grid")
GROUPS = ("rules", "route", "state", "time", "history", "signals")


def _finite(values, fill):
    """A float array of ``values`` with ``None`` replaced by ``fill``."""
    return np.array([fill if v is None else float(v) for v in values], dtype=np.float64)


class Layout:
    """The episode's fixed tables: who is where, what each slot carries, and the scale of every quantity.

    Built once in ``Agent.__init__`` (a few milliseconds) and read every week. Every number comes from ``config``:
    nothing here knows whether it runs on Tiny, Small or Full.
    """

    def __init__(self, config):
        st = config["static"]
        lay = config["layout"]
        inst = st["instance"]
        self.T = int(config["T"])

        nodes, edges, lanes, slots = st["nodes"], st["edges"], st["lanes"], st["action_slots"]
        self.n_nodes = len(nodes["id"])
        self.n_edges = len(edges["id"])
        self.n_slots = len(slots["edge"])
        self.commodities = list(st["commodities"]["id"])
        self.n_k = len(self.commodities)
        self.node_index = {name: i for i, name in enumerate(nodes["id"])}
        self.k_index = {name: i for i, name in enumerate(self.commodities)}

        self.node_type = np.array(
            [NODE_TYPES.index(t) if t in NODE_TYPES else len(NODE_TYPES) for t in nodes["type"]], dtype=np.int64
        )
        self.edge_tail = np.array([int(x) for x in edges["tail"]], dtype=np.int64)
        self.edge_head = np.array([int(x) for x in edges["head"]], dtype=np.int64)
        self.edge_mode = np.array([MODES.index(m) if m in MODES else len(MODES) for m in edges["mode"]], dtype=np.int64)
        self.tau0 = _finite(edges["tau0"], 0.0)
        self.c0 = _finite(edges["c0"], 0.0)
        # an edge without a capacity (a pipeline, a grid link) still needs a scale: take the largest finite one
        u0 = _finite(edges["u0"], np.inf)
        finite = u0[np.isfinite(u0)]
        self.u0_cap = float(finite.max()) if finite.size else 1.0
        self.u0 = np.where(np.isfinite(u0), u0, self.u0_cap)
        self.u0_infinite = ~np.isfinite(u0)

        # the route of every slot: its edges, the chokepoints it passes, where it starts and where it ends
        lane_edges = [[int(e) for e in es] for es in lanes["edges"]]
        lane_chokes = [[int(c) for c in cs] for cs in lanes["chokepoints"]]
        self.route = [
            lane_edges[int(ln)] if ln is not None else [int(e)] for e, ln in zip(slots["edge"], slots["lane"])
        ]
        self.route_chokes = [lane_chokes[int(ln)] if ln is not None else [] for ln in slots["lane"]]
        self.slot_edge = np.array([int(e) for e in slots["edge"]], dtype=np.int64)
        self.slot_k = np.array([int(k) for k in slots["k"]], dtype=np.int64)
        self.slot_tail = np.array([self.edge_tail[r[0]] for r in self.route], dtype=np.int64)
        self.slot_dest = np.array([self.edge_head[r[-1]] for r in self.route], dtype=np.int64)
        self.slot_len = np.array([len(r) for r in self.route], dtype=np.float64)
        self.slot_tau = np.array([float(self.tau0[r].sum()) for r in self.route], dtype=np.float64)
        self.slot_cap0 = self.u0[self.slot_edge]
        self.slot_cost0 = np.array([float(self.c0[r].sum()) for r in self.route], dtype=np.float64)
        # a flat (slot, edge-of-its-route) pair list, so route summaries are one segment-reduce and not a loop
        self.route_slot = np.array([s for s, r in enumerate(self.route) for _ in r], dtype=np.int64)
        self.route_edge = np.array([e for r in self.route for e in r], dtype=np.int64)
        self.chokepoints = [int(c) for c in lay["chokepoints"]]
        self.choke_row = {c: i for i, c in enumerate(self.chokepoints)}
        # the worst chokepoint of a route, as (slot, row of graph_now.open) pairs
        self.choke_slot = np.array(
            [s for s, cs in enumerate(self.route_chokes) for c in cs if c in self.choke_row], dtype=np.int64
        )
        self.choke_row_of = np.array(
            [self.choke_row[c] for cs in self.route_chokes for c in cs if c in self.choke_row], dtype=np.int64
        )

        # stock slots, demands, supplies: the rows of the observation's flat blocks
        self.stock_slots = [(int(n), int(k)) for n, k in lay["stock_slots"]]
        self.stock_row = {p: i for i, p in enumerate(self.stock_slots)}
        self.demands = [(int(n), int(k)) for n, k in lay["demands"]]
        self.supply_slots = [(int(n), int(k)) for n, k in lay["supply_slots"]]
        self.fabs = [int(n) for n in lay["fabs"]]
        self.grids = [int(n) for n in lay["grids"]]
        self.osats = [int(n) for n in lay["osats"]]
        self.fab_row = {n: i for i, n in enumerate(self.fabs)}
        self.grid_row = {n: i for i, n in enumerate(self.grids)}
        self.osat_row = {n: i for i, n in enumerate(self.osats)}
        self.sinks = [(int(n), int(k)) for n, k in zip(st["sinks"]["node"], st["sinks"]["k"])]
        self.sink_price = np.array([float(p) for p in st["sinks"]["pi"]], dtype=np.float64)

        self._needs(inst, nodes)
        self._chain(inst)
        self.warning_units = [(str(kind), int(i)) for kind, i in lay["warning_units"]]
        self.node_region = np.array([-1 if r is None else int(r) for r in nodes["region"]], dtype=np.int64)
        # warning rows that speak about a region, so a node can be given its own score
        self.region_warning = {i: row for row, (kind, i) in enumerate(self.warning_units) if kind == "region"}
        self.choke_warning = {i: row for row, (kind, i) in enumerate(self.warning_units) if kind == "chokepoint"}

    def _needs(self, inst, nodes):
        """``need[node, commodity]``: what the node uses in a normal week. The scale of every stock below."""
        need = np.zeros((self.n_nodes, self.n_k), dtype=np.float64)
        by_id = {n["id"]: n for n in inst["nodes"]}
        for name, node in by_id.items():
            n = self.node_index.get(name)
            if n is None:
                continue
            grid = node.get("grid")
            if grid:  # a grid burns its shares of what it can deliver
                for c, share in grid.get("shares", {}).items():
                    if c in self.k_index:
                        need[n, self.k_index[c]] = float(share) * float(grid["deliverable"])
            fab = node.get("fab")
            if fab:  # a fab eats its capacity in wafers and makes as much raw chip
                cap = float(fab["cap0"])
                for key in ("input", "product"):
                    c = fab.get(key)
                    if c in self.k_index:
                        need[n, self.k_index[c]] = max(need[n, self.k_index[c]], cap)
            osat = node.get("osat")
            if osat:  # a packaging plant passes its throughput through
                thr = float(osat["thr"])
                for c in osat.get("packages", {}) or {}:
                    if c in self.k_index:
                        need[n, self.k_index[c]] = max(need[n, self.k_index[c]], thr)
                for c in self.commodities:
                    if c.endswith("_raw") and need[n, self.k_index[c]] == 0.0:
                        need[n, self.k_index[c]] = thr
        for row, (n, k) in enumerate(self.demands):  # a market needs what it sells
            need[n, k] = max(need[n, k], 1.0)
        # anything still without a need (a terminal, a source) is scaled by what its lanes can bring in a week
        inflow = np.zeros_like(need)
        np.add.at(inflow, (self.slot_dest, self.slot_k), self.slot_cap0)
        outflow = np.zeros_like(need)
        np.add.at(outflow, (self.slot_tail, self.slot_k), self.slot_cap0)
        fallback = np.maximum(inflow, outflow)
        self.need = np.where(need > 0, need, np.maximum(fallback, 1.0))

    def _chain(self, inst):
        """How many weeks from this slot's cargo to a sold chip: the brief's "will it still be useful"."""
        fab_hold, osat_hold = [], []
        for node in inst["nodes"]:
            if node.get("fab"):
                fab_hold.append(float(node["fab"].get("tau", 0.0)))
            if node.get("osat"):
                osat_hold.append(float(node["osat"].get("tau", 0.0)))
        self.fab_tau = float(np.median(fab_hold)) if fab_hold else 0.0
        self.osat_tau = float(np.median(osat_hold)) if osat_hold else 0.0
        # the stages still ahead of a slot's cargo, by what it carries and where it goes
        wafer = self.k_index.get("wafer")
        raws = {self.k_index[c] for c in self.commodities if c.endswith("_raw")}
        finals = {self.k_index[c] for c in self.commodities if c.startswith("chip") and not c.endswith("_raw")}
        ahead = np.zeros(self.n_slots, dtype=np.float64)
        for s in range(self.n_slots):
            k = int(self.slot_k[s])
            if k == wafer:
                ahead[s] = self.fab_tau + self.osat_tau + 2.0
            elif k in raws:
                ahead[s] = self.osat_tau + 1.0
            elif k in finals:
                ahead[s] = 0.0
            else:  # a fuel burns the week it lands
                ahead[s] = 0.0
            ahead[s] += self.slot_tau[s]
        self.slot_ahead = ahead
        self.is_fuel = np.array(
            [self.commodities[int(k)] in ("lng", "crude", "nucfuel") for k in self.slot_k], dtype=np.float64
        )


class History:
    """The counters the brief asks for: how long a strait has been shut, how long an edge has been cut.

    One observation is enough to decide, but how long a thing has lasted is not in it. Kept in the agent, reset
    with the episode, and fed in as features.
    """

    def __init__(self, layout):
        n_choke = len(layout.chokepoints)
        self.choke_weeks = np.zeros(n_choke, dtype=np.float64)
        self.edge_weeks = np.zeros(layout.n_edges, dtype=np.float64)
        self.shed_weeks = np.zeros(len(layout.grids), dtype=np.float64)
        self.lost_weeks = np.zeros(len(layout.demands), dtype=np.float64)

    def update(self, layout, obs):
        open_now = np.asarray(obs["graph_now.open"], dtype=np.float64)
        self.choke_weeks = np.where(open_now < 0.99, self.choke_weeks + 1.0, 0.0)
        u = np.asarray(obs["graph_now.u"], dtype=np.float64)
        cut = u < 0.9 * layout.u0
        self.edge_weeks = np.where(cut, self.edge_weeks + 1.0, 0.0)
        shed = np.asarray(obs["last_week.shed.qty"], dtype=np.float64)
        self.shed_weeks = np.where(shed > 0, self.shed_weeks + 1.0, 0.0)
        lost = np.asarray(obs["last_week.sinks.lost"], dtype=np.float64)
        self.lost_weeks = np.where(lost > 0, self.lost_weeks + 1.0, 0.0)


class Featurizer:
    """Turns one observation plus the rule agent's request into the node, edge and slot tables of one week."""

    def __init__(self, layout, groups=GROUPS):
        self.L = layout
        self.groups = tuple(g for g in GROUPS if g in groups)
        self._sizes = None

    # the only two numbers a caller needs to build a network
    @property
    def sizes(self):
        if self._sizes is None:
            raise RuntimeError("call week() once before reading sizes")
        return self._sizes

    def week(self, obs, rules_flows, history):
        """``(node_feat, edge_feat, slot_feat, global_feat)`` of this week, all float32, all finite."""
        L = self.L
        on = lambda g: g in self.groups  # noqa: E731 - a local, read as "is this group on"

        week = float(np.asarray(obs["week"]).reshape(-1)[0])
        left = max(L.T - week, 0.0)

        u = np.asarray(obs["graph_now.u"], dtype=np.float64)
        c = np.asarray(obs["graph_now.c"], dtype=np.float64)
        tau = np.asarray(obs["graph_now.tau"], dtype=np.float64)
        tariff = np.asarray(obs["graph_now.tariff"], dtype=np.float64)
        prohibited = np.asarray(obs["graph_now.prohibited"], dtype=np.float64)
        open_now = np.asarray(obs["graph_now.open"], dtype=np.float64)
        war = np.asarray(obs["graph_now.war_risk"], dtype=np.float64)
        mask = np.asarray(obs["action_mask"], dtype=np.float64)

        u_rel = np.clip(u / np.maximum(L.u0, EPS), 0.0, 2.0)
        c_rel = np.clip(c / np.maximum(L.c0, 1.0), 0.0, 4.0)
        tau_rel = np.clip(tau / np.maximum(L.tau0, 1.0), 0.0, 4.0)

        # ---- nodes -------------------------------------------------------------------------------------------
        stock = np.zeros((L.n_nodes, L.n_k), dtype=np.float64)
        q = np.asarray(obs["stock.qty"], dtype=np.float64)
        for row, (n, k) in enumerate(L.stock_slots):
            stock[n, k] = q[row]
        cover = np.clip(stock / np.maximum(L.need, EPS), 0.0, 26.0) / 4.0  # weeks of cover, a month is 1.0

        node_cols = [np.eye(len(NODE_TYPES) + 1, dtype=np.float64)[L.node_type]]
        if on("state"):
            node_cols.append(cover)
            grid_state = np.zeros((L.n_nodes, 4), dtype=np.float64)
            gbar = np.asarray(obs["graph_now.grid.G_bar"], dtype=np.float64)
            ybar = np.asarray(obs["graph_now.grid.y_bar"], dtype=np.float64)
            shed = np.asarray(obs["last_week.shed.qty"], dtype=np.float64)
            for row, n in enumerate(L.grids):
                scale = max(gbar[row], EPS)
                grid_state[n] = (
                    np.clip(ybar[row] / scale, 0.0, 2.0),
                    np.clip(shed[row] / scale, 0.0, 2.0),
                    1.0,
                    np.clip(history.shed_weeks[row] / 8.0, 0.0, 4.0) if on("history") else 0.0,
                )
            node_cols.append(grid_state)
            fab_state = np.zeros((L.n_nodes, 4), dtype=np.float64)
            R = np.asarray(obs["graph_now.fab.R"], dtype=np.float64)
            alpha = np.asarray(obs["graph_now.fab.alpha_bar"], dtype=np.float64)
            cap_eff = np.asarray(obs["graph_now.fab.cap_eff"], dtype=np.float64)
            for row, n in enumerate(L.fabs):
                fab_state[n] = (
                    np.clip(R[row], 0.0, 2.0),
                    np.clip(alpha[row], 0.0, 2.0),
                    np.clip(cap_eff[row] / max(L.need[n].max(), EPS), 0.0, 2.0),
                    1.0,
                )
            node_cols.append(fab_state)
            osat_state = np.zeros((L.n_nodes, 3), dtype=np.float64)
            oR = np.asarray(obs["graph_now.osat.R"], dtype=np.float64)
            othr = np.asarray(obs["graph_now.osat.thr_eff"], dtype=np.float64)
            for row, n in enumerate(L.osats):
                osat_state[n] = (
                    np.clip(oR[row], 0.0, 2.0),
                    np.clip(othr[row] / max(L.need[n].max(), EPS), 0.0, 2.0),
                    1.0,
                )
            node_cols.append(osat_state)
            sink_state = np.zeros((L.n_nodes, 4), dtype=np.float64)
            forecast = np.asarray(obs["demand_forecast.qty"], dtype=np.float64)
            served = np.asarray(obs["last_week.sinks.served"], dtype=np.float64)
            lost = np.asarray(obs["last_week.sinks.lost"], dtype=np.float64)
            demand = np.asarray(obs["last_week.sinks.demand"], dtype=np.float64)
            for row, (n, k) in enumerate(L.demands):
                d = max(demand[row], forecast[row, 0], EPS)
                sink_state[n] += (
                    np.clip(served[row] / d, 0.0, 2.0),
                    np.clip(lost[row] / d, 0.0, 2.0),
                    np.clip(forecast[row].mean() / d, 0.0, 2.0),
                    1.0,
                )
            node_cols.append(sink_state)
            supply = np.zeros((L.n_nodes, 2), dtype=np.float64)
            avail = np.asarray(obs["graph_now.supply.avail"], dtype=np.float64)
            for row, (n, k) in enumerate(L.supply_slots):
                supply[n] = (np.clip(avail[row], 0.0, 2.0), 1.0)
            node_cols.append(supply)
        if on("signals"):
            wscore = np.asarray(obs["warning.score"], dtype=np.float64)
            node_sig = np.zeros((L.n_nodes, 1), dtype=np.float64)
            for n in range(L.n_nodes):
                r = L.region_warning.get(int(L.node_region[n]))
                if r is not None:
                    node_sig[n, 0] = wscore[r]
            node_cols.append(node_sig)
        node_feat = np.concatenate(node_cols, axis=1)

        # ---- edges -------------------------------------------------------------------------------------------
        edge_cols = [
            np.eye(len(MODES) + 1, dtype=np.float64)[L.edge_mode],
            u_rel[:, None],
            c_rel[:, None],
            tau_rel[:, None],
            np.clip(L.tau0 / 8.0, 0.0, 4.0)[:, None],
            L.u0_infinite.astype(np.float64)[:, None],
            prohibited.mean(axis=1)[:, None],
            np.clip(tariff.mean(axis=1), 0.0, 4.0)[:, None],
        ]
        if on("history"):
            edge_cols.append(np.clip(history.edge_weeks / 8.0, 0.0, 4.0)[:, None])
        edge_feat = np.concatenate(edge_cols, axis=1)

        # ---- slots -------------------------------------------------------------------------------------------
        e0 = L.slot_edge
        flows = np.asarray(rules_flows, dtype=np.float64)
        cap_now = np.maximum(u[e0], EPS)
        # what else wants this edge this week: the share decides how the simulator splits a scarce edge
        asked = np.zeros(L.n_edges, dtype=np.float64)
        np.add.at(asked, e0, flows)
        share = flows / np.maximum(asked[e0], EPS)
        press = np.clip(asked[e0] / cap_now, 0.0, 4.0)

        # route summaries: worst capacity, worst strait, total transit, any prohibition
        # a min-reduce starts at +inf, not at 1: an edge may read above its nominal capacity and the worst
        # edge of the route would otherwise be reported as 1.0
        worst_u = np.full(L.n_slots, np.inf, dtype=np.float64)
        np.minimum.at(worst_u, L.route_slot, u_rel[L.route_edge])
        route_tau = np.zeros(L.n_slots, dtype=np.float64)
        np.add.at(route_tau, L.route_slot, tau[L.route_edge])
        route_c = np.zeros(L.n_slots, dtype=np.float64)
        np.add.at(route_c, L.route_slot, c[L.route_edge])
        worst_open = np.ones(L.n_slots, dtype=np.float64)
        if L.choke_slot.size:
            np.minimum.at(worst_open, L.choke_slot, open_now[L.choke_row_of])
        worst_war = np.zeros(L.n_slots, dtype=np.float64)
        if L.choke_slot.size:
            np.maximum.at(worst_war, L.choke_slot, war[L.choke_row_of])
        choke_hist = np.zeros(L.n_slots, dtype=np.float64)
        if L.choke_slot.size and on("history"):
            np.maximum.at(choke_hist, L.choke_slot, history.choke_weeks[L.choke_row_of])

        slot_cols = [np.eye(L.n_k, dtype=np.float64)[L.slot_k]]
        if on("rules"):
            req_prev = np.asarray(obs["last_week.clip.requested"], dtype=np.float64)
            exe_prev = np.asarray(obs["last_week.clip.executed"], dtype=np.float64)
            slot_cols += [
                (flows > 0).astype(np.float64)[:, None],
                np.clip(flows / np.maximum(L.slot_cap0, EPS), 0.0, 4.0)[:, None],
                np.clip(flows / cap_now, 0.0, 4.0)[:, None],
                share[:, None],
                press[:, None],
                mask[:, None],
                np.clip(exe_prev / np.maximum(req_prev, EPS), 0.0, 1.0)[:, None],
                np.clip(req_prev / np.maximum(L.slot_cap0, EPS), 0.0, 4.0)[:, None],
            ]
        if on("route"):
            slot_cols += [
                worst_u[:, None],
                worst_open[:, None],
                worst_war[:, None],
                np.clip(L.slot_len / 4.0, 0.0, 2.0)[:, None],
                np.clip(route_c / np.maximum(L.slot_cost0, 1.0), 0.0, 4.0)[:, None],
                np.clip(tariff[e0, L.slot_k], 0.0, 4.0)[:, None],
                prohibited[e0, L.slot_k][:, None],
                L.is_fuel[:, None],
            ]
        if on("state"):
            slot_cols += [
                cover[L.slot_dest, L.slot_k][:, None],
                cover[L.slot_tail, L.slot_k][:, None],
                np.clip(flows / np.maximum(L.need[L.slot_dest, L.slot_k], EPS), 0.0, 4.0)[:, None],
            ]
        if on("time"):
            arrive = route_tau + 1.0
            useful = left - (L.slot_ahead + route_tau - L.slot_tau)
            slot_cols += [
                np.clip(left / 14.0, 0.0, 8.0)[None].repeat(L.n_slots, 0)[:, None],
                np.clip(route_tau / 8.0, 0.0, 4.0)[:, None],
                np.clip(arrive / 8.0, 0.0, 4.0)[:, None],
                np.clip(useful / 14.0, -2.0, 4.0)[:, None],
                (useful > 0).astype(np.float64)[:, None],
            ]
        if on("history"):
            slot_cols.append(np.clip(choke_hist / 8.0, 0.0, 4.0)[:, None])
        if on("signals"):
            wscore = np.asarray(obs["warning.score"], dtype=np.float64)
            sig = np.zeros((L.n_slots, 2), dtype=np.float64)
            for s in range(L.n_slots):
                r = L.region_warning.get(int(L.node_region[L.slot_dest[s]]))
                if r is not None:
                    sig[s, 0] = wscore[r]
                best = 0.0
                for ch in L.route_chokes[s]:
                    row = L.choke_warning.get(ch)
                    if row is not None:
                        best = max(best, wscore[row])
                sig[s, 1] = best
            slot_cols.append(sig)
        slot_feat = np.concatenate(slot_cols, axis=1)

        # ---- the whole week ----------------------------------------------------------------------------------
        costs = np.asarray(obs["last_week.cost_components"], dtype=np.float64)
        total = max(abs(costs).sum(), EPS)
        g = [
            np.clip(week / L.T, 0.0, 1.0),
            np.clip(left / 14.0, 0.0, 8.0),
            float(open_now.min()),
            float(open_now.mean()),
            float(u_rel.mean()),
            float(mask.mean()),
        ]
        g += list(np.clip(costs / total, -2.0, 2.0))
        global_feat = np.array(g, dtype=np.float64)

        out = tuple(
            np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for a in (node_feat, edge_feat, slot_feat, global_feat)
        )
        self._sizes = {
            "node": out[0].shape[1],
            "edge": out[1].shape[1],
            "slot": out[2].shape[1],
            "global": out[3].shape[0],
        }
        return out
