"""strait_wise: `pull`'s requests, then rules that keep cargo from going where it cannot pass or cannot be used.

The base request is `pull`'s (send the maximum, a market only what it can sell). ``Agent.adjust(flows, observation)``
takes any request ``flows`` (this agent's base, or another agent's) and returns them adjusted by the rules below; the
switches and numbers are in ``PARAMS`` (a ``params.json`` beside this file replaces it). Nothing here is typed from
Small: every size and table is read from ``config``.

The rules, in the order they act (each is its own method):

- ``last_weeks`` (rule 4): chip-chain cargo that cannot arrive and be sold or started before the last week is dropped
  (the time a wafer needs in the fab and the plant and the two legs is read from the network).
- ``fuel_cover`` (rule 4): fuel into a grid only up to what the grid can still burn until the last week (full output
  times the weeks left, plus the stock lng must keep above its rationing threshold), less what it has and what is on its
  way. The nuclear stock of Small covers the whole episode, so none is ever sent.
- ``node_balance`` (rule 3): totals for the cargo into a packaging plant (raw chips only up to what its open routes to
  markets can take, and the room in its storage, counting stock, work in process and cargo on its way) and into a fab
  (wafers up to its storage), so nothing is sent where it would only be disposed of or sit.
- ``lane_bottleneck`` (rules 1 and 2): one weighted max-min allocation of the requests over the capacities of the edges
  (``graph_now.u``), the straits (``graph_now.kappa.*``, less as cargo already waiting there piles up), the stocks they
  draw on and the totals above; a commodity with a higher shortage penalty is served first on a shared edge, by what it
  can really send; what a destination loses on a cut route is moved to its other routes that have room.

Tried and left off (``PARAMS``, all off): ``osat_fill``, ``lane_split``, ``tariff_pref``, ``fleet``,
``release_override``; see the report.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = {
    "sink_pull": 1.3,  # pull: margin on a market's forecast demand; None: send the maximum into markets
    "value_first": True,  # pull: the commodity with the higher penalty asks first on a shared route
    "lane_bottleneck": True,  # rule 1: cap every request at what its route can carry, sharing edges and straits
    "queue_w1": 3.0,  # rule 2: weeks of waiting at a strait up to which its capacity is offered in full
    "queue_w2": 8.0,  # rule 2: weeks of waiting at which it is offered nothing (linear in between)
    "balance_osat": True,  # rule 3: raw chips into a plant only up to what the plant can ship on (and room)
    "balance_fab": True,  # rule 3: wafers into a fab only up to what it can start, plus a buffer, and room
    "fab_buffer": None,  # weeks of the fab's effective capacity kept on hand (None: up to its storage)
    "osat_margin": 1.3,  # a market takes at most this times its forecast demand when sizing a plant's outflow
    "osat_fill": False,  # share the edges and straits that a plant's routes have in common when sizing its outflow
    "osat_store": 1.0,  # times its storage that a plant is allowed to fill before raw chips into it are stopped
    "osat_weight": 0.0,  # power of (plant outflow / best plant's) that weights raw chips competing for one stock
    "osat_weeks": None,  # a plant holds at most this many weeks of its outflow (None: up to its storage)
    "tariff_pref": False,  # rule 5: between routes into one (destination, commodity), the cheaper one is served first
    "tariff_scale": 0.05,  # rule 5: extra cost, as a share of the customs value, that cuts a route's weight by e
    "lane_split": False,  # pull's market ask is split over routes by what each can carry, not by its first edge
    "priority_fill": True,  # rule 1: serve the commodity with the higher penalty first, by what it can really send
    "fleet": False,  # rule 1: the simulator's fleet slack (extra transit on Cape, Lombok and east routes is limited)
    "stock_rows": True,  # rule 1: also share each stock among the routes that draw on it (as the simulator's clip does)
    "reroute": 1,  # rule 1: passes that move the volume a destination lost on a cut route to its other routes
    "last_weeks": True,  # rule 4: no cargo that cannot arrive and be used before the end of the episode
    "last_margin": 0,  # rule 4: weeks of safety before the end
    "last_kinds": ["wafer", "chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"],  # rule 4: commodities it applies to
    "release_override": False,  # rule 6: steer the tanker cargo queued at a strait to the routes that can carry it on
    "fuel_cover": True,  # rule 4: fuel into a grid only up to what it can still burn before the end of the episode
    "cover_factor": 1.0,  # rule 4: times the grid's full-output burn that counts as usable
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())

BIG = 1e18


def _waterfill(A, cap, req, weight=None):
    """Weighted max-min allocation x <= req with A @ x <= cap: every request grows in proportion to its weight.

    By default the weight is the request itself, so requests that share an edge or a strait are scaled together and
    the shares are what the simulator's own pro-rata clip would give; but a constraint that binds only some of them
    leaves the capacity the others cannot use to the rest. Returns x (same shape as req).
    """
    n = len(req)
    x = np.zeros(n)
    active = req > 1e-12
    cap = np.asarray(cap, dtype=float)
    resid = cap.copy()
    weight = req if weight is None else np.maximum(np.asarray(weight, dtype=float), 1e-12)
    for _ in range(n + A.shape[0] + 2):
        if not active.any():
            break
        w = np.where(active, weight, 0.0)
        load = A @ w
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cons = np.where(load > 1e-12, resid / load, np.inf)
            t_req = np.where(active, (req - x) / weight, np.inf)
        t = max(min(t_cons.min(), t_req.min()), 0.0)
        if not np.isfinite(t):
            x[active] = req[active]
            break
        x += t * w
        resid = resid - t * load
        saturated = (load > 1e-12) & (resid <= 1e-9 * np.maximum(cap, 1.0))
        frozen = (A[saturated].sum(axis=0) > 0) if saturated.any() else np.zeros(n, dtype=bool)
        done = (req - x) <= 1e-9 * np.maximum(req, 1.0)
        active = active & ~(frozen | done)
    return np.minimum(x, req)


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        self.edge = np.array(slots["edge"])
        self.k = np.array(slots["k"])
        route = [lanes["edges"][lane] if lane is not None else [e] for e, lane in zip(slots["edge"], slots["lane"])]
        self.route = route
        self.dest = np.array([edges["head"][r[-1]] for r in route])  # the node a slot's cargo ends at
        self.lead = np.array([sum(edges["tau0"][e] for e in r) for r in route])
        # the shortage penalty per commodity: the largest over the markets that want it; 0 for an input
        pi = np.zeros(len(static["commodities"]["id"]))
        for k, p in zip(static["sinks"]["k"], static["sinks"]["pi"]):
            pi[k] = max(pi[k], p)
        self.order = sorted(set(self.k.tolist()), key=lambda k: -pi[k])
        self.sold = pi > 0  # a commodity some market pays for; inputs share a route as send-the-maximum does
        sold_order = [k for k in self.order if pi[k] > 0]
        inputs = [k for k in self.order if pi[k] <= 0]
        self.classes = [[k] for k in sold_order] + ([inputs] if inputs else [])  # who is served first on a shared edge
        self.demand_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["demands"])}
        self.horizon = config["spaces"]["observation"]["demand_forecast.qty"]["shape"][1]
        self.T = int(config["T"])
        self.v = np.array(static["commodities"]["v"], dtype=float)  # customs value per unit
        self._build_network(static, layout, lanes, slots)
        self._build_nodes(static, layout)
        self._build_release(static, layout)
        self._wmul = np.ones(len(self.edge))  # weights of the requests that compete for one stock (see node_balance)

    # ------------------------------------------------------------------ the network, read once from config
    def _build_network(self, static, layout, lanes, slots):
        edges = static["edges"]
        n_edges, n_slots = len(edges["id"]), len(self.edge)
        chk_pos = {node: i for i, node in enumerate(layout["chokepoints"])}
        self.n_chk = len(chk_pos)
        pool_index = {"tb": 0, "ct": 1}
        self.edge_pool = np.array([pool_index.get(p, 0) for p in edges["pool"]])
        self.edge_tail_chk = np.array([chk_pos.get(n, -1) for n in edges["tail"]])
        self.edge_head_chk = np.array([chk_pos.get(n, -1) for n in edges["head"]])
        self.edge_head_node = np.array(edges["head"])
        self.slot_pool = np.array([pool_index[static["commodities"]["pool"][k]] for k in self.k])
        self.slot_straits = [
            [chk_pos[c] for c in lanes["chokepoints"][lane]] if lane is not None else [] for lane in slots["lane"]
        ]
        self.lane_dest = [edges["head"][r[-1]] for r in lanes["edges"]]
        self.lane_edges = [list(r) for r in lanes["edges"]]
        self.chk_nodes = list(layout["chokepoints"])
        self.chk_index = chk_pos
        self.lot_ck = [(int(c), int(k)) for c, k, _lane, _e in layout["lot_keys"]]
        self.lot_keys = [(int(c), int(k), None if lane is None else int(lane), int(e)) for c, k, lane, e in layout["lot_keys"]]
        # the edge a lane's cargo takes after a given edge of the lane
        self.next_edge = {}
        for li, route in enumerate(lanes["edges"]):
            for a, b in zip(route[:-1], route[1:]):
                self.next_edge[(li, a)] = b
        # lot keys: (strait ordinal, k, lane, next edge) of every row of queue_lots.qty
        self.lot_c = np.array([chk_pos[c] for c, _k, _lane, _e in layout["lot_keys"]], dtype=int)
        self.lot_e = np.array([e for _c, _k, _lane, e in layout["lot_keys"]], dtype=int)
        self.lot_dest = [
            (self.lane_dest[lane] if lane is not None and lane >= 0 else -1, k) for _c, k, lane, _e in layout["lot_keys"]
        ]
        # constraint rows: every edge on some route, then every (strait, pool) a lane passes
        used_edges = sorted({e for r in self.route for e in r})
        self.row_edge = {e: i for i, e in enumerate(used_edges)}
        pairs = sorted({(c, int(self.slot_pool[s])) for s in range(n_slots) for c in self.slot_straits[s]})
        self.row_chk = {p: len(used_edges) + j for j, p in enumerate(pairs)}
        A = np.zeros((len(used_edges) + len(pairs), n_slots))
        for s in range(n_slots):
            for e in self.route[s]:
                A[self.row_edge[e], s] = 1.0
            for c in self.slot_straits[s]:
                A[self.row_chk[(c, int(self.slot_pool[s]))], s] = 1.0
        self.A_net = A  # edges and straits: what a route can carry
        self.A_mask = A > 0
        self.n_net = A.shape[0]
        self.n_edges = n_edges
        self.slot_tail = np.array([edges["tail"][e] for e in self.edge])
        # one more row per stock a slot draws on: the simulator scales the draws on a stock down to what is on hand
        stocks = sorted({(int(self.slot_tail[s]), int(self.k[s])) for s in range(n_slots)})
        self.row_stock = {p: self.n_net + j for j, p in enumerate(stocks)}
        A_full = np.zeros((self.n_net + len(stocks), n_slots))
        A_full[: self.n_net] = A
        for s in range(n_slots):
            A_full[self.row_stock[(int(self.slot_tail[s]), int(self.k[s]))], s] = 1.0
        self.A = A_full
        self.stock_rows = [(row, key) for key, row in self.row_stock.items()]
        self._build_fleet(static, lanes)
        # per slot, the (strait, next edge) hops of its lane, where its cargo may queue
        self.slot_hops = [
            [(int(self.edge_head_chk[a]), b) for a, b in zip(r[:-1], r[1:]) if self.edge_head_chk[a] >= 0] if len(r) > 1 else []
            for r in self.route
        ]

    def _build_fleet(self, static, lanes):
        """The simulator's fleet slack: the extra transit (weeks) of the flow on duplicate sea routes (Cape, Lombok,
        east of Taiwan) is limited per pool, so a row per pool counts it for the flow along each slot's route."""
        edges = static["edges"]
        alt, mode, tau = edges["alt_of"], edges["mode"], edges["tau0"]
        items = []  # (edge, lane or None, extra weeks)
        for j, ref in enumerate(alt):
            if mode[j] == "sea" and ref is not None:
                replaced = sum(tau[x] for x in lanes["edges"][ref["lane"]]) if "lane" in ref else tau[ref["edge"]]
                items.append((j, None, tau[j] - replaced))
        for li, ref in enumerate(lanes["alt_of"]):
            route = lanes["edges"][li]
            if ref is None or any(mode[x] != "sea" for x in route):
                continue
            if "lane" in ref:
                off = [x for x in route if x not in set(lanes["edges"][ref["lane"]])]
            else:
                off = [x for x in route if edges["tail"][x] == edges["tail"][ref["edge"]] and x != ref["edge"]]
            if not off:
                continue
            if "lane" in ref:
                d = sum(tau[x] for x in route) - sum(tau[x] for x in lanes["edges"][ref["lane"]])
            else:
                d = sum(tau[x] for x in route[route.index(off[0]) :]) - tau[ref["edge"]]
            item = (off[0], li, d) if off[0] == route[0] else (off[0], None, d)
            if item not in items:
                items.append(item)
        slot_lane = static["action_slots"]["lane"]
        coef = np.zeros((2, len(self.edge)))
        for sl in range(len(self.edge)):
            for e, ln, d in items:
                if e in self.route[sl] and (ln is None or (ln == slot_lane[sl] and e == self.route[sl][0])):
                    coef[int(self.slot_pool[sl]), sl] += d
        params = static["instance"]["params"]
        share, measure = params["fleet_share"], params["fleet_measure"]
        self.fleet_A = coef
        self.fleet_cap = np.array([float(share[b]) * float(measure[b]) for b in ("tb", "ct")])

    def _build_nodes(self, static, layout):
        """Stocks, storage, fabs and packaging plants: what the node-balance rules read."""
        ids, kids = static["nodes"]["id"], static["commodities"]["id"]
        k_of = {name: i for i, name in enumerate(kids)}
        node_spec = {n["id"]: n for n in static["instance"]["nodes"]}
        self.stock_pos = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        self.storage = {}
        for i, nid in enumerate(ids):
            for kn, st in (node_spec[nid].get("stock") or {}).items():
                if st.get("storage") is not None:
                    self.storage[(i, k_of[kn])] = float(st["storage"])
        self.in_slots, self.out_slots = {}, {}
        for s in range(len(self.edge)):
            self.in_slots.setdefault((int(self.dest[s]), int(self.k[s])), []).append(s)
            self.out_slots.setdefault((int(self.slot_tail[s]), int(self.k[s])), []).append(s)
        fab_ord = {node: i for i, node in enumerate(layout["fabs"])}
        self.osat_pairs, self.fab_pairs = [], []
        for i, nid in enumerate(ids):
            spec = node_spec[nid]
            if "osat" in spec:
                for raw, packaged in spec["osat"]["packages"].items():
                    self.osat_pairs.append((i, k_of[raw], k_of[packaged], int(spec["osat"]["tau"])))
            if "fab" in spec and i in fab_ord:
                fab = spec["fab"]
                self.fab_pairs.append((i, k_of[fab["input"]], k_of[fab["product"]], int(fab["tau"]), fab_ord[i]))
        self.use_delay = self._use_delays(static, node_spec, k_of)
        self.last_kind = np.array([kids[int(k)] in PARAMS["last_kinds"] for k in self.k])
        # fuel a grid can still burn before the end: its fuels, their burn at full output, the terminals feeding it
        types = static["nodes"]["type"]
        feeder = {}
        for s in range(len(self.edge)):
            if types[self.slot_tail[s]] == "terminal" and types[self.dest[s]] == "grid":
                feeder.setdefault(int(self.dest[s]), set()).add(int(self.slot_tail[s]))
        self.fuel_groups = []
        for i, nid in enumerate(ids):
            if types[i] != "grid":
                continue
            grid = node_spec[nid]["grid"]
            for kn, share in grid["shares"].items():
                if kn not in k_of or share <= 0:
                    continue
                k = k_of[kn]
                nodes = {i} | feeder.get(i, set())
                members = [
                    s
                    for s in range(len(self.edge))
                    if int(self.k[s]) == k and int(self.dest[s]) in nodes and types[self.slot_tail[s]] == "source"
                ]
                # the rationed fuel (lng) must also stay above psi x ibar at the grid, or its output falls in proportion
                keep = float(static["instance"]["params"]["psi"]) * float(grid["ibar"].get(kn, 0.0)) if grid.get("rationed") == kn else 0.0
                if members:
                    self.fuel_groups.append((i, k, members, float(share) * float(grid["deliverable"]), sorted(nodes), keep))

    def _use_delays(self, static, node_spec, k_of):
        """Per slot: weeks from the cargo's arrival until it is burned or sold, by the fastest way on.

        A grid burns fuel in the week it arrives; a terminal passes it on the week after; a fab starts a wafer the
        week it arrives, its lot matures ``tau`` weeks later and leaves the week after that; a plant packages a raw
        chip the week it arrives, ``tau`` weeks later it can leave, the week after that; a market serves on arrival.
        """
        ids, types, kids = static["nodes"]["id"], static["nodes"]["type"], static["commodities"]["id"]
        sold = {(int(n), int(k)) for n, k in zip(static["sinks"]["node"], static["sinks"]["k"])}
        far = 10**6
        memo = {}

        def onward(node, k):
            outs = self.out_slots.get((node, k), [])
            return min((int(self.lead[s]) + ttm(int(self.dest[s]), k) for s in outs), default=far)

        def ttm(node, k):
            key = (node, k)
            if key in memo:
                return memo[key]
            memo[key] = far  # a guard against cycles
            spec, kind = node_spec[ids[node]], types[node]
            if kind == "sink":
                val = 0 if key in sold else far
            elif kind == "grid":
                val = 0 if kids[k] in spec["grid"]["shares"] else far
            elif kind == "terminal":
                val = 1 + onward(node, k)
            elif kind == "fab" and kids[k] == spec["fab"]["input"]:
                val = int(spec["fab"]["tau"]) + 1 + onward(node, k_of[spec["fab"]["product"]])
            elif kind == "osat" and kids[k] in spec["osat"]["packages"]:
                val = int(spec["osat"]["tau"]) + 1 + onward(node, k_of[spec["osat"]["packages"][kids[k]]])
            else:
                val = onward(node, k)
            memo[key] = val
            return val

        return np.array([ttm(int(self.dest[s]), int(self.k[s])) for s in range(len(self.edge))])

    # ------------------------------------------------------------------ the base request (pull)
    @staticmethod
    def _open_slots(observation):
        """Slots without a sanction on their route this week (all of them in a blackout week, when none is shown)."""
        if int(np.asarray(observation["action_mask.observed"]).reshape(-1)[0]) == 0:
            return np.ones(len(observation["action_mask"]), dtype=bool)
        return np.asarray(observation["action_mask"]) == 1

    def base_flows(self, observation, st=None, value_first=None):
        value_first = PARAMS["value_first"] if value_first is None else value_first
        mask = self._open_slots(observation)
        forecast = observation["demand_forecast.qty"]
        left = np.asarray(observation["graph_now.u"], dtype=float).copy()  # capacity not yet asked for, per edge
        left[~np.isfinite(left)] = 0.0
        flows = np.zeros(len(self.edge))
        for k in self.order:
            mine = np.flatnonzero((self.k == k) & mask)
            room = left[self.edge[mine]]
            if PARAMS["lane_split"] and st is not None and self.sold[k]:
                room = np.minimum(room, st["solo"][mine])  # a market's demand is split by what a route can really carry
            ask = room.copy()
            if PARAMS["sink_pull"] is not None:
                for node in set(self.dest[mine].tolist()):
                    row = self.demand_row.get((node, k))
                    if row is None:
                        continue
                    into = self.dest[mine] == node
                    wanted = PARAMS["sink_pull"] * forecast[row, np.minimum(self.lead[mine][into], self.horizon - 1)]
                    total = room[into].sum()
                    ask[into] = np.minimum(room[into], wanted * room[into] / total) if total > 0 else 0.0
            flows[mine] = ask
            if value_first and self.sold[k]:
                np.subtract.at(left, self.edge[mine], ask)
                np.maximum(left, 0.0, out=left)
        return flows

    # ------------------------------------------------------------------ weekly state shared by the rules
    def _state(self, obs):
        u = np.asarray(obs["graph_now.u"], dtype=float).copy()
        u[~np.isfinite(u)] = BIG
        u[np.asarray(obs["graph_now.u.observed"]) == 0] = BIG
        kappa = np.stack([np.asarray(obs["graph_now.kappa.tb"], dtype=float), np.asarray(obs["graph_now.kappa.ct"], dtype=float)])
        for b, name in enumerate(("tb", "ct")):  # a strait whose throughput is not shown (a blackout week) is not blocked
            kappa[b][np.asarray(obs[f"graph_now.kappa.{name}.observed"]) == 0] = BIG
        # cargo waiting at each strait for each next edge: queued, plus in transit to the strait
        load = np.zeros((self.n_chk, self.n_edges))
        queued = np.asarray(obs["queue_lots.qty"], dtype=float).sum(axis=1)
        np.add.at(load, (self.lot_c, self.lot_e), queued)
        # cargo on its way to each (node, commodity): in the pipeline (to the strait or the node) and queued
        inbound = {}
        for q, key in zip(queued, self.lot_dest):
            if q > 0:
                inbound[key] = inbound.get(key, 0.0) + q
        p_edge, p_lane, p_qty = obs["pipeline.edge"], obs["pipeline.lane"], obs["pipeline.qty"]
        p_k = obs["pipeline.k"]
        live_p = np.asarray(obs["pipeline.qty.observed"]) == 1
        has_lane = np.asarray(obs["pipeline.lane.observed"]) == 1
        for i in np.flatnonzero(live_p):
            if has_lane[i]:
                key = (self.lane_dest[int(p_lane[i])], int(p_k[i]))
                c = self.edge_head_chk[p_edge[i]]
                if c >= 0:
                    nxt = self.next_edge.get((int(p_lane[i]), int(p_edge[i])))
                    if nxt is not None:
                        load[c, nxt] += p_qty[i]
            else:
                key = (self.edge_head_node[int(p_edge[i])], int(p_k[i]))
            inbound[key] = inbound.get(key, 0.0) + p_qty[i]
        st = {"u": u, "kappa": kappa, "load": load, "week": int(obs["week"][0]), "inbound": inbound}
        st["cap"] = self._capacities(st)
        st["solo"] = np.where(self.A_mask, st["cap"][:, None], np.inf).min(axis=0)  # what a slot could carry alone
        # weeks a new cargo of each slot would wait in the queues of its straits
        wait = np.zeros(len(self.edge))
        for s, hops in enumerate(self.slot_hops):
            for c, e_out in hops:
                q = load[c, e_out]
                if q > 0:
                    rate = min(u[e_out], kappa[self.edge_pool[e_out], c])
                    wait[s] += q / rate if rate > 0 else 1e3
        st["wait"] = wait
        if PARAMS["tariff_pref"]:
            st["pref"] = self._preference(obs)
        return st

    def _preference(self, obs):
        """Weight of each slot among the routes into the same (destination, commodity): 1 for the cheapest per unit
        (freight plus tariff on the customs value, summed along the route), falling as its extra cost grows."""
        c = np.asarray(obs["graph_now.c"], dtype=float)
        tariff = np.asarray(obs["graph_now.tariff"], dtype=float)
        n = len(self.edge)
        ucost = np.zeros(n)
        for s in range(n):
            k = int(self.k[s])
            ucost[s] = sum(c[e] + tariff[e, k] * self.v[k] for e in self.route[s])
        pref = np.ones(n)
        for (dest, k), members in self.in_slots.items():
            if len(members) < 2:
                continue
            extra = ucost[members] - ucost[members].min()
            pref[members] = np.exp(-extra / max(PARAMS["tariff_scale"] * self.v[k], 1e-9))
        return pref

    @staticmethod
    def _ramp(wait, w1, w2):
        """1 up to w1 weeks of waiting, 0 from w2 on, linear in between."""
        if w2 <= w1:
            return np.where(wait <= w1, 1.0, 0.0)
        return np.clip((w2 - wait) / (w2 - w1), 0.0, 1.0)

    def _capacities(self, st):
        """Per constraint row: the weekly capacity a new cohort can count on (edges, then straits by pool)."""
        u, kappa, load = st["u"], st["kappa"], st["load"]
        w1, w2 = PARAMS["queue_w1"], PARAMS["queue_w2"]
        cap = np.empty(self.n_net)
        # an edge out of a strait offers u, less as the cargo waiting for it piles up
        for e, row in self.row_edge.items():
            c = self.edge_tail_chk[e]
            if c < 0:
                cap[row] = u[e]
                continue
            rate = min(u[e], kappa[self.edge_pool[e], c])
            q = load[c, e]
            wait = 0.0 if q <= 0 else (np.inf if rate <= 0 else q / rate)
            cap[row] = u[e] * float(self._ramp(wait, w1, w2))
        # a strait offers its throughput, less as the cargo it still has to release piles up
        for (c, b), row in self.row_chk.items():
            kap = kappa[b, c]
            if kap <= 0:
                cap[row] = 0.0
                continue
            q = 0.0
            for e in np.flatnonzero(load[c] > 0):
                if self.edge_pool[e] == b:
                    q += min(load[c, e], max(min(u[e], kap), 0.0) * w2)
            cap[row] = kap * float(self._ramp(q / kap, w1, w2))
        return cap

    # ------------------------------------------------------------------ rules 1 and 2
    def lane_bottleneck(self, flows, obs, st, caps=()):
        """Weighted max-min allocation of the requests over the routes' edges and straits (and, when asked, the stocks
        they draw on and the totals ``caps`` that other rules put on sets of slots)."""
        req = np.asarray(flows, dtype=float)
        weight = req * st["pref"] if PARAMS["tariff_pref"] else None  # rule 5: a cheaper route gets the contested capacity
        if PARAMS["osat_weight"]:
            weight = req * self._wmul * (1.0 if weight is None else st["pref"])
        if PARAMS["stock_rows"]:
            stock = np.asarray(obs["stock.qty"], dtype=float)
            cap = np.concatenate([st["cap"], np.full(len(self.stock_rows), BIG)])
            for row, key in self.stock_rows:
                pos = self.stock_pos.get(key)
                if pos is not None:
                    cap[row] = stock[pos]
            A = self.A
        else:
            A, cap = self.A_net, st["cap"]
        if PARAMS["fleet"]:
            A = np.vstack([A, self.fleet_A])
            cap = np.concatenate([cap, self.fleet_cap])
        if caps:
            extra = np.zeros((len(caps), A.shape[1]))
            for i, (members, total) in enumerate(caps):
                extra[i, members] = 1.0
            A = np.vstack([A, extra])
            cap = np.concatenate([cap, [total for _members, total in caps]])
        if PARAMS["priority_fill"]:
            # a sold commodity with a higher shortage penalty is served first on a shared edge, by what it can really
            # send (its stock and routes limit it), and the rest of the capacity is the next one's; inputs share
            x, resid = np.zeros(len(req)), cap.copy()
            for kinds in self.classes:
                part = np.isin(self.k, kinds)
                xk = _waterfill(A, resid, np.where(part, req, 0.0), weight)
                x += xk
                resid = resid - A @ xk
        else:
            x = _waterfill(A, cap, req, weight)
        if PARAMS["stock_rows"]:
            for _ in range(int(PARAMS["reroute"])):
                x = self._reroute(req, x, A, cap)
        return x

    @staticmethod
    def _apply_caps(flows, caps):
        """The totals ``caps`` put on sets of slots, met by scaling the requests of each set in proportion."""
        x = np.asarray(flows, dtype=float).copy()
        for members, total in caps:
            tot = x[members].sum()
            if tot > total:
                x[members] *= total / tot
        return x

    def _reroute(self, req, x, A, cap):
        """Move the volume a group lost (same destination and commodity) to its routes that still have room."""
        short = {}
        for key, members in self.in_slots.items():
            lost = req[members].sum() - x[members].sum()
            if lost > 1e-9 * max(req[members].sum(), 1.0):
                short[key] = lost
        if not short:
            return x
        resid = np.maximum(cap - A @ x, 0.0)
        r2 = x.copy()
        for key, lost in short.items():
            members = np.array(self.in_slots[key])
            sub = A[:, members]
            slack = np.where(sub > 0, resid[:, None] / np.maximum(sub, 1e-12), np.inf).min(axis=0)
            slack = np.where(np.isfinite(slack), slack, 0.0)
            # only routes that were not themselves cut can take more
            slack = np.where(x[members] >= req[members] - 1e-9 * np.maximum(req[members], 1.0), slack, 0.0)
            if slack.sum() <= 0:
                continue
            r2[members] += np.minimum(slack, lost * slack / slack.sum())
        return _waterfill(A, cap, r2)

    # ------------------------------------------------------------------ rule 3
    def node_balance(self, flows, obs, st):
        """Totals for the cargo into a plant or a fab: what the node can pass on or use, counting stock and cargo on its
        way. Returns [(slots into the node, total)]."""
        x = np.asarray(flows, dtype=float)
        caps = []
        self._wmul = np.ones(len(self.edge))
        stock = np.asarray(obs["stock.qty"], dtype=float)
        mask = self._open_slots(obs)
        forecast = np.asarray(obs["demand_forecast.qty"], dtype=float)
        inbound, solo = st["inbound"], st["solo"]
        t, T = st["week"], self.T
        w_node, w_k, w_qty = obs["wip.node"], obs["wip.k"], obs["wip.qty"]
        w_live = np.asarray(obs["wip.qty.observed"]) == 1
        wip = {}
        for i in np.flatnonzero(w_live):
            key = (int(w_node[i]), int(w_k[i]))
            wip[key] = wip.get(key, 0.0) + float(w_qty[i])

        def have(key):
            pos = self.stock_pos.get(key)
            return 0.0 if pos is None else float(stock[pos])

        def mean_lead(into):
            tot = x[into].sum()
            return float((self.lead[into] * x[into]).sum() / tot) if tot > 0 else 0.0

        if PARAMS["balance_osat"]:
            # what each plant can ship on per week: its open routes to markets that want the chips, each limited by
            # what its route can carry and by the market's demand; with osat_fill the edges and straits that routes
            # share (two lanes through one strait) are shared out first
            asks = np.zeros(len(self.edge))
            for o, raw, pk, tau_o in self.osat_pairs:
                for s in self.out_slots.get((o, pk), []):
                    row = self.demand_row.get((int(self.dest[s]), pk))
                    if mask[s] and row is not None:
                        asks[s] = min(float(solo[s]), PARAMS["osat_margin"] * float(forecast[row, 0]))
            ships = asks
            if PARAMS["osat_fill"] == "priority":
                # the commodity with the higher penalty takes a shared edge first (as in the allocation itself)
                ships, resid = np.zeros(len(asks)), st["cap"].copy()
                for kinds in self.classes:
                    part = np.isin(self.k, kinds)
                    xk = _waterfill(self.A_net, resid, np.where(part, asks, 0.0))
                    ships += xk
                    resid = resid - self.A_net @ xk
            elif PARAMS["osat_fill"]:
                ships = _waterfill(self.A_net, st["cap"], asks)
            outflow = {(o, raw): float(sum(ships[s] for s in self.out_slots.get((o, pk), []))) for o, raw, pk, _tau in self.osat_pairs}
            if PARAMS["osat_weight"]:
                # where raw chips compete for one stock, the plant with more to ship on gets the larger share
                best = {}
                for (o, raw), d in outflow.items():
                    best[raw] = max(best.get(raw, 0.0), d)
                for (o, raw), d in outflow.items():
                    ref = best[raw]
                    for s in self.in_slots.get((o, raw), []):
                        self._wmul[s] = ((d + 0.05 * ref + 1e-9) / (ref * 1.05 + 1e-9)) ** PARAMS["osat_weight"]
            for o, raw, pk, tau_o in self.osat_pairs:
                into = self.in_slots.get((o, raw))
                if not into:
                    continue
                if x[into].sum() <= 0:
                    continue
                d = outflow[(o, raw)]
                lead = mean_lead(into)
                load = have((o, pk)) + wip.get((o, pk), 0.0) + have((o, raw)) + inbound.get((o, raw), 0.0)
                held = self.storage.get((o, pk), BIG) * PARAMS["osat_store"]  # what the plant may hold before it disposes
                if PARAMS["osat_weeks"] is not None:  # ... but no more than it can ship in this many weeks
                    held = min(held, d * PARAMS["osat_weeks"])
                room = held - load + d * (lead + tau_o)
                useful = d * (T - t - lead - tau_o - 1) - load
                caps.append((into, max(min(room, useful), 0.0)))
        if PARAMS["balance_fab"]:
            cap_eff = np.asarray(obs["graph_now.fab.cap_eff"], dtype=float)
            for f, win, pout, tau_f, ford in self.fab_pairs:
                into = self.in_slots.get((f, win))
                if not into:
                    continue
                if x[into].sum() <= 0:
                    continue
                starts = wip.get((f, pout), 0.0) / max(tau_f, 1)  # lots started per week, lately
                lead = mean_lead(into)
                load = have((f, win)) + inbound.get((f, win), 0.0)
                target = self.storage.get((f, win), BIG)
                if PARAMS["fab_buffer"] is not None:
                    target = min(target, PARAMS["fab_buffer"] * cap_eff[ford])
                cap_total = max(target - load + starts * (lead + 1), 0.0)
                cap_total = min(cap_total, max(self.storage.get((f, win), BIG) - load + starts * (lead + 1), 0.0))
                caps.append((into, cap_total))
        return caps

    # ------------------------------------------------------------------ rule 4
    def last_weeks(self, flows, obs, st):
        """Nothing that cannot arrive, and be burned or sold, before the end of the episode (it only pays freight)."""
        arrive = st["week"] + self.lead + st["wait"]
        late = (arrive + self.use_delay > self.T - PARAMS["last_margin"]) & self.last_kind
        return np.where(late, 0.0, flows)

    def fuel_cover(self, flows, obs, st):
        """Fuel into a grid only up to what it can still burn before the end of the episode.

        A grid burns at most its fuel's share of its deliverable output per week; stock and cargo on the way (at the
        grid and at the terminals that feed it) beyond that cover cannot be used, and only pay freight and tariff.
        """
        x = np.asarray(flows, dtype=float)
        stock = np.asarray(obs["stock.qty"], dtype=float)
        inbound, t = st["inbound"], st["week"]
        caps = []
        for g, k, members, burn, nodes, keep in self.fuel_groups:
            tot = x[members].sum()
            if tot <= 0:
                continue
            have = sum(
                (float(stock[self.stock_pos[(n, k)]]) if (n, k) in self.stock_pos else 0.0) + inbound.get((n, k), 0.0)
                for n in nodes
            )
            weeks = max(self.T - t + 1, 0)  # the grid burns from now until the end, from what it has and what comes
            caps.append((members, max(burn * weeks * PARAMS["cover_factor"] + keep - have, 0.0)))
        return caps

    # ------------------------------------------------------------------ rule 6
    def _build_release(self, static, layout):
        ov = static["override_slots"]
        self.pairs = [(int(c), int(k)) for c, k in layout["release_pairs"]]
        pair_pos = {p: i for i, p in enumerate(self.pairs)}
        self.ov_pair = [pair_pos[(int(c), int(k))] for c, k in zip(ov["chokepoint"], ov["k"])]
        self.ov_edge = [int(e) for e in ov["out_edge"]]
        self.ov_lane = [None if ln is None else int(ln) for ln in ov["lane"]]
        self.pair_slots = [[j for j, p in enumerate(self.ov_pair) if p == i] for i in range(len(self.pairs))]
        # where each override slot's cargo goes on from its out edge: the rest of its lane
        self.ov_rest = []
        for e, ln in zip(self.ov_edge, self.ov_lane):
            rows = []
            if ln is not None:
                route = self.lane_edges[ln]
                if e in route:
                    after = route[route.index(e) + 1 :]
                    rows += [self.row_edge[x] for x in after if x in self.row_edge]
                    for a, b in zip(route[route.index(e) :], after):
                        c2 = int(self.edge_head_chk[a])
                        if c2 >= 0 and (c2, int(self.edge_pool[b])) in self.row_chk:
                            rows.append(self.row_chk[(c2, int(self.edge_pool[b]))])
            self.ov_rest.append(rows)

    def release_plan(self, obs, st):
        """Tanker cargo queued at a strait leaves, in total up to what the strait can release, by the routes that can
        carry it on (the rest of their lane, edge by edge and strait by strait), in proportion to that capacity; the
        default release splits it first-in-first-out among the lanes in the queue, whether or not they can go on."""
        mode = np.zeros(len(self.pairs), dtype=np.int64)
        qty = np.zeros(len(self.ov_edge))
        week = st["week"]
        queued = np.asarray(obs["queue_lots.qty"], dtype=float).sum(axis=1)
        have, lots = {}, {}
        for q, (c, k), key in zip(queued, self.lot_ck, self.lot_keys):
            have[(c, k)] = have.get((c, k), 0.0) + q
            lots[key] = lots.get(key, 0.0) + q
        p_edge, p_k, p_q, p_w = obs["pipeline.edge"], obs["pipeline.k"], obs["pipeline.qty"], obs["pipeline.arrival_week"]
        for i in np.flatnonzero(np.asarray(obs["pipeline.qty.observed"]) == 1):
            if int(p_w[i]) == week and self.edge_head_chk[p_edge[i]] >= 0:  # arrives at a strait this week
                key = (int(self.chk_nodes[self.edge_head_chk[p_edge[i]]]), int(p_k[i]))
                have[key] = have.get(key, 0.0) + float(p_q[i])
        valid = np.asarray(obs["override_mask"]) == 1
        u, kappa, cap = st["u"], st["kappa"], st["cap"]
        used_edge, used_kappa = {}, {}
        for i, (c, k) in enumerate(self.pairs):
            q = have.get((c, k), 0.0)
            slots = [j for j in self.pair_slots[i] if valid[j]]
            if q <= 0 or not slots:
                continue
            ci = self.chk_index[c]
            b = 0  # the tanker and bulk pool
            room_k = max(kappa[b, ci] - used_kappa.get(ci, 0.0), 0.0)
            eff = []
            for j in slots:
                e = self.ov_edge[j]
                rest = min((cap[r] for r in self.ov_rest[j]), default=np.inf)
                eff.append(max(min(u[e] - used_edge.get(e, 0.0), rest), 0.0))
            eff = np.array(eff)
            budget = min(q, room_k)
            if eff.sum() <= 0 or budget <= 0:
                continue
            if PARAMS["release_override"] == "blocked":
                # steer only when some queued lane cannot go on while another can; otherwise the default is as good
                stuck = any(lots.get((c, k, self.ov_lane[j], self.ov_edge[j]), 0.0) > 0 and e_j <= 0 for j, e_j in zip(slots, eff))
                if not (stuck and (eff > 0).any()):
                    continue
            x = eff * min(1.0, budget / eff.sum())
            for j, xj in zip(slots, x):
                qty[j] = xj
                used_edge[self.ov_edge[j]] = used_edge.get(self.ov_edge[j], 0.0) + xj
            used_kappa[ci] = used_kappa.get(ci, 0.0) + x.sum()
            mode[i] = 1
        return qty, mode

    # ------------------------------------------------------------------ the week
    def adjust(self, flows, observation, st=None):
        """The rules, in order, on any request ``flows`` (this agent's base request, or another agent's): cargo that
        cannot arrive and be used before the end is dropped, totals for what a node can use or pass on join the
        capacities of the routes, and the requests are shared out within them."""
        st = self._state(observation) if st is None else st
        if PARAMS["last_weeks"]:
            flows = self.last_weeks(flows, observation, st)
        caps = []  # totals the rules put on sets of slots: they join the capacities in the allocation below
        if PARAMS["fuel_cover"]:
            caps += self.fuel_cover(flows, observation, st)
        if PARAMS["balance_osat"] or PARAMS["balance_fab"]:
            caps += self.node_balance(flows, observation, st)
        if PARAMS["lane_bottleneck"]:
            return self.lane_bottleneck(flows, observation, st, caps)
        return self._apply_caps(flows, caps)

    def act(self, observation):
        st = self._state(observation)
        flows = self.base_flows(observation, st, value_first=False if PARAMS["priority_fill"] else None)
        flows = np.nan_to_num(self.adjust(flows, observation, st), nan=0.0, posinf=0.0, neginf=0.0)
        action = {"flows": np.maximum(flows, 0.0) * self._open_slots(observation)}
        if PARAMS["release_override"]:
            action["override_qty"], action["release_mode"] = self.release_plan(observation, st)
        return action
