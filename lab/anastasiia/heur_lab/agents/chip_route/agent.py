"""pull's rules for the fuel, explicit routing for the chip chain (wafers, raw chips, packaged chips).

The fuel slots (lng, crude, nucfuel) are exactly ``pull``'s: ``_pull_flows`` is its loop, unchanged. ``fill_chip_flows``
then overwrites the wafer, raw-chip and packaged-chip entries of ``flows`` with three functions, each filling only its own
slots. The simulator executes a request that fits the edge capacities and the stock exactly as asked, so every function
computes a feasible flow instead of asking for the maximum.

- ``_pack_flows``, packaged chips, plant -> market. A market is asked for what it can still sell: its forecast demand up to
  the week the cargo arrives, minus its stock and the cargo already on the way (cargo queued at a strait counts when the
  strait's throughput will have released it). A small max-flow (plant stock, edge capacities, strait throughput) fills the
  needs of all markets in rounds of equal fill ratio; the dearer chip (chip_le) first, the cheaper (chip_mat) on what is
  left of the shared edges, faster routes first, then routes without a strait, then cheaper ones (tariff, freight). What a
  plant still holds after that goes on, over routes without a strait, up to the market's cover (its storage).
- ``_raw_flows``, raw chips, fab -> plant. A plant takes what its storage can hold after its outlets (alive routes to
  markets that want the chip, capped by the plant's throughput; for the cheaper chip only what the dearer one leaves of the
  shared edges this week) have drained what is already there for as long as the new chips need to reach and pass the
  plant; none if it has no outlet or could not pass them on before the horizon ends.
  Plants are filled in rounds by expected wait (chips there / outlets), so a plant with idle outlets comes first; what no
  plant can take stays at the fab, where the fab's own storage keeps it.
- ``_wafer_flows``, wafers, source -> fab. A fab is kept stocked for three weeks of its effective capacity (the stock is
  cheap, a start missed when power returns is not), scaled by ``usefulness``: the share of the fab's capacity whose chips
  can still reach a market. Routes without a strait, lead 1 and cheap ones first; nothing that cannot become a sold chip
  before the horizon ends.

``usefulness`` and ``chip_value`` are for the fuel rules: which grid's fabs can make chips that sell, and the value of the
energy they would use.

Every number is read from ``config``; the same code runs on Tiny, Small and Full. A ``params.json`` beside this file
replaces ``PARAMS``.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = {
    # pull's two rules, kept as they are for the fuel slots
    "sink_pull": 1.3,
    "value_first": True,
    # the chip chain: each rule can be switched off (the entries then keep pull's values)
    "pack": True,
    "raw": True,
    "wafer": True,
    "endgame": True,  # send nothing that cannot become a sold chip before the horizon ends
    # packaged chips
    "pack_margin": 1.3,  # a market's need: margin x forecast demand up to the arrival week, minus stock and cargo
    "pack_levels": 3,  # rounds in which the markets' needs are filled to equal ratios
    "pack_cover": 6.0,  # weeks of demand a market may hold: what plants still have is sent up to it (0: off)
    # raw chips
    "raw_extra": 1.0,  # weeks, beyond the transport and packaging, that a plant's outlets drain before new chips are stored
    "raw_waits": [2.0, 4.0, 8.0, 16.0],  # plants are filled up to these waits (chips there / outlets), shortest first
    # routes through straits
    "strait_pref": True,  # routes without a strait first; a lane through a strait only for what they cannot carry
    "queue_eta": True,  # cargo queued at a strait arrives when its throughput has released what is ahead of it
    # wafers
    "w_weeks": 3.0,  # weeks of effective capacity kept on hand at a fab
    "w_fill": 0.9,  # never more than this share of the wafer storage (above it the stock is thrown away)
    "w_useful": True,  # scale the stock by usefulness
    "w_u_min": 0.02,  # a fab with a smaller share gets no wafers
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())

EPS = 1e-6
FAR = 10**6  # an arrival week that never comes (a strait that is closed)


class Flow:
    """Max-flow (Dinic) on a small graph; capacities can be raised and the flow continued."""

    def __init__(self, n):
        self.g = [[] for _ in range(n)]
        self.to, self.cap = [], []

    def add(self, u, v, c):
        i = len(self.to)
        self.to += [v, u]
        self.cap += [float(c), 0.0]
        self.g[u].append(i)
        self.g[v].append(i + 1)
        return i

    def flow(self, i):
        return self.cap[i ^ 1]

    def raise_to(self, i, c):
        self.cap[i] = max(0.0, float(c) - self.cap[i ^ 1])

    def run(self, s, t):
        n, total = len(self.g), 0.0
        while True:
            level = [-1] * n
            level[s] = 0
            queue = [s]
            for u in queue:
                for i in self.g[u]:
                    if self.cap[i] > EPS and level[self.to[i]] < 0:
                        level[self.to[i]] = level[u] + 1
                        queue.append(self.to[i])
            if level[t] < 0:
                return total
            ptr = [0] * n

            def push(u, f):
                if u == t:
                    return f
                while ptr[u] < len(self.g[u]):
                    i = self.g[u][ptr[u]]
                    v = self.to[i]
                    if self.cap[i] > EPS and level[v] == level[u] + 1:
                        d = push(v, min(f, self.cap[i]))
                        if d > EPS:
                            self.cap[i] -= d
                            self.cap[i ^ 1] += d
                            return d
                    ptr[u] += 1
                return 0.0

            while True:
                f = push(s, float("inf"))
                if f <= EPS:
                    break
                total += f


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        self.T = int(config["T"])
        self.edges, self.lanes = edges, lanes
        node_ix = {n: i for i, n in enumerate(static["nodes"]["id"])}
        com_id = static["commodities"]["id"]
        com_ix = {c: i for i, c in enumerate(com_id)}
        self.v = np.array(static["commodities"]["v"], dtype=float)
        nodes = {node_ix[n["id"]]: n for n in static["instance"]["nodes"]}

        # the action slots and their routes, as pull reads them
        self.n_slots = len(slots["edge"])
        self.edge = np.array(slots["edge"])
        self.k = np.array(slots["k"])
        self.route = [lanes["edges"][ln] if ln is not None else [e] for e, ln in zip(slots["edge"], slots["lane"])]
        self.tail = np.array([edges["tail"][r[0]] for r in self.route])
        self.dest = np.array([edges["head"][r[-1]] for r in self.route])
        self.lead0 = np.array([sum(edges["tau0"][e] for e in r) for r in self.route])
        self.lane_chk = [list(lanes["chokepoints"][ln]) if ln is not None else [] for ln in slots["lane"]]
        self.chk_ord = {int(n): i for i, n in enumerate(layout["chokepoints"])}
        self.u0 = np.array([0.0 if u is None else u for u in edges["u0"]], dtype=float)
        self.lane_pos = {ln: {e: i for i, e in enumerate(es)} for ln, es in enumerate(lanes["edges"])}
        pi = np.zeros(len(com_id))  # the shortage penalty per commodity; 0 for an input
        for k, p in zip(static["sinks"]["k"], static["sinks"]["pi"]):
            pi[k] = max(pi[k], p)
        self.pi = pi
        self.order = sorted(set(self.k.tolist()), key=lambda k: -pi[k])
        self.sold = pi > 0
        self.demand_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["demands"])}
        self.horizon = config["spaces"]["observation"]["demand_forecast.qty"]["shape"][1]

        # the chip chain: fabs (wafer -> raw chip), packaging plants (raw -> packaged chip), markets
        self.stock_ix = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        self.fabs = [int(n) for n in layout["fabs"]]
        self.osats = [int(n) for n in layout["osats"]]
        self.fab_ord = {n: i for i, n in enumerate(self.fabs)}
        self.osat_ord = {n: i for i, n in enumerate(self.osats)}
        self.osat = {}
        for n in self.osats:
            a = nodes[n]["osat"]
            self.osat[n] = dict(pack={com_ix[r]: com_ix[p] for r, p in a["packages"].items()}, tau=int(a["tau"]))
        self.fab = {}
        for n in self.fabs:
            a = nodes[n]["fab"]
            out = com_ix[a["product"]]
            pk = next((o["pack"][out] for o in self.osat.values() if out in o["pack"]), None)
            self.fab[n] = dict(inp=com_ix[a["input"]], out=out, pk=pk, tau=int(a["tau"]), cap0=float(a["cap0"]),
                               e=float(a["e"]), grid=node_ix.get(a["grid"]))
        self.storage = {}
        for n, node in nodes.items():
            for name, d in node.get("stock", {}).items():
                self.storage[(n, com_ix[name])] = float("inf") if d.get("storage") is None else float(d["storage"])
        sink_nodes = {int(n) for n in static["sinks"]["node"]}
        kind = np.zeros(self.n_slots, dtype=int)  # 0 other (fuel), 1 wafer, 2 raw chip, 3 packaged chip
        for s in range(self.n_slots):
            d, t, k = int(self.dest[s]), int(self.tail[s]), int(self.k[s])
            if d in self.fab and k == self.fab[d]["inp"]:
                kind[s] = 1
            elif t in self.fab and k == self.fab[t]["out"]:
                kind[s] = 2
            elif t in self.osat and d in sink_nodes:
                kind[s] = 3
        self.kind = kind
        self.wafer_slots = {n: [s for s in range(self.n_slots) if kind[s] == 1 and self.dest[s] == n] for n in self.fabs}
        self.raw_slots = {n: [s for s in range(self.n_slots) if kind[s] == 2 and self.tail[s] == n] for n in self.fabs}
        self.pack_slots = {}
        for s in range(self.n_slots):
            if kind[s] == 3:
                self.pack_slots.setdefault(int(self.k[s]), []).append(s)
        self.pack_order = sorted(self.pack_slots, key=lambda k: -pi[k])  # the dearer chip first
        self.raw_kinds = sorted({self.fab[n]["out"] for n in self.fabs})
        self.pool_ct = [p == "ct" for p in static["commodities"]["pool"]]  # carried by the container pool
        # Small and Full give the queue at straits as one dense table, Tiny as a padded list of lots
        self.lot_keys = [tuple(int(x) for x in key) for key in layout["lot_keys"]] if layout.get("lot_keys") else None
        self.last_mask = np.ones(self.n_slots, dtype=bool)
        self.last_forecast = None

    # ------------------------------------------------------------------ the whole week
    def act(self, observation):
        flows = self._pull_flows(observation)
        self.fill_chip_flows(observation, flows)
        return {"flows": flows}

    def fill_chip_flows(self, observation, flows):
        """Overwrite the wafer, raw-chip and packaged-chip entries of ``flows`` (the fuel slots are not touched)."""
        c = self._context(observation)
        c.cap_before = {}  # packaged chip -> edge capacity left for it after the dearer chips this week
        if PARAMS["pack"]:
            self._pack_flows(c, flows)
        if PARAMS["raw"]:
            self._raw_flows(c, flows)
        if PARAMS["wafer"]:
            self._wafer_flows(c, flows)
        return flows

    def chip_value(self, observation):
        """grid node -> [(USD per GWh, GWh a week)] for each fab on the grid.

        The energy a fab uses is worth its chips' shortage penalty per GWh (penalty / GWh per lot) as long as the chips can
        reach a market; the GWh a week are what the fab can still use that way (its effective capacity times
        ``usefulness``, times the GWh per lot). A grid's fabs run on the output above its base load.
        """
        c = self._context(observation)
        use = self.usefulness(c)
        out = {}
        for n, fab in self.fab.items():
            if fab["grid"] is None or fab["pk"] is None:
                continue
            gwh = fab["e"] * float(c.cap_eff[self.fab_ord[n]]) * use[n]
            out.setdefault(fab["grid"], []).append((self.pi[fab["pk"]] / max(fab["e"], EPS), gwh))
        return out

    # ------------------------------------------------------------------ pull, unchanged
    def _pull_flows(self, observation):
        mask = observation["action_mask"] == 1
        forecast = observation["demand_forecast.qty"]
        left = np.asarray(observation["graph_now.u"], dtype=float).copy()  # capacity not yet asked for, per edge
        left[~np.isfinite(left)] = 0.0
        flows = np.zeros(self.n_slots)
        for k in self.order:
            mine = np.flatnonzero((self.k == k) & mask)
            room = left[self.edge[mine]]
            ask = room.copy()
            if PARAMS["sink_pull"] is not None:
                for node in set(self.dest[mine].tolist()):
                    row = self.demand_row.get((node, k))
                    if row is None:
                        continue
                    into = self.dest[mine] == node
                    wanted = PARAMS["sink_pull"] * forecast[row, np.minimum(self.lead0[mine][into], self.horizon - 1)]
                    total = room[into].sum()
                    ask[into] = np.minimum(room[into], wanted * room[into] / total) if total > 0 else 0.0
            flows[mine] = ask
            if PARAMS["value_first"] and self.sold[k]:
                np.subtract.at(left, self.edge[mine], ask)
                np.maximum(left, 0.0, out=left)
        return flows

    # ------------------------------------------------------------------ the week's picture
    def _context(self, obs):
        c = type("Ctx", (), {})()
        c.obs = obs
        c.t = int(obs["week"][0])
        u = np.asarray(obs["graph_now.u"], dtype=float)
        c.u = np.where(np.asarray(obs["graph_now.u.observed"]) == 1, u, self.u0)
        c.u[~np.isfinite(c.u)] = 0.0
        c.tau = np.asarray(obs["graph_now.tau"], dtype=int)
        c.c = np.asarray(obs["graph_now.c"], dtype=float)
        c.tariff = np.asarray(obs["graph_now.tariff"], dtype=float)
        if int(obs["action_mask.observed"][0]) == 1:  # a blackout week shows every slot as allowed: keep the last mask
            self.last_mask = np.asarray(obs["action_mask"]) == 1
        c.mask = self.last_mask
        c.stock = np.asarray(obs["stock.qty"], dtype=float)
        forecast = np.asarray(obs["demand_forecast.qty"], dtype=float)
        if np.all(np.asarray(obs["demand_forecast.qty.observed"]) == 1):
            self.last_forecast = forecast
        c.forecast = self.last_forecast if self.last_forecast is not None else forecast
        c.kappa = np.asarray(obs["graph_now.kappa.ct"], dtype=float)  # a strait's weekly throughput for containers
        c.cap_eff = np.asarray(obs["graph_now.fab.cap_eff"], dtype=float)
        c.thr_eff = np.asarray(obs["graph_now.osat.thr_eff"], dtype=float)
        # what each chip-chain route can carry, how long it takes, what it costs per unit (freight and tariff)
        n = self.n_slots
        c.cap, c.lead, c.cost = np.zeros(n), np.zeros(n, dtype=int), np.zeros(n)
        c.strait = np.zeros(n, dtype=bool)  # the route passes a strait
        for s in np.flatnonzero(self.kind > 0):
            r, k = self.route[s], int(self.k[s])
            cap = min(c.u[e] for e in r)
            for chk in self.lane_chk[s]:
                cap = min(cap, c.kappa[self.chk_ord[int(chk)]])
            c.strait[s] = len(self.lane_chk[s]) > 0
            c.cap[s] = cap if c.mask[s] else 0.0
            c.lead[s] = sum(c.tau[e] for e in r)
            c.cost[s] = sum(c.c[e] + c.tariff[e, k] * self.v[k] for e in r)
        c.transit = self._transit(c)
        c.wip = self._wip(c)
        return c

    def stock_of(self, c, node, k):
        i = self.stock_ix.get((int(node), int(k)))
        return 0.0 if i is None else float(c.stock[i])

    def _queue_lots(self, obs):
        """[(strait, k, lane, next edge, arrival week, quantity)] of the cargo waiting at straits."""
        out = []
        qty = np.asarray(obs["queue_lots.qty"], dtype=float)
        seen = np.asarray(obs["queue_lots.qty.observed"]) == 1
        if self.lot_keys is None:  # Tiny: a padded list of lots
            lane, nxt = np.asarray(obs["queue_lots.lane"]), np.asarray(obs["queue_lots.next_edge"])
            chk, k = np.asarray(obs["queue_lots.chokepoint"]), np.asarray(obs["queue_lots.k"])
            week = np.asarray(obs["queue_lots.arrival_week"])
            for i in np.flatnonzero(seen):
                out.append((int(chk[i]), int(k[i]), int(lane[i]), int(nxt[i]), int(week[i]), float(qty[i])))
            return out
        for row in np.flatnonzero(seen.any(axis=1)):  # one row per (strait, k, lane, next edge), one column per week
            chk, k, ln, nxt = self.lot_keys[row]
            for col in np.flatnonzero(seen[row]):
                out.append((chk, k, ln, nxt, int(col) + 1, float(qty[row, col])))
        return out

    def _transit(self, c):
        """(node, k) -> [(week of arrival there, quantity)] of the cargo on its way, queued cargo included.

        With ``queue_eta``, cargo that waits at a strait (or will join its queue) arrives when the strait's throughput
        has released what is ahead of it, oldest cargo first: cargo stuck behind a closed strait never arrives.
        """
        obs, edges, lanes, tau = c.obs, self.edges, self.lanes, c.tau
        out = {}
        lots = self._queue_lots(obs)
        queue_by_week = {}  # strait -> {week of arrival: container cargo}
        for chk, k, _ln, _nxt, week, q in lots:
            if self.pool_ct[k]:
                book = queue_by_week.setdefault(chk, {})
                book[week] = book.get(week, 0.0) + q
        queue_now = {chk: sum(book.values()) for chk, book in queue_by_week.items()}
        live = np.asarray(obs["pipeline.qty.observed"]) == 1
        lane_seen = np.asarray(obs["pipeline.lane.observed"]) == 1
        e_a, k_a = np.asarray(obs["pipeline.edge"]), np.asarray(obs["pipeline.k"])
        l_a, q_a = np.asarray(obs["pipeline.lane"]), np.asarray(obs["pipeline.qty"])
        w_a = np.asarray(obs["pipeline.arrival_week"])
        for i in np.flatnonzero(live):
            e, k, q, aw = int(e_a[i]), int(k_a[i]), float(q_a[i]), int(w_a[i])
            if lane_seen[i]:
                ln = int(l_a[i])
                es = lanes["edges"][ln]
                pos = self.lane_pos[ln].get(e)
                rest = sum(tau[x] for x in es[pos + 1 :]) if pos is not None else 0
                dest = edges["head"][es[-1]]
            else:
                rest, dest = 0, edges["head"][e]
            head = edges["head"][e]
            if PARAMS["queue_eta"] and self.pool_ct[k] and head in self.chk_ord:  # it will join the queue at that strait
                kap = float(c.kappa[self.chk_ord[head]])
                left = queue_now.get(head, 0.0) - kap * max(0, aw - c.t)
                rest += int(np.ceil(max(0.0, left) / kap)) if kap > EPS else FAR
            out.setdefault((dest, k), []).append((aw + rest, q))
        for chk, k, ln, nxt, week, q in lots:
            if ln < 0:
                continue
            es = lanes["edges"][ln]
            pos = self.lane_pos[ln].get(nxt)
            rest = sum(tau[x] for x in es[pos:]) if pos is not None else 1
            wait = 0
            if PARAMS["queue_eta"] and self.pool_ct[k]:
                kap = float(c.kappa[self.chk_ord[chk]])
                book = queue_by_week[chk]
                ahead = sum(v for w, v in book.items() if w < week) + 0.5 * book[week]
                wait = int(np.ceil(ahead / kap)) if kap > EPS else FAR
            out.setdefault((edges["head"][es[-1]], k), []).append((c.t + wait + rest, q))
        return out

    def _wip(self, c):
        """node -> {week of output: {k: quantity}} of the work in process (fabs and plants)."""
        obs = c.obs
        live = np.asarray(obs["wip.qty.observed"]) == 1
        out = {}
        for n, k, q, w in zip(np.asarray(obs["wip.node"])[live], np.asarray(obs["wip.k"])[live], np.asarray(obs["wip.qty"])[live], np.asarray(obs["wip.out_week"])[live]):
            book = out.setdefault(int(n), {}).setdefault(int(w), {})
            book[int(k)] = book.get(int(k), 0.0) + float(q)
        return out

    def arrivals(self, c, node, k, by):
        return sum(q for aw, q in c.transit.get((int(node), int(k)), ()) if aw <= by)

    def _throughput(self, c, o):
        """What plant ``o`` can package a week now (its effective throughput); unlimited if the observation does not say."""
        t = float(c.thr_eff[self.osat_ord[o]])
        return t if np.isfinite(t) else float("inf")

    # ------------------------------------------------------------------ packaged chips: plant -> market
    def _need(self, c, node, k, lead):
        """What the market at ``node`` can still sell of ``k`` when cargo sent now arrives in ``lead`` weeks."""
        row = self.demand_row.get((int(node), int(k)))
        if row is None:
            return 0.0
        h = min(int(lead), self.horizon - 1)
        want = PARAMS["pack_margin"] * float(c.forecast[row, : h + 1].sum())
        have = self.stock_of(c, node, k) + self.arrivals(c, node, k, c.t + lead)
        return max(0.0, want - have)

    def _pack_flows(self, c, flows):
        cap_left = c.u.copy()  # edge capacity not yet given to a dearer chip
        used = {}  # (plant, k) -> stock already assigned
        for k in self.pack_order:
            c.cap_before[k] = cap_left.copy()  # what is left of the edges for this chip, for its plants' outlets
            for s in self.pack_slots[k]:
                flows[s] = 0.0
            slots = [s for s in self.pack_slots[k] if c.cap[s] > EPS and c.t + c.lead[s] <= self.T]
            if not slots:
                continue
            slots.sort(key=lambda s: (c.lead[s], c.strait[s] and PARAMS["strait_pref"], c.cost[s]))
            first = min(c.lead[s] for s in slots)  # the fastest routes first (air), then the slower ones (sea)
            given = {}  # market -> quantity assigned in earlier passes
            for ps in ([s for s in slots if c.lead[s] == first], [s for s in slots if c.lead[s] > first]):
                if ps:
                    self._pack_pass(c, k, ps, cap_left, used, given, flows)

    def _pack_pass(self, c, k, ps, cap_left, used, given, flows):
        plants = sorted({int(self.tail[s]) for s in ps})
        markets = sorted({int(self.dest[s]) for s in ps})
        need = {}
        for m in markets:
            lead = min(c.lead[s] for s in ps if self.dest[s] == m)
            need[m] = max(0.0, self._need(c, m, k, lead) - given.get(m, 0.0))
        if not any(need[m] > EPS for m in markets):
            return
        idx, n = {}, 2  # nodes: 0 source, 1 sink, then plants and markets, then one gate per shared first edge
        for x in plants + markets:
            idx[x] = n
            n += 1
        by_edge = {}
        for s in ps:
            by_edge.setdefault(int(self.edge[s]), []).append(s)
        gate_of = {}
        for e, ss in by_edge.items():
            if len(ss) > 1:  # lanes that start on one edge share its capacity
                gate_of[e] = n
                n += 1
        fl = Flow(n)
        for o in plants:
            fl.add(0, idx[o], max(0.0, self.stock_of(c, o, k) - used.get((o, k), 0.0)))
        arc, gate_in = {}, {}
        for s in ps:  # in the order of preference: the flow tries earlier arcs first
            o, m, e = int(self.tail[s]), int(self.dest[s]), int(self.edge[s])
            cap = min(c.cap[s], cap_left[e])
            if cap <= EPS:
                continue
            if e in gate_of:
                if (o, e) not in gate_in:
                    gate_in[(o, e)] = fl.add(idx[o], gate_of[e], cap_left[e])
                arc[s] = fl.add(gate_of[e], idx[m], cap)
            else:
                arc[s] = fl.add(idx[o], idx[m], cap)
        sink_arc = {m: fl.add(idx[m], 1, 0.0) for m in markets}
        levels = max(1, int(PARAMS["pack_levels"]))
        for j in range(1, levels + 1):  # needs are filled to the same ratio in each market
            for m in markets:
                fl.raise_to(sink_arc[m], need[m] * j / levels)
            fl.run(0, 1)
        if PARAMS["pack_cover"] > 0:  # what the plants still hold goes on, up to a market's cover
            for s, a in arc.items():
                if c.strait[s]:
                    fl.cap[a] = 0.0  # but not over a strait lane; what it carries so far stays
            for m in markets:
                row = self.demand_row.get((m, k))
                if row is None:
                    continue
                weekly = float(c.forecast[row].mean())
                have = self.stock_of(c, m, k) + self.arrivals(c, m, k, c.t + 50) + given.get(m, 0.0)
                room = min(PARAMS["pack_cover"] * weekly, self.storage.get((m, k), float("inf"))) - have
                fl.raise_to(sink_arc[m], max(need[m], room))
            fl.run(0, 1)
        for s, a in arc.items():
            f = fl.flow(a)
            if f > EPS:
                o, m, e = int(self.tail[s]), int(self.dest[s]), int(self.edge[s])
                flows[s] = f
                cap_left[e] -= f
                used[(o, k)] = used.get((o, k), 0.0) + f
                given[m] = given.get(m, 0.0) + f

    # ------------------------------------------------------------------ raw chips: fab -> plant
    def _outlet(self, c, o, p):
        """Weekly capacity of plant ``o`` to ship the packaged chip ``p`` to markets that want it (alive routes)."""
        total = 0.0
        for s in self.pack_slots.get(p, ()):
            if self.tail[s] == o and c.cap[s] > EPS and c.t + c.lead[s] <= self.T:
                row = self.demand_row.get((int(self.dest[s]), p))
                if row is not None and c.forecast[row].sum() > EPS:
                    cap = c.cap[s]
                    if p in c.cap_before:  # a cheaper chip shares its edges with the dearer ones: it gets what they leave
                        cap = min(cap, c.cap_before[p][int(self.edge[s])])
                    total += cap
        return min(total, self._throughput(c, o))

    def _plant_state(self, c, o, r, lead_in):
        """(room, outlet, backlog) of plant ``o`` for raw chips ``r`` sent now over a route of ``lead_in`` weeks.

        outlet: weekly capacity to ship the packaged chip to markets; backlog: chips already at the plant or on the way
        (packaged, in process, raw); room: how many more it can take. Chips sent now are packaged and can leave after
        ``late`` weeks; by then the outlets have drained the backlog for ``late + raw_extra`` weeks, and what is left plus
        the new chips must fit the plant's storage. A plant with no outlet takes none, nor does one that cannot pass
        them on before the horizon ends.
        """
        p = self.osat[o]["pack"][r]
        outlet = self._outlet(c, o, p)
        lead_out = min((c.lead[s] for s in self.pack_slots.get(p, ()) if self.tail[s] == o and c.cap[s] > EPS), default=None)
        backlog = (
            self.stock_of(c, o, p)
            + self.stock_of(c, o, r)
            + self.arrivals(c, o, r, c.t + 50)
            + sum(book.get(p, 0.0) for book in c.wip.get(o, {}).values())
        )
        if lead_out is None or outlet <= EPS:
            return 0.0, 0.0, backlog
        late = lead_in + self.osat[o]["tau"]
        if PARAMS["endgame"] and self.T - c.t - late - lead_out <= 0:
            return 0.0, outlet, backlog
        drained = (late + PARAMS["raw_extra"]) * outlet
        room = self.storage.get((o, p), float("inf")) - max(0.0, backlog - drained)
        return max(0.0, room), outlet, backlog

    def _raw_flows(self, c, flows):
        for r in self.raw_kinds:
            fabs = [n for n in self.fabs if self.fab[n]["out"] == r]
            for n in fabs:
                for s in self.raw_slots[n]:
                    flows[s] = 0.0
            slots = [s for n in fabs for s in self.raw_slots[n] if c.cap[s] > EPS]
            if not slots:
                continue
            slots.sort(key=lambda s: (c.strait[s] and PARAMS["strait_pref"], c.cost[s], c.lead[s]))
            plants = sorted({int(self.dest[s]) for s in slots})
            sources = sorted({int(self.tail[s]) for s in slots})
            idx, n_ = {}, 2
            for x in sources + plants:
                idx[x] = n_
                n_ += 1
            by_edge = {}
            for s in slots:
                by_edge.setdefault(int(self.edge[s]), []).append(s)
            gate_of = {}
            for e, ss in by_edge.items():
                if len(ss) > 1:
                    gate_of[e] = n_
                    n_ += 1
            fl = Flow(n_)
            for f in sources:
                fl.add(0, idx[f], self.stock_of(c, f, r))
            arc, gate_in = {}, {}
            for s in slots:
                f, o, e = int(self.tail[s]), int(self.dest[s]), int(self.edge[s])
                cap = min(c.cap[s], c.u[e])
                if e in gate_of:
                    if (f, e) not in gate_in:
                        gate_in[(f, e)] = fl.add(idx[f], gate_of[e], c.u[e])
                    arc[s] = fl.add(gate_of[e], idx[o], cap)
                else:
                    arc[s] = fl.add(idx[f], idx[o], cap)
            state, sink = {}, {}
            for o in plants:
                lead_in = min(c.lead[s] for s in slots if self.dest[s] == o)
                state[o] = self._plant_state(c, o, r, lead_in)
                sink[o] = fl.add(idx[o], 1, 0.0)
            # plants are filled in rounds: first up to a short wait (backlog / outlet), then longer ones, then their room
            for wait in list(PARAMS["raw_waits"]) + [None]:
                for o in plants:
                    room, outlet, backlog = state[o]
                    fl.raise_to(sink[o], room if wait is None else min(room, max(0.0, outlet * wait - backlog)))
                fl.run(0, 1)
            for s, a in arc.items():
                flows[s] = fl.flow(a)

    # ------------------------------------------------------------------ what a fab's chips can still become
    def usefulness(self, c):
        """fab node -> share of the fab's nominal capacity whose chips can still reach a market (0..1)."""
        return {f: min(1.0, phi / max(self.fab[f]["cap0"], EPS)) for f, phi in self.sale_rates(c).items()}

    def sale_rates(self, c):
        """fab node -> chips a week that can still go from the fab to markets.

        A max-flow of the fab alone: fab -> plants (alive raw routes; a plant packages at most its throughput) -> markets
        (alive packaged routes; a market takes at most its mean forecast demand). Lanes that start on one edge share it.
        """
        out = {}
        for f in self.fabs:
            r = self.fab[f]["out"]
            rs = [s for s in self.raw_slots[f] if c.cap[s] > EPS]
            plants = sorted({int(self.dest[s]) for s in rs})
            ps = []
            for o in plants:
                p = self.osat[o]["pack"][r]
                ps += [s for s in self.pack_slots.get(p, ()) if self.tail[s] == o and c.cap[s] > EPS and c.t + c.lead[s] <= self.T]
            markets = sorted({int(self.dest[s]) for s in ps})
            if not rs or not ps:
                out[f] = 0.0
                continue
            idx, n = {}, 3  # nodes: 0 source, 1 sink, 2 the fab, then plants (in), markets, plants (out), gates
            for x in plants + markets:
                idx[x] = n
                n += 1
            idx_out = {}
            for o in plants:
                idx_out[o] = n
                n += 1
            gate_of = {}
            for group in (rs, ps):
                by_edge = {}
                for s in group:
                    by_edge.setdefault(int(self.edge[s]), []).append(s)
                for e, ss in by_edge.items():
                    if len(ss) > 1:
                        gate_of[e] = n
                        n += 1
            fl = Flow(n)
            cap0 = self.fab[f]["cap0"]
            fl.add(0, 2, cap0)
            for o in plants:
                fl.add(idx[o], idx_out[o], self._throughput(c, o))
            gate_in = {}
            for src_node, group in ((2, rs), (None, ps)):
                for s in group:
                    a = src_node if src_node is not None else idx_out[int(self.tail[s])]
                    b = idx[int(self.dest[s])]
                    e = int(self.edge[s])
                    cap = min(c.cap[s], c.u[e])
                    if e in gate_of:
                        if (a, e) not in gate_in:
                            gate_in[(a, e)] = fl.add(a, gate_of[e], c.u[e])
                        fl.add(gate_of[e], b, cap)
                    else:
                        fl.add(a, b, cap)
            p = self.osat[plants[0]]["pack"][r]
            for m in markets:
                row = self.demand_row.get((m, p))
                fl.add(idx[m], 1, 0.0 if row is None else float(c.forecast[row].mean()))
            out[f] = fl.run(0, 1)
        return out

    # ------------------------------------------------------------------ wafers: source -> fab
    def _wafer_flows(self, c, flows):
        rate = self.sale_rates(c) if PARAMS["w_useful"] else {}
        for n in self.fabs:
            for s in self.wafer_slots[n]:
                flows[s] = 0.0
            cap_eff = float(c.cap_eff[self.fab_ord[n]])
            inp, tau = self.fab[n]["inp"], self.fab[n]["tau"]
            slots = [s for s in self.wafer_slots[n] if c.cap[s] > EPS]
            if cap_eff <= EPS or not slots:
                continue
            raw = [s for s in self.raw_slots[n] if c.cap[s] > EPS]
            if not raw:
                continue
            # wafers sent now arrive, are started, mature, are shipped, packaged and shipped again: the week of the first sale
            lead_w = min(c.lead[s] for s in slots)
            lead_r = min(c.lead[s] for s in raw)
            tau_o = min(self.osat[int(self.dest[s])]["tau"] for s in raw)
            lead_p = min((c.lead[s] for p in self.pack_slots for s in self.pack_slots[p] if c.cap[s] > EPS), default=1)
            first_sale = c.t + lead_w + tau + 1 + lead_r + tau_o + 1 + lead_p
            if PARAMS["endgame"] and first_sale > self.T:
                continue
            target = min(PARAMS["w_weeks"] * cap_eff, PARAMS["w_fill"] * self.storage.get((n, inp), float("inf")))
            if PARAMS["w_useful"]:  # a fab whose chips can hardly leave gets fewer wafers, so a smaller claim on power
                u = min(1.0, rate[n] / max(self.fab[n]["cap0"], EPS))
                target *= 0.0 if u < PARAMS["w_u_min"] else u
            # next week's stock if this week's starts use every wafer there is (a window of power may be opening)
            on_hand = self.stock_of(c, n, inp)
            started = min(cap_eff, on_hand + self.arrivals(c, n, inp, c.t))
            want = max(0.0, target - (on_hand + self.arrivals(c, n, inp, c.t + 1) - started))
            slots.sort(key=lambda s: (c.strait[s] and PARAMS["strait_pref"], c.lead[s] > 1, c.cost[s], c.lead[s]))
            for s in slots:
                give = min(want, c.cap[s], c.u[int(self.edge[s])])
                if give <= EPS:
                    continue
                flows[s] = give
                want -= give
                if want <= EPS:
                    break
