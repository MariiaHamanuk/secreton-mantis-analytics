"""pull's rules for the fuel, explicit routing for the chip chain (wafers, raw chips, packaged chips).

The fuel slots (lng, crude, nucfuel) are exactly ``pull``'s: ``_pull_flows`` is its loop, unchanged. ``fill_chip_flows``
then overwrites the wafer, raw-chip and packaged-chip entries of ``flows`` with three functions, each filling only its
own slots. The simulator executes a request that fits the edge capacities and the stock exactly as asked, so every
function computes a feasible flow instead of asking for the maximum.

- ``_pack_flows``, packaged chips, plant -> market. A market is asked for what it can still sell: its forecast demand up
  to the week the cargo arrives, minus its stock and the cargo already on the way (cargo queued at a strait counts when
  the strait's throughput will have released it). A small max-flow (plant stock, edge capacities, strait throughput)
  fills the needs of all markets in rounds of equal fill ratio; the dearer chip (chip_le) first, the cheaper (chip_mat)
  on what is left of the shared edges, faster routes first, then routes without a strait, then cheaper ones (tariff,
  freight). What a plant still holds after that goes on, over routes without a strait, up to the market's cover (its
  storage).
- ``_raw_flows``, raw chips, fab -> plant. A plant takes what its storage can hold after its outlets (alive routes to
  markets that want the chip, capped by the plant's throughput; for the cheaper chip only what the dearer one leaves of
  the shared edges this week) have drained what is already there for as long as the new chips need to reach and pass the
  plant; none if it has no outlet or could not pass them on before the horizon ends. Plants are filled in rounds by
  expected wait (chips there / outlets), so a plant with idle outlets comes first; what no plant can take stays at the
  fab, where the fab's own storage keeps it. Fabs whose alive routes reach a single plant are served first
  (``raw_captive``): a fab with a choice of plants must not take the room of one that has none.
- ``_wafer_flows``, wafers, source -> fab. A fab is kept stocked for three weeks of its effective capacity (the stock is
  cheap, a start missed when power returns is not), scaled by ``usefulness``: the share of the fab's capacity whose
  chips can still reach a market. Routes without a strait, lead 1 and cheap ones first; nothing that cannot become a
  sold chip before the horizon ends. The stock is restored each week as if every fab started lots every week; with
  ``PARAMS["stock_fix"]`` (off: as before) the order must also fit the storage, counting only the starts there will be
  (a fab whose grid served no base load last week starts nothing): a dark fab at its storage limit was sent about a
  fifth of its storage every week and the simulator threw it away (about 9 M wafers an episode on Full, 0.4 M on
  Small). With ``PARAMS["pull"]`` (off: the rules above alone) the wafers are also handled as a scarce resource
  where a source's supply is cut (``wafer_sources``: supply below ``scarce_frac`` of its nominal
  supply, true at week 1 in about three episodes of four on Small, and then the stock is all there will be; the base
  draws it all in week 1, pro rata to what every fab asks, and starts the lots while the chain is still full of the
  initial work in process):

  - a request never exceeds what its source holds, and the fabs are served the dearer chip first, those whose grid
    served its base load last week before the others (``power_order``);
  - the stock of a cut source goes only to the fabs of the dearest chip that can still reach a market
    (``reserve_dear``, ``reserve_useful``): the cheaper chip's fabs take nothing from it;
  - what a fab draws from a cut source is capped by ``pull_plan``: two max-flows per chip, the first over the weeks a
    wafer needs to become a sold chip (every capacity a week's capacity times that), selling what the chain already
    holds through the routes that are alive; the second over ``pull_buffer`` weeks, selling what is left of it, and
    then adding production, fab by fab, where an outlet is still idle. A fab's production is what it must still make,
    so wafers wait at the source until the outlets they would feed are about to run dry.

  Ordering the same way from sources that are not cut (``pull_all``), or braking the order on the chain's inventory
  position, saved energy and disposal but lost more in sales: the outlets must be kept full every week, and a chip
  sold is worth several lots. The same holds for ``PARAMS["rate_cap"]`` (off), a fab sent at most that many times its
  sale rate in wafers a week: 1.0 halves the raw chips thrown away on Full and saves 7 TWh of fab energy an episode,
  but sells fewer chips (score against the base: Full +0.0031 and Small +0.0021, without the cap +0.0012 and +0.0044);
  0.8 loses 0.011 on Full and 0.6 loses 0.07. ``PARAMS["cap"]`` is the brake that loses no sale: a fab whose raw-chip
  store is seen overflowing (a balance of last week's stock, the lots that finished and what was shipped gives the
  chips thrown away exactly) is sent fewer wafers by ``cap_gain`` times its recent overflow, never less than
  ``cap_floor`` of what it has been shipping. Its shipments are limited elsewhere, so the lots cut are lots whose
  chips would be thrown away.

``usefulness`` and ``chip_value`` are for the fuel rules: which grid's fabs can make chips that sell, and the value of
the energy they would use.

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
    "raw_extra": 1.0,  # weeks beyond transport and packaging that a plant's outlets drain before new chips are stored
    "raw_arrive": None,  # cargo in transit counts as a plant's backlog only if it arrives within this many weeks after
    # the new chips are packaged (None: all of it, a cargo stuck behind a strait too)
    "raw_captive": True,  # fabs with a single plant to go to are served before fabs with a choice ("tiers": by choices)
    "raw_over": 0.0,  # a plant's storage counts as this much larger when sizing its room (chips are then disposed there)
    "raw_floor": 0.0,  # a plant whose room is spent still takes this many weeks of its outlet (0: off)
    "raw_cap": 0.0,  # a plant takes at most this many weeks of its outlet in a week (0: off)
    "raw_waits": [2.0, 4.0, 8.0, 16.0],  # plants are filled up to these waits (chips there / outlets), shortest first
    # routes through straits
    "strait_pref": True,  # routes without a strait first; a lane through a strait only for what they cannot carry
    "queue_eta": True,  # cargo queued at a strait arrives when its throughput has released what is ahead of it
    # wafers
    "w_weeks": 3.0,  # weeks of effective capacity kept on hand at a fab
    "w_fill": 0.9,  # never more than this share of the wafer storage (above it the stock is thrown away)
    "w_useful": True,  # scale the stock by usefulness
    "w_u_min": 0.02,  # a fab with a smaller share gets no wafers
    # pull: wafers from a source whose supply is cut are ordered only for chips the outlets will still need
    "stock_fix": True,  # a wafer order must fit the storage: a fab whose grid served no base load starts nothing
    "cap": True,  # reactive cap: a fab's wafer order is cut by the rate at which its raw-chip store is seen overflowing
    "cap_gain": 16.0,  # wafers cut per chip a week of the (averaged) overflow
    "cap_win": 6,  # weeks over which the overflow rate is averaged
    "cap_floor": 0.8,  # a capped order is not cut below this share of the fab's recent raw-chip shipping rate
    "cap_sent_win": 6,  # weeks over which that shipping rate is averaged
    "cap_chips": [],  # [] all fabs, else only fabs whose chip's penalty rank is listed (0 = dearest chip)
    "cap_shed_n": 0,  # > 0: only at grids that shed base load in at least this many of the last cap_shed_win weeks
    "cap_shed_win": 13,
    "rate_cap": 0.0,  # > 0: a fab is sent at most this many times its sale rate in wafers a week (0: no cap)
    "pull": True,  # False: the wafer rule above alone
    "pull_buffer": 12.0,  # weeks beyond the time a wafer needs to become a sold chip that the plan looks ahead
    "scarce_frac": 0.5,  # a wafer source supplying less than this share of its nominal supply is cut: its stock is all
    "reserve_dear": True,  # a cut source's stock goes to the fabs of the dearest chip that can take it
    "pull_all": False,  # the plan limits the wafers drawn from every source, not only from those whose supply is cut
    "reserve_useful": True,  # only fabs whose chips can still reach a market keep a claim on a cut source's stock
    "power_order": True,  # a cut source's stock goes first to fabs whose grid served its base load last week
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
            self.fab[n] = dict(
                inp=com_ix[a["input"]],
                out=out,
                pk=pk,
                tau=int(a["tau"]),
                cap0=float(a["cap0"]),
                e=float(a["e"]),
                grid=node_ix.get(a["grid"]),
            )
        self.storage, self.supply_rate = {}, {}
        for n, node in nodes.items():
            for name, d in node.get("stock", {}).items():
                self.storage[(n, com_ix[name])] = float("inf") if d.get("storage") is None else float(d["storage"])
                self.supply_rate[(n, com_ix[name])] = float(d.get("supply_rate") or 0.0)
        self.supply_ix = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["supply_slots"])}
        self.grid_ix = {int(n): i for i, n in enumerate(layout["grids"])}  # grid node -> its place in grid arrays
        self.fab_inputs = {self.fab[n]["inp"] for n in self.fabs}
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
        self.wafer_slots = {
            n: [s for s in range(self.n_slots) if kind[s] == 1 and self.dest[s] == n] for n in self.fabs
        }
        self.raw_slots = {n: [s for s in range(self.n_slots) if kind[s] == 2 and self.tail[s] == n] for n in self.fabs}
        self.pack_slots = {}
        for s in range(self.n_slots):
            if kind[s] == 3:
                self.pack_slots.setdefault(int(self.k[s]), []).append(s)
        self.pack_order = sorted(self.pack_slots, key=lambda k: -pi[k])  # the dearer chip first
        self.raw_kinds = sorted({self.fab[n]["out"] for n in self.fabs})
        self.raw_of = {p: r for o in self.osats for r, p in self.osat[o]["pack"].items()}  # packaged chip -> raw chip
        self.pool_ct = [p == "ct" for p in static["commodities"]["pool"]]  # carried by the container pool
        # Small and Full give the queue at straits as one dense table, Tiny as a padded list of lots
        self.lot_keys = [tuple(int(x) for x in key) for key in layout["lot_keys"]] if layout.get("lot_keys") else None
        self.last_mask = np.ones(self.n_slots, dtype=bool)
        self.last_forecast = None
        self.sent_hist = {n: [] for n in self.fabs}  # fab -> raw chips shipped, week by week
        self.shed_hist = []  # per week, which grids shed base load the week before
        self.ov_prev = None  # (week, stock, lots finishing by week) of the last observation, for the overflow balance
        self.ov_reset = 0  # last week in which the work in process at reset comes out
        self.ov_hist = {n: [] for n in self.fabs}  # fab -> [(week, raw chips found disposed of that week)]

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

        The energy a fab uses is worth its chips' shortage penalty per GWh (penalty / GWh per lot) as long as the chips
        can reach a market; the GWh a week are what the fab can still use that way (its effective capacity times
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
        c.avail = np.asarray(obs["graph_now.supply.avail"], dtype=float)  # supply at each source this week
        c.shed = np.asarray(obs["last_week.shed.qty"], dtype=float)  # base load each grid could not serve last week
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
            if (
                PARAMS["queue_eta"] and self.pool_ct[k] and head in self.chk_ord
            ):  # it will join the queue at that strait
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
        for n, k, q, w in zip(
            np.asarray(obs["wip.node"])[live],
            np.asarray(obs["wip.k"])[live],
            np.asarray(obs["wip.qty"])[live],
            np.asarray(obs["wip.out_week"])[live],
        ):
            book = out.setdefault(int(n), {}).setdefault(int(w), {})
            book[int(k)] = book.get(int(k), 0.0) + float(q)
        return out

    def arrivals(self, c, node, k, by):
        return sum(q for aw, q in c.transit.get((int(node), int(k)), ()) if aw <= by)

    def _throughput(self, c, o):
        """What plant ``o`` can package this week (its effective throughput); unlimited if the observation is silent."""
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
                    if (
                        p in c.cap_before
                    ):  # a cheaper chip shares its edges with the dearer ones: it gets what they leave
                        cap = min(cap, c.cap_before[p][int(self.edge[s])])
                    total += cap
        return min(total, self._throughput(c, o))

    def _plant_state(self, c, o, r, lead_in):
        """(room, outlet, backlog) of plant ``o`` for raw chips ``r`` sent now over a route of ``lead_in`` weeks.

        outlet: weekly capacity to ship the packaged chip to markets; backlog: chips already at the plant or on the way
        (packaged, in process, raw); room: how many more it can take. Chips sent now are packaged and can leave after
        ``late`` weeks; by then the outlets have drained the backlog for ``late + raw_extra`` weeks, and what is left
        plus the new chips must fit the plant's storage. A plant with no outlet takes none, nor does one that cannot
        pass them on before the horizon ends.
        """
        p = self.osat[o]["pack"][r]
        outlet = self._outlet(c, o, p)
        lead_out = min(
            (c.lead[s] for s in self.pack_slots.get(p, ()) if self.tail[s] == o and c.cap[s] > EPS), default=None
        )
        late = lead_in + self.osat[o]["tau"]
        by = c.t + 50 if PARAMS["raw_arrive"] is None else c.t + late + PARAMS["raw_arrive"]
        backlog = (
            self.stock_of(c, o, p)
            + self.stock_of(c, o, r)
            + self.arrivals(c, o, r, by)
            + sum(book.get(p, 0.0) for book in c.wip.get(o, {}).values())
        )
        if lead_out is None or outlet <= EPS:
            return 0.0, 0.0, backlog
        if PARAMS["endgame"] and self.T - c.t - late - lead_out <= 0:
            return 0.0, outlet, backlog
        drained = (late + PARAMS["raw_extra"]) * outlet
        room = self.storage.get((o, p), float("inf")) * (1.0 + PARAMS["raw_over"]) - max(0.0, backlog - drained)
        room = max(0.0, room)
        if PARAMS["raw_floor"] > 0:
            room = max(room, PARAMS["raw_floor"] * outlet)
        if PARAMS["raw_cap"] > 0:
            room = min(room, PARAMS["raw_cap"] * outlet)
        return room, outlet, backlog

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
            reach = {}  # fab -> the plants its alive routes reach
            for s in slots:
                reach.setdefault(int(self.tail[s]), set()).add(int(self.dest[s]))
            captive = [s for s in slots if len(reach[int(self.tail[s])]) == 1]
            if PARAMS["raw_captive"] == "tiers":  # fewest plants to choose from first, one pass per number of choices
                taken = {}
                for d in sorted({len(p) for p in reach.values()}):
                    self._raw_pass(c, r, [s for s in slots if len(reach[int(self.tail[s])]) == d], flows, taken)
            elif PARAMS["raw_captive"] and captive and len(captive) < len(slots):
                # a fab with one plant to go to is served first: a fab with a choice must not take its room
                taken = {}
                self._raw_pass(c, r, captive, flows, taken)
                self._raw_pass(c, r, [s for s in slots if len(reach[int(self.tail[s])]) > 1], flows, taken)
            else:
                self._raw_pass(c, r, slots, flows, {})

    def _raw_pass(self, c, r, slots, flows, taken):
        """Max-flow of the raw chip ``r`` over ``slots`` into the plants' rooms less what ``taken`` already holds."""
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
            room, outlet, backlog = self._plant_state(c, o, r, lead_in)
            held = taken.get(o, 0.0)
            state[o] = (max(0.0, room - held), outlet, backlog + held)
            sink[o] = fl.add(idx[o], 1, 0.0)
        # plants are filled in rounds: up to a short wait (backlog / outlet) first, then longer ones, then the room
        for wait in list(PARAMS["raw_waits"]) + [None]:
            for o in plants:
                room, outlet, backlog = state[o]
                fl.raise_to(sink[o], room if wait is None else min(room, max(0.0, outlet * wait - backlog)))
            fl.run(0, 1)
        for s, a in arc.items():
            flows[s] = fl.flow(a)
            taken[int(self.dest[s])] = taken.get(int(self.dest[s]), 0.0) + flows[s]

    # ------------------------------------------------------------------ what a fab's chips can still become
    def usefulness(self, c):
        """fab node -> share of the fab's nominal capacity whose chips can still reach a market (0..1)."""
        return {f: min(1.0, phi / max(self.fab[f]["cap0"], EPS)) for f, phi in self.sale_rates(c).items()}

    def sale_rates(self, c):
        """fab node -> chips a week that can still go from the fab to markets.

        A max-flow of the fab alone: fab -> plants (alive raw routes; a plant packages at most its throughput) ->
        markets (alive packaged routes; a market takes at most its mean forecast demand). Lanes that start on one edge
        share it.
        """
        out = {}
        for f in self.fabs:
            r = self.fab[f]["out"]
            rs = [s for s in self.raw_slots[f] if c.cap[s] > EPS]
            plants = sorted({int(self.dest[s]) for s in rs})
            ps = []
            for o in plants:
                p = self.osat[o]["pack"][r]
                ps += [
                    s
                    for s in self.pack_slots.get(p, ())
                    if self.tail[s] == o and c.cap[s] > EPS and c.t + c.lead[s] <= self.T
                ]
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

    # ------------------------------------------------------------------ what the chain still has to make
    def _lead_to_sale(self, c, n):
        """Weeks from a wafer sent now to fab ``n`` until its chip is sold; None if a leg has no live route."""
        slots = [s for s in self.wafer_slots[n] if c.cap[s] > EPS]
        raw = [s for s in self.raw_slots[n] if c.cap[s] > EPS]
        packs = [c.lead[s] for p in self.pack_slots for s in self.pack_slots[p] if c.cap[s] > EPS]
        if not slots or not raw or not packs:
            return None
        lead_w = min(c.lead[s] for s in slots)
        lead_r = min(c.lead[s] for s in raw)
        tau_o = min(self.osat[int(self.dest[s])]["tau"] for s in raw)
        return lead_w + self.fab[n]["tau"] + 1 + lead_r + tau_o + 1 + min(packs)

    def pull_plan(self, c):
        """fab -> chips it still has to make so that its outlets are not idle when its new chips would reach them.

        Two max-flows per chip, each over a window of weeks in which every capacity is a week's capacity times the
        window. The first lasts as long as a wafer needs to arrive as a sale: the chips the chain holds (wafers and
        work in process at fabs, raw chips at fabs, on the way to plants and at plants, work in process and packaged
        chips at plants, chips on the way to markets and at markets) are sold through the routes that are alive. What
        is left of them is carried into the second, ``pull_buffer`` weeks long, the weeks in which chips ordered now
        would arrive: there the chips left are sold first, then each fab, the ones with the fewest plants first, adds
        production (its capacity for the window) where an outlet is still idle. What a fab adds is what it must still
        make. An outlet that holds enough chips, or that cannot be reached, asks nothing; a plant whose stock runs dry
        asks its feeders to make as much as it can ship. The dearer chip first; the cheaper one gets what is left of
        shared edges.
        """
        left = c.u.copy()  # edge capacity not yet taken by a dearer chip's routes to markets
        plan = {n: 0.0 for n in self.fabs}
        by = c.t + 50
        for p in self.pack_order:
            r = self.raw_of.get(p)
            lead = {
                n: self._lead_to_sale(c, n)
                for n in self.fabs
                if self.fab[n]["out"] == r and c.cap_eff[self.fab_ord[n]] > EPS
            }
            lead = {n: w for n, w in lead.items() if w is not None}
            if not lead:
                continue
            first, extra = sum(lead.values()) / len(lead), PARAMS["pull_buffer"]
            rs = {n: [s for s in self.raw_slots[n] if c.cap[s] > EPS] for n in lead}
            plants = sorted(
                {int(self.dest[s]) for ss in rs.values() for s in ss}
                | {o for o in self.osats if r in self.osat[o]["pack"] and self._outlet(c, o, p) > EPS}
            )
            ps = [
                s
                for o in plants
                for s in self.pack_slots.get(p, ())
                if self.tail[s] == o and c.cap[s] > EPS and c.t + c.lead[s] <= self.T
            ]
            markets = sorted({int(self.dest[s]) for s in ps} | {m for (m, k) in self.demand_row if k == p})
            idx, n_ = {}, 2  # nodes: 0 source, 1 sink, fabs, plants (in), markets, plants (out), gates (shared edges)
            for x in list(lead) + plants + markets:
                idx[x] = n_
                n_ += 1
            idx_out = {}
            for o in plants:
                idx_out[o] = n_
                n_ += 1
            gate_of = {}
            for group in ([s for ss in rs.values() for s in ss], ps):
                by_edge = {}
                for s in group:
                    by_edge.setdefault(int(self.edge[s]), []).append(s)
                for e, ss in by_edge.items():
                    if len(ss) > 1:
                        gate_of[e] = n_
                        n_ += 1
            order = sorted(
                lead, key=lambda n: (len({int(self.dest[s]) for s in rs[n]}), self.fab[n]["e"], self.fab_ord[n])
            )
            held = {}  # what the chain holds, by node
            for n in lead:
                inp = self.fab[n]["inp"]
                held[idx[n]] = self.stock_of(c, n, inp) + self.arrivals(c, n, inp, by) + self.stock_of(c, n, r)
                held[idx[n]] += sum(book.get(r, 0.0) for book in c.wip.get(n, {}).values())
            for o in plants:
                if r in self.osat[o]["pack"]:
                    held[idx[o]] = self.stock_of(c, o, r) + self.arrivals(c, o, r, by)
                    packaging = sum(book.get(p, 0.0) for book in c.wip.get(o, {}).values())
                    held[idx_out[o]] = self.stock_of(c, o, p) + packaging
            for m in markets:
                held[idx[m]] = self.stock_of(c, m, p) + self.arrivals(c, m, p, by)

            def build(weeks, supply):
                """The network over ``weeks`` weeks, with ``supply`` (node -> chips) from the source."""
                fl = Flow(n_)
                make = {n: fl.add(0, idx[n], 0.0) for n in order}  # what a fab can still make: raised later
                for o in plants:
                    fl.add(idx[o], idx_out[o], self._throughput(c, o) * weeks)
                gate_in = {}
                for n in rs:
                    for s in rs[n]:
                        e, o = int(self.edge[s]), int(self.dest[s])
                        cap = min(c.cap[s], c.u[e]) * weeks
                        if e in gate_of:
                            if (idx[n], e) not in gate_in:
                                gate_in[(idx[n], e)] = fl.add(idx[n], gate_of[e], c.u[e] * weeks)
                            fl.add(gate_of[e], idx[o], cap)
                        else:
                            fl.add(idx[n], idx[o], cap)
                pack_arc = {}
                for s in ps:
                    o, m, e = int(self.tail[s]), int(self.dest[s]), int(self.edge[s])
                    cap = min(c.cap[s], left[e]) * weeks
                    if cap <= EPS:
                        continue
                    if e in gate_of:
                        if (idx_out[o], e) not in gate_in:
                            gate_in[(idx_out[o], e)] = fl.add(idx_out[o], gate_of[e], left[e] * weeks)
                        pack_arc[s] = fl.add(gate_of[e], idx[m], cap)
                    else:
                        pack_arc[s] = fl.add(idx_out[o], idx[m], cap)
                for m in markets:
                    row = self.demand_row.get((m, p))
                    fl.add(idx[m], 1, 0.0 if row is None else float(c.forecast[row].mean()) * weeks)
                give = {node: fl.add(0, node, q) for node, q in supply.items() if q > EPS}
                return fl, make, give, pack_arc

            fa, _make, give, pack_a = build(first, held)
            fa.run(0, 1)
            rest = {node: fa.cap[arc] for node, arc in give.items()}  # what the first window leaves of the chips held
            fb, make, _give, pack_b = build(extra, rest)
            fb.run(0, 1)
            for n in order:  # then what each fab must still make
                fb.raise_to(make[n], float(c.cap_eff[self.fab_ord[n]]) * max(0.0, min(extra, first + extra - lead[n])))
                fb.run(0, 1)
            for n in order:
                plan[n] = fb.flow(make[n])
            for s in pack_a:
                e = int(self.edge[s])
                use = max(fa.flow(pack_a[s]) / first, fb.flow(pack_b[s]) / extra if s in pack_b else 0.0)
                left[e] = max(0.0, left[e] - use)
        return plan

    def wafer_sources(self, c):
        """Wafer source nodes whose supply is cut (below ``scarce_frac`` of nominal): their stock is all there is."""
        finite = set()
        for (n, k), i in self.supply_ix.items():
            if k in self.fab_inputs and self.supply_rate.get((n, k), 0.0) > EPS:
                if float(c.avail[i]) < PARAMS["scarce_frac"] * self.supply_rate[(n, k)]:
                    finite.add(n)
        return finite

    # ------------------------------------------------------------------ wafers: source -> fab
    def _overflow_update(self, c):
        """Record, per fab, the raw chips its store disposed of last week, from a balance of what is observed.

        Week t-1 closed with stock I(t-1) = min(storage, I(t-2) + lots finished - raw chips shipped); the disposal is
        the difference, and it is only nonzero when the store ended at its limit. Lots finished in week t-1 are the
        work in process seen last week with ``out_week == t-1``; raw chips shipped are ``last_week.clip.executed``.
        """
        t = c.t
        done = {}
        for n, book in c.wip.items():
            if n in self.fab:
                done[n] = {w: q.get(self.fab[n]["out"], 0.0) for w, q in book.items()}
        if self.ov_prev is None:
            self.ov_reset = max((w for d in done.values() for w in d), default=0)
        else:
            t0, stock0, done0 = self.ov_prev
            if t0 + 1 == t:
                sent = np.asarray(c.obs["last_week.clip.executed"], dtype=float)
                for n in self.fabs:
                    k = self.fab[n]["out"]
                    i = self.stock_ix[(n, k)]
                    est = 0.0
                    if c.stock[i] >= self.storage[(n, k)] - 1e-6:
                        est = stock0[i] + done0.get(n, {}).get(t - 1, 0.0) - sum(sent[s] for s in self.raw_slots[n])
                        est = max(0.0, est - c.stock[i])
                    self.ov_hist[n].append((t - 1, est))
                    self.sent_hist[n].append(sum(sent[s] for s in self.raw_slots[n]))
        self.ov_prev = (t, c.stock.copy(), done)

    def sent_rate(self, n):
        """Raw chips a week fab ``n`` has been shipping to plants over the last ``cap_sent_win`` weeks."""
        hist = self.sent_hist[n][-PARAMS["cap_sent_win"] :]
        return sum(hist) / len(hist) if hist else 0.0

    def overflow_rate(self, c, n):
        """Chips a week fab ``n``'s store has been disposing of over the last ``cap_win`` weeks (reset lots ignored)."""
        hist = [q for w, q in self.ov_hist[n] if w > self.ov_reset and w > c.t - 1 - PARAMS["cap_win"]]
        return sum(hist) / PARAMS["cap_win"]

    def _cap_applies(self, c, n):
        """Where the cap acts: the chips whose penalty rank is listed (0 = dearest), the grids that shed lately."""
        pk = self.fab[n]["pk"]
        if PARAMS["cap_chips"]:
            pens = sorted({float(p) for p in self.pi[self.sold]}, reverse=True)
            if pk is None or pens.index(float(self.pi[pk])) not in PARAMS["cap_chips"]:
                return False
        if PARAMS["cap_shed_n"] > 0:
            g = self.grid_ix.get(self.fab[n]["grid"])
            if g is None or sum(row[g] for row in self.shed_hist[-PARAMS["cap_shed_win"] :]) < PARAMS["cap_shed_n"]:
                return False
        return True

    def _wafer_flows(self, c, flows):
        self.shed_hist.append(c.shed > EPS)  # base load each grid could not serve, week by week
        self._overflow_update(c)
        rate = self.sale_rates(c) if PARAMS["w_useful"] else {}
        plan = self.pull_plan(c) if PARAMS["pull"] else None
        finite = self.wafer_sources(c) if PARAMS["pull"] else set()
        src_left = {}
        top = {}  # source with its supply cut -> the highest chip penalty among the fabs that can take its wafers
        for n in self.fabs:
            for s in self.wafer_slots[n]:
                flows[s] = 0.0
            pk = self.fab[n]["pk"]
            if PARAMS["reserve_dear"] and pk is not None and c.cap_eff[self.fab_ord[n]] > EPS:
                floor = PARAMS["w_u_min"] * self.fab[n]["cap0"]
                useful = rate.get(n, float("inf")) >= floor or not PARAMS["reserve_useful"]
                if useful and any(c.cap[s] > EPS for s in self.raw_slots[n]):
                    for s in self.wafer_slots[n]:
                        if c.cap[s] > EPS and int(self.tail[s]) in finite:
                            top[int(self.tail[s])] = max(top.get(int(self.tail[s]), 0.0), self.pi[pk])

        def dark(n):  # the fab's grid could not serve its base load last week: no power for lots now
            g = self.grid_ix.get(self.fab[n]["grid"])
            return g is not None and c.shed[g] > EPS

        def value(n):  # minus the penalty of the chip a fab makes: the dearer chip sorts first
            pk = self.fab[n]["pk"]
            return -self.pi[pk] if pk is not None else 0.0

        # the dearer chip's fabs first, those whose grid has power before the others: what a source holds goes to them
        fab_order = self.fabs
        if PARAMS["pull"]:
            fab_order = sorted(self.fabs, key=lambda n: (value(n), PARAMS["power_order"] and dark(n), self.fab_ord[n]))
        for n in fab_order:
            cap_eff = float(c.cap_eff[self.fab_ord[n]])
            inp, tau = self.fab[n]["inp"], self.fab[n]["tau"]
            slots = [s for s in self.wafer_slots[n] if c.cap[s] > EPS]
            if cap_eff <= EPS or not slots:
                continue
            raw = [s for s in self.raw_slots[n] if c.cap[s] > EPS]
            if not raw:
                continue
            # wafers sent now arrive, are started, mature, are shipped, packaged, shipped again: the first sale's week
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
            arriving = self.arrivals(c, n, inp, c.t + 1)
            started = min(cap_eff, on_hand + self.arrivals(c, n, inp, c.t))
            want = max(0.0, target - (on_hand + arriving - started))
            if PARAMS["stock_fix"]:  # what is ordered must fit the storage, counting only the starts there will be
                gone = 0.0 if dark(n) else started  # a fab whose grid served no base load last week starts nothing
                room = self.storage.get((n, inp), float("inf")) - (on_hand + arriving - gone)
                want = min(want, max(0.0, room))
            if PARAMS["cap"]:  # cut the order by what the fab is demonstrably throwing away
                over = self.overflow_rate(c, n)
                if over > 0 and self._cap_applies(c, n):
                    # never below what the fab has been shipping: its outlets must stay busy
                    floor = PARAMS["cap_floor"] * self.sent_rate(n)
                    want = min(want, max(want - PARAMS["cap_gain"] * over, floor))
            if PARAMS["rate_cap"] > 0:  # a fab is sent no more wafers a week than its chips can be sold at
                want = min(want, PARAMS["rate_cap"] * rate[n])
            slots.sort(
                key=lambda s: (
                    int(self.tail[s]) in finite,
                    c.strait[s] and PARAMS["strait_pref"],
                    c.lead[s] > 1,
                    c.cost[s],
                    c.lead[s],
                )
            )
            budget = float("inf")  # wafers this fab may take from sources whose supply is cut: what its outlets need
            if plan is not None and (finite or PARAMS["pull_all"]):
                budget = plan[n]
            pk_n = self.fab[n]["pk"]
            for s in slots:
                src = int(self.tail[s])
                if src in finite and pk_n is not None and self.pi[pk_n] < top.get(src, 0.0) - 1.0:
                    continue  # a dearer chip's fab can still take this scarce stock: it keeps it
                if src not in src_left:
                    src_left[src] = self.stock_of(c, src, inp) if PARAMS["pull"] else float("inf")
                capped = src in finite or PARAMS["pull_all"]
                give = min(want, c.cap[s], c.u[int(self.edge[s])], src_left[src])  # never more than the source holds
                if capped:
                    give = min(give, budget)
                if give <= EPS:
                    continue
                flows[s] = give
                src_left[src] -= give
                want -= give
                if capped:
                    budget -= give
                if want <= EPS:
                    break
