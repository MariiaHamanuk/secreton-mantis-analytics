"""pull's rules everywhere except on fuel; on the fuel slots (lng, crude, nucfuel) a planner that fills grids in time.

Everything that is not a fuel slot is ``pull``'s code, unchanged (``Agent._pull``): value_first on shared routes and
sink_pull into markets. The fuel slots are filled by ``FuelRules.fill(observation, flows)``, which overwrites the fuel
entries of ``flows`` and touches no other entry, so it can sit beside any other rules for the non-fuel slots
(``FuelRules(config, FUEL)`` needs only ``config``). Every number comes from ``config`` and the observation.

What ``FuelRules`` does, in the order of ``fill``:

1. Per grid and fuel: the weekly burn at full output (share x G_bar), the rationing threshold psi x ibar of the
   rationed fuel (lng), the storage of the grid's stock slot. The simulator rations a grid's lng output on LAST week's
   closing stock, and a grid always burns what it holds, so the grid must close every week above the threshold.
2. Terminal-to-grid slots (lead time 0) are the valve between a terminal's stock and the grid. They move stock so that
   the grid closes the week at threshold + ``ss_weeks`` weeks of burn (crude starts at the terminals only: these slots
   must move it every week). With the throttle (``FuelRules._throttle``) the rationed fuel's level is instead
   ``rho`` x threshold: a grid offers its free segment, every fuel's output and the rationed fuel's share x G_bar x
   ration, takes base load first and then what its fabs ask, and runs every segment at the same load factor, so an offer
   above the load burns fuel in place of the free segment. ``rho`` is the ration that makes next week's offer equal
   next week's load (base load + the fabs' ask, which follows from the wafers on hand and on the way; the chip part has
   filled this week's wafer flows before the fuel part runs). In the last weeks, when a lot started cannot be sold, the
   fabs' ask does not count. A terminal fuller than ``grid_buffer`` of its storage passes the surplus on to the
   grid's own storage (any fuel in mode "on"): a second buffer against a lane that closes later; the throttle then
   stands back.
3. Time concentration (``_hold``), for the rationed fuel of grids with fabs whose inflow is below their burn. A grid
   shed-for-a-week-then-complete beats a steady partial burn: the same fuel is burned, and shed is linear in it, but
   only a complete week makes the fab sliver (output above base load, worth far more than VOLL). So such a grid holds
   its fuel at the terminal (hold), moves the threshold to the grid when terminal + grid stock cover the threshold plus
   ``prime_weeks`` weeks of burn (prime), then keeps moving one week's burn a week (run) until the terminal is dry.
   Its other fuels wait at the terminal too while it is off (bank), if they are scarce. Grids that cannot gain (little
   sliver per unit of fuel), and the last ``end_weeks`` weeks (a lot started then is not sold), are always on.
   A fuel that is not rationed but whose weekly burn exceeds the sliver (``FuelRules.gate``: crude at JP and EU on
   Small) gates the fabs the same way, without a threshold: a week's fab energy is the sliver minus every fuel's
   shortfall, so it moves a whole week's burn when terminal and grid hold it and the week can be complete (the other
   fuels cover the load, the fabs ask for power), and nothing otherwise.
4. Orders from the sources, an order-up-to policy on the inventory position (stock at the grid and its terminals plus
   cargo in transit to either, from ``pipeline.*`` whose lane gives the final destination, plus cargo waiting at
   straits from ``queue_lots``), lane by lane (fastest first), never more than the lane's tightest edge and strait
   can carry this week and the source's stock, so the requests fit the simulator's clip and are executed as asked.
   Grids are served in priority order (chip value their fabs make per unit of fuel they burn), in two passes: first
   every grid's essential level (crude before lng; half of it for every grid, then the rest in priority order), then
   a top-up level from what is left, so a grid's buffer never takes supply another grid needs to burn.
5. Nuclear fuel (stocked at the grid, no terminal) is shipped only if stock plus cargo in transit will not last the
   weeks that remain, and then at once (``nuc_early``), not just in time: the grid stores the whole horizon's fuel
   and a lane open today may be closed later (on Full the stock at reset lasts 60 of the 104 weeks; on Small it
   lasts the episode and nothing is shipped).
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = {
    "sink_pull": 1.3,  # margin on a market's forecast demand; None: send the maximum into markets
    "value_first": True,  # the commodity with the higher penalty asks first on a shared route
}
FUEL = {
    "fuel": True,  # False: pull's behaviour on the fuel slots too
    # orders
    "cover_weeks": 0.0,  # weeks of full burn the essential level holds beyond the lead time (first pass)
    "top_weeks": 8.0,  # the top-up level (second pass, from what the first pass left); <= cover_weeks: none
    "share_first": 0.5,  # before the strict-priority pass, every grid gets this share of its essential need
    "crude_first": True,  # in each pass serve every grid's non-rationed fuels (crude) before the rationed fuel
    "queue_frac": 0.15,  # share of the cargo waiting at a strait counted as using this week's throughput
    "queue_horizon": 4.0,  # weeks of throughput a strait's queue counts in the inventory position (rest is stuck)
    "fleet_frac": 1.0,  # share of the tanker/bulk fleet slack (7) duplicate lanes may use (past 1 it scales them)
    "grid_buffer": 0.6,  # a terminal fuller than this share passes the surplus to the grid's storage (None: off)
    "grid_buffer_all": True,  # ... for every fuel with a terminal, not only the rationed one
    "nuc_early": True,  # stocked fuel: ship the whole remaining shortfall as soon as lanes carry it, not just in time
    "nuc_buffer": 6.0,  # stocked fuel: weeks of burn kept beyond the lead time
    "nuc_safety": 1.5,  # stocked fuel: ship at capacity once shortfall x this exceeds what the lanes can still carry
    # what the grid holds
    "ss_weeks": 0.3,  # safety stock of the rationed fuel above its threshold, in weeks of full burn
    "grid_cover_weeks": 0.5,  # non-rationed fuels: closing stock kept at the grid, in weeks of full burn
    # time concentration (FuelRules._hold)
    "pulse": True,  # False: always move the terminal's fuel to the grid as the level asks
    "pulse_min": 0.1,  # only grids with a sliver premium (sliver x chip value over VOLL per base value of fuel) >= this
    "on_ratio": 0.9,  # inflow at or above this share of the full burn: stay on, move everything
    "prime_weeks": 2.0,  # start a stretch once terminal + grid stock cover the threshold plus this many weeks of burn
    "prime_arrivals": True,  # count cargo landing at the terminal this week in that decision
    "run_band": 0.9,  # a grid whose closing stock is at least this share of the threshold counts as primed
    "end_weeks": 16.0,  # no pulsing in the last weeks: a lot started then is not sold before the horizon
    "overflow_frac": 0.92,  # a terminal this full starts a stretch whatever it holds (overflow is disposed of)
    # fuels that gate the sliver (FuelRules.gate): kept at a terminal, not rationed, weekly burn above the sliver
    "gate": True,  # such a fuel goes to the grid in whole weeks: all of a week's burn, or nothing
    "gate_ratio": 1.0,  # ... "above the sliver": weekly burn over this many times the sliver
    "gate_on": 0.98,  # its inflow at or above this share of the burn: no need, move as the level asks
    "gate_ask": 0.2,  # ... nor when the fabs ask less than this share of the sliver this week (nothing to complete)
    "gate_cover": 0.9,  # the week can be complete: with this fuel in full the offer covers this share of the ask
    "bank_crude": True,  # hold the grid's other fuels at the terminal while it holds, if their inflow is scarce
    "bank_ratio": 0.9,  # ... scarce: inflow below this share of their full burn
    # throttle (FuelRules._throttle): the grid is offered the load, not more
    "throttle": True,  # False: the rationed fuel's level is threshold + ss_weeks x burn, as before
    "thr_endgame": True,  # the fabs' ask counts only if a lot started then can still be sold (else base load only)
    "thr_useful": 0.5,  # ... "counts" when this share of the ask is for lots that can still be sold (pro rata sharing)
    "thr_margin": 0.0,  # power offered above base load + the fabs' ask, as a share of that ask
    "thr_floor": 0.0,  # ... and in weeks of the rationed fuel's full burn
    "thr_hold": True,  # the hold / prime / run decision works on the throttled threshold and burn
    "thr_min": 0.0,  # the lowest ration the throttle asks for
    "thr_full": 0.6,  # no lower level while the grid's terminals hold more than this share of their storage (0: always)
    "thr_prime": True,  # a prime week moves what brings the grid to the level after its burn (else: ignores the burn)
    "thr_clip": True,  # next week's wafers: this week's requests as the simulator will clip them (else: as asked)
    "trace": False,  # keep a per-week record in FuelRules.trace (diagnostics)
}
if (HERE / "params.json").is_file():  # a params.json beside this file replaces any of the numbers above
    _override = json.loads((HERE / "params.json").read_text())
    PARAMS |= {k: v for k, v in _override.items() if k in PARAMS}
    FUEL |= {k: v for k, v in _override.items() if k in FUEL}

EPS = 1e-9


class FuelRules:
    """Fills the fuel entries of ``flows`` (see the module docstring)."""

    def __init__(self, config, params):
        self.P = params
        static, layout, inst = config["static"], config["layout"], config["static"]["instance"]
        self.T = int(config["T"])
        self.psi = float(inst["params"]["psi"])
        ids, types = static["nodes"]["id"], static["nodes"]["type"]
        ni = {n: i for i, n in enumerate(ids)}
        ki = {c: i for i, c in enumerate(static["commodities"]["id"])}
        ed, ln, sl = static["edges"], static["lanes"], static["action_slots"]
        self.ed_head = ed["head"]
        self.n_slots = len(sl["edge"])
        self.row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        self.grid_row = {int(n): i for i, n in enumerate(layout["grids"])}
        self.chk_row = {int(n): i for i, n in enumerate(layout["chokepoints"])}
        self.lot_keys = [[int(x) for x in key] for key in layout.get("lot_keys", [])]
        self.pool_of = list(static["commodities"]["pool"])
        self.v = list(static["commodities"]["v"])
        self.u0 = np.array([np.inf if x is None else float(x) for x in ed["u0"]])
        lane_edges = ln["edges"]
        self.lane_dest = [ed["head"][es[-1]] for es in lane_edges]
        self.lane_lead = [float(sum(ed["tau0"][x] for x in es)) for es in lane_edges]
        self.ed_tau = [float(x) for x in ed["tau0"]]

        # ---- grids: fuels, shares, thresholds, storage ------------------------------------------------------------
        nodes_by_id = {n["id"]: n for n in inst["nodes"]}
        self.storage = {  # a stock slot's storage; no limit where the instance gives none
            (ni[n["id"]], ki[c]): (np.inf if s.get("storage") is None else float(s["storage"]))
            for n in inst["nodes"]
            for c, s in n.get("stock", {}).items()
            if c in ki
        }
        self.grids = {}
        for g in layout["grids"]:
            gd = nodes_by_id[ids[g]]["grid"]
            fuels = {ki[c]: float(s) for c, s in gd["shares"].items() if c in ki}
            rationed = ki[gd["rationed"]] if gd.get("rationed") in ki else None
            ibar = {ki[c]: float(x) for c, x in gd.get("ibar", {}).items() if c in ki}
            self.grids[int(g)] = {
                "fuels": fuels,
                "rationed": rationed,
                "thr": {k: self.psi * ibar[k] for k in fuels if k == rationed and k in ibar},
                "deliverable": float(gd["deliverable"]),
            }
        self.fuel = sorted({k for g in self.grids.values() for k in g["fuels"]})
        self.fuel_set = set(self.fuel)
        self.deliverable = np.array([self.grids[int(g)]["deliverable"] for g in layout["grids"]])

        # ---- fuel slots: route, destination, lead time -------------------------------------------------------------
        self.slot = {}
        for s in range(self.n_slots):
            k = int(sl["k"][s])
            if k not in self.fuel_set:
                continue
            e, lane = int(sl["edge"][s]), sl["lane"][s]
            route = list(lane_edges[lane]) if lane is not None else [e]
            self.slot[s] = {
                "k": k,
                "edge": e,
                "route": route,
                "tail": int(ed["tail"][route[0]]),
                "dest": int(ed["head"][route[-1]]),
                "lead": float(sum(ed["tau0"][x] for x in route)),
                "chk": [self.chk_row[c] for c in (ln["chokepoints"][lane] if lane is not None else [])],
                "pool": self.pool_of[k],
                "lane": lane,
            }
        # terminal -> grid slots, and the slots that ship into a grid or into one of its terminals
        self.tg, self.feeders, self.ship = defaultdict(list), defaultdict(set), defaultdict(list)
        for s, d in self.slot.items():
            if d["lane"] is None and d["dest"] in self.grids and types[d["tail"]] == "terminal":
                self.tg[(d["dest"], d["k"])].append(s)
                self.feeders[(d["dest"], d["k"])].add(d["tail"])
        for s, d in self.slot.items():
            if types[d["tail"]] == "terminal":
                continue
            for g, gd in self.grids.items():
                if d["k"] in gd["fuels"] and (d["dest"] == g or d["dest"] in self.feeders[(g, d["k"])]):
                    self.ship[(g, d["k"])].append(s)
        # fuel kept as stock at the grid: not rationed and no terminal (nuclear)
        self.stocked = {
            (g, k) for g, gd in self.grids.items() for k in gd["fuels"] if k != gd["rationed"] and (g, k) not in self.tg
        }

        # ---- grid priority: chip value the fabs make per week at full capacity, per unit of fuel burned -------------
        pi = defaultdict(float)
        for k, p in zip(static["sinks"]["k"], static["sinks"]["pi"]):
            pi[k] = max(pi[k], p)
        packaged = {}
        for n in inst["nodes"]:
            for raw, pk in n.get("osat", {}).get("packages", {}).items():
                packaged[ki[raw]] = ki[pk]
        worth, need = defaultdict(float), defaultdict(float)  # per grid: chip value and GWh of its fabs at capacity
        for n in inst["nodes"]:
            fab = n.get("fab")
            if fab and fab.get("grid") in ni:
                prod = ki[fab["product"]]
                worth[ni[fab["grid"]]] += fab["cap0"] * pi[packaged.get(prod, prod)]
                need[ni[fab["grid"]]] += fab["cap0"] * fab["e"]
        burn = {g: sum(s * gd["deliverable"] for s in gd["fuels"].values()) for g, gd in self.grids.items()}
        self.priority = sorted(self.grids, key=lambda g: -worth[g] / max(burn[g], EPS))
        # premium of running a grid complete rather than short: the fab sliver (output above base load) is worth the
        # chip value per GWh, base load only VOLL; relative to the base value of all the fuel the grid burns
        self.premium, self.sliver = {}, {}
        for g in self.grids:
            grid = nodes_by_id[ids[g]]["grid"]
            sliver = min(max(grid["deliverable"] - grid["base_load"], 0.0), need[g])
            chip = worth[g] / need[g] if need[g] > 0 else 0.0  # USD per GWh the fabs turn into chips
            self.premium[g] = sliver * max(chip - grid["voll"], 0.0) / max(burn[g] * grid["voll"], EPS)
            self.sliver[g] = sliver

        # ---- the grid's free segment, base load, and the fabs on it (for the throttle) -----------------------------
        self.null_share = {  # output that burns no stocked fuel: the shares that name no commodity
            g: sum(s for c, s in nodes_by_id[ids[g]]["grid"]["shares"].items() if c not in ki) for g in self.grids
        }
        self.base_load = np.array([float(nodes_by_id[ids[g]]["grid"]["base_load"]) for g in layout["grids"]])
        n_sl = self.n_slots
        sl_route = [
            list(lane_edges[sl["lane"][s]]) if sl["lane"][s] is not None else [int(sl["edge"][s])] for s in range(n_sl)
        ]
        sl_tail = [int(ed["tail"][r[0]]) for r in sl_route]
        sl_dest = [int(ed["head"][r[-1]]) for r in sl_route]
        sl_lead = [float(sum(ed["tau0"][x] for x in r)) for r in sl_route]
        sl_k = [int(x) for x in sl["k"]]
        osat_tau = {ni[n["id"]]: int(n["osat"]["tau"]) for n in inst["nodes"] if n.get("osat")}
        sink_nodes = {int(n) for n in static["sinks"]["node"]}
        pack_leads = [sl_lead[s] for s in range(n_sl) if sl_tail[s] in osat_tau and sl_dest[s] in sink_nodes]
        lead_p = min(pack_leads, default=1.0)
        self.fabs, self.grid_fabs = {}, defaultdict(list)
        for fo, n in enumerate(layout["fabs"]):
            fab = nodes_by_id[ids[n]]["fab"]
            if fab.get("grid") not in ni:
                continue
            inp, out = ki[fab["input"]], ki[fab["product"]]
            raw = [s for s in range(n_sl) if sl_tail[s] == n and sl_k[s] == out and sl_dest[s] in osat_tau]
            lead_r = min((sl_lead[s] for s in raw), default=1.0)
            tau_o = min((osat_tau[sl_dest[s]] for s in raw), default=2)
            self.fabs[int(n)] = {
                "node": int(n),
                "ord": fo,  # the fab's place in graph_now.fab.*
                "grid": ni[fab["grid"]],
                "e": float(fab["e"]),
                "cap0": float(fab["cap0"]),
                "row": self.row[(int(n), inp)],
                "inp": inp,
                # weeks from the start of a lot to its first sale (as the chip part counts it for wafers)
                "lag": int(fab["tau"]) + 1 + lead_r + tau_o + 1 + lead_p,
                # wafer slots into this fab that arrive within the week they are sent: (slot, lead)
                "wafer_slots": [s for s in range(n_sl) if sl_dest[s] == n and sl_k[s] == inp and sl_lead[s] <= 1.0],
            }
            self.grid_fabs[ni[fab["grid"]]].append(int(n))
        # every slot that carries a fab's input: (slot, first edge, source stock row): the simulator clips them together
        self.wafer_sl = [
            (s, int(sl["edge"][s]), self.row.get((sl_tail[s], sl_k[s])))
            for s in range(n_sl)
            if sl_k[s] in {f["inp"] for f in self.fabs.values()}
        ]

        # ---- fleet slack (7): extra weeks per unit sent on a sea duplicate lane or edge ------------------------------
        self.dtau = {}
        e_alt, l_alt = ed["alt_of"], ln["alt_of"]
        for s, d in self.slot.items():
            lane, e = d["lane"], d["edge"]
            dt = 0.0
            if lane is not None and l_alt[lane] is not None and all(ed["mode"][x] == "sea" for x in lane_edges[lane]):
                ref = l_alt[lane]
                if "lane" in ref:
                    dt = sum(ed["tau0"][x] for x in lane_edges[lane]) - sum(
                        ed["tau0"][x] for x in lane_edges[ref["lane"]]
                    )
                else:
                    r = ref["edge"]
                    off = [x for x in lane_edges[lane] if ed["tail"][x] == ed["tail"][r] and x != r]
                    if off:
                        dt = (
                            sum(ed["tau0"][x] for x in lane_edges[lane][lane_edges[lane].index(off[0]) :])
                            - ed["tau0"][r]
                        )
            elif lane is None and e_alt[e] is not None and ed["mode"][e] == "sea":
                ref = e_alt[e]
                replaced = (
                    sum(ed["tau0"][x] for x in lane_edges[ref["lane"]]) if "lane" in ref else ed["tau0"][ref["edge"]]
                )
                dt = ed["tau0"][e] - replaced
            self.dtau[s] = max(0.0, float(dt))
        fp = inst["params"]
        self.fleet_cap = {p: float(fp["fleet_share"][p]) * float(fp["fleet_measure"][p]) for p in fp["fleet_share"]}
        # a fuel kept at a terminal whose weekly burn exceeds the sliver decides, as the rationed fuel does, whether
        # the fabs get any power: a little short of it and no lot starts
        self.gate = {
            (g, k)
            for g, gd in self.grids.items()
            for k in gd["fuels"]
            if k != gd["rationed"]
            and (g, k) in self.tg
            and self.sliver[g] > EPS
            and gd["fuels"][k] * gd["deliverable"] > params["gate_ratio"] * self.sliver[g]
        }
        self._last = {}
        self.trace = []

    # ------------------------------------------------------------------------------------------------------------
    def _hold(self, g, k, move, s0, landed, thr, b_nom, left_weeks, rate, stock, landing, end=None):
        """Time concentration of the rationed fuel at a grid with fabs: (fuel to move this week, mode).

        The grid burns whatever it holds, and its output of the rationed fuel is share x G_bar x min(1, last week's
        closing stock / threshold). A grid whose inflow is below its burn therefore does better in on and off weeks
        than at a steady partial level: the weeks it is off take the whole shortfall as shed base load (cheap), the
        weeks it is on make the fab sliver (dear). The terminal-to-grid slot is the valve:

        - on: the inflow covers the burn, or too few weeks remain for a lot to be sold: move as the level asks;
        - run: the grid is primed (it closed last week at ``run_band`` of the threshold or more): move as the level
          asks, which keeps it primed while the terminal lasts;
        - hold: not primed and too little upstream: move nothing, the fuel piles up at the terminal;
        - prime: not primed and terminal + grid stock cover the threshold plus ``prime_weeks`` weeks of burn: move up to
          the threshold (the week still produces by last week's low stock; the next one runs).

        With the throttle (``end`` given), ``thr`` and ``b_nom`` are the threshold and the weekly burn at the ration
        that fits next week's load (see ``_throttle``), ``end`` is the level the week closes at, and a prime week moves
        what brings the grid to that level after its own burn (``move``, as in a run week).
        """
        P = self.P
        feeders = self.feeders.get((g, k), ())
        inflow = rate[(g, k)] + sum(rate[(f, k)] for f in feeders)
        useful = left_weeks - P["end_weeks"]  # weeks in which a lot started now is still sold
        if useful <= 1.0 or inflow >= P["on_ratio"] * b_nom:
            return move, "on"
        if s0 >= P["run_band"] * thr:
            return move, "run"
        upstream = sum(stock(f, k) for f in feeders)
        if P["prime_arrivals"]:  # cargo landing at the terminal this week can be moved from next week on
            upstream += sum(landing[(f, k)] for f in feeders)
        weeks = P["prime_weeks"] if useful > P["prime_weeks"] + 2.0 else 1.0
        store = sum(self.storage.get((f, k), np.inf) for f in feeders)
        if upstream + s0 + landed >= thr + weeks * b_nom or upstream >= P["overflow_frac"] * store:
            if end is not None and P["thr_prime"]:  # throttled: the week closes at the level whatever the grid burns
                return move, "prime"
            if end is not None:  # the unthrottled formula, on the throttled level
                return max(0.0, end - s0 - landed), "prime"
            return max(0.0, thr + P["ss_weeks"] * b_nom - s0 - landed), "prime"
        return 0.0, "hold"

    # ------------------------------------------------------------------------------------------------------------
    def _fab_state(self, obs, flows, t, wafer_in, S):
        """fab node -> the numbers behind what it asks of its grid this week and next.

        A fab asks e x min(alpha x cap0, wafers on hand / R) GWh (e per lot, R the restoration factor, alpha the power
        multiplier). Wafers on hand next week = on hand now + arriving now - started now + arriving next week. Arriving
        next week = cargo on the way that lands then (the pipeline's entries for the last leg) + this week's wafer
        flows with a one-week lead, as the simulator will clip them (the chip part has filled them in already). Cargo
        waiting at a strait is not counted: when it is released it lands a week later or more.
        """
        alpha = self._graph(obs, "graph_now.fab.alpha_bar", 1.0)
        R = self._graph(obs, "graph_now.fab.R", 1.0)
        done = (  # what the simulator will make of this week's wafer requests
            self._executed(obs, flows, S) if self.P["thr_clip"] else {s: float(flows[s]) for s, _e, _r in self.wafer_sl}
        )
        out = {}
        for n, f in self.fabs.items():
            o = f["ord"]
            out[n] = {
                "alpha": float(alpha[o]),  # power multiplier: the fab can start alpha x R x cap0 lots
                "R": float(R[o]),
                "W": float(S[f["row"]]),
                "a_now": wafer_in.get((n, t), 0.0),
                "a_next": wafer_in.get((n, t + 1), 0.0) + sum(done.get(s, 0.0) for s in f["wafer_slots"]),
            }
        return out

    def _executed(self, obs, flows, S):
        """slot -> what the simulator executes of this week's wafer requests: each edge's capacity, then the stock of
        the source, shared pro rata by everything that is asked of it (the clip (4)-(5) of the design)."""
        mask = np.asarray(obs["action_mask"]) == 1
        u = self._graph(obs, "graph_now.u", self.u0)
        u = np.where(np.isfinite(u), u, 0.0)
        ask = {s: float(flows[s]) for s, _e, _r in self.wafer_sl if mask[s] and flows[s] > 0}
        by_e = defaultdict(float)
        for s, e, _r in self.wafer_sl:
            if s in ask:
                by_e[e] += ask[s]
        f_edge = {e: min(1.0, max(float(u[e]), 0.0) / tot) for e, tot in by_e.items()}
        by_row = defaultdict(float)
        for s, e, r in self.wafer_sl:
            if s in ask:
                by_row[r] += ask[s] * f_edge[e]
        f_stock = {r: min(1.0, float(S[r]) / tot) if r is not None else 1.0 for r, tot in by_row.items()}
        return {s: ask[s] * f_edge[e] * f_stock[r] for s, e, r in self.wafer_sl if s in ask}

    def _throttle(self, g, t, G, y_bar, stock, landing, arrive_next, fstate):
        """What the grid's rationed fuel has to be: this week's load, and the ration that fits next week's load.

        The grid offers ``g_av`` = free segment + each fuel's output (share x G_bar, times the ration for the rationed
        fuel, but never more than the fuel on hand). Base load goes first, then the fabs' ask; every segment runs at
        ``load = (y + fab energy) / g_av``, so an offer above the load burns fuel in place of the free segment. The
        rationed fuel's output is share x G_bar x min(1, last week's closing stock / threshold): closing the week at
        ``rho x threshold`` makes next week's offer equal next week's load, ``rho`` being what is left of the load after
        the free segment and the other fuels, over the rationed fuel's full output. None if the grid has no valve.
        """
        P, gd = self.P, self.grids[g]
        kr = gd["rationed"]
        thr = gd["thr"].get(kr) if kr is not None else None
        if not thr or (g, kr) not in self.tg:
            return None
        b_r = gd["fuels"][kr] * G  # the rationed fuel's output at full ration
        if b_r <= EPS:
            return None
        null_out = self.null_share[g] * G
        # ---- this week: what the grid offers and what it takes
        s0 = stock(g, kr)
        av_r = min(b_r * min(1.0, s0 / thr), s0 + landing[(g, kr)] + sum(stock(f, kr) for f in self.feeders[(g, kr)]))
        oth = {}
        for k, share in gd["fuels"].items():
            if k != kr:
                have = stock(g, k) + landing[(g, k)] + sum(stock(f, k) for f in self.feeders.get((g, k), ()))
                oth[k] = min(share * G, have)
        g_av = null_out + av_r + sum(oth.values())
        fabs = [(self.fabs[n], fstate[n]) for n in self.grid_fabs.get(g, ())]
        phat = [min(s["alpha"] * s["R"] * f["cap0"], s["W"] + s["a_now"]) for f, s in fabs]
        ask = [f["e"] * p / s["R"] if s["R"] > 0 else 0.0 for (f, s), p in zip(fabs, phat)]
        tot = sum(ask)
        y = min(y_bar, g_av)
        frac = min(1.0, max(g_av - y, 0.0) / tot) if tot > EPS else 0.0  # share of the fabs' ask that gets power
        load = (y + frac * tot) / g_av if g_av > EPS else 0.0
        # ---- next week: the fabs' ask, the other fuels' output, the ration that makes the offer equal the load
        ask_next, useful = 0.0, 0.0
        for (f, s), p in zip(fabs, phat):
            w1 = s["W"] + s["a_now"] - p * frac  # wafers on hand next week before arrivals
            p1 = min(s["alpha"] * s["R"] * f["cap0"], w1 + s["a_next"])
            a1 = f["e"] * p1 / s["R"] if s["R"] > 0 else 0.0
            ask_next += a1
            if t + 1 + f["lag"] <= self.T:
                useful += a1
        offer = ask_next if (not P["thr_endgame"] or useful >= P["thr_useful"] * ask_next) else 0.0
        oth_next = 0.0
        for k, av_k in oth.items():
            feeders = self.feeders.get((g, k), ())
            nxt = (
                stock(g, k)
                + landing[(g, k)]
                + sum(stock(f, k) + landing[(f, k)] for f in feeders)
                - av_k * load
                + arrive_next[(g, k)]
            )
            oth_next += min(gd["fuels"][k] * G, max(nxt, 0.0))
        want = min(y_bar + offer * (1.0 + P["thr_margin"]), G)
        b_nom = gd["fuels"][kr] * gd["deliverable"]
        rho = min(1.0, max(P["thr_min"], (want - null_out - oth_next) / b_r))
        return {
            "rho": rho,
            "thr": rho * thr,
            "b": rho * b_nom,
            "end": rho * thr + P["thr_floor"] * b_nom,
            "burn": av_r * load,
            "load": load,
            "ask": tot,
            "y_bar": y_bar,
            "offer_now": {k: g_av - av_k + gd["fuels"][k] * G for k, av_k in oth.items()},  # with fuel k in full
            "ask_next": ask_next,
            "offer": offer,
            "frac": frac,
            "fabs": [(f["node"], s["W"], s["a_now"], s["a_next"]) for f, s in fabs] if P["trace"] else None,
        }

    def _lots(self, obs):
        """(strait node, k, lane, next edge, quantity) of every group of cargo waiting at a strait."""
        if self.lot_keys:  # small and full: a dense block, one row per key, one column per arrival week
            qty = np.where(np.asarray(obs["queue_lots.qty.observed"]) == 1, obs["queue_lots.qty"], 0.0).sum(axis=1)
            return [(*self.lot_keys[r], float(qty[r])) for r in np.flatnonzero(qty > 0)]
        if "queue_lots.qty" in obs:  # tiny: a padded list of lots
            live = np.asarray(obs["queue_lots.qty.observed"]) == 1
            cols = [obs[f"queue_lots.{n}"][live] for n in ("chokepoint", "k", "lane", "next_edge", "qty")]
            return [(int(c), int(k), int(ln), int(nx), float(q)) for c, k, ln, nx, q in zip(*cols) if q > 0]
        return []

    def _graph(self, obs, name, fallback):
        """A graph_now array as observed; a blackout week (nothing observed) repeats the last one seen."""
        x = np.asarray(obs[name], dtype=float)
        seen = np.asarray(obs[name + ".observed"]) == 1
        if seen.any() or name not in self._last:
            self._last[name] = np.where(seen, x, fallback)
        return self._last[name]

    def fill(self, obs, flows):
        """Overwrite the fuel entries of ``flows``; return extra action entries ({} here).

        ``flows`` should already hold the wafer entries (the throttle reads them to know what the fabs will ask).
        """
        P = self.P
        t = int(obs["week"][0])
        left_weeks = self.T - t + 1
        for s in self.slot:
            flows[s] = 0.0
        S = np.asarray(obs["stock.qty"], dtype=float)
        mask = np.asarray(obs["action_mask"]) == 1
        u = self._graph(obs, "graph_now.u", self.u0)
        u = np.where(np.isfinite(u), u, 0.0)
        kappa = {p: self._graph(obs, f"graph_now.kappa.{p}", 0.0) for p in {self.pool_of[k] for k in self.fuel}}
        c_edge = self._graph(obs, "graph_now.c", 0.0)
        tariff = np.asarray(obs["graph_now.tariff"], dtype=float)
        prohibited = np.asarray(obs["graph_now.prohibited"]) == 1
        G = self._graph(obs, "graph_now.grid.G_bar", self.deliverable)

        # ---- cargo in transit and waiting at straits, by (destination, fuel) --------------------------------------
        transit, landing, rate = defaultdict(float), defaultdict(float), defaultdict(float)
        arrive_next, wafer_in = defaultdict(float), defaultdict(float)
        live = np.asarray(obs["pipeline.qty.observed"]) == 1
        has_lane = np.asarray(obs["pipeline.lane.observed"]) == 1
        for e, k, lane, q, aw, hl in zip(
            obs["pipeline.edge"][live],
            obs["pipeline.k"][live],
            obs["pipeline.lane"][live],
            obs["pipeline.qty"][live],
            obs["pipeline.arrival_week"][live],
            has_lane[live],
        ):
            if int(k) not in self.fuel_set:
                fab = self.fabs.get(self.ed_head[int(e)])
                if fab is not None and fab["inp"] == int(k):  # wafers on their last leg: when they reach the fab
                    wafer_in[(fab["node"], int(aw))] += float(q)
                continue
            dest = self.lane_dest[int(lane)] if hl else self.ed_head[int(e)]
            transit[(dest, int(k))] += float(q)
            # Little's law: cargo in transit over its transit time is the rate it arrives at
            rate[(dest, int(k))] += float(q) / max(self.lane_lead[int(lane)] if hl else self.ed_tau[int(e)], 1.0)
            if self.ed_head[int(e)] == dest and int(aw) == t:
                landing[(dest, int(k))] += float(q)  # arrives this week, before the grids burn
            elif self.ed_head[int(e)] == dest and int(aw) == t + 1:
                arrive_next[(dest, int(k))] += float(q)
        queued, queue_at = defaultdict(float), defaultdict(float)
        for c, k, lane, nxt, q in self._lots(obs):
            if k not in self.fuel_set:
                continue
            queue_at[(self.chk_row[c], self.pool_of[k])] += q
            if not prohibited[nxt, k]:  # a lot behind a prohibited edge releases nothing: not on its way
                queued[(self.lane_dest[lane], k, c)] += q  # (cargo waiting is a backlog, not an inflow rate)
        # the queue counts in the position only as far as the strait can pass it within queue_horizon weeks
        q_in = defaultdict(float)
        for (dest, k, c), q in queued.items():
            cap = max(float(kappa[self.pool_of[k]][self.chk_row[c]]), 0.0) * P["queue_horizon"]
            q_in[(dest, k)] += q * min(1.0, cap / max(queue_at[(self.chk_row[c], self.pool_of[k])], EPS))

        # ---- capacities left this week ---------------------------------------------------------------------------
        edge_left = u.copy()
        kap_left = {
            p: np.maximum(
                kappa[p] - P["queue_frac"] * np.array([queue_at.get((r, p), 0.0) for r in range(len(kappa[p]))]), 0.0
            )
            for p in kappa
        }
        src_left = {}
        fleet_left = {p: P["fleet_frac"] * c for p, c in self.fleet_cap.items()}

        def stock(n, k):
            return float(S[self.row[(n, k)]]) if (n, k) in self.row else 0.0

        def room(s):
            """What slot s can still carry this week: tightest edge and strait, the source's stock, the fleet slack."""
            d = self.slot[s]
            r = min(edge_left[x] for x in d["route"])
            for c in d["chk"]:
                r = min(r, kap_left[d["pool"]][c])
            r = min(r, src_left.setdefault((d["tail"], d["k"]), stock(d["tail"], d["k"]) * (1 - 1e-9)))
            if self.dtau[s] > 0:
                r = min(r, fleet_left.get(d["pool"], np.inf) / self.dtau[s])
            return max(r, 0.0)

        def take(s, q):
            d = self.slot[s]
            for x in d["route"]:
                edge_left[x] -= q
            for c in d["chk"]:
                kap_left[d["pool"]][c] -= q
            src_left[(d["tail"], d["k"])] -= q
            if self.dtau[s] > 0:
                fleet_left[d["pool"]] -= self.dtau[s] * q
            flows[s] += q

        def lane_cost(s):
            d = self.slot[s]
            return sum(c_edge[x] + tariff[x, d["k"]] * self.v[d["k"]] for x in d["route"])

        def eff_lead(s):
            """Weeks until cargo sent on slot s can be burned: transit, wait at its straits, a week at a terminal."""
            d = self.slot[s]
            wait = sum(queue_at.get((c, d["pool"]), 0.0) / max(float(kappa[d["pool"]][c]), 1.0) for c in d["chk"])
            return d["lead"] + wait + (1.0 if d["dest"] not in self.grids else 0.0)

        # ---- terminal -> grid: close the week at the level --------------------------------------------------------
        plan = {}
        y_bar = self._graph(obs, "graph_now.grid.y_bar", self.base_load)
        fstate = self._fab_state(obs, flows, t, wafer_in, S) if P["throttle"] else {}
        for g in self.priority:
            gd = self.grids[g]
            grid_mode = "on"
            th = None
            if P["throttle"]:  # the ration next week's load needs, and this week's load
                row = self.grid_row[g]
                th = self._throttle(g, t, float(G[row]), float(y_bar[row]), stock, landing, arrive_next, fstate)
            for k in sorted(gd["fuels"], key=lambda k: (k != gd["rationed"], k)):  # the rationed fuel decides the mode
                share = gd["fuels"][k]
                b_now = share * float(G[self.grid_row[g]])
                b_nom = share * gd["deliverable"]
                s0 = stock(g, k)
                thr = gd["thr"].get(k)
                ration = min(1.0, s0 / thr) if thr else 1.0
                burn = b_now * ration
                store_g = self.storage.get((g, k), np.inf)
                end = (thr + P["ss_weeks"] * b_nom) if thr else P["grid_cover_weeks"] * b_nom
                end = min(end, 0.98 * store_g)
                if th is not None and k == gd["rationed"]:  # the throttled level, and the burn the grid will make
                    feeders_k = self.feeders.get((g, k), ())  # terminals more than thr_full full: the fuel is plentiful
                    store_f = sum(self.storage.get((f, k), np.inf) for f in feeders_k)
                    plenty = P["thr_full"] and sum(stock(f, k) for f in feeders_k) > P["thr_full"] * store_f
                    end, burn = (end if plenty else min(th["end"], 0.98 * store_g)), th["burn"]  # plenty: base level
                plan[(g, k)] = {"b_nom": b_nom, "s0": s0, "thr": thr, "tg": 0.0, "ship": 0.0, "mode": "on", "th": th}
                if (g, k) not in self.tg:
                    continue
                move = max(0.0, end + burn - s0 - landing[(g, k)])
                if thr and k == gd["rationed"] and P["pulse"] and self.premium[g] >= P["pulse_min"]:
                    if th is not None and P["thr_hold"]:
                        move, grid_mode = self._hold(
                            g, k, move, s0, landing[(g, k)], th["thr"], th["b"], left_weeks, rate, stock, landing, end
                        )
                    else:
                        move, grid_mode = self._hold(
                            g, k, move, s0, landing[(g, k)], thr, b_nom, left_weeks, rate, stock, landing
                        )
                    plan[(g, k)]["mode"] = grid_mode
                elif P["gate"] and (g, k) in self.gate and left_weeks - P["end_weeks"] > 1.0:
                    # a week is complete in this fuel or it is not: move a whole week's burn when the terminal and
                    # the grid hold it and the week can be complete, else keep it for a week that can be
                    feeders = self.feeders.get((g, k), ())
                    inflow = rate[(g, k)] + sum(rate[(f, k)] for f in feeders)
                    full = sum(self.storage.get((f, k), np.inf) for f in feeders)
                    held = sum(stock(f, k) for f in feeders)
                    kr = gd["rationed"]
                    asked, can = True, True  # the fabs ask for power; the other fuels let the week be complete
                    if th is not None:
                        asked = th["ask"] >= P["gate_ask"] * self.sliver[g]
                        can = th["offer_now"][k] >= th["y_bar"] + P["gate_cover"] * th["ask"]
                    elif kr is not None and gd["thr"].get(kr):
                        on_hand = stock(g, kr) + landing[(g, kr)] + plan[(g, kr)]["tg"]
                        can = stock(g, kr) >= gd["thr"][kr] and on_hand >= gd["fuels"][kr] * float(G[self.grid_row[g]])
                    if held >= P["overflow_frac"] * full or inflow >= P["gate_on"] * b_nom or not asked:
                        pass  # plentiful, about to overflow, or nothing to complete: as the level asks
                    elif not can:
                        move, plan[(g, k)]["mode"] = 0.0, "bank"
                    elif s0 + landing[(g, k)] + held >= b_now:
                        move, plan[(g, k)]["mode"] = max(0.0, b_now - s0 - landing[(g, k)]), "crun"
                    else:
                        move, plan[(g, k)]["mode"] = 0.0, "chold"
                elif P["bank_crude"] and grid_mode in ("hold", "prime") and k != gd["rationed"]:
                    # the other fuels wait at the terminal too while the grid is off, if they are scarce: they are
                    # worth most in the weeks the grid runs (they are part of the fab sliver there)
                    feeders = self.feeders.get((g, k), ())
                    inflow = rate[(g, k)] + sum(rate[(f, k)] for f in feeders)
                    full = sum(self.storage.get((f, k), np.inf) for f in feeders)
                    held = sum(stock(f, k) for f in feeders)
                    if inflow < P["bank_ratio"] * b_nom and held < P["overflow_frac"] * full:
                        move, plan[(g, k)]["mode"] = 0.0, "bank"
                buffered = (thr and k == gd["rationed"]) or P["grid_buffer_all"]
                if P["grid_buffer"] is not None and buffered and plan[(g, k)]["mode"] == "on":
                    # a terminal that fills up passes its surplus on: the grid's own storage is a second buffer
                    # against a lane that closes later (the grid burns what the load takes, never the surplus)
                    feeders = self.feeders.get((g, k), ())
                    full = sum(self.storage.get((f, k), np.inf) for f in feeders)
                    held = sum(stock(f, k) for f in feeders)
                    if np.isfinite(full) and held > P["grid_buffer"] * full:
                        room_g = 0.9 * store_g - s0 - landing[(g, k)] + burn
                        move = max(move, min(held - P["grid_buffer"] * full, room_g))
                move = min(move, max(0.0, store_g - s0 - landing[(g, k)] + burn))
                for s in self.tg[(g, k)]:
                    if not mask[s] or move <= 0:
                        continue
                    q = min(move, room(s))
                    take(s, q)
                    move -= q
                    plan[(g, k)]["tg"] += q

        # ---- sources -> terminal or grid: order up to the level, grids in priority order ---------------------------
        # Two passes: first every grid's essential level (cover_weeks), then, from the capacity and stock that are
        # left, the top-up level (top_weeks) that lets a grid accumulate ahead of a run. A higher-priority grid's
        # buffer therefore never takes supply that a lower-priority grid needs to burn.
        position = {}
        for (g, k), pl in plan.items():
            feeders = self.feeders.get((g, k), ())
            position[(g, k)] = (
                pl["s0"]
                + sum(stock(f, k) for f in feeders)
                + sum(transit[(n, k)] + q_in[(n, k)] for n in (g, *feeders))
            )
        rank = {g: i for i, g in enumerate(self.priority)}
        # stages: share_first of every grid's essential need, the rest of it (strict priority), then the top-up level
        stages = [
            (P["cover_weeks"], P["share_first"]),
            (P["cover_weeks"], 1.0),
            (max(P["top_weeks"], P["cover_weeks"]), 1.0),
        ]
        for cover, frac in stages if P["share_first"] < 1.0 else stages[1:]:
            # the rationed fuel first inside a grid, or (crude_first) every grid's other fuels before the rationed ones
            if P["crude_first"]:
                turns = sorted(plan, key=lambda gk: (gk[1] == self.grids[gk[0]]["rationed"], rank[gk[0]], gk[1]))
            else:
                turns = sorted(plan, key=lambda gk: (rank[gk[0]], gk[1] != self.grids[gk[0]]["rationed"], gk[1]))
            for g, k in turns:
                pl = plan[(g, k)]
                stocked = (g, k) in self.stocked
                if stocked and (cover != P["cover_weeks"] or frac < 1.0):
                    continue
                usable = [s for s in self.ship.get((g, k), []) if mask[s] and room(s) > EPS]
                if not usable:
                    continue
                lead = max(eff_lead(s) for s in usable)
                if stocked:
                    # last the weeks that remain: routine order-up-to, or at capacity once the lanes run out of time
                    routine = pl["b_nom"] * min(float(left_weeks), lead + P["nuc_buffer"])
                    short = pl["b_nom"] * left_weeks - position[(g, k)]
                    carry = sum(room(s) for s in usable) * max(left_weeks - min(eff_lead(s) for s in usable) - 1.0, 0.0)
                    need = max(routine - position[(g, k)], short if short * P["nuc_safety"] >= carry else 0.0)
                    if P["nuc_early"]:
                        # a lane open today may be closed when the stock runs low, and the grid stores the whole
                        # horizon's fuel: bring what the remaining weeks will burn as soon as the lanes carry it
                        need = max(need, short)
                else:
                    base = (
                        (pl["thr"] + P["ss_weeks"] * pl["b_nom"]) if pl["thr"] else P["grid_cover_weeks"] * pl["b_nom"]
                    )
                    need = base + pl["b_nom"] * min(float(left_weeks), lead + cover) - position[(g, k)]
                need = max(need, 0.0) * frac
                for s in sorted(usable, key=lambda s: (round(eff_lead(s), 3), lane_cost(s))):  # fastest lane first
                    if need <= EPS:
                        break
                    q = min(need, room(s))
                    if q > EPS:
                        take(s, q)
                        need -= q
                        pl["ship"] += q
                        position[(g, k)] += q
        if P["trace"]:
            for (g, k), pl in plan.items():
                feeders = self.feeders.get((g, k), ())
                self.trace.append(
                    {
                        "week": t,
                        "grid": g,
                        "k": k,
                        "mode": pl["mode"],
                        "s0": round(pl["s0"]),
                        "term": round(sum(stock(f, k) for f in feeders)),
                        "transit": round(sum(transit[(n, k)] for n in (g, *feeders))),
                        "position": round(position[(g, k)]),
                        "ship": round(pl["ship"]),
                        "tg": round(pl["tg"]),
                        "thr": round(pl["thr"] or 0),
                        "b": round(pl["b_nom"]),
                        **(
                            {n: round(v, 3) if isinstance(v, float) else v for n, v in pl["th"].items()}
                            if pl["th"] and k == self.grids[g]["rationed"]
                            else {}
                        ),
                    }
                )
        return {}


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        self.edge = np.array(slots["edge"])
        self.k = np.array(slots["k"])
        route = [lanes["edges"][lane] if lane is not None else [e] for e, lane in zip(slots["edge"], slots["lane"])]
        self.dest = np.array([edges["head"][r[-1]] for r in route])  # the node a slot's cargo ends at
        self.lead = np.array([sum(edges["tau0"][e] for e in r) for r in route])
        # the shortage penalty per commodity: the largest over the markets that want it; 0 for an input
        pi = np.zeros(len(static["commodities"]["id"]))
        for k, p in zip(static["sinks"]["k"], static["sinks"]["pi"]):
            pi[k] = max(pi[k], p)
        self.order = sorted(set(self.k.tolist()), key=lambda k: -pi[k])
        self.sold = pi > 0  # a commodity some market pays for; inputs share a route as send-the-maximum does
        self.demand_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["demands"])}
        self.horizon = config["spaces"]["observation"]["demand_forecast.qty"]["shape"][1]
        self.fuel = FuelRules(config, FUEL) if FUEL["fuel"] else None

    def act(self, observation):
        flows = self._pull(observation)
        extra = self.fuel.fill(observation, flows) if self.fuel is not None else {}
        return {"flows": flows, **extra}

    def _pull(self, observation):
        """pull's flows for every slot (the fuel entries are overwritten by FuelRules.fill)."""
        mask = observation["action_mask"] == 1
        forecast = observation["demand_forecast.qty"]
        left = np.asarray(observation["graph_now.u"], dtype=float).copy()  # capacity not yet asked for, per edge
        left[~np.isfinite(left)] = 0.0
        flows = np.zeros(len(self.edge))
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
                    wanted = PARAMS["sink_pull"] * forecast[row, np.minimum(self.lead[mine][into], self.horizon - 1)]
                    total = room[into].sum()
                    ask[into] = np.minimum(room[into], wanted * room[into] / total) if total > 0 else 0.0
            flows[mine] = ask
            if PARAMS["value_first"] and self.sold[k]:
                np.subtract.at(left, self.edge[mine], ask)
                np.maximum(left, 0.0, out=left)
        return flows
