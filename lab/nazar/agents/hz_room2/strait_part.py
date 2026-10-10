"""Tanker cargo waiting at a strait behind a throttled out-edge is released onto another out-edge that has room.

By default a strait releases each queued lot onto the next edge of the lane it was dispatched on, within that edge's
capacity and the strait's throughput. When that edge's capacity is cut, the cargo already on its way waits there for
weeks, sometimes to the end, while another edge out of the same strait is idle. For tanker cargo the agent may set
the release itself: ``release_mode`` 1 for a (strait, commodity) pair and a quantity per override slot (an out-edge
and the lane the cargo continues on). The simulator then takes the oldest cargo of that commodity, whatever lane it
came on, and turns the default release of the pair off for the week.

``StraitRules.fill(observation)`` returns ``override_qty`` and ``release_mode``. A pair is switched to override only
in a week when that adds flow: the strait has throughput to spare, cargo of the pair would be left waiting, another
out-edge has capacity to spare, what lies beyond that edge can pass it on, and the destination wants the fuel. Every
slot of the pair then asks what the default release would have sent on its own lane, plus the extra; the requests fit
the edge capacities, the queue and the throughput, so they are executed as asked. Duplicate sea routes (east,
lombok, cape, turn-back) share the pool's fleet slack with everything else released on them that week; the extra
stays within what is left of it.

Every number is read from ``config`` and the observation.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


STRAIT = {
    "strait": True,  # False: the default release everywhere
    "need_weeks": 8.0,  # a destination wants fuel while its grid, terminals and cargo on the way hold less than this
    "room_frac": 0.9,  # never fill a terminal beyond this share of its storage
    "min_extra": 0.02,  # switch a pair to override only for an extra of at least this share of a week's burn
    "strait_fleet": 0.95,  # share of the fleet slack the releases may use, the extra included
    "trace": False,
}
HERE = Path(__file__).resolve().parent
if (HERE / "params.json").is_file():  # a params.json beside this file replaces any of the numbers above
    STRAIT |= {k: v for k, v in json.loads((HERE / "params.json").read_text()).items() if k in STRAIT}
EPS = 1e-9


class StraitRules:
    def __init__(self, config, params=None):
        self.P = dict(STRAIT, **(params or {}))
        static, layout, inst = config["static"], config["layout"], config["static"]["instance"]
        ed, ln, ov = static["edges"], static["lanes"], static["override_slots"]
        self.n_ov = len(ov["chokepoint"])
        self.pairs = [(int(c), int(k)) for c, k in layout.get("release_pairs", [])]
        self.pair_row = {p: i for i, p in enumerate(self.pairs)}
        self.slots = [
            (int(c), int(k), int(e), None if lane is None else int(lane))
            for c, k, e, lane in zip(ov["chokepoint"], ov["k"], ov["out_edge"], ov["lane"])
        ]
        self.chk_row = {int(n): i for i, n in enumerate(layout["chokepoints"])}
        self.pool_of = list(static["commodities"]["pool"])
        self.head, self.tail = [int(x) for x in ed["head"]], [int(x) for x in ed["tail"]]
        self.tau0 = [float(x) for x in ed["tau0"]]
        self.u0 = np.array([np.inf if x is None else float(x) for x in ed["u0"]])
        self.lane_edges = [[int(x) for x in es] for es in ln["edges"]]
        self.lot_keys = [[int(x) for x in key] for key in layout.get("lot_keys") or []]
        self.row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        ids, types = static["nodes"]["id"], static["nodes"]["type"]
        ni = {n: i for i, n in enumerate(ids)}
        ki = {c: i for i, c in enumerate(static["commodities"]["id"])}
        nodes_by_id = {n["id"]: n for n in inst["nodes"]}
        self.storage = {
            (ni[n["id"]], ki[c]): (np.inf if s.get("storage") is None else float(s["storage"]))
            for n in inst["nodes"]
            for c, s in n.get("stock", {}).items()
            if c in ki
        }
        # grids: weekly burn of each fuel at full output; the terminals that feed a grid
        self.burn = {}
        for g in layout["grids"]:
            gd = nodes_by_id[ids[g]]["grid"]
            for c, share in gd["shares"].items():
                if c in ki:
                    self.burn[(int(g), ki[c])] = float(share) * float(gd["deliverable"])
        self.fed = defaultdict(set)  # (terminal or grid node, k) -> grids it supplies
        for g, k in self.burn:
            self.fed[(g, k)].add(g)
        for e in range(len(self.head)):
            if types[self.tail[e]] == "terminal":
                for k in ed["K"][e]:
                    if (self.head[e], int(k)) in self.burn:
                        self.fed[(self.tail[e], int(k))].add(self.head[e])
        self.feeders = defaultdict(set)  # (grid, k) -> terminals
        for (n, k), gs in self.fed.items():
            for g in gs:
                if n != g:
                    self.feeders[(g, k)].add(n)
        # extra weeks of a duplicate sea edge (the fleet slack counts them): edges a duplicate lane leaves its route by
        self.dtau = {}
        for lane, ref in enumerate(ln["alt_of"]):
            if ref is None or "lane" not in ref:
                continue
            base = set(self.lane_edges[ref["lane"]])
            off = [x for x in self.lane_edges[lane] if x not in base]
            if off and ed["mode"][off[0]] == "sea":
                d = sum(self.tau0[x] for x in self.lane_edges[lane]) - sum(
                    self.tau0[x] for x in self.lane_edges[ref["lane"]]
                )
                if d > 0:
                    self.dtau[off[0]] = max(self.dtau.get(off[0], 0.0), d)
        for e, ref in enumerate(ed["alt_of"]):
            if ref is not None and ed["mode"][e] == "sea":
                rep = (
                    sum(self.tau0[x] for x in self.lane_edges[ref["lane"]]) if "lane" in ref else self.tau0[ref["edge"]]
                )
                if self.tau0[e] - rep > 0:
                    self.dtau[e] = max(self.dtau.get(e, 0.0), self.tau0[e] - rep)
        fp = inst["params"]
        self.fleet_cap = {p: float(fp["fleet_share"][p]) * float(fp["fleet_measure"][p]) for p in fp["fleet_share"]}
        self.T = int(config["T"])
        self.trace = []
        self._last = {}

    def _graph(self, obs, name, fallback):
        x = np.asarray(obs[name], dtype=float)
        seen = np.asarray(obs[name + ".observed"]) == 1
        if seen.any() or name not in self._last:
            self._last[name] = np.where(seen, x, fallback)
        return self._last[name]

    def fill(self, obs):
        """{"override_qty": ..., "release_mode": ...}: zeros (the default release) unless an override adds flow."""
        qty = np.zeros(self.n_ov)
        mode = np.zeros(len(self.pairs), dtype=np.int64)
        out = {"override_qty": qty, "release_mode": mode}
        P = self.P
        if not P["strait"] or not self.lot_keys or not self.slots:
            return out
        if "override_mask.observed" in obs and int(np.asarray(obs["override_mask.observed"])[0]) != 1:
            return out  # a blackout week: leave the default release
        t = int(obs["week"][0])
        omask = np.asarray(obs["override_mask"]) == 1
        S = np.asarray(obs["stock.qty"], dtype=float)
        u = self._graph(obs, "graph_now.u", self.u0)
        u = np.where(np.isfinite(u), u, 0.0)
        prohibited = np.asarray(obs["graph_now.prohibited"]) == 1
        pools = sorted({self.pool_of[k] for _c, k, _e, _l in self.slots})
        kappa = {p: self._graph(obs, f"graph_now.kappa.{p}", 0.0) for p in pools}
        fuels = sorted({k for _c, k, _e, _l in self.slots})

        # ---- the lot book as the strait will see it this week: waiting cargo and this week's arrivals ----------------
        book = defaultdict(float)  # (strait, k, lane, next edge) -> quantity
        q_obs = np.where(np.asarray(obs["queue_lots.qty.observed"]) == 1, obs["queue_lots.qty"], 0.0).sum(axis=1)
        for r in np.flatnonzero(q_obs > 0):
            c, k, lane, nxt = self.lot_keys[r]
            if k in fuels:
                book[(c, k, lane, nxt)] += float(q_obs[r])
        transit = defaultdict(float)  # (destination node, k) -> cargo on its way (final destination of its lane)
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
            e, k, q = int(e), int(k), float(q)
            if k not in fuels:
                continue
            if hl:
                es = self.lane_edges[int(lane)]
                transit[(self.head[es[-1]], k)] += q
                if int(aw) == t and self.head[e] in self.chk_row and e in es and es.index(e) + 1 < len(es):
                    book[(self.head[e], k, int(lane), es[es.index(e) + 1])] += q  # joins the book before the release
            else:
                transit[(self.head[e], k)] += q
        if not book:
            return out
        queued_to = defaultdict(float)  # (destination node, k) -> cargo waiting at straits on a lane to it
        for (c, k, lane, _nxt), q in book.items():
            queued_to[(self.head[self.lane_edges[lane][-1]], k)] += q

        # ---- what the default release sends: per out-edge within its capacity, then within the strait's throughput --
        on_edge = defaultdict(float)  # (strait, pool, edge) -> cargo of the pool queued for it (not prohibited)
        for (c, k, _lane, nxt), q in book.items():
            if not prohibited[nxt, k]:
                on_edge[(c, self.pool_of[k], nxt)] += q
        rel = {}  # (strait, pool, edge) -> default release
        spare_kap = {}
        for (c, p, e), q in on_edge.items():
            rel[(c, p, e)] = min(q, float(u[e]))
        for c in {c for c, _p, _e in rel}:
            for p in pools:
                tot = sum(x for (cc, pp, _e), x in rel.items() if cc == c and pp == p)
                kap = max(float(kappa[p][self.chk_row[c]]), 0.0)
                if tot > kap:
                    for key in [key for key in rel if key[0] == c and key[1] == p]:
                        rel[key] *= kap / max(tot, EPS)
                    tot = kap
                spare_kap[(c, p)] = kap - tot
        used = defaultdict(float)  # edge -> what the default release sends on it
        for (_c, _p, e), x in rel.items():
            used[e] += x
        fleet_left = {p: P["strait_fleet"] * self.fleet_cap.get(p, np.inf) for p in pools}
        for (_c, p, e), x in rel.items():
            fleet_left[p] -= self.dtau.get(e, 0.0) * x

        # ---- what each destination still wants -----------------------------------------------------------------------
        def stock(n, k):
            return float(S[self.row[(n, k)]]) if (n, k) in self.row else 0.0

        want = {}
        for (g, k), b in self.burn.items():
            if k not in fuels:
                continue
            nodes = (g, *self.feeders.get((g, k), ()))
            have = sum(stock(n, k) + transit[(n, k)] for n in nodes)
            want[(g, k)] = max(0.0, min(P["need_weeks"], self.T - t) * b - have)
        room = {}  # (terminal or grid, k) -> storage left after what is there and on its way

        def room_of(n, k):
            if (n, k) not in room:
                full = P["room_frac"] * self.storage.get((n, k), np.inf)
                room[(n, k)] = max(0.0, full - stock(n, k) - transit[(n, k)])
            return room[(n, k)]

        def beyond(c, k, e, lane):
            """What the route after out-edge ``e`` can pass on this week: later straits' spare throughput and edges."""
            r = np.inf
            if lane is None:
                return r
            es = self.lane_edges[lane]
            for x in es[es.index(e) + 1 :] if e in es else ():
                r = min(r, float(u[x]) - used[x])
                c2 = self.tail[x]
                if c2 in self.chk_row:
                    p = self.pool_of[k]
                    waiting = sum(q for (cc, pp, _e), q in on_edge.items() if cc == c2 and pp == p)
                    r = min(r, float(kappa[p][self.chk_row[c2]]) - waiting)
            return max(r, 0.0)

        # ---- pairs with cargo left waiting: the extra, the dearest fuel (least burned per week) first ---------------
        by_pair = defaultdict(float)
        for (c, k, _lane, _nxt), q in book.items():
            by_pair[(c, k)] += q
        scale = {k: min((b for (g, kk), b in self.burn.items() if kk == k), default=1.0) for k in fuels}
        for c, k in sorted(by_pair, key=lambda ck: (scale[ck[1]], ck)):
            if (c, k) not in self.pair_row:
                continue
            p = self.pool_of[k]
            mine = {key: q for key, q in book.items() if key[0] == c and key[1] == k}
            own = {}  # (lane, next edge) -> what the default release sends of this pair
            for (_c, _k, lane, nxt), q in mine.items():
                tot = on_edge.get((c, p, nxt), 0.0)
                own[(lane, nxt)] = 0.0 if prohibited[nxt, k] or tot <= EPS else rel[(c, p, nxt)] * q / tot
            waiting = sum(mine.values()) - sum(own.values())
            kap_left = spare_kap.get((c, p), max(float(kappa[p][self.chk_row[c]]), 0.0))
            if waiting <= EPS or kap_left <= EPS:
                continue
            cands = []
            for s, (cc, kk, e, lane) in enumerate(self.slots):
                if (cc, kk) != (c, k) or not omask[s] or prohibited[e, k]:
                    continue
                dest = self.head[self.lane_edges[lane][-1]] if lane is not None else self.head[e]
                grids = self.fed.get((dest, k), ())
                if not grids:
                    continue
                cands.append((sum(want.get((g, k), 0.0) for g in grids), s, e, lane, dest, grids))
            extra = {}
            for need, s, e, lane, dest, grids in sorted(cands, key=lambda x: (-x[0], x[1])):
                x = min(
                    waiting,
                    kap_left,
                    float(u[e]) - used[e],
                    beyond(c, k, e, lane),
                    need,
                    room_of(dest, k),
                    fleet_left[p] / self.dtau[e] if self.dtau.get(e, 0.0) > 0 else np.inf,
                )
                if x <= EPS:
                    continue
                extra[s] = x
                waiting -= x
                kap_left -= x
                used[e] += x
                room[(dest, k)] -= x
                fleet_left[p] -= self.dtau.get(e, 0.0) * x
                left = x
                for g in grids:  # the fuel now on its way to these grids
                    take = min(left, want.get((g, k), 0.0))
                    want[(g, k)] = want.get((g, k), 0.0) - take
                    left -= take
            if sum(extra.values()) <= P["min_extra"] * scale[k]:
                for s, x in extra.items():  # give back what was booked
                    _cc, _kk, e, _lane = self.slots[s]
                    used[e] -= x
                    fleet_left[p] += self.dtau.get(e, 0.0) * x
                continue
            spare_kap[(c, p)] = kap_left
            mode[self.pair_row[(c, k)]] = 1
            for s, (cc, kk, e, lane) in enumerate(self.slots):
                if (cc, kk) == (c, k) and omask[s]:
                    qty[s] = own.get((lane, e), 0.0) + extra.get(s, 0.0)
            if P["trace"]:
                self.trace.append({"week": t, "strait": c, "k": k, "extra": {s: round(x) for s, x in extra.items()}})
        return out
