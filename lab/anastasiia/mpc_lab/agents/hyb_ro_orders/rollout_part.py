"""Pick this week's fuel entries by playing 2-4 candidates forward in the vendored simulator (persistence forecast).

The candidates differ only in the fuel entries of the week's action (everything else, strait releases included, is
the same): (a) as assembled; (b) ``valve_boost``: the terminal -> grid valves x ``boost``; (c) ``valve_hold``: the
valves closed; (d) ``order_boost``: the fuel orders from the sources x ``boost``. A candidate that changes nothing, or
duplicates another, is skipped.

Each candidate is judged by the same ``H``-week rollout from this week's observed start state (``lp_common``'s
``observed_start``: stock, pipeline, queue lots, WIP, backlog), on the persistence forecast (``persistence_arrays``:
the network as observed, announced prohibitions taking effect, demand from the forecast, no scaling):

- week 1 plays the candidate (entries the validator would drop under the window's prohibitions dropped first);
- weeks 2..H play a fixed continuation, the same for every candidate:
  * non-fuel slots: week 1's non-fuel requests repeated (the simulator clips them to stock and capacity);
  * fuel orders: the mean of the last ``order_memory`` weeks' played orders per slot (this week's as-is included);
  * valves: FuelRules' level rule without pulse or throttle: move what brings the grid to its level after its burn,
    ``max(0, level + share x G_bar - grid stock - landing)`` (level = psi ibar + ss_weeks burn for the rationed fuel,
    grid_cover_weeks burn otherwise, at most 0.98 of the grid's storage), taken from the feeder terminals in slot order,
    never more than a terminal holds.
- score = sum of the H weekly costs - an end-of-window credit. When the window ends at T the credit is the
  simulator's own salvage; otherwise fuel (stock outside supply nodes, cargo in transit and at straits) at v_k, but
  only up to ``credit_weeks`` weeks of the fuel's full burn over all grids (beyond that, fuel is not worth v_k: without
  the cap an order boost always wins by credit), plus (``credit_chips``) every other commodity outside supply nodes at
  v_k and WIP at the v of its product.

The lowest score is played; ties keep the as-is action. ``Rollout.choose`` raises on any failure; the caller keeps the
as-is action then.
"""

import dataclasses
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sbfv.dynamics import sim as S  # noqa: E402
from sbfv.dynamics.cost import salvage  # noqa: E402
from sbfv.policies import lp_common as L  # noqa: E402

ROLLOUT = {
    "rollout": True,  # False: play the as-is action (the base agent)
    "rollout_h": 16,  # weeks rolled forward
    "boost": 1.5,  # factor of valve_boost and order_boost
    "valve_boost": True,
    "valve_hold": True,
    "order_boost": True,
    "order_memory": 4,  # weeks of played orders the continuation averages
    "credit_weeks": 8.0,  # fuel credited at v_k up to this many weeks of its full burn
    "credit_chips": True,  # credit non-fuel commodities and WIP at v too
    "time_share": 0.6,  # skip the rollout when the week's CPU already exceeds this share of the budget
    "abort_share": 0.85,
    "min_gain": 0.0,  # USD: another candidate replaces as-is only if its score is lower by more than this  # stop evaluating candidates past this share (play the best evaluated so far, as-is first)
}
BUDGET_S = {"small": 2.0, "full": 4.0}
_MEMO_KEYS = ("dynamics.sim", "dynamics.cost", "dynamics.clip.dup_terms", "marks.osat_thr")  # T-free tables


class Rollout:
    def __init__(self, config, fuel, planner, params: dict | None = None):
        self.p = ROLLOUT | (params or {})
        self.fuel, self.planner = fuel, planner
        inst = self.inst = planner.inst
        self.budget = BUDGET_S.get(inst.kind, 2.0)
        self.T = inst.T
        self.valves = sorted(s for slots in fuel.tg.values() for s in slots)
        vset = set(self.valves)
        self.orders = sorted(s for s in fuel.slot if s not in vset)
        self.fuel_slots = np.array(sorted(fuel.slot), dtype=int)
        self.nonfuel = np.array([s for s in range(len(inst.action_slots)) if s not in fuel.slot], dtype=int)
        self.slot_ek = [(e, k) for e, k, _lane in inst.action_slots]
        self.ov_ek = [(o[2], o[1]) for o in inst.override_slots]
        layout, static = config["layout"], config["static"]
        self.pairs = [(int(c), int(k)) for c, k in layout.get("release_pairs", [])]
        pair = {p: i for i, p in enumerate(self.pairs)}
        ovs = static["override_slots"]
        self.opair = [pair.get((int(c), int(k))) for c, k in zip(ovs["chokepoint"], ovs["k"])]
        self.tail_row = {s: fuel.row.get((d["tail"], d["k"])) for s, d in fuel.slot.items()}
        # the level rule of the continuation, per (grid node, fuel)
        P = fuel.P
        self.level = []
        for (g, k), slots in sorted(fuel.tg.items()):
            gd = fuel.grids[g]
            share, thr = gd["fuels"][k], gd["thr"].get(k)
            b_nom = share * gd["deliverable"]
            end = (thr + P["ss_weeks"] * b_nom) if thr else P["grid_cover_weeks"] * b_nom
            end = min(end, 0.98 * fuel.storage.get((g, k), np.inf))
            terms = [(s, inst.slot_index[(fuel.slot[s]["tail"], k)]) for s in slots]
            self.level.append((g, k, share, inst.grid_ordinal[g], inst.slot_index[(g, k)], end, terms))
        # end-of-window credit
        self.v = np.array([c.v for c in inst.commodities])
        self.fuel_k = sorted(fuel.fuel_set)
        burn = {k: 0.0 for k in self.fuel_k}
        for g, gd in fuel.grids.items():
            for k, share in gd["fuels"].items():
                burn[k] += share * gd["deliverable"]
        self.fuel_cap = {k: self.p["credit_weeks"] * burn[k] for k in self.fuel_k}
        supply = set(inst.supply_nodes)
        self.stock_k = np.array([sl.k for sl in inst.stock_slots])
        self.stock_live = np.array([sl.node not in supply for sl in inst.stock_slots])
        self.fab_product = [inst.nodes[f].fab.product for f in inst.fabs]
        self.history = deque(maxlen=max(1, int(self.p["order_memory"])))
        self.week1: list = []
        self.wins: dict = {}  # candidate name -> weeks it was played (diagnostics)
        self.log_path = os.environ.get("ROLLOUT_LOG")

    # ----- the week -------------------------------------------------------------------------------------------------
    def remember(self, flows: np.ndarray) -> None:
        """Keep the order entries of the action played (the continuation's orders)."""
        self.history.append(np.asarray(flows, dtype=float)[self.orders].copy())

    def candidates(self, flows, extra, S_obs):
        out = [("as_is", flows)]
        b = float(self.p["boost"])
        v = np.array(self.valves, dtype=int)
        o = np.array(self.orders, dtype=int)
        if self.p["valve_boost"] and len(v) and (flows[v] > 0).any():
            f = flows.copy()
            cap = np.array([np.inf if self.tail_row[s] is None else float(S_obs[self.tail_row[s]]) for s in v])
            f[v] = np.minimum(flows[v] * b, np.maximum(cap, flows[v]))
            out.append(("valve_boost", f))
        if self.p["valve_hold"] and len(v) and (flows[v] > 0).any():
            f = flows.copy()
            f[v] = 0.0
            out.append(("valve_hold", f))
        if self.p["order_boost"] and len(o) and (flows[o] > 0).any():
            f = flows.copy()
            f[o] = flows[o] * b
            out.append(("order_boost", f))
        uniq = []
        for name, f in out:
            if not any(np.array_equal(f, g) for _n, g in uniq):
                uniq.append((name, f))
        return uniq

    def choose(self, observation, flows: np.ndarray, extra: dict, t0: float):
        """(name, flows) of the cheapest candidate; t0 is the week's CPU start. Raises on failure."""
        p = self.p
        state = getattr(self.planner, "last_state", None)
        t = int(observation["week"][0])
        if state is None or state["week"] != t:
            raise RuntimeError("no start state for this week")
        S_obs = np.asarray(observation["stock.qty"], dtype=float)
        cands = self.candidates(flows, extra, S_obs)
        if len(cands) < 2:
            return cands[0]
        H = L.window_length(int(p["rollout_h"]), t, self.T)
        inst = self.inst
        arrays = L.persistence_arrays(inst, state, self.planner.memory, H)
        start, backlog = L.observed_start(inst, state)
        inst_r = dataclasses.replace(inst, T=H, initial_state=start)
        for key in _MEMO_KEYS:
            if key in inst._memo:
                inst_r._memo[key] = inst._memo[key]
        inst_r._memo["content_digest"] = "rollout"  # nothing in the simulator checks it; skips hashing the instance
        marks = L.window_marks(inst_r, arrays, backlog)
        overrides, holds = self._releases(extra, marks.prohibited[0])
        cont_orders = np.mean(np.vstack(list(self.history) + [flows[self.orders]]), axis=0)
        to_end = t + H - 1 == self.T
        limit = t0 + p["abort_share"] * self.budget
        scores, self.week1 = [], []
        for name, f in cands:
            if scores and time.process_time() > limit:
                break
            scores.append(self._score(inst_r, marks, f, overrides, holds, cont_orders, H, to_end))
        self._last_scores = scores
        best = int(np.argmin(scores))  # first of equal scores: as-is wins ties
        if scores[0] - scores[best] <= p["min_gain"]:
            best = 0
        name = cands[best][0]
        self.week1 = [self.week1[best]]  # diagnostics: the played candidate's week-1 cost in the rollout
        self.wins[name] = self.wins.get(name, 0) + 1
        if self.log_path:
            with open(self.log_path, "a") as fh:
                sc = " ".join(f"{n}={s:.6e}" for (n, _f), s in zip(cands, scores))
                fh.write(f"{id(self)} {inst.kind} t={t} H={H} pick={name} {sc}\n")
        return cands[best]

    # ----- pieces ---------------------------------------------------------------------------------------------------
    def _releases(self, extra, Z):
        """overrides {override slot: qty} and holds {(chokepoint, k)} of the strait part's arrays, validated."""
        modes = np.asarray(extra.get("release_mode", np.zeros(len(self.pairs), dtype=int)))
        oq = np.asarray(extra.get("override_qty", np.zeros(len(self.opair))), dtype=float)
        overrides = {}
        for o, pi in enumerate(self.opair):
            if pi is None or modes[pi] != 1:
                continue
            e, k = self.ov_ek[o]
            q = float(oq[o])
            if np.isfinite(q) and q >= 0 and not Z[e, k]:
                overrides[o] = q
        holds = frozenset(self.pairs[i] for i in range(len(self.pairs)) if modes[i] == 2)
        return overrides, holds

    def _valid(self, flows, Z) -> dict:
        out = {}
        for s in np.flatnonzero(flows > 0):
            q = float(flows[s])
            e, k = self.slot_ek[s]
            if np.isfinite(q) and not Z[e, k]:
                out[int(s)] = q
        return out

    def _score(self, inst_r, marks, f, overrides, holds, cont_orders, H, to_end) -> float:
        st = S.initial_state(inst_r)
        st.last = None
        total = S.step(inst_r, marks, st, self._valid(f, marks.prohibited[0]), overrides, holds).costs.total()
        self.week1.append(total)  # diagnostics: the candidate's week-1 cost in the rollout
        base = np.zeros_like(f)
        base[self.nonfuel] = f[self.nonfuel]
        base[self.orders] = cont_orders
        for w in range(2, H + 1):
            req = base.copy()
            self._valves(st, marks, w, req)
            total += S.step(inst_r, marks, st, self._valid(req, marks.prohibited[w - 1])).costs.total()
        return total - (salvage(inst_r, st) if to_end else self._credit(st))

    def _valves(self, st, marks, w, req) -> None:
        """The continuation's level rule on the valve entries of ``req`` (week w of the window)."""
        landing = {}
        for sh in st.pipeline:
            if sh.arrival_week == w:
                key = (self.inst.edges[sh.edge].head, sh.k)
                landing[key] = landing.get(key, 0.0) + sh.qty
        stock = st.stock
        for g, k, share, go, gslot, end, terms in self.level:
            burn = share * float(marks.G_bar[w - 1, go])
            move = max(0.0, end + burn - float(stock[gslot]) - landing.get((g, k), 0.0))
            for s, tslot in terms:
                q = min(move, max(float(stock[tslot]), 0.0))
                req[s] = q
                move -= q

    def _credit(self, st) -> float:
        """The end-of-window credit (USD): fuel at v_k up to credit_weeks of its burn, the rest (optionally) at v."""
        qty = np.zeros(len(self.v))
        np.add.at(qty, self.stock_k[self.stock_live], np.asarray(st.stock, dtype=float)[self.stock_live])
        for sh in st.pipeline:
            qty[sh.k] += sh.qty
        # queue lots need no term: a chokepoint's stock slot holds their total (stock_live keeps it)
        credit = 0.0
        for k in self.fuel_k:
            credit += self.v[k] * min(max(qty[k], 0.0), self.fuel_cap[k])
            qty[k] = 0.0
        if self.p["credit_chips"]:
            credit += float(self.v @ qty)
            for fi, book in st.fab_wip.items():
                credit += self.v[self.fab_product[fi]] * sum(book.values())
            for book in st.osat_wip.values():
                for by_k in book.values():
                    credit += sum(self.v[k] * q for k, q in by_k.items())
        return credit
