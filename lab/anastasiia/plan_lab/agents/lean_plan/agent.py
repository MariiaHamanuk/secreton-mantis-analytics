"""The regime planner made cheap: the rules order the fuel, a carried plan decides everything else.

Lab agent (it loads ``agents/anastasiia_hybrid_hub``, ``lab/anastasiia/rollout_lab/model.py`` and
``lab/anastasiia/plan_lab/regime.py`` by path; not a submission folder). ``agents/regime_plan`` showed where a plan
on the network "as observed" pays and where it does not: it pays on the entries whose effect follows from cargo
already on the way (the valves from terminals into grids, the tanker releases at the straits, wafers, raw and
packaged chips), and it loses on the fuel orders from the sources, weeks away through straits, where a forecast
without disruptions makes every buffer look idle. So here the fuel rules keep the orders, always, and the plan
decides the rest. That also makes the week cheap, because a rollout no longer needs the chip rules (more than half
of the rules' time):

1. the fuel rules' memory is copied; the simulator's state and the window's network are set from the observation;
2. the reference: last week's plan from its second week on, played on the model with the fuel rules ordering each
   week (they read the plan's wafers). Without a plan, and every ``rules_every`` weeks beside it, the rules alone
   play the window; the cheaper trajectory is the reference;
3. the simulator's yes/no answers are read off the reference, the plan is solved under them with the orders held at
   the reference's, played open loop on the model, and kept if it is cheaper than the reference;
4. the first week of the best plan is sent with the rules' orders of this week (computed on the plan's wafers);
   when there is no plan the hybrid's own action stands (its linear program runs only then).

``PLAN_LAB_PARAMS`` (``harness.py --params``) replaces entries of ``PARAMS``.
"""

import builtins
import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "agents" / "anastasiia_hybrid_hub"
NOW = {"u": "u_now", "o": "o_now", "kappa": "kappa_now", "supply": "supply_now", "G_bar": "G_bar_now",
       "y_bar": "y_bar_now", "R": "R_now", "alpha_bar": "alpha_now"}  # fmt: skip
PARAMS = {
    "truth": False,  # the window's network: the true one (from the harness) instead of "as observed"
    "horizon": 0,  # weeks planned; 0: to the end of the episode
    "rules_every": 4,  # weeks between rollouts of the rules alone beside the carried plan (0: only without a plan)
    "valves": True,  # the plan decides the valves from terminals into grids (False: the fuel rules keep them too)
    "valves_by": None,  # "grid": every ``rules_every`` weeks a grid whose weeks the rules alone close more often has
    # its valves played by the fuel rules instead of the plan, if the window is cheaper so; "grid+wafer": and the
    # wafers of its fabs by the chip rules
    "hand_by": "rollout",  # what a try of ``valves_by`` is judged by: "rollout" (the carried plan played with the
    # grid's entries by the rules) or "plan" (the plan solved under that rollout's answers and played; a rollout
    # leaves the chips after the fab at the carried plan's flows, so lots the rules add never reach a market in it)
    "hand_every": 0,  # weeks between tries of ``valves_by`` (0: every ``rules_every`` weeks); a multiple of it
    "hand_parts": False,  # notes keep what each try of ``valves_by`` changed on the model, by cost component
    "chips": True,  # ... wafers, raw and packaged chips (False: the chip rules, and the hybrid's program in the action)
    "strait": True,  # ... the tanker releases at the straits (False: the strait rules)
    "pack": True,  # the plants' rule among the rows
    "stores": True,  # ... the sources' lift and the stores' disposal
    "release": True,  # ... the default release of container cargo at the straits
    "min_gain": 0.0,  # a new plan replaces the reference only if it is this share of the window's cost cheaper
    "pending": False,  # announced prohibitions planned as taking effect
    "grids": None,  # a grid in a spell of low output: None (it stays low), "step" or "blend" (it comes back as spells
    # of its age do: lab/anastasiia/rollout_lab/grid_recovery.json), or a number of weeks after which a spell of at
    # most 3 weeks is planned as over
    "end_fuel": 0.0,  # USD a unit of fuel left in the system at the window's end is worth (0: its salvage only)
    "end_chip": 0.0,  # share of its penalty a chip left in the chain at the window's end is worth
    "end_straits": True,  # False: fuel in a strait's queue, or on its way into one, is worth nothing at the window's end
    "method": "highs",
} | dict(getattr(builtins, "PLAN_LAB_PARAMS", {}))


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_base = _load("plan_lab_hub_base", BASE / "agent.py")  # its lp_part puts sbfv/ on the path
for folder in (ROOT / "lab/anastasiia/rollout_lab", ROOT / "lab/anastasiia/plan_lab"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import model as rmodel  # noqa: E402
import regime  # noqa: E402
from sbfv.dynamics import sim  # noqa: E402
from sbfv.dynamics.state import cents  # noqa: E402
from sbfv.marks import FabHit, osat_throughput  # noqa: E402
from sbfv.oracle.lp import build_lp  # noqa: E402
from sbfv.policies import lp_common as L  # noqa: E402


class Agent(_base.Agent):
    def __init__(self, config=None):
        super().__init__(config)
        self.p = dict(PARAMS)
        self.model = None
        if self.planner is not None:
            rule = self.p["grids"]
            if isinstance(rule, str):
                import json

                table = json.loads((ROOT / "lab/anastasiia/rollout_lab/grid_recovery.json").read_text())
                self.model = rmodel.Model(config, self.planner, table, rule)
            else:
                self.model = rmodel.Model(config, self.planner)
        self.truth = None
        self.prev_x = None  # last week's plan from its second week on, in the window's columns
        self.tables = None
        self._pred = None
        self.notes = {"planned": 0, "carried": 0, "rules": 0, "failed": 0, "cpu": []}
        kind = np.asarray(self.chips.kind)  # 0 fuel, 1 wafer, 2 raw chip, 3 packaged chip
        self.chip_slots = kind != 0
        if self.model is not None:  # a valve: a fuel slot of one edge from a terminal into a grid (it acts this week)
            inst = self.model.inst
            fed = {e.head for e in inst.edges}
            self.valves = np.array([
                lane is None and inst.nodes[inst.edges[e].head].grid is not None and inst.edges[e].tail in fed
                for e, _k, lane in inst.action_slots
            ]) & (kind == 0)
            self.orders = np.flatnonzero((kind == 0) & ~self.valves)
            self.grid_slots = {}  # grid ordinal -> the entries a grid handed to the rules takes from them
            grids = list(inst.grids)
            for s, (e, _k, lane) in enumerate(inst.action_slots):
                head = inst.nodes[inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head]
                if self.valves[s]:
                    gi = grids.index(inst.edges[e].head)
                elif kind[s] == 1 and head.fab is not None and self.p["valves_by"] == "grid+wafer":
                    gi = grids.index(head.fab.grid)
                else:
                    continue
                self.grid_slots.setdefault(gi, np.zeros(self.n_slots, bool))[s] = True
            self.by_rules = np.zeros(self.n_slots, bool)  # entries the rules play although the plan decides their kind
            self.like = {"override_qty": np.zeros(len(self.model.ov_pair)),
                         "release_mode": np.zeros(len(self.model.pairs), dtype=np.int64)}
            self.pair_index = {pair: i for i, pair in enumerate(self.model.pairs)}
            # what a unit of each commodity left in the system at the end of a window is worth, USD
            value = np.zeros(len(inst.commodities))
            for k in {k for g in inst.grids for k in inst.nodes[g].grid.fuels}:
                value[k] = float(self.p["end_fuel"])
            pi = np.zeros(len(inst.commodities))
            for d in inst.demands:
                pi[d.k] = max(pi[d.k], d.pi)
            for o in inst.osats:
                for raw, packed in inst.nodes[o].osat.packages.items():
                    pi[raw] = max(pi[raw], pi[packed])
            self.end_k = value + float(self.p["end_chip"]) * pi
            supply = set(inst.supply_nodes)
            self.end_stock = np.array([0.0 if sl.node in supply else self.end_k[sl.k] for sl in inst.stock_slots])
            # fuel that the end of a window does not credit while it waits at a strait or sails to one
            self.queued = set() if self.p["end_straits"] else {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
            chk = set(inst.chokepoints)
            self.to_strait = {e for e, edge in enumerate(inst.edges) if edge.head in chk}
            for s, sl in enumerate(inst.stock_slots):
                if sl.node in chk and sl.k in self.queued:
                    self.end_stock[s] = 0.0
        self.end_on = False  # this week's window ends before the episode and something is credited at its end

    def tell_truth(self, truth) -> None:
        if self.p["truth"]:
            marks = truth["marks"]
            self.truth = ({name: np.asarray(getattr(marks, name)) for name in L.WINDOW_FIELDS}, tuple(marks.fab_hits))

    # ----- the model's pieces -----------------------------------------------------------------------------------------
    def _network(self, week: int, weeks: int, observation=None):
        if self.truth is None:
            after = self.p["grids"]
            if isinstance(after, (int, float)) and not isinstance(after, bool) and after:
                planner = self.planner
                young = (planner.grid_age > 0) & (planner.grid_age <= 3)
                if young.any() and weeks > int(after):
                    arrays = {k: np.array(a) for k, a in self.model.forecast(observation, weeks, bool(self.p["pending"])).items()}
                    arrays["G_bar"][int(after):, young] = planner.calm["grid_G"][young]
                    arrays["G_bar_now"][int(after) + 1:, young] = planner.calm["grid_G"][young]
                    return arrays, ()
            return None, ()
        arrays, all_hits = self.truth
        real = {name: a[week - 1 : week - 1 + weeks] for name, a in arrays.items()}
        hits = [FabHit(h.fab, h.onset - (week - 1), h.severity) for h in all_hits if week <= h.onset_week < week + weeks]
        return real, hits

    def _end(self, w) -> float:
        """USD credited for what the window's last state holds beyond its salvage (the window is not the episode)."""
        if not self.end_on:
            return 0.0
        inst, st = self.model.inst, w.state
        total = float(np.dot(self.end_stock, np.asarray(st.stock, dtype=float)))
        total += sum(self.end_k[s.k] * s.qty for s in st.pipeline
                     if not (s.k in self.queued and s.edge in self.to_strait))
        for fi, book in st.fab_wip.items():
            total += self.end_k[inst.nodes[inst.fabs[fi]].fab.product] * sum(book.values())
        for book in st.osat_wip.values():
            for lots in book.values():
                total += sum(self.end_k[k] * q for k, q in lots.items())
        return total

    def _cost(self, w, records) -> int:
        return sum(r.cost_cents for r in records) - cents(sim.terminal_salvage(w.inst, w.state) + self._end(w))

    def _tables(self, lp):
        """Column of every action slot and of the tanker releases of a plan's week, by the package's action mapping."""
        inst, tm = self.model.inst, lp.meta["template"]
        slot_col = np.array([tm[("x", e, k, lane)] for e, k, lane in inst.action_slots])
        first = {}
        for o, (c, k, e, _lane) in enumerate(inst.override_slots):
            first.setdefault((c, k, e), o)
        releases = [(o, c, k, e, tm[("x", e, k, None)]) for (c, k, e), o in first.items() if ("x", e, k, None) in tm]
        return slot_col, releases

    def _week(self, lp_nc: int, x, h: int, prohibited):
        """Week ``h`` of a plan as the simulator takes it: requests, tanker releases, held pairs."""
        slot_col, releases = self.tables
        q = x[h * lp_nc + slot_col]
        banned = prohibited[self.model.slot_edge, self.model.slot_k]
        flows = {int(s): float(q[s]) for s in np.flatnonzero((q > 1e-9) & ~banned)}
        overrides, sendable = {}, {}
        for o, c, k, e, j in releases:
            ok = not bool(prohibited[e, k])
            sendable[(c, k)] = sendable.get((c, k), False) or ok
            if ok:
                overrides[o] = max(float(x[h * lp_nc + j]), 0.0)
        holds = frozenset(ck for ck, ok in sendable.items() if not ok)
        return flows, overrides, holds

    def _flat(self, nc: int, x, h: int, prohibited) -> dict:
        """Week ``h`` of a plan as the environment's flat arrays: flows, override quantities, release modes."""
        flows, overrides, holds = self._week(nc, x, h, prohibited)
        f = np.zeros(self.n_slots)
        for s, q in flows.items():
            f[s] = q
        qty, mode = np.zeros_like(self.like["override_qty"]), np.zeros_like(self.like["release_mode"])
        for o, amount in overrides.items():
            qty[o] = amount
            mode[self.model.ov_pair[o]] = rmodel.OVERRIDE
        for pair in holds:
            mode[self.pair_index[pair]] = rmodel.HOLD
        return {"flows": f, "override_qty": qty, "release_mode": mode}

    def _with_orders(self, fuel, observation, plan: dict, chips=None, strait=None, by_rules=None) -> dict:
        """A plan's week with the fuel rules' orders of that week (the rules read the plan's wafers).

        ``chips`` and ``strait`` are copies of those rules for the entries the plan does not decide; ``by_rules``
        marks entries of the plan's kinds that the rules play this window.
        """
        flows = np.zeros(self.n_slots)
        if self.p["chips"]:
            flows[self.chip_slots] = plan["flows"][self.chip_slots]
            if by_rules is not None and (by_rules & self.chip_slots).any():
                ruled = np.zeros(self.n_slots)
                chips.fill_chip_flows(observation, ruled)
                handed = by_rules & self.chip_slots
                flows[handed] = ruled[handed]
        else:
            chips.fill_chip_flows(observation, flows)
        fuel.fill(observation, flows)
        if self.p["valves"]:
            mine = self.valves if by_rules is None else self.valves & ~by_rules
            flows[mine] = plan["flows"][mine]
        if self.p["strait"]:
            return {"flows": flows, "override_qty": plan["override_qty"], "release_mode": plan["release_mode"]}
        return {"flows": flows, **strait.fill(observation)}

    def _carried(self, observation, w, fuel, nc: int, x, chips=None, strait=None, by_rules=None):
        """The window played by a plan with the fuel rules ordering every week: records and cost."""
        seen, records = observation, []
        weeks = w.inst.T
        for h in range(weeks):
            plan = self._flat(nc, x, h, np.asarray(w.marks.prohibited[h]))
            records.append(self.model.step(w, self._with_orders(fuel, seen, plan, chips, strait, by_rules)))
            if h + 1 < weeks:
                seen = self.model.flat(observation, w)
        return records, self._cost(w, records)

    def _rules(self, observation, w, chips, fuel, strait):
        """The window played by the rules alone (no linear program)."""
        seen, records = observation, []
        weeks = w.inst.T
        for h in range(weeks):
            flows = np.zeros(self.n_slots)
            chips.fill_chip_flows(seen, flows)
            fuel.fill(seen, flows)
            records.append(self.model.step(w, {"flows": flows, **strait.fill(seen)}))
            if h + 1 < weeks:
                seen = self.model.flat(observation, w)
        return records, self._cost(w, records)

    def _play(self, w, nc: int, x):
        """A plan played open loop on a fresh copy of the window, its orders as written: records and cost."""
        w2 = self.model.restart(w)
        records = []
        for h in range(w2.inst.T):
            flows, overrides, holds = self._week(nc, x, h, np.asarray(w2.marks.prohibited[h]))
            records.append(sim.step(w2.inst, w2.marks, w2.state, flows, overrides, holds))
        return records, self._cost(w2, records)

    def _hand_grids(self, observation, w, fuel0, rules0, nc: int, x, best, best_records, ruled, week: int):
        """Grid by grid, who plays its valves in the carried plan: the plan or the rules, whichever window is cheaper.

        Tried for the grids the rules have now and for those where the rules alone (``ruled``) close more weeks than
        the carried plan does. Returns the cheapest trajectory found and its cost; ``self.by_rules`` follows it.
        """
        def closed(records):
            return (np.array([r.shed for r in records]) < 1e-6).sum(axis=0)

        def parts(w2, records, gi):  # bn USD: shed, unsold, other costs, the end's credit and salvage; lots of the grid, M
            names = list(records[0].costs.as_dict())
            c = np.sum([list(r.costs.as_dict().values()) for r in records], axis=0)
            shed, unsold = c[names.index("shed")], c[names.index("shortage")]
            lots = sum(float(np.asarray(r.lots_started)[list(w2.inst.grid_fabs[gi])].sum()) for r in records)
            return np.array([shed, unsold, c.sum() - shed - unsold, -self._end(w2),
                             -sim.terminal_salvage(w2.inst, w2.state), lots * 1e3]) / 1e9

        mine, theirs = closed(best_records), closed(ruled)
        base = None
        for gi, slots in self.grid_slots.items():
            handed = bool(self.by_rules[slots].any())
            if not handed and theirs[gi] <= mine[gi]:
                continue
            mask = self.by_rules & ~slots if handed else self.by_rules | slots
            chips, strait = copy.deepcopy(rules0)
            w2 = self.model.restart(w)
            records, j = self._carried(observation, w2, copy.deepcopy(fuel0), nc, x, chips, strait, mask)
            took = j < best
            if self.p["hand_parts"]:
                if base is None:  # the trajectory the tries are set against, played again to read its end
                    chips, strait = copy.deepcopy(rules0)
                    w1 = self.model.restart(w)
                    base = (w1, self._carried(observation, w1, copy.deepcopy(fuel0), nc, x, chips, strait, self.by_rules)[0])
                delta = parts(w2, records, gi) - parts(base[0], base[1], gi)
                self.notes.setdefault("hand_parts", []).append([week, gi, int(handed)] + [round(float(v), 2) for v in delta])
                if took:
                    base = None
            self.notes.setdefault("hand", []).append(
                [week, gi, int(handed), int(mine[gi]), int(theirs[gi]), int(closed(records)[gi]), round((j - best) / 1e11, 2), int(took)])
            if took:
                best, best_records, self.by_rules = j, records, mask
        return best, best_records

    def _hold_orders(self, R, nc: int, records, by_rules=None) -> None:
        """Bounds of ``R``: the fuel orders, and every entry the plan does not decide, stay at the reference's flows."""
        slot_col, releases = self.tables
        held = np.zeros(self.n_slots, bool)
        held[self.orders] = True
        held |= self.by_rules if by_rules is None else by_rules
        if not self.p["valves"]:
            held |= self.valves
        if not self.p["chips"]:
            held |= self.chip_slots
        held = np.flatnonzero(held)
        inst = self.model.inst
        for h, rec in enumerate(records):
            done = np.array([rec.executed.get(int(s), 0.0) for s in held])
            cols = h * nc + slot_col[held]
            R.lb[cols] = R.ub[cols] = done
            if not self.p["strait"]:
                out = {}
                for (e, k, _lane), q in rec.x.items():
                    if inst.edges[e].tail in inst.chokepoint_ordinal:
                        out[(e, k)] = out.get((e, k), 0.0) + q
                for _o, _c, k, e, j in releases:
                    R.lb[h * nc + j] = R.ub[h * nc + j] = out.get((e, k), 0.0)

    def _end_values(self, lp) -> np.ndarray:
        """What the plan's objective credits for fuel and chips still in the system at the window's end."""
        inst = self.model.inst
        nc, T = lp.meta["nc"], lp.T
        credit = np.zeros(len(lp.lb))
        last = (T - 1) * nc
        for key, j in lp.meta["template"].items():
            if key[0] == "I":
                credit[last + j] = self.end_stock[key[1]]
            elif key[0] == "Q":
                credit[last + j] = 0.0 if key[2] in self.queued else self.end_k[key[2]]
            elif key[0] == "x":  # dispatched in the window, arriving after it
                if key[2] in self.queued and key[1] in self.to_strait:
                    continue
                for t in range(max(0, T - inst.edges[key[1]].tau), T):
                    credit[t * nc + j] = self.end_k[key[2]]
            elif key[0] == "p":  # lots still in process
                fab = inst.nodes[inst.fabs[key[1]]].fab
                for t in range(max(0, T - fab.tau), T):
                    credit[t * nc + j] = self.end_k[fab.product]
            elif key[0] == "xi":
                for t in range(max(0, T - inst.nodes[inst.osats[key[1]]].osat.tau), T):
                    credit[t * nc + j] = self.end_k[key[2]]
        return credit

    # ----- the week ---------------------------------------------------------------------------------------------------
    def _audit(self, observation) -> None:
        pred, self._pred = self._pred, None
        if pred is None:
            return
        got = np.asarray(observation["last_week.cost_components"], dtype=float)
        want = np.array(list(pred.costs.as_dict().values()))
        self.notes["audit_bn"] = self.notes.get("audit_bn", 0.0) + float(np.abs(got - want).sum()) / 1e9

    def act(self, observation):
        t0 = time.process_time()
        self._audit(observation)
        week = int(observation["week"][0])
        every = int(self.p["rules_every"])
        alone = self.prev_x is None or (every > 0 and (week - 1) % every == 0)
        fuel0 = copy.deepcopy(self.fuel)  # the rules' memory before this week
        # a rollout with the plan needs these rules too
        whole = not (self.p["chips"] and self.p["strait"]) or (
            self.model is not None and bool((self.by_rules & self.chip_slots).any()))
        rules0 = copy.deepcopy((self.chips, self.strait)) if alone or whole else None
        self.alone = alone
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # the rules' chip entries; keeps their memory
        plan = None
        if self.model is not None:
            try:
                self.planner._remember(observation)
                plan = self._plan(observation, week, fuel0, rules0)
            except Exception as error:  # the hybrid's action stands
                self.notes["failed"] += 1
                self.notes["error"] = repr(error)[:300]
        if plan is not None and self.p["chips"]:
            mine = self.chip_slots & ~self.by_rules
            flows[mine] = plan["flows"][mine]
        elif self.planner is not None:  # the hybrid's program for the chips after the fab, when no plan decides them
            remember, self.planner._remember = self.planner._remember, (lambda o: None)
            try:
                planned = self._planned(observation)
            finally:
                self.planner._remember = remember
            if planned is not None:
                flows[self.take] = planned[self.take]
        self.fuel.fill(observation, flows)  # this week's orders (and the rules' valves); keeps the fuel rules' memory
        action = {"flows": flows, **self.strait.fill(observation)}
        if plan is not None:
            if self.p["valves"]:
                mine = self.valves & ~self.by_rules
                flows[mine] = plan["flows"][mine]
            if self.p["strait"]:
                action["override_qty"], action["release_mode"] = plan["override_qty"], plan["release_mode"]
        self.notes["cpu"].append(round(time.process_time() - t0, 3))
        return action

    def _plan(self, observation, week: int, fuel0, rules0):
        """The first week of the best plan as flat arrays, or None: the rules alone are the cheapest."""
        left = self.model.inst.T - week + 1
        weeks = left if not self.p["horizon"] else min(int(self.p["horizon"]), left)
        arrays, hits = self._network(week, weeks, observation)
        w = self.model.window(observation, weeks, arrays, hits, pending=bool(self.p["pending"]))
        lp = build_lp(w.inst, w.marks)
        nc = lp.meta["nc"]
        if self.tables is None:
            self.tables = self._tables(lp)
        self.end_on = weeks < left and bool(self.p["end_fuel"] or self.p["end_chip"])
        credit = self._end_values(lp) if self.end_on else None
        best, best_x, best_records = None, None, None
        tries = None
        if self.prev_x is not None:
            x = self.prev_x
            if len(x) < weeks * nc:  # a capped window moved on by a week: its last week repeats the one before
                x = np.concatenate([x, np.tile(x[-nc:], weeks - len(x) // nc)])
            x = x[: weeks * nc]
            chips, strait = copy.deepcopy(rules0) if rules0 is not None else (None, None)
            best_records, best = self._carried(observation, self.model.restart(w), copy.deepcopy(fuel0), nc, x,
                                               chips, strait, self.by_rules)
            best_x = x
        if self.alone:
            due = not self.p["hand_every"] or (week - 1) % int(self.p["hand_every"]) == 0
            both = copy.deepcopy(rules0) if self.p["valves_by"] and best_x is not None and due else None
            records, j = self._rules(observation, self.model.restart(w), rules0[0], copy.deepcopy(fuel0), rules0[1])
            if best is None or j < best:
                best, best_x, best_records = j, None, records
            elif both is not None and self.p["hand_by"] == "rollout":
                best, best_records = self._hand_grids(observation, w, fuel0, both, nc, x, best, best_records, records, week)
            elif both is not None:
                tries = (both, records)
        j_ref = best
        reg = regime.regimes(w.inst, w.marks, best_records, sim.initial_stock(w.inst),
                             osat_throughput(w.inst, w.marks.R_osat))
        R = regime.rows(w.inst, w.marks, lp, reg, sim.initial_stock(w.inst), pack=bool(self.p["pack"]),
                        stores=bool(self.p["stores"]), release=bool(self.p["release"]))
        self._hold_orders(R, nc, best_records)
        res, _secs = regime.solve(lp, R, method=self.p["method"], credit=credit)
        if res.x is not None and res.status == 0:
            x = np.asarray(res.x)
            records, j = self._play(w, nc, x)
            if j < best - self.p["min_gain"] * abs(best):
                best, best_x, best_records = j, x, records
                self.notes["planned"] += 1
        else:
            self.notes["unsolved"] = self.notes.get("unsolved", 0) + 1
        if tries is not None:  # grid by grid: the plan under the answers of the carried plan with that grid by the rules
            def closed(records):
                return (np.array([r.shed for r in records]) < 1e-6).sum(axis=0)

            mine, theirs = closed(best_records if best_x is x else records), closed(tries[1])
            found = None
            for gi, slots in self.grid_slots.items():
                handed = bool(self.by_rules[slots].any())
                if not handed and theirs[gi] <= mine[gi]:
                    continue
                mask = self.by_rules & ~slots if handed else self.by_rules | slots
                chips, strait = copy.deepcopy(tries[0])
                ref, _j = self._carried(observation, self.model.restart(w), copy.deepcopy(fuel0), nc, x, chips, strait, mask)
                reg2 = regime.regimes(w.inst, w.marks, ref, sim.initial_stock(w.inst), osat_throughput(w.inst, w.marks.R_osat))
                R2 = regime.rows(w.inst, w.marks, lp, reg2, sim.initial_stock(w.inst), pack=bool(self.p["pack"]),
                                 stores=bool(self.p["stores"]), release=bool(self.p["release"]))
                self._hold_orders(R2, nc, ref, mask)
                res2, _secs = regime.solve(lp, R2, method=self.p["method"], credit=credit)
                took, j2 = False, None
                if res2.x is not None and res2.status == 0:
                    x2 = np.asarray(res2.x)
                    played, j2 = self._play(w, nc, x2)
                    took = j2 < best and (found is None or j2 < found[0])
                    if took:
                        found = (j2, x2, played, mask)
                self.notes.setdefault("hand", []).append(
                    [week, gi, int(handed), int(mine[gi]), int(theirs[gi]), int(closed(ref)[gi]),
                     None if j2 is None else round((j2 - best) / 1e11, 2), int(took)])
            if found is not None:  # one grid changes hands a time
                best, best_x, best_records, self.by_rules = found
        if best_x is None:
            self.notes["rules"] += 1
            self.prev_x = None
            return None
        self.notes["carried"] += int(j_ref == best)
        self.notes["gain_bn"] = self.notes.get("gain_bn", 0.0) + (j_ref - best) / 1e11
        self._pred = best_records[0]
        self.prev_x = best_x[nc:].copy()
        return self._flat(nc, best_x, 0, np.asarray(w.marks.prohibited[0]))
