"""A weekly planner whose plan the simulator executes: the rules decide every yes/no, a linear program the quantities.

Lab agent (it loads ``agents/anastasiia_hybrid_hub``, ``lab/anastasiia/rollout_lab/model.py`` and
``lab/anastasiia/plan_lab/regime.py`` by path; not a submission folder). Every week:

1. the simulator's state is rebuilt from the observation and the network of the coming weeks is set: as observed
   ("it stays as it is"), or the true one when the harness tells the truth (``truth``);
2. a copy of the rules (no linear program) plays those weeks on that model: a reference trajectory. Last week's
   plan, from its second week on, gives another: the same rollout with the entries the plan decides taken from it.
   The cheaper of the two is the reference. A plan carried from week to week matters: the rules' yes/no answers
   change with every state a plan puts them in, and a plan re-made under new answers each week never collects what
   the one before it was built for;
3. ``regime.regimes`` reads the simulator's yes/no answers off the reference (which grid sheds, which fuel runs out,
   which fab has wafers for its capacity), ``regime.rows`` writes the simulator's rules under those answers into the
   package's plan of the window, and the plan is solved: the cheapest flows under the same answers. Entries the
   plan does not decide (``take``) are held at the reference's. The reference satisfies the rows, so on the model
   the plan is never worse than it;
4. the plan is played on the model (``rounds`` > 1: its answers are read and it is solved again), and the first week
   of the cheapest of the references and the plans is sent; the hybrid's own entries stand where the plan does not
   decide, and when nothing beats the rules by ``min_gain``.

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
FAMILIES = ("order", "valve", "wafer", "raw", "pack", "strait")  # what a plan can decide; "strait": tanker releases
ALIASES = {"fuel": ("order", "valve", "strait"), "chips": ("wafer", "raw", "pack"), "all": FAMILIES}
NOW = {"u": "u_now", "o": "o_now", "kappa": "kappa_now", "supply": "supply_now", "G_bar": "G_bar_now",
       "y_bar": "y_bar_now", "R": "R_now", "alpha_bar": "alpha_now"}  # fmt: skip
GROUPS = {  # parts of the network that ``known`` can make true while the rest stays as observed
    "power": ("G_bar", "y_bar"),
    "edges": ("u", "c"),
    "straits": ("o", "kappa", "wr_class", "h_queue", "c_wr"),
    "bans": ("prohibited", "tariff"),
    "fabs": ("R", "alpha_bar", "R_osat", "sigma_scr"),
    "supply": ("supply",),
    "demand": ("demand",),
}
PARAMS = {
    "truth": False,  # the window's network: the true one (from the harness) instead of "as observed"
    "known": [],  # ... or only these groups of it true (names of GROUPS), the rest as observed
    "ahead": 0,  # > 0: the true network of only this many coming weeks is known; after them it stays as in the last
    "horizon": 0,  # weeks planned; 0: to the end of the episode
    "rounds": 1,  # plans solved a week: each next one under the answers of the previous plan played on the model
    "pack": True,  # the plants' rule (package everything or fill the throughput) among the rows
    "stores": True,  # ... the sources' lift and the stores' disposal
    "release": True,  # ... the default release of container cargo at the straits
    "floor": 0.0,  # > 0: by every week the plan has ordered, for each grid and fuel, at least this share of what the
    # reference has ordered by then (the plan may send more, sooner or by another lane, never less)
    "take": "all",  # the entries the plan decides: "all", or a list of FAMILIES ("fuel" and "chips" name several)
    "free": None,  # the entries the plan may move in its program (None: those it decides); the rest are held at the
    # reference's. Freer than take: the plan counts on moves the rules, not it, will have to make
    "min_gain": 0.0,  # the plan is played only if the model puts it this share of the window's cost below the rules
    "pending": False,  # announced prohibitions planned as taking effect
    "persist": True,  # last week's plan is a reference too
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


class Rules:
    """The hybrid's rules without its linear program: cheap to copy, cheap to play on the model."""

    def __init__(self, chips, fuel, strait, n_slots):
        self.chips, self.fuel, self.strait, self.n_slots = chips, fuel, strait, n_slots

    def act(self, observation, early=None, late=None):
        """``early(flows)`` changes the chip entries before the fuel rules read them, ``late(action)`` the rest."""
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)
        if early is not None:
            early(flows)
        self.fuel.fill(observation, flows)
        action = {"flows": flows, **self.strait.fill(observation)}
        if late is not None:
            late(action)
        return action


class Agent(_base.Agent):
    def __init__(self, config=None):
        super().__init__(config)
        self.p = dict(PARAMS)
        self.model = rmodel.Model(config, self.planner) if self.planner is not None else None
        self.truth = None
        self.prev_x = None  # last week's plan from its second week on, in the window's columns
        self._pred = None
        self.notes = {"planned": 0, "kept": 0, "failed": 0, "gain_bn": 0.0, "gap_bn": 0.0, "cpu": []}
        kind = np.asarray(self.chips.kind)  # 0 fuel, 1 wafer, 2 raw chip, 3 packaged chip
        take = [self.p["take"]] if isinstance(self.p["take"], str) else list(self.p["take"])
        take = {name for x in take for name in ALIASES.get(x, (x,))}
        slots = {"wafer": kind == 1, "raw": kind == 2, "pack": kind == 3}
        if self.model is not None:  # a valve: a fuel slot of one edge from a terminal into a grid (it acts this week)
            inst = self.model.inst
            fed = {e.head for e in inst.edges}
            valve = np.array([
                lane is None and inst.nodes[inst.edges[e].head].grid is not None and inst.edges[e].tail in fed
                for e, _k, lane in inst.action_slots
            ]) & (kind == 0)
            slots |= {"valve": valve, "order": (kind == 0) & ~valve}
        self.mine = np.any([slots[name] for name in take if name in slots] or [np.zeros(self.n_slots, bool)], axis=0)
        self.order_groups = {}  # (grid node, fuel) -> the order slots whose lane ends at the grid or at its terminal
        if self.model is not None:
            behind = {}  # (terminal node, fuel) -> the grid its valve feeds
            for s in np.flatnonzero(slots["valve"]):
                e, k, _lane = inst.action_slots[s]
                behind[(inst.edges[e].tail, k)] = inst.edges[e].head
            for s in np.flatnonzero(slots["order"]):
                e, k, lane = inst.action_slots[s]
                end = inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head
                grid = end if inst.nodes[end].grid is not None else behind.get((end, k))
                if grid is not None:
                    self.order_groups.setdefault((grid, k), []).append(int(s))
        self.early = self.mine & (kind != 0)  # the chip entries, set before the fuel rules read the wafers
        self.late = self.mine & (kind == 0)
        self.strait_mine = "strait" in take
        self.all_mine = bool(self.mine.all()) and self.strait_mine
        free = self.p["free"]
        if free is None:
            self.free, self.strait_free = self.mine, self.strait_mine
        else:
            free = {name for x in ([free] if isinstance(free, str) else free) for name in ALIASES.get(x, (x,))} | take
            self.free = np.any([slots[name] for name in free if name in slots], axis=0)
            self.strait_free = "strait" in free
        self.all_free = bool(self.free.all()) and self.strait_free

    def tell_truth(self, truth) -> None:
        if self.p["truth"] or self.p["known"] or self.p["ahead"]:
            marks = truth["marks"]
            self.truth = ({name: np.asarray(getattr(marks, name)) for name in L.WINDOW_FIELDS}, tuple(marks.fab_hits))

    # ----- the model's pieces -----------------------------------------------------------------------------------------
    def _network(self, observation, week: int, weeks: int):
        """The window's network: None (as observed), or arrays with the true weeks of all or some of its fields."""
        if self.truth is None:
            return None, ()
        arrays, all_hits = self.truth
        real = {name: a[week - 1 : week - 1 + weeks] for name, a in arrays.items()}
        hits = [FabHit(h.fab, h.onset - (week - 1), h.severity) for h in all_hits if week <= h.onset_week < week + weeks]
        if self.p["truth"]:
            return real, hits
        if self.p["ahead"]:
            h = int(self.p["ahead"])
            if weeks > h:
                real = {name: np.concatenate([a[:h], np.repeat(a[h - 1 : h], weeks - h, axis=0)]) for name, a in real.items()}
                hits = [x for x in hits if x.onset_week <= h]
            return real, hits
        seen = {name: np.array(a) for name, a in self.model.forecast(observation, weeks, bool(self.p["pending"])).items()}
        for group in self.p["known"]:
            for name in GROUPS[group]:
                seen[name] = real[name]
                if name in NOW:
                    seen[NOW[name]] = real[NOW[name]]
        return seen, (hits if "fabs" in self.p["known"] else ())

    @staticmethod
    def _cost(w, records) -> int:
        return sum(r.cost_cents for r in records) - cents(sim.terminal_salvage(w.inst, w.state))

    def _tables(self, lp):
        """Column of every action slot and of the tanker releases of a plan's week, by the package's action mapping."""
        inst, tm = self.model.inst, lp.meta["template"]
        slot_col = np.array([tm[("x", e, k, lane)] for e, k, lane in inst.action_slots])
        first = {}
        for o, (c, k, e, _lane) in enumerate(inst.override_slots):
            first.setdefault((c, k, e), o)
        releases = [(o, c, k, e, tm[("x", e, k, None)]) for (c, k, e), o in first.items() if ("x", e, k, None) in tm]
        return slot_col, releases

    def _week(self, lp, x, h: int, prohibited, tables):
        """Week ``h`` of a plan as the simulator takes it: requests, tanker releases, held pairs."""
        nc = lp.meta["nc"]
        slot_col, releases = tables
        q = x[h * nc + slot_col]
        banned = prohibited[self.model.slot_edge, self.model.slot_k]
        flows = {int(s): float(q[s]) for s in np.flatnonzero((q > 1e-9) & ~banned)}
        overrides, sendable = {}, {}
        for o, c, k, e, j in releases:
            ok = not bool(prohibited[e, k])
            sendable[(c, k)] = sendable.get((c, k), False) or ok
            if ok:
                overrides[o] = max(float(x[h * nc + j]), 0.0)
        holds = frozenset(ck for ck, ok in sendable.items() if not ok)
        return flows, overrides, holds

    def _flat(self, like, lp, x, h: int, prohibited, tables) -> dict:
        """Week ``h`` of a plan as flat arrays shaped as the action ``like``: flows, override quantities, modes."""
        flows, overrides, holds = self._week(lp, x, h, prohibited, tables)
        f = np.zeros(self.n_slots)
        for s, q in flows.items():
            f[s] = q
        qty = np.zeros_like(np.asarray(like["override_qty"], dtype=float))
        mode = np.zeros_like(np.asarray(like["release_mode"]))
        for o, amount in overrides.items():
            qty[o] = amount
            mode[self.model.ov_pair[o]] = rmodel.OVERRIDE
        index = {pair: i for i, pair in enumerate(self.model.pairs)}
        for pair in holds:
            mode[index[pair]] = rmodel.HOLD
        return {"flows": f, "override_qty": qty, "release_mode": mode}

    def _merge(self, rules: Rules, observation, lp, x, h: int, prohibited, tables) -> dict:
        """The rules' action of a week with the plan's entries where the plan decides."""
        plan = {}

        def early(flows):
            plan.update(self._flat({"override_qty": np.zeros(len(self.model.ov_pair)),
                                    "release_mode": np.zeros(len(self.model.pairs), dtype=np.int64)},
                                   lp, x, h, prohibited, tables))
            flows[self.early] = plan["flows"][self.early]

        def late(action):
            action["flows"][self.late] = plan["flows"][self.late]
            if self.strait_mine:
                action["override_qty"], action["release_mode"] = plan["override_qty"], plan["release_mode"]

        return rules.act(observation, early, late)

    def _rollout(self, observation, w, rules: Rules, lp=None, x=None, tables=None):
        """The window played on the model by the rules, the plan's entries (if a plan is given) in their place."""
        seen, records = observation, []
        weeks = w.inst.T
        for h in range(weeks):
            if x is None:
                action = rules.act(seen)
            else:
                action = self._merge(rules, seen, lp, x, h, np.asarray(w.marks.prohibited[h]), tables)
            records.append(self.model.step(w, action))
            if h + 1 < weeks:
                seen = self.model.flat(observation, w)
        return records, self._cost(w, records)

    def _play(self, w, lp, x, tables):
        """A plan of every entry played open loop on a fresh copy of the window: records and cost."""
        w2 = self.model.restart(w)
        records = []
        for h in range(w2.inst.T):
            flows, overrides, holds = self._week(lp, x, h, np.asarray(w2.marks.prohibited[h]), tables)
            records.append(sim.step(w2.inst, w2.marks, w2.state, flows, overrides, holds))
        return records, self._cost(w2, records)

    def _evaluate(self, observation, w, rules, lp, x, tables):
        """What a plan costs on the model: played open loop, or with the rules on the entries it does not decide."""
        if self.all_mine:
            return self._play(w, lp, x, tables)
        return self._rollout(observation, self.model.restart(w), copy.deepcopy(rules), lp, x, tables)

    def _hold(self, R, lp, records, tables) -> None:
        """Bounds of ``R``: the entries the plan does not decide stay at the reference's executed flows."""
        nc = lp.meta["nc"]
        slot_col, releases = tables
        others = np.flatnonzero(~self.free)
        inst = self.model.inst
        for h, rec in enumerate(records):
            done = np.zeros(self.n_slots)
            for s, q in rec.executed.items():
                done[s] = q
            cols = h * nc + slot_col[others]
            R.lb[cols] = R.ub[cols] = done[others]
            if not self.strait_free:
                out = {}
                for (e, k, _lane), q in rec.x.items():
                    if inst.edges[e].tail in inst.chokepoint_ordinal:
                        out[(e, k)] = out.get((e, k), 0.0) + q
                for _o, _c, k, e, j in releases:
                    R.lb[h * nc + j] = R.ub[h * nc + j] = out.get((e, k), 0.0)

    def _floor(self, R, lp, records, tables) -> None:
        """Rows of R: by every week, per grid and fuel, the plan has ordered floor of the reference's orders."""
        nc = lp.meta["nc"]
        slot_col = tables[0]
        share = float(self.p["floor"])
        for (grid, k), group in self.order_groups.items():
            if not self.free[group].any():
                continue
            cols, total = [], 0.0
            for h, rec in enumerate(records):
                week = sum(rec.executed.get(s, 0.0) for s in group)
                cols += [(h * nc + int(slot_col[s]), 1.0) for s in group]
                total += week
                if week > 0:
                    R.add(list(cols), share * total, np.inf, ("floor", h + 1, grid, k), share * total)

    # ----- the week ---------------------------------------------------------------------------------------------------
    def _audit(self, observation) -> None:
        """Last week as the model played it against what the environment reports (diagnostics)."""
        pred, self._pred = self._pred, None
        if pred is None:
            return
        got = np.asarray(observation["last_week.cost_components"], dtype=float)
        want = np.array(list(pred.costs.as_dict().values()))
        self.notes["audit_bn"] = self.notes.get("audit_bn", 0.0) + float(np.abs(got - want).sum()) / 1e9

    def act(self, observation):
        t0 = time.process_time()
        self._audit(observation)
        rules = Rules(*copy.deepcopy((self.chips, self.fuel, self.strait)), self.n_slots)  # before this week's memory
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)
        planned = self._planned(observation)  # the hybrid's program for the chips after the fab
        if planned is not None:
            flows[self.take] = planned[self.take]
            self.planned_weeks += 1
        plan = None
        if self.model is not None:
            try:
                plan = self._plan(observation, rules)
            except Exception as error:  # the hybrid's action stands
                self.notes["failed"] += 1
                self.notes["error"] = repr(error)[:300]
        if plan is not None:
            flows[self.early] = plan["flows"][self.early]
        self.fuel.fill(observation, flows)  # reads the wafer entries; keeps the fuel rules' memory
        action = {"flows": flows, **self.strait.fill(observation)}
        if plan is not None:
            flows[self.late] = plan["flows"][self.late]
            if self.strait_mine:
                action["override_qty"], action["release_mode"] = plan["override_qty"], plan["release_mode"]
        self.notes["cpu"].append(round(time.process_time() - t0, 3))
        return action

    def _plan(self, observation, rules):
        """The first week of the best plan as flat arrays, or None: the rules' rollout is not beaten."""
        week = int(observation["week"][0])
        left = self.model.inst.T - week + 1
        weeks = left if not self.p["horizon"] else min(int(self.p["horizon"]), left)
        arrays, hits = self._network(observation, week, weeks)
        w = self.model.window(observation, weeks, arrays, hits, pending=bool(self.p["pending"]))
        records, j_ref = self._rollout(observation, w, copy.deepcopy(rules))
        lp = build_lp(w.inst, w.marks)
        tables = self._tables(lp)
        nc = lp.meta["nc"]
        i0 = sim.initial_stock(w.inst)
        thr = osat_throughput(w.inst, w.marks.R_osat)
        best, best_x, best_records = j_ref, None, records
        prev = self.prev_x if self.p["persist"] else None
        if prev is not None and len(prev) >= weeks * nc:  # last week's plan from its second week on
            prev = prev[: weeks * nc]
            records, j_prev = self._evaluate(observation, w, rules, lp, prev, tables)
            self.notes["prev_wins"] = self.notes.get("prev_wins", 0) + int(j_prev < best)
            if j_prev < best:
                best, best_x, best_records = j_prev, prev, records
        for _round in range(int(self.p["rounds"])):
            reg = regime.regimes(w.inst, w.marks, best_records, i0, thr)
            R = regime.rows(w.inst, w.marks, lp, reg, i0, pack=bool(self.p["pack"]), stores=bool(self.p["stores"]),
                            release=bool(self.p["release"]))
            if not self.all_free:
                self._hold(R, lp, best_records, tables)
            if self.p["floor"] > 0:
                self._floor(R, lp, best_records, tables)
            res, _secs = regime.solve(lp, R, method=self.p["method"])
            if res.x is None or res.status != 0:
                self.notes["unsolved"] = self.notes.get("unsolved", 0) + 1
                break
            x = np.asarray(res.x)
            records, j_plan = self._evaluate(observation, w, rules, lp, x, tables)
            self.notes["gap_bn"] += (j_plan - float(res.fun) * 100.0) / 1e11 if _round == 0 else 0.0
            if j_plan >= best:
                break
            best, best_x, best_records = j_plan, x, records
        if best_x is None or j_ref - best < self.p["min_gain"] * abs(j_ref):
            self.notes["kept"] += 1
            self.prev_x = None
            return None
        self.notes["planned"] += 1
        self.notes["gain_bn"] += (j_ref - best) / 1e11
        self._pred = best_records[0]
        self.prev_x = best_x[nc:].copy()
        like = {"override_qty": np.zeros(len(self.model.ov_pair)),
                "release_mode": np.zeros(len(self.model.pairs), dtype=np.int64)}
        return self._flat(like, lp, best_x, 0, np.asarray(w.marks.prohibited[0]), tables)
