"""The hybrid's rules as a weekly starting plan, the cell program on top of it (a lab agent, not a submission).

Every week:

1. the hybrid (``agents/anastasiia_hybrid_hub``) answers the observation: its action is the fallback and the first
   week of the rules' plan;
2. the rules alone (the hybrid without its linear program) play the weeks left on the simulator's model of them
   (``lab/anastasiia/rollout_lab/model.py``: the state from the observation, the network as observed): the rules'
   plan, with the regimes the rules would live in;
3. last week's plan, minus its first week, is the other candidate (``carry``);
4. the better of the two on the model is improved by ``core.descend`` (the cell program, ``passes`` passes) and its
   first week is sent, when the model says it beats the rules' plan by ``min_gain``.

The hybrid's parts (``hybrid_agent.py``, its ``*_part.py`` and ``sbfv/``), ``sim_model.py`` and ``plan_core.py`` sit beside
this file: ``../../build.py`` copies them into a folder that stands alone. The program is solved with SciPy's
bundled HiGHS. Numbers of ``PARAMS`` come from ``regime.json`` beside this file when there is one.
"""

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OVERRIDE, HOLD = 1, 2  # release_mode codes

PARAMS = {
    "passes": 2,  # descent passes a week
    "first_passes": 6,  # in the first week
    "carry": True,  # last week's plan is a candidate
    "pending": True,  # announced prohibitions take effect in the forecast
    "min_gain": 0.0,  # bn USD the plan must gain on the rules' plan, on the model, to be played
    "hints": True,
    "solve_seconds": 5.0,  # HiGHS's time limit for one cell; a cell not solved in time ends the week's passes
    "play": "plan",  # "plan", or "rules": plan but send the hybrid's action (a dry run for the timings)
    # the forecast: the chance per week that a cut lasts, by field and pool ("u_ct", "kappa_ct": edges and straits of
    # container cargo; "u_tb", "kappa_tb": of tankers; "G_bar"); a field not named stays as observed
    "recover": {},
    # the plan sends at least this share of what the rules' plan sends, slot by slot and week by week, by the kind
    # of slot: "wafer", "order" (a fuel from its source), "valve" (a fuel from a terminal into its grid), "raw", "pack"
    "floor": {},
    "floor_weeks": 0,  # only in the first weeks of the plan (0: every week)
    # the plan keeps at least this share of the fuel the rules' plan keeps at the terminals, week by week: the
    # rules' buffer against a route that fails (the forecast knows no such thing)
    "stock_floor": 0.0,
    # a lot started is worth this share of its chip's shortage penalty beyond what the forecast lets it sell (a chip
    # in stock can leave when a way out opens; the forecast knows no such thing); not paid in the last 14 weeks
    "lot_bonus": 0.0,
    # USD per unit a slot sends above or below the rules' plan, by the kind of slot (as "floor"): the plan leaves
    # the rules' plan only where that pays
    "anchor": {},
    # the forecast of a cut edge by what cut it, read from the cut's depth, and by its age (the generator's laws):
    # a sanction's side effect (a quarter left) ends as sanctions do, a port's stoppage (0.07 left) within weeks, a
    # slowdown within months; a conflict's cut and a cut that was there in week 1 stay. The capacity planned is the
    # expected one. "ct" / "tb": for edges of container cargo / of tankers. "straits": a strait's throughput by the
    # age of its disruption (hub/FINDINGS.md: 0.66 to 0.85 a week while young, 0.91 after 4 weeks, 0.98 after 6)
    "typed": {},
}
if (HERE / "regime.json").is_file():
    PARAMS |= json.loads((HERE / "regime.json").read_text())


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_hybrid = _load(f"{HERE.name}_hybrid", HERE / "hybrid_agent.py")  # its lp_part puts sbfv/ on the path
_model = _load(f"{HERE.name}_model", HERE / "sim_model.py")
_core = _load(f"{HERE.name}_core", HERE / "plan_core.py")
_core.PKG = "sbfv"


class Agent:
    def __init__(self, config=None):
        self.p = dict(PARAMS)
        self.base = _hybrid.Agent(config)
        self.rules = _hybrid.Agent(config)
        self.rules.planner = None  # the rules alone: cheap enough to play every week left
        self.model = None if self.base.planner is None else _model.Model(config, self.base.planner)
        self.acts = None  # the plan carried from week to week: (flows, overrides, holds) per week left
        if self.model is not None:
            inst = self.model.inst
            fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
            self.tanker_edge = np.array([bool(set(e.K) & fuels) for e in inst.edges])
            fed = {e.head for e in inst.edges}
            chips = np.asarray(self.base.chips.kind)
            self.kinds = []  # per action slot, as lab/anastasiia/search_lab/search.py names them
            for s, (e, k, lane) in enumerate(inst.action_slots):
                edge = inst.edges[e]
                if k in fuels:
                    valve = lane is None and inst.nodes[edge.head].grid is not None and edge.tail in fed
                    self.kinds.append("valve" if valve else "order")
                else:
                    self.kinds.append({1: "wafer", 2: "raw", 3: "pack"}.get(int(chips[s]), "other"))
            price_of = {}
            for dm in inst.demands:
                price_of[dm.k] = max(price_of.get(dm.k, 0.0), float(dm.pi))
            packed = {raw: pk for o in inst.osats for raw, pk in inst.nodes[o].osat.packages.items()}
            self.chip_worth = np.array([price_of.get(packed.get(inst.nodes[f].fab.product, -1), 0.0) for f in inst.fabs])
            E = len(inst.edges)
            self.cut_age, self.cut_left, self.cut_seen = np.zeros(E, dtype=int), np.ones(E), np.zeros(E, dtype=bool)
            self.strait_age = np.zeros(len(inst.chokepoints), dtype=int)
            self.terminal_fuel = sorted({inst.slot_index[(inst.edges[e].tail, k)]
                                         for s, (e, k, _lane) in enumerate(inst.action_slots) if self.kinds[s] == "valve"})
        self.log = []  # per week: CPU seconds, the model's cost of the rules' plan, of the plan sent, which one

    def act(self, observation):
        t0 = time.process_time()
        fallback = self.base.act(observation)
        ruled = self.rules.act(observation)  # keeps the rules' own memory in step with the episode
        if self.model is None:
            return fallback
        try:
            action, note = self._plan(observation, fallback)
        except Exception as exc:  # the hybrid's action is a complete week
            action, note = None, ("error", repr(exc)[:200])
            self.acts = None
        self.log.append((time.process_time() - t0, *note))
        del ruled
        return fallback if action is None or self.p["play"] == "rules" else action

    def _plan(self, observation, first):
        m, p = self.model, self.p
        week = int(np.asarray(observation["week"]).ravel()[0])
        H = m.inst.T - week + 1
        arrays = None
        if p["typed"]:
            self._ages(week)
        if p["recover"] or p["typed"]:
            arrays = {k: np.array(a) for k, a in m.forecast(observation, H, pending=p["pending"]).items()}
            if p["recover"]:
                self._recover(arrays)
            if p["typed"]:
                self._typed(arrays)
        w = m.window(observation, H, arrays, pending=p["pending"])
        policy, action, ruled = copy.deepcopy(self.rules), first, []
        for h in range(H):  # the rules' plan: the hybrid's action now, the rules after it
            ruled.append(m.wire(action, np.asarray(w.marks.prohibited[w.state.week])))
            m.step(w, action)
            if h + 1 < H:
                action = policy.act(m.flat(observation, w))
        ep = _core.Episode(w.inst, w.marks, anchored=bool(p["anchor"]))
        recs_rules, J_rules = ep.simulate(ruled)
        start, J_start, name = ruled, J_rules, "rules"
        if p["carry"] and self.acts is not None and len(self.acts) > 1:
            carried = ep.clean(self.acts[1:])
            _recs, J_carried = ep.simulate(carried)
            if J_carried < J_rules:
                start, J_start, name = carried, J_carried, "carried"
        passes = p["first_passes"] if self.acts is None else p["passes"]
        tweak = None
        if p["floor"] or p["stock_floor"] > 0:
            last = p["floor_weeks"] or H
            floor = [(ep.col("x", t, *m.inst.action_slots[s]), p["floor"][self.kinds[s]] * q)
                     for t, (fl, _ov, _ho) in enumerate(ruled[:last], start=1) for s, q in fl.items()
                     if p["floor"].get(self.kinds[s], 0) > 0]
            if p["stock_floor"] > 0:
                floor += [(ep.col("I", t, s), p["stock_floor"] * float(rec.stock[s]))
                          for t, rec in enumerate(recs_rules[:last], start=1) for s in self.terminal_fuel if rec.stock[s] > 0]

            def tweak(C):
                for j, q in floor:
                    C.lb[j] = max(C.lb[j], min(q, C.ub[j]))

        bonus = (p["lot_bonus"] * self.chip_worth, H - 14) if p["lot_bonus"] > 0 else None
        price = np.array([p["anchor"].get(kind, 0.0) for kind in self.kinds]) if p["anchor"] else None
        d = (_core.descend(ep, start, iters=passes, hints=p["hints"], tweak=tweak, time_limit=p["solve_seconds"],
                           anchor=ruled if price is not None else None, price=price, bonus=bonus,
                           force_first=tweak is not None)
             if passes > 0 else {"acts": start, "J": J_start})
        self.acts = d["acts"]
        self.last = (ep, ruled, d)  # for a look inside from a lab script
        gain = (J_rules - d["J"]) / 1e11
        if gain < p["min_gain"]:
            return None, (J_rules / 1e11, d["J"] / 1e11, name + ":kept rules")
        return self._flat(d["acts"][0]), (J_rules / 1e11, d["J"] / 1e11, name)

    def _recover(self, arrays: dict) -> None:
        """Let cuts end in the forecast, in expectation: a capacity below its calm value closes the gap by 1 - s^h
        after h weeks, s the chance per week that the cut lasts; this week stays as observed."""
        calm, rates = self.base.planner.calm, self.p["recover"]
        H = len(arrays["u"])
        h = np.arange(H, dtype=float)

        def back(cur, target, s, mask):
            gap = np.where(np.isfinite(target), np.maximum(target - cur, 0.0), 0.0) * mask
            return cur + gap * (1.0 - s**h).reshape(-1, *([1] * (cur.ndim - 1)))

        for name, mask in (("u_ct", ~self.tanker_edge), ("u_tb", self.tanker_edge)):
            if name in rates:
                arrays["u"] = back(arrays["u"], calm["u"], rates[name], mask)
        for name, pool in (("kappa_tb", 0), ("kappa_ct", 1)):
            if name in rates:
                mask = np.zeros(2)
                mask[pool] = 1.0
                arrays["kappa"] = back(arrays["kappa"], calm["kappa"], rates[name], mask)
        if "G_bar" in rates:
            arrays["G_bar"] = back(arrays["G_bar"], calm["grid_G"], rates["G_bar"], 1.0)
        for now, week in (("u_now", "u"), ("kappa_now", "kappa"), ("G_bar_now", "G_bar")):
            arrays[now] = arrays[week].copy()

    def _ages(self, week: int) -> None:
        """How long every cut edge and disrupted strait has been as it is now (in weeks seen)."""
        v, calm = self.base.planner.memory.values, self.base.planner.calm
        with np.errstate(invalid="ignore", divide="ignore"):
            left = np.where(np.isfinite(calm["u"]) & (calm["u"] > 0), v["u"] / calm["u"], 1.0)
        cut = left < 0.999
        same = cut & (np.abs(left - self.cut_left) < 1e-3)
        self.cut_seen = np.where(same, self.cut_seen, cut & (week > 1))  # its start was seen, not carried in
        self.cut_age = np.where(same, self.cut_age + 1, cut.astype(int))
        self.cut_left = left
        low = (v["kappa"] < 0.999 * calm["kappa"]).any(axis=1)
        self.strait_age = np.where(low, self.strait_age + 1, 0)

    def _typed(self, arrays: dict) -> None:
        """The expected capacity of cut edges and disrupted straits over the weeks left (``PARAMS["typed"]``)."""
        from scipy.special import ndtr

        calm, on = self.base.planner.calm, self.p["typed"]
        H = len(arrays["u"])
        h = np.arange(H, dtype=float)

        def lasts(kind: str, age: float) -> np.ndarray:
            """The chance that a cut of this kind, ``age`` weeks old, still holds h weeks from now."""
            if kind == "sanction":  # LN(ln 52 + 2 x 0.4307, 2.0) days
                S = lambda w: 1.0 - ndtr((np.log(7.0 * np.maximum(w, 1e-9)) - 4.8125) / 2.0)  # noqa: E731
            elif kind == "stoppage":  # LN(0.48, 0.67) weeks
                S = lambda w: 1.0 - ndtr((np.log(np.maximum(w, 1e-9)) - 0.48) / 0.67)  # noqa: E731
            else:  # a slowdown: LN(1.93, 0.77) weeks
                S = lambda w: 1.0 - ndtr((np.log(np.maximum(w, 1e-9)) - 1.93) / 0.77)  # noqa: E731
            return S(age + h) / max(float(S(age)), 1e-9)

        u = arrays["u"]
        for e in np.flatnonzero((self.cut_age > 0) & self.cut_seen):
            if not on.get("tb" if self.tanker_edge[e] else "ct"):
                continue
            left = float(self.cut_left[e])
            kind = ("sanction" if abs(left - 0.25) < 0.005 else "stoppage" if abs(left - 0.07) < 0.005
                    else "slowdown" if 0.405 < left < 0.805 else None)  # 0.40, 0.27, 0.90 and their products: a conflict
            if kind is None:
                continue
            gone = 1.0 - lasts(kind, float(self.cut_age[e]))
            gone[0] = 0.0  # this week as observed
            u[:, e] = u[:, e] + (calm["u"][e] - u[:, e]) * gone
        arrays["u_now"] = u.copy()
        if on.get("straits"):
            stay = np.array([0.75, 0.75, 0.80, 0.91, 0.91] + [0.98] * 200)
            for c in np.flatnonzero(self.strait_age > 0):
                a = int(self.strait_age[c])
                holds = np.concatenate([[1.0], np.cumprod(stay[a - 1 : a - 1 + H - 1])])
                arrays["kappa"][:, c, :] += (calm["kappa"][c] - arrays["kappa"][:, c, :]) * (1.0 - holds)[:, None]
            arrays["kappa_now"] = arrays["kappa"].copy()

    def _flat(self, act):
        """One week's (flows, overrides, holds) as the environment's flat arrays."""
        m = self.model
        fl, ov, ho = act
        flows = np.zeros(len(m.slot_edge))
        for s, q in fl.items():
            flows[s] = q
        mode, qty = np.zeros(len(m.pairs), dtype=np.int64), np.zeros(len(m.ov_pair))
        for o, q in ov.items():
            mode[m.ov_pair[o]], qty[o] = OVERRIDE, q
        index = {pair: i for i, pair in enumerate(m.pairs)}
        for pair in ho:
            mode[index[pair]] = HOLD
        return {"flows": flows, "override_qty": qty, "release_mode": mode}
