"""Futures of the network sampled from the generator itself, blind to everything that has not happened yet.

A scenario at week ``t`` (the observation shows the instant ``s = t - 1``) is a list of events:

- every event of the episode that has started by ``s`` and whose effect is not over, with its own type, target, onset
  and severity (what an agent reads off the observations) and a duration DRAWN from the generator's law of that type,
  given what the observations up to ``s`` say about it (``World.facts``: it still runs; for a regional conflict the
  phase of its war profile, and the step between the phases when it fell inside the episode);
- the events of an independent draw of the same generator (root ``ROOT_NEW``, never the episode's own root) whose
  onset is later than ``s``, as they are.

``World.marks(j, week)`` composes the weekly marks of scenario ``j`` with the package's own rules
(``shockbench_flow.marks``), for all weeks of the episode; a planner reads the rows of its window. A scenario keeps
the durations it drew from week to week and draws one again only when the observations contradict it.

Two options replace what the first form takes from outside the observations:

- ``new="conditional"``: the new events are not an independent draw of the generator but one joint draw given what
  is seen at ``s`` (``info/filter.py``, ``Futures``): the announcement threads alive at ``s`` with their chance of
  being real and their lead law, the children of the events already seen, immigrants at the rate the warning
  scores and the conflicts seen imply. The weather closures and port strikes, which nothing announces or excites,
  stay those of the independent draw;
- ``ages="mixture"``: an event carried in from before the episode has no known onset. Its age is drawn from the
  stationary law of the ages of events of its kind that still run (and its duration given that age); for a
  regional conflict from the same law given what its war profile and its war-risk class have shown.

The episode's truth is read in a few places only, each returning what is observable at ``s`` and nothing else:
``World.facts`` (the events that have started), ``World.evidence`` (what a carried event has shown),
``World.seen`` (which events started inside the episode, and when they were first seen), ``World.messages`` and
``World.scores`` (the announcements and warning scores shown by ``s``). ``tampered`` rewrites the truth in everything
that is not observable (the events that have not started are dropped, the ends still ahead are moved, with
``ages="mixture"`` the onsets of the carried events too; ``tampered_feed`` drops the messages and scores of later
weeks); ``blind_check`` plays two worlds, one on the truth and one on the tampered truth, and asserts that every
scenario of every week is the same in both.
"""

import math
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT_NEW = 999  # the root of the independent draws of new events (free: see CLAUDE.md, the roots in use)
ROOT = Path(__file__).resolve().parents[4]
DRAWS = ROOT / "outputs" / "frontier_lab" / "scen" / "draws"
INFO = Path(__file__).resolve().parents[1] / "info"  # the generator as a filtering model (``filter.Futures``)
CONFLICT, CLOSURE, STRIKE = "regional_conflict", "militarised_closure", "port_strike"
UNEXCITED = ("weather_closure", "port_strike")  # the generator's Poisson components: nothing announces or excites them
# the fields of the forecast a scenario replaces (groups of the lab agent's ``TRUTH``): everything the events make,
# but not the plants' recovery after a conflict (its dead time is an end still ahead) and not the demand
GROUPS = ("edges", "straits", "grids", "prohibitions", "supply", "tariffs")


def generator(task: str):
    from shockbench_flow.hosting.tasks import task_generator

    return task_generator(task)


def drawn_events(task: str, episode: int) -> tuple:
    """The events of episode ``episode`` of the root ``ROOT_NEW`` (kept on disk: a draw takes seconds)."""
    path = DRAWS / f"{task}_{ROOT_NEW}_{episode}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    from shockbench_flow import marks as M
    from shockbench_flow.disruption.sampler import sample_omega

    inst, params = generator(task)
    omega = sample_omega(inst, params, ROOT_NEW, episode, "train")
    events = M.read_events(inst.at_digest(str(omega.arrays["meta_instance_digest"])), omega.arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.tmp")
    tmp.write_bytes(pickle.dumps(events))
    tmp.replace(path)
    return events


def draw(law, lo: float, hi: float, u: float) -> float:
    """A duration from ``law`` given that it lies in (``lo``, ``hi``], by inversion at the uniform ``u``."""
    a = law.cdf(lo) if lo > 0 else 0.0
    b = law.cdf(hi) if math.isfinite(hi) else 1.0
    if b - a < 1e-10:  # the law has (numerically) nothing there: just past ``lo``, or the middle of the interval
        return lo * 1.5 + 1.0 if math.isinf(hi) else 0.5 * (lo + hi)
    d = law.quantile(min(max(a + u * (b - a), 1e-12), 1.0 - 1e-12))
    return float(min(max(d, lo + 1e-6), hi))


def _solve(cdf, u: float, lo: float, hi: float) -> float:
    """The point of [``lo``, ``hi``] where the non-decreasing ``cdf`` reaches ``u`` (bisection)."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if cdf(mid) < u else (lo, mid)
    return 0.5 * (lo + hi)


class World:
    def __init__(self, task: str, inst, real: tuple, K: int, seed: int, episode: int, new: bool | str = True,
                 ends: bool = True, mark_params=None, ages: str = "true", feed: dict | None = None) -> None:
        """``real``: the episode's own events (``marks.read_events``); ``inst``: the episode's instance. ``new``:
        True (an independent draw of the generator), False (none) or "conditional"; ``ages``: "true" or "mixture"
        (the module docstring). ``feed``: what the observations show of the announcements and the warning, {"columns":
        the messages' columns, "counts": how many of them each week shows, "scores": (T, units)}; "conditional"
        needs it."""
        if ages not in ("true", "mixture") or new not in (True, False, "conditional"):
            raise ValueError(f"ages: 'true' or 'mixture'; new: True, False or 'conditional'; got {ages!r}, {new!r}")
        from shockbench_flow import marks as M
        from shockbench_flow.disruption import targets
        from shockbench_flow.omega import codes

        self.M, self.inst, self.K, self.new, self.ends, self.ages = M, inst, K, new, ends, ages
        self._real = tuple(real)  # read by ``facts``, ``evidence`` and ``seen`` alone
        self._feed = feed  # read by ``messages`` and ``scores`` alone
        _gen_inst, gen = generator(task)
        self.mp = mark_params if mark_params is not None else gen.marks
        self.W = float(self.mp.war_profile_window)
        self.name = codes.EVENT_TYPES
        self.laws = dict(gen.laws.duration)
        self.strike = (gen.poisson.stoppage_duration, gen.poisson.slowdown_duration)
        self.strait_weeks = float(gen.laws.strait_closure_weeks)
        self.strait_sev = float(gen.laws.strait_closure_severity)
        self.straits = set(targets.strait_chokepoints(inst, gen).values())  # the nodes of the dyad straits
        self.adjacent = {m for regions in inst.chokepoint_adjacency.values() for m in regions}
        self.rng = [np.random.default_rng([int(seed) % (1 << 32), 7919, j]) for j in range(K)]
        self.dur: list[dict] = [{} for _ in range(K)]  # scenario -> {the event's row: the duration it drew}
        self.draws = [drawn_events(task, episode * 100 + j) if new else () for j in range(K)]
        self.t = M._weeks(inst.T)
        self.k_mu = np.array([inst.nodes[c].chokepoint.kappa0 for c in inst.chokepoints]).reshape(-1, 2)
        yb = np.array([inst.nodes[g].grid.base_load for g in inst.grids], dtype=float)
        self.y_bar = np.broadcast_to(yb, (inst.T, len(yb))).copy()
        self.redrawn = 0  # durations drawn again because the observations contradicted them
        self.key = (ROOT_NEW, int(seed) % (1 << 32), int(episode))  # of the draws the two options add
        self.burn_in = float(gen.burn_in)
        # scenario -> {a carried event's row: (its evidence, the onset and the duration drawn, how many draws)}
        self.age: list[dict] = [{} for _ in range(K)]
        self.restoration = M.restoration_regions(inst)
        self.futures = None
        if new == "conditional":
            if feed is None:
                raise ValueError("new='conditional' reads the announcements and the warning scores shown (feed)")
            if str(INFO) not in sys.path:
                sys.path.insert(0, str(INFO))
            from filter import Futures, Tables

            self.futures = Futures(Tables.of(task), K, seed, episode, ROOT_NEW)

    # ----- the only reader of the truth --------------------------------------------------------------------------
    def facts(self, s: float) -> list:
        """What the observations up to the instant ``s`` show of the episode's events: for each event that has
        started and whose effect is not over, (the event WITHOUT its duration, lo, hi, known): its duration lies in
        (lo, hi], or is ``known`` because its end, or the step that fixes it, has been seen."""
        out = []
        for q in self._real:
            a = s - q.onset
            if a < 0:
                continue  # not started: nothing of it is seen
            d, kind = q.duration, self.name[q.type]
            lo, hi, known = a, math.inf, None
            if kind == CONFLICT and q.counterpart >= 0:
                w0 = max(d, self.W)  # the first phase of the war profile lasts max(duration, window) weeks
                if a >= w0 + self.W:
                    continue  # both phases are over
                if a >= w0:  # second phase
                    if q.onset + w0 > 0:
                        known = d  # the step between the phases was seen inside the episode
                    else:  # carried in after the step: it fell before the episode and less than a window ago
                        lo, hi = (a - self.W if a - self.W >= self.W else 0.0), -q.onset
                elif a < self.W:  # first phase, younger than the window: only a war-risk class tells if it still runs
                    if q.region in self.adjacent:
                        if d > a:
                            lo, hi = a, math.inf
                        elif q.onset + d > 0:
                            known = d  # the class was seen to go off
                        else:
                            lo, hi = 0.0, -q.onset
                    else:
                        lo, hi = 0.0, math.inf
            elif a >= d:
                continue  # over
            out.append((replace(q, duration=math.nan), lo, hi, known))
        return out

    def evidence(self, s: float) -> dict:
        """What an event carried in from before the episode, and still in effect at ``s``, has shown by then, never
        its onset or its duration: {row: the evidence}.

        ("plain",): it runs. ("strait", row): the closure of a dyad's strait, which lasts as long as the first phase
        of the conflict ``row``. A regional conflict with a counterpart shows the phase of its war profile and, if
        its region lies at a strait, its war-risk class: ("W1", the instant of the step) when the step between the
        phases fell inside the episode, ("W1",) when before it; ("W0", "on") while the class is on, ("W0", "off0")
        when it was off from the start, ("W0", "off", the instant it went off) when it went off inside the episode,
        ("W0", None) where no strait shows it."""
        out = {}
        for q in self._real:
            if q.onset > 0:
                continue
            ev = self._evidence(q, s)
            if ev is not None:
                out[q.row] = ev
        return out

    def _evidence(self, q, s: float) -> tuple | None:
        kind, d = self.name[q.type], q.duration
        if kind == CONFLICT and q.counterpart >= 0:
            w0 = q.onset + max(d, self.W)  # the end of the first phase
            if s >= w0 + self.W:
                return None
            if s >= w0:
                return ("W1", w0) if w0 > 0 else ("W1",)
            if q.region not in self.adjacent:
                return ("W0", None)
            off = q.onset + d  # the war-risk class lasts the conflict's own duration
            return ("W0", "on") if off > s else ("W0", "off0") if off <= 0 else ("W0", "off", off)
        if s - q.onset >= d:
            return None
        if self._derived(q):
            mine = (q.onset, q.region)
            of = [c.row for c in self._real if self.name[c.type] == CONFLICT and (c.onset, c.region) == mine]
            if of:
                return ("strait", of[0])
        return ("plain",)

    def seen(self, s: float) -> tuple:
        """Which events of the cluster process the observations up to ``s`` have shown to start inside the episode,
        [(row, type, region, target, the instant it was first seen, whether it excites others)], and the regional
        conflicts running at ``s``, [(region, the instant it was first seen, or None if it was carried in)]. An event
        is seen at the first observed instant after its onset, if it still runs then."""
        started, running = [], []
        for q in self._real:
            kind = self.name[q.type]
            if q.onset > s or kind in UNEXCITED:
                continue
            first, span = math.ceil(q.onset), q.duration
            if kind == CONFLICT:
                if q.counterpart >= 0:
                    span = max(q.duration, self.W) + self.W
                elif q.region not in self.restoration and q.region not in self.adjacent:
                    continue  # a conflict without a counterpart, a plant or a strait changes nothing an agent sees
                if s < q.onset + span:
                    running.append((q.region, first if q.onset > 0 else None))
            if q.onset > 0 and first < q.onset + span:
                started.append((q.row, q.type, q.region, q.target, first, not self._derived(q)))
        return started, running

    def messages(self, s: float) -> dict:
        """The columns of the messages the observation of the instant ``s`` shows (week ``s + 1``'s)."""
        n = self._feed["counts"][int(s)]
        return {name: column[:n] for name, column in self._feed["columns"].items()}

    def scores(self, s: float):
        """The warning scores the observation of the instant ``s`` shows (regions, dyads, straits)."""
        return self._feed["scores"][int(s)]

    # ----- scenarios ------------------------------------------------------------------------------------------------
    def _law(self, q):
        kind = self.name[q.type]
        if kind == STRIKE:
            return self.strike[0] if q.severity >= 0.9 else self.strike[1]
        return self.laws[kind]

    def _derived(self, q) -> bool:
        """A closure of a dyad strait that a conflict brings with it (the generator gives it severity 1)."""
        return self.name[q.type] == CLOSURE and q.target in self.straits and q.severity == self.strait_sev

    def events(self, j: int, week: int) -> tuple:
        """Scenario ``j`` as week ``week``'s observation leaves it: a tuple of ``marks.Event``."""
        s = week - 1.0
        out = self._running(j, s) if self.ages == "true" else self._running_mix(j, s)
        if self.new == "conditional":
            out += [q for q in self.draws[j] if q.onset > s and self.name[q.type] in UNEXCITED]
            out += self.futures.draw(j, s, *self.seen(s), self.messages(s), self.scores(s))
        elif self.new:
            out += [q for q in self.draws[j] if q.onset > s]
        return tuple(out)

    def fresh(self, j: int, week: int) -> tuple:
        """The events of scenario ``j`` that have not started by week ``week``'s observation."""
        return tuple(q for q in self.events(j, week) if q.onset > week - 1.0)

    def _running(self, j: int, s: float) -> list:
        """The events that run at ``s``, each with a duration drawn given its own (true) age."""
        seen = self.facts(s)
        dur, rng, out = self.dur[j], self.rng[j], []
        conflicts = {(q.onset, q.region): q.row for q, _lo, _hi, _k in seen if self.name[q.type] == CONFLICT}
        alive = set()
        for q, lo, hi, known in seen:
            if self._derived(q) and (q.onset, q.region) in conflicts:
                continue  # after its conflict, below
            d = dur.get(q.row)
            if known is not None:
                d = known
            elif not self.ends:  # a lab's variant, the model's own forecast of the ends: a short cut is over at the
                # median of its law given its age, every other event stays ("as observed")
                short = self.name[q.type] in (STRIKE, "weather_closure")
                d = draw(self._law(q), lo, hi, 0.5) if short or math.isfinite(hi) else max(s - q.onset, 0.0) + 10_000.0
            elif d is None or not (lo < d <= hi):
                self.redrawn += d is not None
                d = draw(self._law(q), lo, hi, float(rng.random()))
            dur[q.row] = d
            alive.add(q.row)
            out.append(replace(q, duration=d))
        for q, _lo, _hi, _known in seen:  # the strait of a conflict is shut for the conflict's first phase
            if self._derived(q) and (q.onset, q.region) in conflicts:
                out.append(replace(q, duration=max(dur[conflicts[(q.onset, q.region)]], self.strait_weeks)))
        for row in [r for r in dur if r not in alive]:
            del dur[row]
        return out

    def _running_mix(self, j: int, s: float) -> list:
        """The events that run at ``s``; one carried in from before the episode with an onset and a duration drawn
        from the stationary law of the events of its kind that have shown what it has (``evidence``), kept from
        week to week until the observations contradict them. An event of the episode is as in ``_running``."""
        shown = self.evidence(s)
        seen = [(q, lo, hi, known) for q, lo, hi, known in self.facts(s) if q.onset > 0]
        dur, rng, out = self.dur[j], self.rng[j], []
        conflicts = {(q.onset, q.region): q.row for q, _lo, _hi, _k in seen if self.name[q.type] == CONFLICT}
        alive = set()
        for q, lo, hi, known in seen:
            if self._derived(q) and (q.onset, q.region) in conflicts:
                continue
            d = dur.get(q.row)
            if known is not None:
                d = known
            elif not self.ends:
                short = self.name[q.type] in (STRIKE, "weather_closure")
                d = draw(self._law(q), lo, hi, 0.5) if short or math.isfinite(hi) else max(s - q.onset, 0.0) + 10_000.0
            elif d is None or not (lo < d <= hi):
                self.redrawn += d is not None
                d = draw(self._law(q), lo, hi, float(rng.random()))
            dur[q.row] = d
            alive.add(q.row)
            out.append(replace(q, duration=d))
        for q, _lo, _hi, _known in seen:
            if self._derived(q) and (q.onset, q.region) in conflicts:
                out.append(replace(q, duration=max(dur[conflicts[(q.onset, q.region)]], self.strait_weeks)))
        for row in [r for r in dur if r not in alive]:
            del dur[row]
        by_row = {q.row: q for q in self._real}  # only the fields an agent reads off the cut: type, target, depth
        drawn = {}
        for row, ev in shown.items():
            if ev[0] != "strait":
                drawn[row] = self._age(j, by_row[row], ev, s)
                out.append(replace(by_row[row], onset=drawn[row][0], duration=drawn[row][1]))
        for row, ev in shown.items():
            if ev[0] == "strait":
                on, d = drawn[ev[1]] if ev[1] in drawn else self._age(j, by_row[row], ("plain",), s)
                out.append(replace(by_row[row], onset=on, duration=max(d, self.strait_weeks) if ev[1] in drawn else d))
        for row in [r for r in self.age[j] if r not in shown]:
            del self.age[j][row]
        return out

    # ----- the age of a carried event -------------------------------------------------------------------------------
    def _age(self, j: int, q, ev: tuple, s: float) -> tuple:
        """(onset, duration) of the carried event ``q`` in scenario ``j``: the kept draw while it agrees with what
        the event has shown (``ev``), else a new one."""
        if not self.ends:  # the model's own forecast of the ends: a short cut is over at the median of its law, a
            # conflict whose step or class dates it at the median of what is left, every other event stays
            short = self.name[q.type] in (STRIKE, "weather_closure")
            if short or ev[0] == "W1" or ev[:2] in (("W0", "off0"), ("W0", "off")):
                return self._draw_age(q, ev, s, np.array([0.5, 0.5]))
            return -1.0, s + 10_001.0
        kept = self.age[j].get(q.row)
        if kept is not None and kept[0] == ev and self._fits(ev, kept[1], kept[2], s):
            return kept[1], kept[2]
        n = kept[3] + 1 if kept is not None else 0
        self.redrawn += kept is not None
        u = np.random.default_rng([*self.key, j, 11, q.row, n]).random(2)
        on, d = self._draw_age(q, ev, s, np.clip(u, 1e-12, 1.0 - 1e-12))
        self.age[j][q.row] = (ev, on, d, n)
        return on, d

    def _fits(self, ev: tuple, on: float, d: float, s: float) -> bool:
        """Whether a carried event with this onset and duration would have shown ``ev`` at ``s`` (and still run)."""
        if ev[0] == "plain" or ev[:2] == ("W0", "on"):
            return on + d > s
        if ev[0] == "W1":
            return len(ev) == 2 or on + max(d, self.W) + self.W > s
        return on + max(d, self.W) > s  # the first phase, the class off or not shown

    def _draw_age(self, q, ev: tuple, s: float, u) -> tuple:
        """An (onset, duration) of the carried event ``q`` from the stationary law of the events of its kind (onsets
        at a constant rate over the burn-in, durations by the kind's law) given what it has shown by ``s``."""
        law, W = self._law(q), self.W
        F, tail = law.cdf, law.tail_integral
        if ev[0] == "plain" or ev[:2] == ("W0", "on"):  # it runs: the age has the density S(age + s), then the duration
            top = max(tail(s) - tail(s + self.burn_in), 1e-300)
            age = _solve(lambda a: (tail(s) - tail(s + a)) / top, u[0], 0.0, self.burn_in)
            return -age, draw(law, age + s, math.inf, float(u[1]))
        if ev[0] == "W1":  # second phase: it ends a window after the step
            if len(ev) == 2:
                return ev[1] - W, W
            return -float(u[0]) * max(W - s, 1e-6) - W, W  # the step fell before the episode, at most W - s ago
        left = max(W - s, 1e-6)  # the first phase is not over: with a duration under W it began less than W - s ago
        if ev[1] == "off0":  # the class was off at the start: the duration is under the age, the age under W - s
            mass = lambda a: a - (tail(0.0) - tail(a))  # noqa: E731 - the integral of F up to a
            age = _solve(lambda a: mass(a) / max(mass(left), 1e-300), u[0], 0.0, left)
            return -age, draw(law, 0.0, age, float(u[1]))
        if ev[1] == "off":  # the class went off at ev[2]: the duration is the age then
            d = draw(law, ev[2], ev[2] + left, float(u[0]))
            return ev[2] - d, d
        # no class shown: the first phase lasts max(duration, W) and is seen in proportion to what is left of it
        m = max(W, s)
        atom = F(W) * max(W - s, 0.0)
        rest = tail(m) + (m - s) * (1.0 - F(m))
        if float(u[0]) * (atom + rest) < atom:
            return -float(u[1]) * (W - s), W
        v = (float(u[0]) * (atom + rest) - atom) / rest
        d = _solve(lambda x: 1.0 - (tail(x) + (x - s) * (1.0 - F(x))) / rest, v, m, m + 20.0 * self.burn_in)
        return -float(u[1]) * (d - s), d

    def marks_of(self, events: tuple) -> SimpleNamespace:
        """The weekly marks of a list of events, as ``marks.compute_marks`` forms them (all weeks of the episode)."""
        M, inst = self.M, self.inst
        g = M.graph_marks(inst, events, self.mp)
        wr = M._war_risk_class(inst, events, self.mp, self.t)
        hq, cwr = M._queue_and_transit(inst, wr)
        return SimpleNamespace(
            u=g["u"], u_now=g["u_now"], c=g["c"], o=g["o"], o_now=g["o_now"],
            kappa=self.k_mu[None, :, :] * g["o"][:, :, None], kappa_now=self.k_mu[None, :, :] * g["o_now"][:, :, None],
            supply=g["supply"], supply_now=g["supply_now"], G_bar=g["G_bar"], G_bar_now=g["G_bar_now"],
            y_bar=self.y_bar, y_bar_now=self.y_bar, prohibited=g["prohibited"], tariff=g["tariff"], wr_class=wr,
            h_queue=hq, c_wr=cwr,
        )  # fmt: skip

    def marks(self, j: int, week: int) -> SimpleNamespace:
        return self.marks_of(self.events(j, week))


PRESENT = ("u_now", "o_now", "G_bar_now", "supply_now", "prohibited", "tariff", "wr_class")


def present_bad(scenario: SimpleNamespace, truth, week: int) -> list:
    """The fields in which a scenario's present (the row week ``week``'s observation shows) is not the episode's."""
    row = week - 1
    return [name for name in PRESENT
            if not np.allclose(np.asarray(getattr(scenario, name)[row], dtype=float),
                               np.asarray(getattr(truth, name)[row], dtype=float), rtol=1e-9, atol=1e-9)]  # fmt: skip


def tampered(world: World, s: float) -> tuple:
    """The episode's events with everything the observations up to ``s`` do not show changed: an event that has not
    started is dropped, a duration that is not known is moved to another value the observations allow; with
    ``ages="mixture"`` the onset of an event carried in from before the episode is moved too (``_moved``)."""
    seen = {q.row: (lo, hi, known) for q, lo, hi, known in world.facts(s)}
    moved = _moved(world, s) if world.ages == "mixture" else {}
    out = []
    for q in world._real:
        if q.onset > s:
            continue
        if q.row in moved:
            out.append(moved[q.row])
            continue
        if q.row not in seen or seen[q.row][2] is not None:
            out.append(q)
            continue
        lo, hi, _known = seen[q.row]
        out.append(replace(q, duration=lo + 777.0 if math.isinf(hi) else lo + 0.37 * (hi - lo)))
    return tuple(out)


def _until(onset: float, end: float) -> float | None:
    """A duration that ends an event of this onset at ``end`` to the bit (None: the floats have none)."""
    d = end - onset
    for _ in range(16):
        if onset + d == end:
            return d
        d = math.nextafter(d, math.inf if onset + d < end else -math.inf)
    return None


def _moved(world: World, s: float, shift: float = 123.4) -> dict:
    """The carried events with other onsets and durations that would have shown the same at every instant up to
    ``s`` (asserted on ``World.evidence``), by row: what ``ages="mixture"`` must not depend on."""
    W, moved = world.W, {}
    carried = [q for q in world._real if q.onset <= 0]
    for q in carried:
        on, d, new = q.onset, q.duration, None
        if world._derived(q):
            continue  # with its conflict, below
        if world.name[q.type] == CONFLICT and q.counterpart >= 0:
            w0, off, at_strait = on + max(d, W), on + d, q.region in world.adjacent
            if s >= w0 + W:
                pass  # over: both phases were seen
            elif w0 <= 0:  # second phase since before the episode: the step fell at most W - s before it
                step = -0.37 * (W - s)
                new = (step - W - shift, W + shift)
            elif w0 <= s:  # the step was seen; the onset is free unless a class that went off earlier dates it
                if not at_strait or d >= W:
                    new = (on - shift, _until(on - shift, w0))
            elif not at_strait:
                new = (on - shift, max(d, W) + shift + 777.0)
            elif off > s:
                new = (on - shift, s - (on - shift) + 777.0)
            elif off <= 0:
                age = 0.37 * (W - s)
                new = (-age, 0.5 * age)
            else:
                age = 0.37 * (W - s)
                new = (-age, _until(-age, off))
        elif on + d <= s:  # over: its end was seen
            new = (on - shift, _until(on - shift, on + d))
        else:
            new = (on - shift, s - (on - shift) + 777.0)
        moved[q.row] = replace(q, onset=new[0], duration=new[1]) if new is not None and new[1] is not None else q
    for q in carried:  # the strait of a conflict is shut for the conflict's first phase
        if world._derived(q):
            of = [c for c in carried if world.name[c.type] == CONFLICT and (c.onset, c.region) == (q.onset, q.region)]
            c = moved[of[0].row] if of else None
            follows = c is not None and c is not of[0]
            moved[q.row] = replace(q, onset=c.onset, duration=max(c.duration, world.strait_weeks)) if follows else q
    # the moved truth shows what the truth shows, at every instant observed so far
    twin = World.__new__(World)
    twin.__dict__.update(world.__dict__)
    twin._real = tuple(moved.get(q.row, q) for q in world._real)
    for t in range(int(s) + 1):
        a, b = world.evidence(float(t)), twin.evidence(float(t))
        assert a == b, (t, {r: (a.get(r), b.get(r)) for r in set(a) | set(b) if a.get(r) != b.get(r)})
    return moved


def tampered_feed(feed: dict, s: float) -> dict:
    """The announcements and warning scores with everything the observation of the instant ``s`` does not show
    taken out: the later messages are dropped, the later scores overwritten."""
    n = feed["counts"][int(s)]
    scores = np.array(feed["scores"], dtype=float, copy=True)
    scores[int(s) + 1 :] = 9.9
    return {"columns": {name: tuple(column[:n]) for name, column in feed["columns"].items()},
            "counts": tuple(min(c, n) for c in feed["counts"]), "scores": scores}  # fmt: skip


def blind_check(task: str, inst, real: tuple, K: int, seed: int, episode: int, weeks=(2, 9, 20, 33, 47), feed=None,
                **kw) -> int:  # fmt: skip
    """Asserts that the scenarios of weeks 1..w do not depend on anything unseen at week w, for each w of ``weeks``:
    a world on the truth and a world on the truth tampered at w's instant give the same marks in every week up to w.
    Returns how many (week, scenario) pairs were compared."""
    compared = 0
    for w in weeks:
        if w > inst.T:
            continue
        a = World(task, inst, real, K, seed, episode, feed=feed, **kw)
        b = World(task, inst, tampered(a, w - 1.0), K, seed, episode,
                  feed=None if feed is None else tampered_feed(feed, w - 1.0), **kw)  # fmt: skip
        moved = sum(1 for x, y in zip(sorted(a._real, key=lambda q: q.row), sorted(b._real, key=lambda q: q.row))
                    if x.row == y.row and x.duration != y.duration) + (len(a._real) - len(b._real))  # fmt: skip
        assert moved > 0 or len(a.facts(w - 1.0)) == 0, "the tampering changed nothing"
        for week in range(1, w + 1):
            for j in range(K):
                ma, mb = a.marks(j, week), b.marks(j, week)
                for name, x in vars(ma).items():
                    assert np.array_equal(np.asarray(x), np.asarray(getattr(mb, name))), (w, week, j, name)
                compared += 1
    return compared
