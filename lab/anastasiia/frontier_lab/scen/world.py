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

The episode's true events are read in ONE place, ``World.facts``, which returns only what is observable at ``s``.
``tampered`` rewrites the truth in everything that is not (the events that have not started are dropped, the ends
still ahead are moved); ``blind_check`` plays two worlds, one on the truth and one on the tampered truth, and asserts
that every scenario of every week is the same in both.
"""

import math
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT_NEW = 999  # the root of the independent draws of new events (free: see CLAUDE.md, the roots in use)
ROOT = Path(__file__).resolve().parents[4]
DRAWS = ROOT / "outputs" / "frontier_lab" / "scen" / "draws"
CONFLICT, CLOSURE, STRIKE = "regional_conflict", "militarised_closure", "port_strike"
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


class World:
    def __init__(self, task: str, inst, real: tuple, K: int, seed: int, episode: int, new: bool = True,
                 ends: bool = True, mark_params=None) -> None:
        """``real``: the episode's own events (``marks.read_events``); ``inst``: the episode's instance."""
        from shockbench_flow import marks as M
        from shockbench_flow.disruption import targets
        from shockbench_flow.omega import codes

        self.M, self.inst, self.K, self.new, self.ends = M, inst, K, new, ends
        self._real = tuple(real)  # read by ``facts`` alone
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
        if self.new:
            out += [q for q in self.draws[j] if q.onset > s]
        return tuple(out)

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
    started is dropped, a duration that is not known is moved to another value the observations allow."""
    seen = {q.row: (lo, hi, known) for q, lo, hi, known in world.facts(s)}
    out = []
    for q in world._real:
        if q.onset > s:
            continue
        if q.row not in seen or seen[q.row][2] is not None:
            out.append(q)
            continue
        lo, hi, _known = seen[q.row]
        out.append(replace(q, duration=lo + 777.0 if math.isinf(hi) else lo + 0.37 * (hi - lo)))
    return tuple(out)


def blind_check(task: str, inst, real: tuple, K: int, seed: int, episode: int, weeks=(2, 9, 20, 33, 47), **kw) -> int:
    """Asserts that the scenarios of weeks 1..w do not depend on anything unseen at week w, for each w of ``weeks``:
    a world on the truth and a world on the truth tampered at w's instant give the same marks in every week up to w.
    Returns how many (week, scenario) pairs were compared."""
    compared = 0
    for w in weeks:
        if w > inst.T:
            continue
        a = World(task, inst, real, K, seed, episode, **kw)
        b = World(task, inst, tampered(a, w - 1.0), K, seed, episode, **kw)
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
