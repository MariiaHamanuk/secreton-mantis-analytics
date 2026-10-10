"""Short disruptions told from the observations alone, and when they are likely to be over.

The generator draws three kinds of event that cut a capacity for days or weeks, not for the rest of the episode, and
each leaves a mark an agent can read off ``graph_now`` (``lab/anastasiia/hazard_lab/GENERATOR.md``):

- a weather closure of a strait: its openness is exactly 0 (the only other event that closes a strait fully is a
  closure that comes with a war-risk class, and it lasts a year or more);
- a port strike that stops a port: the capacity of its sea edges falls to 0.07 of what it was;
- a port strike that slows a port: the capacity falls to between 0.4 and 0.8 of what it was (a conflict leaves
  exactly 0.40, 0.27 or 0.90, a sanction's side effect exactly 0.25).

``Watch.see`` is called once a week with the capacities and the straits as last observed. It keeps, per edge and per
strait, the short cuts running there and the week each was first seen. ``Watch.patch`` writes a forecast of the
window in which each of them is over when an event of its kind that has run as long is over with a given chance
(``LAWS``, the generator's laws of duration); everything else in the forecast stays as observed.
"""

import numpy as np
from scipy.special import ndtr


# the generator's laws of duration by kind of event (``shockbench_flow.disruption.profiles``): ("lognormal", mu,
# sigma, days in the unit) of ln(duration), or a mix [(weight, law), ...]
LAWS = {
    "weather_closure": ("lognormal", 1.79, 0.8, 7.0),
    "port_strike_stoppage": ("lognormal", 0.48, 0.67, 1.0),
    "port_strike_slowdown": ("lognormal", 1.93, 0.77, 1.0),
    "energy_shock": ("lognormal", 2.6391, 1.0, 7.0),
    "regional_conflict": ("lognormal", 5.531, 1.453, 7.0),
    "militarised_closure": [(0.6, ("lognormal", 4.519, 1.251, 7.0)), (0.4, ("lognormal", 4.6444, 1.0, 1.0))],
}
STOPPAGE = 0.07  # what a stopped port's sea edges keep
SLOWDOWN = (0.4, 0.8)  # what a slowed port's sea edges keep
# what the long events leave of an edge's capacity: a sanction's side effect, a conflict's third parties, its first
# year between the two sides, the step from that to its second year (0.27 / 0.40)
LONG = (0.25, 0.90, 0.40, 0.675)
TOL = 1e-6


def law_cdf(law, weeks: float) -> float:
    """P(duration <= ``weeks``) under a law of ``LAWS``."""
    if isinstance(law, list):
        return float(sum(w * law_cdf(part, weeks) for w, part in law))
    _kind, mu, sigma, per_week = law
    return float(ndtr((np.log(max(weeks * per_week, 1e-12)) - mu) / sigma))


def law_end(law, elapsed: float, q: float) -> float:
    """The duration (weeks) by which an event that has run ``elapsed`` weeks is over with chance ``q``."""
    done = law_cdf(law, elapsed)
    target = done + q * (1.0 - done)
    lo, hi = max(elapsed, 1e-6), max(2.0 * elapsed, 1.0)
    while law_cdf(law, hi) < target and hi < 1e5:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if law_cdf(law, mid) < target else (lo, mid)
    return hi


def _products() -> list:
    """Every product of a few long factors: (value, how many factors)."""
    out = {}
    for a in range(9):  # an edge may lie under many sanctions at once
        for b in range(6):
            for c in range(3):
                for d in range(3):
                    n = a + b + c + d
                    value = LONG[0] ** a * LONG[1] ** b * LONG[2] ** c * LONG[3] ** d
                    if value not in out or n < out[value]:
                        out[value] = n
    return sorted(out.items(), key=lambda item: item[1])


PRODUCTS = _products()


def split(change: float) -> list | None:
    """A fall of an edge's capacity (this week's over last week's, below 1) as the short cuts in it:
    [("port_strike_stoppage" | "port_strike_slowdown", factor)], the long factors left out; None: not a product of
    the factors the generator makes. A reading of long factors alone stands before one with a stopped port, either
    before one with a slowed port (whose factor is whatever is left), and among equals the one with fewer factors."""
    best = None
    for stops in (0, 1):
        for value, n in PRODUCTS:
            rest = change / (value * STOPPAGE**stops)
            short = [("port_strike_stoppage", STOPPAGE)] * stops
            if abs(rest - 1.0) < TOL:
                cand = ((0, stops, n), short)
            elif SLOWDOWN[0] - TOL <= rest <= SLOWDOWN[1] + TOL:
                cand = ((1, stops, n), short + [("port_strike_slowdown", rest)])
            else:
                continue
            if best is None or cand[0] < best[0]:
                best = cand
    return None if best is None else best[1]


class Watch:
    def __init__(self, inst) -> None:
        self.u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges], dtype=float)
        self.edges = np.flatnonzero(np.isfinite(self.u0) & (self.u0 > 0))
        fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
        self.fuel_edge = np.array([bool(fuels.intersection(e.K)) for e in inst.edges])  # no sea edge carries both
        self.ratio = np.ones(len(self.u0))  # each edge's capacity over its nominal one, as last seen
        self.cuts: dict[int, list] = {}  # edge -> [[kind, factor, the week it was first seen, carried in]]
        self.open = np.ones(len(inst.chokepoints))
        self.risk = np.zeros(len(inst.chokepoints), dtype=int)
        self.shut: dict[int, list] = {}  # strait -> [the week it was first seen shut, the openness it comes back to]
        self.week = 0
        self.unread = 0  # changes of an edge's capacity that no product of the generator's factors explains

    def see(self, week: int, u: np.ndarray, opened: np.ndarray, risk: np.ndarray) -> None:
        """Take week ``week``'s observation: the edges' capacities, the straits' openness and war-risk class."""
        if week <= self.week:  # one reading a week
            return
        first, self.week = self.week == 0, week
        ratio = np.ones(len(self.u0))
        ratio[self.edges] = np.asarray(u, dtype=float)[self.edges] / self.u0[self.edges]
        for e in self.edges[np.abs(ratio[self.edges] - self.ratio[self.edges]) > TOL * np.maximum(self.ratio[self.edges], 1e-9)]:
            e = int(e)
            old, new = float(self.ratio[e]), float(ratio[e])
            cuts = self.cuts.get(e, [])
            read = self._read(cuts, new / max(old, 1e-12))
            if read is None:  # no reading of the change: the level is read afresh, as in week 1
                read, cuts = ([], split(new) or []), []
                self.unread += 1
            gone, found = read
            cuts = [cut for i, cut in enumerate(cuts) if i not in gone] + [[kind, factor, week, first] for kind, factor in found]
            if cuts:
                self.cuts[e] = cuts
            else:
                self.cuts.pop(e, None)
        self.ratio = ratio
        opened, risk = np.asarray(opened, dtype=float), np.asarray(risk).astype(int)
        for c in range(len(opened)):
            if opened[c] <= TOL:
                if c not in self.shut:
                    # a full closure that comes with a war-risk class is the long kind: not watched
                    armed = risk[c] > (0 if first else self.risk[c])
                    back = 1.0 if first else float(self.open[c])
                    if not armed and back > TOL:
                        self.shut[c] = [week, back, first]
            else:
                self.shut.pop(c, None)
        self.open, self.risk = opened.copy(), risk.copy()

    @staticmethod
    def _read(cuts: list, change: float) -> tuple | None:
        """A change of an edge's capacity (this week's over last week's) as (the indices of ``cuts`` that ended, the
        short cuts that started: ``split``'s list), long events starting or ending beside them; None: no reading.
        A reading without a new slowed port (whose factor could be anything) stands before one with it, and among
        those the one with the fewest ended cuts: a cut may end in the very week another starts."""
        n = len(cuts)
        for slowed in (False, True):
            for size in range(n + 1):
                for mask in range(1 << n):
                    picked = [i for i in range(n) if mask >> i & 1]
                    if len(picked) != size:
                        continue
                    rest = change * float(np.prod([cuts[i][1] for i in picked])) if picked else change
                    if rest <= 1.0 + TOL:  # what is left is a fall: new cuts, short or long
                        found = split(min(rest, 1.0))
                        if found is not None and any(kind == "port_strike_slowdown" for kind, _f in found) == slowed:
                            return picked, found
                    elif not slowed and any(abs(rest * value - 1.0) < TOL for value, _n in PRODUCTS):
                        return picked, []  # a long event ended as well
        return None

    def _chance(self, chance: dict, kind: str, e: int | None = None):
        """The chance set for ``kind``: a number, or {"fuel": q, "chips": q} by what the edge carries (None: the
        kind, or that class of edge, stays as observed)."""
        q = chance.get(kind)
        if isinstance(q, dict):
            q = q.get("fuel" if e is not None and self.fuel_edge[e] else "chips")
        return q

    def running(self) -> dict:
        """What is watched now: {"edges": {edge: [(kind, factor, first seen, carried in)]}, "straits": {...}}."""
        return {"edges": {e: [tuple(c) for c in cuts] for e, cuts in self.cuts.items()},
                "straits": {c: tuple(v) for c, v in self.shut.items()}}

    def _share(self, kind: str, since: int, carried: bool, q: float, H: int, forget: float) -> tuple | None:
        """Of the window's weeks, the part each still lies under a cut of ``kind`` first seen in week ``since``, and
        whether the cut still runs at the instant each week's observation shows; None: it has run too long to be
        the kind it was read as."""
        elapsed = self.week - since + 0.5  # seen first at the instant ``since - 1``, begun within the week before it
        if elapsed > forget:
            return None
        left = law_end(LAWS[kind], elapsed, q) - elapsed
        weeks = np.arange(H, dtype=float)  # the window's week h covers [h, h + 1) from the instant observed
        return np.clip(left - weeks, 0.0, 1.0), (weeks < left).astype(float)

    def back_weeks(self, chance: dict, forget: float = 40.0) -> dict:
        """The first week of the window (1: next week) the forecast of ``patch`` has each watched element free of its
        short cuts: {"edges": {edge: week}, "straits": {strait: week}}; an element whose cut is not forecast leaves
        no entry."""
        out = {"edges": {}, "straits": {}}
        for e, cuts in self.cuts.items():
            weeks = []
            for kind, _factor, since, _carried in cuts:
                elapsed = self.week - since + 0.5
                q = self._chance(chance, kind, e)
                if q is None or elapsed > forget:
                    weeks = None  # a cut on this edge stays in the forecast: the edge is not back in the window
                    break
                weeks.append(int(np.ceil(law_end(LAWS[kind], elapsed, float(q)) - elapsed)))
            if weeks:
                out["edges"][e] = max(1, max(weeks))
        if chance.get("weather_closure") is not None:
            for c, (since, _back, _carried) in self.shut.items():
                elapsed = self.week - since + 0.5
                if elapsed <= min(forget, 8.0):
                    left = law_end(LAWS["weather_closure"], elapsed, float(chance["weather_closure"])) - elapsed
                    out["straits"][c] = max(1, int(np.ceil(left)))
        return out

    def soon(self, chance: dict, forget: float = 40.0) -> dict:
        """The chance that each watched element is free of its short cuts within the week that starts now, by the
        kinds' laws and the time each cut has run: {"edges": {edge: p}, "straits": {strait: p}} (the elements of
        ``back_weeks``)."""

        def within(kind: str, elapsed: float) -> float:
            done = law_cdf(LAWS[kind], elapsed)
            return (law_cdf(LAWS[kind], elapsed + 1.0) - done) / max(1.0 - done, 1e-9)

        out = {"edges": {}, "straits": {}}
        for e, cuts in self.cuts.items():
            if all(self._chance(chance, kind, e) is not None and self.week - since + 0.5 <= forget for kind, _f, since, _c in cuts):
                out["edges"][e] = float(np.prod([within(kind, self.week - since + 0.5) for kind, _f, since, _c in cuts]))
        if chance.get("weather_closure") is not None:
            for c, (since, _back, _carried) in self.shut.items():
                if self.week - since + 0.5 <= min(forget, 8.0):
                    out["straits"][c] = within("weather_closure", self.week - since + 0.5)
        return out

    def patch(self, arrays: dict, H: int, chance: dict, kappa0: np.ndarray, forget: float = 40.0) -> int:
        """Write the short cuts' ends into the forecast ``arrays`` (``u``, ``o``, ``kappa`` and their ``_now``): each
        cut of a kind named in ``chance`` is over when a cut of its kind that has run as long is over with that
        chance. Returns how many edges and straits were changed."""
        changed = 0
        for e, cuts in self.cuts.items():
            avg, now, hit = np.ones(H), np.ones(H), False
            for kind, factor, since, carried in cuts:
                q = self._chance(chance, kind, e)
                if q is None:
                    continue
                parts = self._share(kind, since, carried, float(q), H, forget)
                if parts is None:
                    continue
                share, live = parts
                # the cut's factor taken out of every week, and put back for the part of the week it still runs
                avg *= (1.0 - (1.0 - factor) * share) / factor
                now *= (1.0 - (1.0 - factor) * live) / factor
                hit = True
            if hit:
                arrays["u"][:, e] = arrays["u"][:, e] * avg
                arrays["u_now"][:, e] = arrays["u_now"][:, e] * now
                changed += 1
        if chance.get("weather_closure") is not None:
            for c, (since, back, carried) in self.shut.items():
                parts = self._share("weather_closure", since, carried, float(chance["weather_closure"]), H, min(forget, 8.0))
                if parts is None:
                    continue
                share, live = parts
                arrays["o"][:, c] = back * (1.0 - share)
                arrays["o_now"][:, c] = back * (1.0 - live)
                arrays["kappa"][:, c, :] = kappa0[c][None, :] * arrays["o"][:, c][:, None]
                arrays["kappa_now"][:, c, :] = kappa0[c][None, :] * arrays["o_now"][:, c][:, None]
                changed += 1
        return changed
