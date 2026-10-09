"""The cycle, not the week, as the unit of decision (paradigm_lab P17).

``core._rounded`` turns a solve with the hull into whole weeks by banking the shares of a whole week the relaxation
asked of every short week and marking a week whole the moment the bank reaches that week's price. The price is not
one week's burn everywhere: for a week whose rationed fuel is under its ration and that does **not** follow a whole
week it is ``psi * ibar / cap``, the stock the ration asks for the week before, which is more than a week's burn;
for a week that follows a whole week it is exactly one week's burn. So the price has the shape of a **setup cost**:
a run of consecutive whole weeks pays the ration's priming once and one week's burn for each week after it.

``_rounded`` spends the bank greedily, earliest first, and that is where a run is lost. With a priming price of 2.5
and 0.35 of a week banked each week, the greedy marks week 8 (the bank is 2.8), is left with 0.3, and then needs the
whole priming again, so about seven weeks pass before the next whole week: one whole week. Waiting to week 11 banks
3.85, which buys week 11 (the priming) and then weeks 12 and 13 at one week's burn each: three whole weeks over the
same span. The decision is not "is this week whole" but "when does the cycle start and how long does it run", and
that is what this module decides.

The problem one grid poses is exactly:

    weeks j = 1..n in order, with a_j the share of a whole week the hull asked of week j, and a price
    prime_j (the ration's stock) or 1 (one week's burn) depending on whether week j - 1 is whole;
    choose W subset of the weeks maximising sum of v_j over W, subject to the bank never going negative.

``greedy`` replays ``core._rounded``'s own policy on that problem (identical to the cent, which the probe checks),
``solve`` takes its optimum by a dynamic program over (week, bank, was the last week whole) whose states are kept on
a Pareto frontier of (bank, value), and ``proposals`` returns a ranked set of schedules - the optimum under several
readings of what a whole week is worth, plus the greedy's own set, so the judge can never do worse than today.

Nothing here decides anything: every schedule it returns is a **proposal**, and the judge stays the exact cell and
the simulator's replay (``core.descend``'s ``search``). That is the one shape of idea that has paid in this
repository (``hub/tried/mpc.md``: a proposal judged by the replay pays, a choice made by the model's own accounting
does not).
"""

import numpy as np


TOL = 1e-6


class Problem:
    """One grid's whole-week schedule problem, read off a solve with the hull.

    ``weeks``  the short weeks the hull covered, in order;
    ``share``  a_j, the share of a whole week the relaxation asked of week j;
    ``prime``  the price of week j when it does not follow a whole week (the ration's stock, >= 1);
    ``primes`` whether week j primes the ration at all: its cap, its reference buffer and its fuel regime "R". A
               week that primes it may **not** pay with its own share, however small the price - the ration is
               lifted by last week's stock - and that holds even where the price is only one week's burn;
    ``rmax``   the fab ratio a whole week would give at week j (``Cell.hull``);
    ``peak``   the energy above the base load at week j, in GWh: the size of the whole week there;
    ``fixed``  {week: whether that week is already whole} for the weeks the hull did not cover, which reset the run.
    """

    def __init__(self, gi: int, weeks: list, share: list, prime: list, primes: list, rmax: list, peak: list,
                 fixed: dict) -> None:
        self.gi = gi
        self.weeks = list(weeks)
        self.share = np.asarray(share, dtype=float)
        self.prime = np.asarray(prime, dtype=float)
        self.primes = np.asarray(primes, dtype=bool)
        self.rmax = np.asarray(rmax, dtype=float)
        self.peak = np.asarray(peak, dtype=float)
        self.fixed = dict(fixed)

    def __len__(self) -> int:
        return len(self.weeks)

    def entries(self) -> list:
        """The grid's weeks in order as (kind, index): ("hull", j) for a week the hull covered, ("fixed", week) for
        one it did not, which sets the run's state from ``fixed`` exactly as ``_rounded`` does."""
        out = []
        at = {t: j for j, t in enumerate(self.weeks)}
        for t in sorted(set(self.weeks) | set(self.fixed)):
            out.append(("hull", at[t]) if t in at else ("fixed", t))
        return out

    def price(self, j: int, last_whole: bool) -> tuple[float, bool]:
        """(the price of making week j whole, whether this week's own share counts towards paying it).

        ``_rounded``: a week that follows a whole week, or whose fuel is not rationed-and-short, costs one week's
        burn and may pay with its own share; a week that primes the ration costs ``prime`` and may pay only with
        what was banked before it, because the ration is lifted by **last** week's stock. Whether its own share
        counts turns on ``primes`` and not on the price: ``_rounded`` reads the bank before the week whenever the
        ration's three conditions hold, even where ``psi * ibar / cap`` is under one week's burn and the price is
        therefore one. That is the whole of the setup cost on such a grid, and it is what makes waiting pay."""
        if last_whole or not self.primes[j]:
            return 1.0, True
        return float(self.prime[j]), False


# ----- reading the problem off a solve with the hull -------------------------------------------------------------------
def problems(ep, C, x: np.ndarray, mode: dict) -> dict:
    """{grid: ``Problem``} from a solved hull cell, read exactly as ``core._rounded`` reads it.

    ``C.hull`` names the (week, grid) written as "HULL" and the fab ratio each would give; ``x`` is that cell's
    solution; ``mode`` the regimes it was built from. The prime of a week is ``psi * ibar / cap`` under the same
    three conditions ``_rounded`` tests (a cap above zero, a reference buffer above zero, and the rationed fuel's
    regime read as "R"), and 1 otherwise, so that ``greedy`` below reproduces ``_rounded`` to the cent.
    """
    inst, marks, psi = ep.inst, ep.marks, ep.psi
    out = {}
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        k = ga.rationed
        weeks, share, prime, primes, rmax_l, peak, fixed = [], [], [], [], [], [], {}
        for t in range(1, ep.T + 1):
            rmax = C.hull.get((t, gi))
            if rmax is None:
                fixed[t] = mode["grid"][(t, gi)] != "OFF"
                continue
            cap = ga.shares[k] * float(marks.G_bar[t - 1][gi]) if k in ga.fuels else 0.0
            rat = bool(cap > 0 and psi * ga.ibar[k] > 0 and mode["fuel"][(t, gi, k)] == "R")
            p = max(1.0, psi * ga.ibar[k] / cap) if rat else 1.0
            weeks.append(t)
            share.append(float(x[ep.jrho(t, gi)]) / rmax if rmax > 0 else 0.0)
            prime.append(p)
            primes.append(rat)
            rmax_l.append(float(rmax))
            peak.append(max(0.0, float(marks.G_bar[t - 1][gi]) - float(marks.y_bar[t - 1][gi])))
        if weeks:
            out[gi] = Problem(gi, weeks, share, prime, primes, rmax_l, peak, fixed)
    return out


# ----- the greedy ``_rounded`` plays, on the same problem ----------------------------------------------------------
def greedy(p: Problem, lean: float = 0.0) -> list:
    """The weeks ``core._rounded`` marks on this problem: its own policy, spending the bank as early as it can."""
    bank, last_whole, out = 0.0, False, []
    for kind, j in p.entries():
        if kind == "fixed":
            last_whole = bool(p.fixed[j])
            continue
        need, own = p.price(j, last_whole)
        ready = bank + p.share[j] if own else bank
        bank += p.share[j]
        last_whole = ready >= need - lean - TOL
        if last_whole:
            bank -= need
            out.append(p.weeks[j])
    return out


# ----- the optimum of the same problem ------------------------------------------------------------------------------
def solve(p: Problem, value: np.ndarray, kept: int = 6, forbid: set | None = None) -> list:
    """Schedules that maximise ``sum(value[j] for j in W)`` under the bank, best first: [[weeks], ...].

    A dynamic program over the grid's weeks in order. The state after a week is (the bank, whether that week was
    made whole); for each of the two run states the states reachable with the same number of weeks behind them are
    kept on a **Pareto frontier** of (bank, value), because a state with at least as much banked and at least as
    much value already can only do at least as well from here on. The frontier is what keeps this small: its size is
    bounded by the count of distinct values, so the program is O(n^2) and runs in microseconds.

    ``kept``: how many of the frontier's tips at the last week are returned. The tips differ in how much of the bank
    is left unspent, so they are genuinely different schedules (one more whole week against a fuller bank), and the
    judge, not this function, says which is worth more. ``forbid``: weeks that may not be made whole, which is how
    ``by_start`` asks for the cycles that begin late.
    """
    block = forbid or set()
    n = len(p)
    if n == 0:
        return []
    # a state: (bank, value, trail) where trail is the tuple of weeks made whole, kept per run state
    front = {False: [(0.0, 0.0, ())], True: []}
    for kind, j in p.entries():
        if kind == "fixed":
            after = bool(p.fixed[j])
            merged = _pareto(front[False] + front[True])
            front = {after: merged, not after: []}
            continue
        nxt = {False: [], True: []}
        for last_whole in (False, True):
            for bank, val, trail in front[last_whole]:
                nxt[False].append((bank + p.share[j], val, trail))  # the week left short
                if p.weeks[j] in block:
                    continue
                need, own = p.price(j, last_whole)
                ready = bank + p.share[j] if own else bank
                if ready >= need - TOL:
                    nxt[True].append((bank + p.share[j] - need, val + float(value[j]), trail + (p.weeks[j],)))
        front = {s: _pareto(nxt[s]) for s in (False, True)}
    tips = sorted(_pareto(front[False] + front[True]), key=lambda s: (-s[1], -s[0]))
    return [list(t[2]) for t in tips[:kept]]


def _pareto(states: list) -> list:
    """The states no other state dominates: nothing has both at least as much banked and at least as much value."""
    if not states:
        return []
    # by value falling, then bank falling: a state is kept only when it banks more than every state kept before it
    out, best_bank = [], -np.inf
    for bank, val, trail in sorted(states, key=lambda s: (-s[1], -s[0])):
        if bank > best_bank + 1e-9:
            out.append((bank, val, trail))
            best_bank = bank
    return out


# ----- the proposals ------------------------------------------------------------------------------------------------
def values(p: Problem, name: str) -> np.ndarray:
    """What a whole week at each of the grid's weeks is taken to be worth, under one reading.

    None of these is the truth: the truth is what the simulator pays, and that is why several are tried and the
    replay chooses. ``count`` asks only for as many whole weeks as the fuel buys (the reading ``_rounded`` makes
    implicitly, but without its earliest-first bias); ``early`` and ``late`` break its ties towards the near and the
    far end of the window; ``peak`` and ``ratio`` weigh a week by how much of a fab a whole week there opens;
    ``share`` follows where the relaxation itself put the fuel.
    """
    n = len(p)
    if name == "count":
        return np.ones(n)
    if name == "early":
        return 1.0 + 1e-3 * (n - np.arange(n)) / max(1, n)
    if name == "late":
        return 1.0 + 1e-3 * np.arange(n) / max(1, n)
    if name == "peak":
        return np.maximum(1e-9, p.rmax * p.peak)
    if name == "ratio":
        return np.maximum(1e-9, p.rmax)
    if name == "share":
        return np.maximum(1e-9, p.share)
    raise ValueError(name)


READINGS = ("count", "early", "late", "peak", "ratio", "share")


def by_start(p: Problem, kept: int = 1) -> list:
    """The best cycle that starts no earlier than each of the grid's weeks: [[weeks], ...], distinct, in week order.

    This is P17's unit written out. A cycle is "bank from week s, burn from week t onwards", and the family of
    cycles is small enough to enumerate: one member per week the cycle may begin at. For each such s the program
    is asked for the most whole weeks it can buy while forbidden to spend anything before s, so the bank that the
    greedy would have spent on an isolated early week is carried forward and buys a run instead.

    Why this and not another reading of what a week is worth: the readings guess at a value the program does not
    know, while a start week is a **decision**, and the one the greedy makes without being asked. The weeks offered
    as starts are those the relaxation put fuel into and those the greedy itself marked - elsewhere there is nothing
    to bank and nothing to move.
    """
    if not len(p):
        return []
    want = sorted({p.weeks[j] for j in range(len(p)) if p.share[j] > TOL} | set(greedy(p)))
    val = values(p, "early")  # the most weeks, the earliest of the schedules that hold that many
    out, seen = [], set()
    for s in want:
        got = solve(p, val, kept=kept, forbid={t for t in p.weeks if t < s})
        for w in got:
            key = tuple(w)
            if w and key not in seen:
                seen.add(key)
                out.append(w)
    return out


def forward(p: Problem, asked_here: list) -> list:
    """Sets that ask for whole weeks the bank does not pay for, early in the window: [[weeks], ...].

    Read off the ceiling of the family (``ceiling.py`` on Small 444 episodes 0-7, about 600 plays an episode,
    +0.0030 over the rounding): the sets the replay likes are **earlier and larger** than the rounding's. On
    episode 6 the rounding asks for weeks 11, 37 and 39 and the family's best asks for 5, 7, 9, 16, 33 and 37; on
    episode 1 it asks 22, 23, 23 against 5, 8, 9, 21 and 32. Week 5 of 52 is a week the hull's share accounting
    says nothing is affordable in.

    Why it is affordable anyway: a week asked to be whole is written "SOFT", which is a **price** on its shed base
    load (``core.SOFT``) and not a requirement. The cell has a solution either way - the week closes where the fuel
    can be brought to it and stays short where it cannot - so asking lets the exact cell **bring fuel forward**,
    which the hull's own bank never considers because the bank only ever counts fuel that has already arrived. The
    rounding never asks, so the fuel never comes forward. That is the cycle run forwards rather than backwards, and
    it is the half of P17 the bank hid.

    Not ``hull_lean``, which was measured a loss (-0.0009 Small / -0.0061 Full): ``lean`` marks a week whose bank is
    within a share of the price, so it only ever moves a near miss by a week. These sets are not near misses - they
    sit at the first short weeks of the window with no bank at all - and they are **proposals**, judged by the
    replay, not a change to what the rounding believes.
    """
    if not len(p):
        return []
    ts = p.weeks
    n_now = max(1, len(asked_here))
    out = [
        list(ts[:1]),                       # the grid's earliest short week alone
        sorted(set(asked_here) | set(ts[:1])),   # keep what is asked and add the earliest: "more and earlier"
        sorted(set(asked_here) | set(ts[:2])),
        list(ts[:n_now]),                   # the same count, moved to the front of the window
        list(ts[: n_now + 1]),              # one more than is asked, at the front
        sorted(set(asked_here) | set(ts[: n_now + 2])),
    ]
    seen, kept = set(), []
    for w in out:
        key = tuple(sorted(w))
        if w and key not in seen and sorted(w) != sorted(asked_here):
            seen.add(key)
            kept.append(sorted(w))
    return kept


def proposals(probs: dict, asked: frozenset, readings: tuple = READINGS, kept: int = 3,
              limit: int = 24, fwd: bool = False) -> list:
    """Ranked sets of whole weeks from the cycle schedules, as ``core._other_weeks`` returns them.

    [(the reading that proposed it, frozenset of (week, grid))]. Every grid is scheduled on its own - the bank is a
    grid's own fuel - and the grids' schedules are then put together, with the other grids left as ``asked`` has
    them, so one proposal changes one grid. The grid with the most weeks to win is proposed first: a set is ranked
    by how many more whole weeks it holds than ``asked`` does, and the readings' own order breaks the ties.

    ``asked`` is what the greedy rounding marked, which is what the first cell was solved with; it is never
    proposed again. ``limit`` caps the list, since each proposal costs the judge a cell, a solve and a replay.
    """
    ranked = []
    for gi, p in probs.items():
        mine = frozenset(key for key in asked if key[1] != gi)
        here = sorted(t for (t, g) in asked if g == gi)
        # ``fwd``: the forward cycle, early weeks the bank does not pay for. Off by default - it was measured to
        # lose where it goes first, not because the sets are bad but because it crowds out ``by_start``, which pays
        # (Small 444 ep 0-7, 60 tries: 0.9238 with it first against 0.9245 without; 12 tries: 0.9233 against 0.9237)
        if fwd:
            for rank, weeks in enumerate(forward(p, here)):
                cand = mine | frozenset((t, gi) for t in weeks)
                ranked.append((0, 2, -rank, "fwd", cand))
        for rank, weeks in enumerate(by_start(p)):  # P17's own family: where does the cycle begin
            cand = mine | frozenset((t, gi) for t in weeks)
            ranked.append((len(cand) - len(asked), 1, -rank, "start", cand))
    for order, name in enumerate(readings):
        for gi, p in probs.items():
            v = values(p, name)
            mine = frozenset(key for key in asked if key[1] != gi)
            for rank, weeks in enumerate(solve(p, v, kept=kept)):
                cand = mine | frozenset((t, gi) for t in weeks)
                ranked.append((len(cand) - len(asked), -order, -rank, name, cand))
    ranked.sort(key=lambda r: (-r[1], -r[0], -r[2]))
    seen, out = {asked}, []
    for gain, _o, _r, name, cand in ranked:
        if cand in seen:
            continue
        seen.add(cand)
        out.append((f"cyc_{name}{gain:+d}", cand))
        if len(out) >= limit:
            break
    return out
