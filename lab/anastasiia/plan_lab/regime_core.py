"""The simulator's automatic steps as rows of a linear program, every yes/no taken from a played trajectory.

The package's plan (``oracle/lp.py``) leaves free what the simulator does by itself: how a grid's output is split
between base load and fabs, how much fuel each segment burns, how many lots a fab starts, what a plant packages. Those
rules are linear except for a few yes/no questions per week:

- a grid: is its available output below the base load (``S``: it sheds, the fabs get nothing, every segment runs
  flat out), between the base load and the base load plus the fabs' ask (``P``: no shed, the fabs take the rest,
  every segment flat out), or above (``F``: the fabs get all they ask and every segment runs at the same load factor
  below 1, so fuel burns in place of the free segment);
- a fuel segment: does the fuel on hand cover its cap (``cap``) or not (``stock``: it burns all it has);
- the rationed fuel (lng): did last week's closing stock reach the threshold (ration 1) or not (ration = stock /
  threshold);
- a fab with power: has it wafers for its whole capacity (``cap``) or not (it starts all it has);
- a plant: does it package every raw chip it holds (``all``) or is its throughput full;
- a source: is it lifted by all the week's supply, or to its storage limit;
- a store: is it thrown away from (the stock is then at the storage limit) or not;
- container cargo queued at a strait (the default release; tanker cargo is the agent's to release): is the queue of
  a lane emptied, or is the lane's edge out of the strait full, or the strait's throughput.

With the answers fixed the simulator's week is linear in the flows, so a plan solved under them is one the simulator
executes as planned, provided the plan keeps to the answers, and that is a set of bounds and rows too. ``regimes``
reads the answers off a played trajectory, ``rows`` writes them into a model, ``solve`` solves it. The trajectory the
answers came from is itself feasible (``check`` verifies that on real episodes: it is the test that the rows are the
simulator's rules), so the solution is never worse than it on paper.

One case is not linear: ``F`` with a segment whose availability is not constant (lng under the ration, a fuel
burning its last stock). There the load factor is frozen at the trajectory's (``frozen``); or the week is turned
into ``P``, which asks the plan to offer the grid exactly its load (no fuel burned in place of the free segment).

This module needs numpy and SciPy only and takes the instance, the marks and the simulator's records as they are
(the package's or the copy an agent ships), so an agent's folder can carry it. The commands that play episodes are in
``regime.py``.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse


S, P, F = 0, 1, 2
EMPTY, EDGE, THROUGH, LOOSE = 0, 1, 2, 3  # a lane's queue at a strait: emptied; its edge full; the throughput; neither
TOL = 1e-7  # relative tolerance of a regime read off a trajectory
TIGHT = 1e-9  # ... of "an edge or a strait's throughput is full" (the simulator fills them to rounding)
SLACK = 4 * TOL  # a rule read with a tolerance is written with this band (relative), so the trajectory stays inside


# ----- the yes/no answers of a trajectory -----------------------------------------------------------------------------
@dataclass
class Regimes:
    mode: np.ndarray  # (T, G): S, P or F
    lam: np.ndarray  # (T, G): the load factor of the grid's segments
    fuel_cap: np.ndarray  # (T, G, fuels): the fuel on hand covers the segment's cap
    ration_one: np.ndarray  # (T, G): last week's stock of the rationed fuel reached the threshold
    fab_cap: np.ndarray  # (T, F): wafers on hand cover the capacity
    osat_all: np.ndarray  # (T, O): every raw chip is packaged
    frozen: np.ndarray = field(default=None)  # (T, G): an F week whose load factor is kept (not linear otherwise)
    lift_all: np.ndarray = field(default=None)  # (T, slots): a source lifted by all the week's supply
    dumping: np.ndarray = field(default=None)  # (T, slots): a store thrown away from (it is at its storage limit)
    queue: dict = field(default=None)  # (week index, strait, commodity, lane) -> EMPTY, EDGE, THROUGH or LOOSE

    def copy(self) -> "Regimes":
        return Regimes(*(None if a is None else a.copy() for a in (
            self.mode, self.lam, self.fuel_cap, self.ration_one, self.fab_cap, self.osat_all, self.frozen,
            self.lift_all, self.dumping, self.queue)))


def regimes(inst, marks, records, i0, thr_osat) -> Regimes:
    """The answers the simulator gave in every week of a trajectory (its records).

    ``i0`` is the stock the trajectory started from (``sim.initial_stock``) and ``thr_osat`` the plants' weekly
    throughput (``marks.osat_throughput``): arguments, so that the function works with the package's objects and with
    the copy of its modules an agent ships.
    """
    T, G, nF, nO = len(records), len(inst.grids), len(inst.fabs), len(inst.osats)
    width = max(len(inst.nodes[g].grid.fuels) for g in inst.grids)
    r = Regimes(np.zeros((T, G), int), np.ones((T, G)), np.ones((T, G, width), bool), np.ones((T, G), bool),
                np.ones((T, nF), bool), np.ones((T, nO), bool), np.zeros((T, G), bool))
    psi = inst.params.psi
    prev = i0
    n_slots = len(inst.stock_slots)
    r.lift_all, r.dumping, r.queue = np.ones((T, n_slots), bool), np.zeros((T, n_slots), bool), {}
    supply, straits, pool = set(inst.supply_nodes), inst.chokepoint_ordinal, inst.commodity_pool
    lanes = [  # the queues the default release empties: (strait, commodity, lane, the lane's edge out of the strait)
        (c, k, li, inst.lane_through[(li, c)][1])
        for li, ln in enumerate(inst.lanes)
        for c in ln.chokepoints
        for k in inst.lane_K[li]
        if not inst.commodities[k].override
    ]
    for ti, rec in enumerate(records):
        stock, disp = rec.stock, rec.disposal
        for s, sl in enumerate(inst.stock_slots):
            if sl.node in supply:
                r.lift_all[ti, s] = rec.lift[s] >= float(marks.supply[ti, s]) * (1 - TOL)
            elif sl.node not in straits:
                r.dumping[ti, s] = disp[s] > TOL * max(1.0, stock[s])
        on_edge, through = {}, {}
        for (e, k, _lane), q in rec.x.items():
            on_edge[e] = on_edge.get(e, 0.0) + q
            tail = inst.edges[e].tail
            if tail in straits:
                through[(tail, pool[k])] = through.get((tail, pool[k]), 0.0) + q
        for c, k, li, e in lanes:
            if rec.queue.get((c, k, li), 0.0) <= 0.0:
                r.queue[(ti, c, k, li)] = EMPTY
            elif marks.prohibited[ti][e][k]:
                r.queue[(ti, c, k, li)] = LOOSE
            elif np.isfinite(marks.u[ti][e]) and on_edge.get(e, 0.0) >= float(marks.u[ti][e]) * (1 - TIGHT):
                r.queue[(ti, c, k, li)] = EDGE
            elif through.get((c, pool[k]), 0.0) >= float(marks.kappa[ti][straits[c]][pool[k]]) * (1 - TIGHT):
                r.queue[(ti, c, k, li)] = THROUGH
            else:
                r.queue[(ti, c, k, li)] = LOOSE
        phat = np.zeros(nF)
        for fi, f in enumerate(inst.fabs):
            fab = inst.nodes[f].fab
            cap = marks.alpha_bar[ti, fi] * marks.R[ti, fi] * fab.cap0
            s = inst.slot_index[(f, fab.input)]
            on_hand = stock[s] + disp[s] + rec.lots_started[fi]
            r.fab_cap[ti, fi] = on_hand >= cap * (1 - TOL)
            phat[fi] = min(cap, on_hand)
        for gi, g in enumerate(inst.grids):
            grid = inst.nodes[g].grid
            g_bar, y_bar = float(marks.G_bar[ti, gi]), float(marks.y_bar[ti, gi])
            g_av, const = grid.shares.get(None, 0.0) * g_bar, True
            for j, k in enumerate(grid.fuels):
                s = inst.slot_index[(g, k)]
                on_hand = stock[s] + disp[s] + rec.segment[(gi, k)]
                cap = grid.shares[k] * g_bar
                if k == grid.rationed:
                    thr = psi * grid.ibar[k]
                    one = thr <= 0 or prev[s] >= thr
                    r.ration_one[ti, gi] = one
                    if not one:
                        cap *= prev[s] / thr
                        const = False
                at_cap = on_hand >= cap * (1 - TOL)
                r.fuel_cap[ti, gi, j] = at_cap
                const &= bool(at_cap)
                g_av += min(cap, on_hand)
            ask = sum(inst.nodes[inst.fabs[fi]].fab.e * phat[fi] / marks.R[ti, fi]
                      for fi in inst.grid_fabs[gi] if marks.R[ti, fi] > 0)
            if rec.shed[gi] > TOL * max(y_bar, 1.0):
                r.mode[ti, gi] = S
            elif g_av <= (y_bar + ask) * (1 + TOL):
                r.mode[ti, gi] = P
            else:
                r.mode[ti, gi] = F
                r.lam[ti, gi] = (y_bar + ask) / g_av
                r.frozen[ti, gi] = not const
        for oi, o in enumerate(inst.osats):
            raw = [inst.slot_index[(o, kr)] for kr in inst.nodes[o].osat.packages]
            left = sum(stock[s] + disp[s] for s in raw)
            done = sum(q for (oo, _k), q in rec.packaged.items() if oo == oi)
            r.osat_all[ti, oi] = left <= TOL * max(done, 1.0) or done < thr_osat[ti][oi] * (1 - TOL)
        prev = stock
    return r


def summary(inst, reg: Regimes) -> str:
    out = []
    for gi, g in enumerate(inst.grids):
        m = reg.mode[:, gi]
        out.append(f"{inst.nodes[g].id}: S {int((m == S).sum())} P {int((m == P).sum())} F {int((m == F).sum())}"
                   f" (frozen {int(reg.frozen[:, gi].sum())}, mean load {reg.lam[m == F, gi].mean() if (m == F).any() else 1:.3f})")
    return "; ".join(out)


# ----- the answers as bounds and rows of a model ----------------------------------------------------------------------
class Rows:
    """Rows lo <= a.x <= hi appended to a model, and its bounds (copies) tightened in place.

    An answer is read off a trajectory with a tolerance, so what it fixes is written with a band of SLACK (times
    the quantity's size): the trajectory the answers came from then satisfies every row and bound, and the plan gains
    nothing it could use.
    """

    def __init__(self, model):
        self.lb, self.ub = model.lb.copy(), model.ub.copy()
        self.r, self.c, self.v, self.lo, self.hi, self.names = [], [], [], [], [], []

    def add(self, coefs, lo, hi, name=None, scale: float = 0.0) -> None:
        """scale > 0 widens the row by the band of a quantity of that size."""
        i = len(self.lo)
        for j, a in coefs:
            self.r.append(i), self.c.append(j), self.v.append(a)
        band = SLACK * scale
        self.lo.append(lo - band), self.hi.append(hi + band), self.names.append(name)

    def fix(self, j, value) -> None:
        """The column at value, within the band and its own bounds."""
        band = SLACK * max(1.0, abs(value))
        self.lb[j], self.ub[j] = max(self.lb[j], value - band), min(self.ub[j], value + band)

    def top(self, j) -> None:
        """The column at its upper bound."""
        if np.isfinite(self.ub[j]):
            self.lb[j] = max(self.lb[j], self.ub[j] - SLACK * max(1.0, abs(self.ub[j])))

    def none(self, j, scale: float = 1.0) -> None:
        """The column at zero (to the band of a quantity of size scale)."""
        self.ub[j] = min(self.ub[j], max(self.lb[j], SLACK * max(1.0, scale)))

    def matrix(self, n_col):
        return sparse.coo_matrix((self.v, (self.r, self.c)), shape=(len(self.lo), n_col)).tocsr()


def rows(inst, marks, model, reg: Regimes, i0, pack: bool = True, t_from: int = 1, t_to: int | None = None,
         pro_rata: dict | None = None, stores: bool = False, release: bool = False) -> Rows:
    """The simulator's rules under the answers ``reg``, for weeks ``t_from``..``t_to`` of ``model``.

    ``i0`` is the stock at the model's start (the ration of its first week reads it). ``pro_rata`` {(week, fab):
    share of its grid's fab energy} shares a grid's power between two fabs as a previous plan did (the simulator
    shares pro rata to what each asks, which is not linear when both asks vary); None leaves the split to the plan.
    ``stores`` adds the sources' lift and the stores' disposal, ``release`` the default release of container cargo at
    the straits (which lanes of a full edge or strait go first is still the plan's choice, not the simulator's
    first-come order).
    """
    nc, tm = model.meta["nc"], model.meta["template"]
    T = len(model.lb) // nc
    t_to = T if t_to is None else min(t_to, T)
    R = Rows(model)
    psi = inst.params.psi
    thr_pack = None

    def col(tag, t, *key):
        return (t - 1) * nc + tm[(tag, *key)]

    def has(tag, *key):
        return (tag, *key) in tm

    def end_stock(t, s):
        """The columns of 'what the slot holds after the week's production': stock plus what is disposed of."""
        return [(col("I", t, s), 1.0)] + ([(col("O", t, s), 1.0)] if has("O", s) else [])

    on_edge, through = {}, {}  # the flow columns of a week's template on each edge, and out of each strait by pool
    if release:
        for key, j in tm.items():
            if key[0] == "x":
                on_edge.setdefault(key[1], []).append(j)
                tail = inst.edges[key[1]].tail
                if tail in inst.chokepoint_ordinal:
                    through.setdefault((tail, inst.commodity_pool[key[2]]), []).append(j)
    for t in range(t_from, t_to + 1):
        ti = t - 1
        if stores:
            for s, sl in enumerate(inst.stock_slots):
                if has("lift", s):
                    R.top(col("lift", t, s) if reg.lift_all[ti, s] else col("I", t, s))
                elif has("O", s):
                    if not reg.dumping[ti, s]:
                        R.none(col("O", t, s), float(sl.storage) if np.isfinite(sl.storage) else 1.0)
                    else:
                        R.top(col("I", t, s))
        if release:
            full_edges, full_straits = set(), set()
            for (tj, c, k, li), code in reg.queue.items():
                if tj != ti or not has("Q", c, k, li):
                    continue
                if code == EMPTY:
                    R.none(col("Q", t, c, k, li))
                elif code == EDGE:
                    full_edges.add(inst.lane_through[(li, c)][1])
                elif code == THROUGH:
                    full_straits.add((c, inst.commodity_pool[k]))
            for e in sorted(full_edges):
                cap = float(marks.u[ti][e])
                R.add([((t - 1) * nc + j, 1.0) for j in on_edge[e]], cap, cap, ("edge full", t, e), cap)
            for c, b in sorted(full_straits):
                cap = float(marks.kappa[ti][inst.chokepoint_ordinal[c]][b])
                R.add([((t - 1) * nc + j, 1.0) for j in through[(c, b)]], cap, cap, ("strait full", t, c, b), cap)
        for gi, g in enumerate(inst.grids):
            grid = inst.nodes[g].grid
            if grid.priority != "base_first":
                raise ValueError(f"{inst.nodes[g].id}: priority {grid.priority} is not written here")
            g_bar = float(marks.G_bar[ti, gi])
            fabs = list(inst.grid_fabs[gi])
            mode, lam = int(reg.mode[ti, gi]), float(reg.lam[ti, gi])
            segs = list(grid.fuels) + ([None] if None in grid.shares else [])
            G = {k: col("G", t, gi, k) for k in segs}
            E = {fi: col("E", t, fi) for fi in fabs}
            # the grid's output is its load: what it serves and what the fabs take
            R.add([(j, 1.0) for j in G.values()] + [(col("y", t, gi), -1.0)] + [(j, -1.0) for j in E.values()],
                  0.0, 0.0, ("load", t, gi))
            for fi in fabs:  # a lot takes e / R of energy, no more and no less
                fab = inst.nodes[inst.fabs[fi]].fab
                if fab.e <= 0:
                    continue
                if marks.R[ti, fi] > 0:
                    R.add([(col("p", t, fi), fab.e), (E[fi], -float(marks.R[ti, fi]))], 0.0, 0.0, ("lot", t, fi))
                else:
                    R.ub[col("p", t, fi)] = R.ub[E[fi]] = 0.0
            if mode == S:
                for fi in fabs:
                    R.ub[E[fi]] = 0.0
                    if inst.nodes[inst.fabs[fi]].fab.e > 0:
                        R.ub[col("p", t, fi)] = 0.0
            else:
                R.none(col("ysh", t, gi), float(marks.y_bar[ti, gi]))
            if mode == F:
                for fi in fabs:
                    _full_fab(inst, marks, R, col, end_stock, t, fi, bool(reg.fab_cap[ti, fi]))
            elif mode == P and pro_rata is not None and len(fabs) > 1:
                shares = [pro_rata.get((t, fi)) for fi in fabs]
                if all(sh is not None for sh in shares):
                    for fi, sh in zip(fabs[:-1], shares[:-1]):
                        R.add([(E[fj], (1.0 if fj == fi else 0.0) - sh) for fj in fabs], 0.0, 0.0, ("share", t, fi))
            constant = mode == F and not reg.frozen[ti, gi]
            if constant:  # every segment at its cap: the load factor is the load over the sum of the caps
                total = sum(grid.shares[k] * g_bar for k in segs)
                for k in segs:
                    R.add([(G[k], total)] + [(col("y", t, gi), -grid.shares[k] * g_bar)]
                          + [(j, -grid.shares[k] * g_bar) for j in E.values()], 0.0, 0.0, ("factor", t, gi, k),
                          total * grid.shares[k] * g_bar)
            factor = lam if mode == F else 1.0
            for j, k in enumerate(grid.fuels):
                s = inst.slot_index[(g, k)]
                cap = grid.shares[k] * g_bar
                rationed = k == grid.rationed and psi * grid.ibar[k] > 0
                thr = psi * grid.ibar[k] if rationed else 0.0
                one = bool(reg.ration_one[ti, gi]) if rationed else True
                if rationed:  # the answer about last week's stock is a bound on it
                    if t > 1:
                        jp = col("I", t - 1, s)
                        if one:
                            R.lb[jp] = max(R.lb[jp], thr)
                        else:
                            R.ub[jp] = min(R.ub[jp], thr)
                if reg.fuel_cap[ti, gi, j]:
                    if constant:  # the fuel on hand covers the cap: stock + burn >= cap
                        R.add(end_stock(t, s) + [(G[k], 1.0)], cap, np.inf, ("covers", t, gi, k), cap)
                    elif one:
                        R.fix(G[k], factor * cap)
                        if factor < 1.0:
                            R.add(end_stock(t, s) + [(G[k], 1.0)], cap, np.inf, ("covers", t, gi, k), cap)
                    else:  # thr G = factor cap I(t-1)
                        if t > 1:
                            R.add([(G[k], thr), (col("I", t - 1, s), -factor * cap)], 0.0, 0.0, ("ration", t, gi),
                                  thr * cap)
                        else:
                            R.fix(G[k], factor * cap * float(i0[s]) / thr)
                        if factor < 1.0:  # on hand >= the rationed cap
                            lhs = end_stock(t, s) + [(G[k], 1.0)]
                            if t > 1:
                                R.add(lhs + [(col("I", t - 1, s), -cap / thr)], 0.0, np.inf, ("covers", t, gi, k), cap)
                elif factor >= 1.0:  # it burns all it has
                    for jj, _a in end_stock(t, s):
                        R.none(jj, cap)
                else:  # G = factor (stock + burn)
                    R.add([(G[k], 1.0 - factor)] + [(jj, -factor) for jj, _a in end_stock(t, s)], 0.0, 0.0,
                          ("last", t, gi, k), cap)
            if None in grid.shares and not constant:
                R.fix(G[None], factor * grid.shares[None] * g_bar)
        for fi, f in enumerate(inst.fabs):  # a fab that needs no power starts all it can
            fab = inst.nodes[f].fab
            if fab.grid is None or fab.e <= 0:
                _full_fab(inst, marks, R, col, end_stock, t, fi, bool(reg.fab_cap[ti, fi]))
        if pack:
            for oi, o in enumerate(inst.osats):
                osat = inst.nodes[o].osat
                if reg.osat_all[ti, oi]:
                    for kr in osat.packages:
                        for jj, _a in end_stock(t, inst.slot_index[(o, kr)]):
                            R.none(jj, float(thr_pack[ti][oi]) if thr_pack is not None else 1.0)
                else:
                    xs = [(col("xi", t, oi, kp), 1.0) for kp in osat.packages.values()]
                    if len(xs) == 1:
                        R.top(xs[0][0])
                    else:
                        thr = float(model.b_ub[(t - 1) * len(model.ub_rows) + model.ub_rows.index(("osat", oi))])
                        R.add(xs, thr, thr, ("pack", t, oi), thr)
    return R


def _full_fab(inst, marks, R, col, end_stock, t, fi, at_cap: bool) -> None:
    """A fab with all the power it asks: it starts its capacity, or every wafer it holds."""
    fab = inst.nodes[inst.fabs[fi]].fab
    jp = col("p", t, fi)
    if at_cap:
        R.top(jp)
    else:
        for jj, _a in end_stock(t, inst.slot_index[(inst.fabs[fi], fab.input)]):
            R.none(jj, float(R.ub[jp]))


def residual(model, R: Rows, z: np.ndarray) -> tuple[float, tuple]:
    """How far ``z`` is from the bounds and rows ``R``, relative to the row's largest term; the worst one's name."""
    worst, name = 0.0, ()
    over = np.maximum(R.lb - z, np.where(np.isinf(R.ub), -np.inf, z - R.ub)) / np.maximum(1.0, np.abs(z))
    j = int(np.argmax(over))
    if over[j] > worst:
        worst, name = float(over[j]), ("bound", *model.key(j), float(R.lb[j]), float(z[j]), float(R.ub[j]))
    if R.lo:
        A = R.matrix(len(z))
        val = A @ z
        lo, hi = np.array(R.lo), np.array(R.hi)
        scale = np.maximum(1.0, np.asarray(abs(A).multiply(np.abs(z)[None, :]).max(axis=1).todense()).ravel())
        viol = np.maximum(lo - val, val - hi) / scale
        i = int(np.argmax(viol))
        if viol[i] > worst:
            worst, name = float(viol[i]), (*R.names[i], float(lo[i]), float(val[i]), float(hi[i]))
    return worst, name


def solve(model, R: Rows, time_limit: float | None = None, method: str = "highs", credit=None):
    """The plan under the answers: the model's own cost (no service prices) with the bounds and rows of ``R``.

    ``credit`` (USD per unit of each column, or None) is taken off the cost: what a window that ends before the
    episode does leaves behind is worth more than its salvage.
    """
    from scipy.optimize import linprog

    n = len(model.lb)
    A_eq, b_eq = [model.A_eq], [model.b_eq]
    A_ub, b_ub = [model.A_ub], [model.b_ub]
    if R.lo:
        A = R.matrix(n)
        lo, hi = np.array(R.lo), np.array(R.hi)
        eq = lo == hi
        if eq.any():
            A_eq.append(A[eq]), b_eq.append(lo[eq])
        up, down = ~eq & np.isfinite(hi), ~eq & np.isfinite(lo)
        if up.any():
            A_ub.append(A[up]), b_ub.append(hi[up])
        if down.any():
            A_ub.append(-A[down]), b_ub.append(-lo[down])
    options = {} if time_limit is None else {"time_limit": time_limit}
    t0 = time.time()
    objective = model.cost - model.salvage if credit is None else model.cost - model.salvage - credit
    res = linprog(objective, A_ub=sparse.vstack(A_ub).tocsr(), b_ub=np.concatenate(b_ub),
                  A_eq=sparse.vstack(A_eq).tocsr(), b_eq=np.concatenate(b_eq),
                  bounds=np.column_stack([R.lb, R.ub]), method=method, options=options)
    return res, time.time() - t0
