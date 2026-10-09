"""The episode as a linear program whose simulator regimes are fixed (a "cell"), and the simulator as its judge.

The simulator's automatic steps are all of the form "the smaller of two limits": a fuel segment burns the smaller of
its capacity, its ration and the fuel on hand; a grid serves base load first and gives the fabs what is left; a fab
starts the smaller of its capacity and the wafers on hand; a plant packages the smaller of its throughput and its raw
chips; a source lifts up to its storage; a strait releases what its throughput allows. Once it is known which limit
binds in every (week, element), the "regime", every one of these steps is a linear equation, and the whole episode is
a linear program with no integer variable. The package's oracle program leaves these steps free (a relaxation, score
1); a mixed-integer program writes the choice of the limit as a binary (``../mpc_lab/planners/planner_H.py``); here
the regimes are read from a trajectory the simulator played and the program is solved inside them.

``Episode`` holds one scenario with the whole future known:

- ``simulate``   weekly actions -> the simulator's records and cost (the judge);
- ``regimes``    records -> the regime of every (week, element) and the point the few bilinear rules are linearised at;
- ``cell``       regimes -> column bounds and extra rows on top of the oracle program;
- ``solve``      the cell's optimum (HiGHS), with the duals;
- ``actions``    a solution's dispatches and tanker releases as weekly actions.

Exact inside a cell: grids (base load first, common load factor at full load), fuel burn at full load, lot starts
with the fab at or under capacity, packaging of one product, disposal, supply lift, straits that pass everything.
Linearised at the reference point: fuel burn at a load factor below 1 when the fuel is rationed or short, lot starts
of a wafer-limited fab on a short grid, the split of a plant's throughput between two products, a strait that passes
part of a lane's queue.
"""

import importlib
import math
import time
import types

import numpy as np
import scipy.sparse as sp


PKG = "shockbench_flow"  # the package the simulator and the oracle program come from; an agent sets its copy ("sbfv")
TOL = 1e-7  # relative tolerance when a regime is read from the simulator's numbers
SLACK = 4.0  # a regime's equality is written with a band of SLACK * TOL * scale, so a tie is inside both cells
BIG = 20000  # columns above which a cell is solved by the interior-point method ("auto")
SOFT = 1e8  # USD per unit of base load shed in a week asked to be whole: above what a unit of energy is worth to any fab
INF = float("inf")


def pkg(module: str):
    """A module of the benchmark package, or of the copy an agent ships (``PKG``)."""
    return importlib.import_module(f"{PKG}.{module}")


def highs():
    """HiGHS's own interface: SciPy's bundled copy (all the server has), else the highspy package."""
    try:
        import scipy.optimize._highspy._core as hs

        return hs, hs._Highs
    except ImportError:  # an older SciPy: the highspy package, where there is one (never on the server)
        hs = importlib.import_module("highspy")
        return hs, hs.Highs


def world(task: str, entropy: int, n: int):
    """(instance, marks) of an episode: the true network of every week. The benchmark package only (a lab's tool)."""
    sample_omega = importlib.import_module("shockbench_flow.disruption.sampler").sample_omega
    task_generator = importlib.import_module("shockbench_flow.hosting.tasks").task_generator
    compute_marks = importlib.import_module("shockbench_flow.marks").compute_marks
    inst, params = task_generator(task)
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train"))
    return inst.at_digest(marks.instance_digest), marks


class Cell:
    """Column bounds and extra rows of one choice of regimes, with a tag on every regime constraint."""

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:
        self.lb, self.ub = lb.copy(), ub.copy()
        self.cost = None  # added to the episode's objective (the price of leaving an anchor plan)
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []
        self.tags: list[tuple] = []  # (kind, key, role, "col" | "row", index): a constraint that states a regime
        self.hull: dict = {}  # (week, grid) written as "HULL": the fab ratio a whole week would give there

    def row(self, coefs, lo: float, hi: float, tag: tuple | None = None) -> int:
        i = len(self.lo)
        for j, a in coefs:
            if a != 0.0:
                self.r.append(i), self.c.append(j), self.v.append(a)
        self.lo.append(lo), self.hi.append(hi)
        if tag is not None:
            self.tags.append((*tag, "row", i))
        return i

    def skip(self, n: int = 1) -> None:
        """``n`` empty rows: every cell has the same rows in the same order, so a basis carries over between cells."""
        self.lo.extend([-INF] * n), self.hi.extend([INF] * n)

    def fix(self, j: int, value: float, tag: tuple | None = None) -> None:
        self.lb[j] = self.ub[j] = value
        if tag is not None:
            self.tags.append((*tag, "col", j))

    def cap(self, j: int, value: float, tag: tuple | None = None) -> None:
        self.ub[j] = min(self.ub[j], value)
        if tag is not None:
            self.tags.append((*tag, "col", j))

    def floor(self, j: int, value: float, tag: tuple | None = None) -> None:
        self.lb[j] = max(self.lb[j], value)
        if tag is not None:
            self.tags.append((*tag, "col", j))


class Episode:
    """Some weeks of a network that is known week by week (a whole scenario, or an agent's window with its forecast):
    the oracle program, the simulator, and the cells between them."""

    def __init__(self, inst, marks, task: str | None = None, entropy: int | None = None, n: int | None = None,
                 anchored: bool = False, end: tuple | None = None) -> None:
        initial_stock = pkg("dynamics.sim").initial_stock
        self.task, self.entropy, self.n = task, entropy, n
        self.inst, self.marks = inst, marks
        self.m = m = pkg("oracle.lp").build_lp(inst, self.marks)
        self.nc, self.tm, self.T = m.meta["nc"], m.meta["template"], m.T
        self.n0 = len(m.lb)
        self.G = len(inst.grids)
        self.n1 = self.n0 + 2 * self.G * self.T  # the load factor and the fabs' ratio of every (week, grid)
        # ``anchored``: two more columns per (week, action slot), what the plan sends above and below an anchor plan
        self.anchored, self.S = anchored, len(inst.action_slots)
        self.N = self.n1 + (2 * self.S * self.T if anchored else 0)
        self.n2 = self.N  # where the anchor's columns end
        # a capped end credit for fuel (``end`` with a third item, weeks): one more column a (grid, fuel) and a fuel
        self.pools, self.owner, self.rest = self._pools(end) if end is not None and len(end) > 2 and end[2] else ([], {}, {})
        self.N += len(self.pools)
        self.i0 = np.asarray(initial_stock(inst), dtype=float)
        self.offset = -math.fsum(m.meta["salvage_const"])  # the credit of what is still out at the end
        self.nub, self.neq = len(m.ub_rows), len(m.eq_rows)
        self.ubi = {r: i for i, r in enumerate(m.ub_rows)}
        self.eqi = {r: i for i, r in enumerate(m.eq_rows)}
        self.base = sp.vstack([m.A_ub, m.A_eq], format="csr")
        self.base = sp.hstack([self.base, sp.csr_matrix((self.base.shape[0], self.N - self.n0))], format="csr")
        self.obj = np.concatenate([m.objective(), np.zeros(self.N - self.n0)])
        # ``end``: (USD per unit by stock slot, USD per unit by commodity) for what a window that stops before the
        # episode does leaves in the system, beyond its salvage: stock and queues at the last week, cargo and lots
        # still on the way. Part of the cost here and in ``simulate`` alike (``lab/anastasiia/plan_lab``'s rule).
        # A third item, weeks: fuel is worth its value only up to what the grids can use (``_pools``); without it a
        # plan that orders lifts fuel to park it wherever the window ends.
        self.end = end
        if end is not None:
            self.obj[: self.n0] -= self._end_columns()
            for i, pool in enumerate(self.pools):
                self.obj[self.n2 + i] = -pool["value"]
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        self.plain = [s for s, st in enumerate(inst.stock_slots) if st.node not in chk]
        self.supply_slots = [s for s in self.plain if inst.stock_slots[s].node in supply]
        self.disposal_slots = [s for s in self.plain if inst.stock_slots[s].node not in supply]
        self.lanes = [key[1:] for key in self.tm if key[0] == "Q" and key[3] is not None]  # (c, k, lane)
        self.psi = inst.params.psi
        # how ``solve`` treats a cell that does not solve at once (set by whoever owns the episode; None: as before),
        # and a note of every run of the solver
        self.warm_limit: float | None = None  # seconds the simplex may take from a basis before it starts cold
        self.tol_retry: float | None = None  # primal feasibility tolerance of one more run of a cell left unsolved
        # seconds: a run with less than this left of its limit does not start. HiGHS's interior point takes a limit
        # shorter than its presolve for no limit at all (a run given 0.06 s took 0.59 s on a cell of Full)
        self.least = 0.0
        self.rough_tol = 1e-4  # the interior point's optimality tolerance in a "rough" solve of an exact cell
        self.tol: float | None = None  # primal feasibility tolerance of every run (None: HiGHS's own, 1e-7)
        self.solves: list[dict] = []

    def _pools(self, end: tuple) -> tuple[list, dict, dict]:
        """The capped end credit for fuel: (pools, the pool of a (node, fuel), the pool of the rest of a fuel).

        A pool is what one grid can still use of one fuel: its stock, its terminals' and the cargo on the way to
        them, worth the fuel's value up to the grid's own stock level plus ``weeks`` of burn at its cap. Whatever
        else holds the fuel past its sources (a strait's queue, cargo on the way to a strait) is one pool a fuel,
        worth the value up to ``weeks`` of all the grids' burn. Each pool is a column of the program between 0 and
        its cap, no larger than the sum of its columns of the last week (``cell``), and paid in the objective.
        """
        inst, T, nc = self.inst, self.T, self.nc
        _stock, end_k, weeks = end[0], end[1], float(end[2])
        after = float(end[3]) if len(end) > 3 and end[3] is not None else None  # plan_lab: weeks of the episode past the window
        supply, chk = set(inst.supply_nodes), set(inst.chokepoints)
        fuels = sorted({k for g in inst.grids for k in inst.nodes[g].grid.fuels if end_k[k] > 0})
        pools, owner, rest = [], {}, {}
        for g in inst.grids:
            ga = inst.nodes[g].grid
            tails = dict.fromkeys(inst.edges[e].tail for e in inst.in_edges[g])
            terms = [x for x in tails if x not in supply and x not in chk and inst.nodes[x].grid is None
                     and all(inst.edges[e].head == g for e in inst.out_edges[x])]  # a terminal of this grid alone
            for k in ga.fuels:
                burn = ga.shares[k] * ga.deliverable
                if k not in fuels or burn <= 0:
                    continue
                for x in [g] + [x for x in terms if (x, k) in inst.slot_index]:
                    owner[(x, k)] = len(pools)
                cap = float(ga.ibar.get(k, 0.0)) + weeks * burn
                if after is not None:  # plan_lab: no more than the grid can still burn before the episode ends
                    cap = min(cap, after * burn)
                pools.append({"k": k, "value": float(end_k[k]), "cap": cap, "cols": []})
        for k in fuels:
            burn = sum(inst.nodes[g].grid.shares.get(k, 0.0) * inst.nodes[g].grid.deliverable for g in inst.grids)
            rest[k] = len(pools)
            pools.append({"k": k, "value": float(end_k[k]), "cap": weeks * burn if after is None else min(weeks, after) * burn,
                          "cols": []})
        last = (T - 1) * nc
        for key, j in self.tm.items():
            if key[0] == "I":
                sl = inst.stock_slots[key[1]]
                if sl.k in rest and sl.node not in supply:
                    pools[owner.get((sl.node, sl.k), rest[sl.k])]["cols"].append(last + j)
            elif key[0] == "Q" and key[2] in rest:
                pools[rest[key[2]]]["cols"].append(last + j)
            elif key[0] == "x" and key[2] in rest:  # dispatched in the window, arriving after it
                edge = inst.edges[key[1]]
                i = owner.get((edge.head, key[2]), rest[key[2]])
                pools[i]["cols"].extend(t * nc + j for t in range(max(0, T - edge.tau), T))
        return pools, owner, rest

    def _end_columns(self) -> np.ndarray:
        """The end credit per column of the program (fuel with a capped credit is paid through its pools)."""
        inst, nc, T = self.inst, self.nc, self.T
        end_stock, end_k = self.end[0], self.end[1]
        credit = np.zeros(self.n0)
        last = (T - 1) * nc
        for key, j in self.tm.items():
            if key[0] == "I":
                if inst.stock_slots[key[1]].k not in self.rest:
                    credit[last + j] = end_stock[key[1]]
            elif key[0] == "Q":
                if key[2] not in self.rest:
                    credit[last + j] = end_k[key[2]]
            elif key[0] == "x":  # dispatched in the window, arriving after it
                if key[2] in self.rest:
                    continue
                for t in range(max(0, T - inst.edges[key[1]].tau), T):
                    credit[t * nc + j] = end_k[key[2]]
            elif key[0] == "p":  # lots still in process
                fab = inst.nodes[inst.fabs[key[1]]].fab
                for t in range(max(0, T - fab.tau), T):
                    credit[t * nc + j] = end_k[fab.product]
            elif key[0] == "xi":
                for t in range(max(0, T - inst.nodes[inst.osats[key[1]]].osat.tau), T):
                    credit[t * nc + j] = end_k[key[2]]
        return credit

    def cost(self, recs: list, state) -> int:
        """The cost in cents of the weeks ``recs`` that end in ``state``: the weeks' costs minus the terminal salvage
        and, in a window with ``end``, minus what the state still holds."""
        value = pkg("dynamics.sim").terminal_salvage(self.inst, state)
        if self.end is not None:
            inst = self.inst
            end_stock, end_k = self.end[0], self.end[1]
            stock = np.asarray(state.stock, dtype=float)
            if self.pools:  # fuel: each pool's holding, worth its value up to the pool's cap
                held = np.zeros(len(self.pools))
                supply = set(inst.supply_nodes)
                for sl_i, sl in enumerate(inst.stock_slots):
                    if sl.k in self.rest:
                        if sl.node not in supply:
                            held[self.owner.get((sl.node, sl.k), self.rest[sl.k])] += stock[sl_i]
                    else:
                        value += float(end_stock[sl_i] * stock[sl_i])
                for sh in state.pipeline:
                    if sh.k in self.rest:
                        held[self.owner.get((inst.edges[sh.edge].head, sh.k), self.rest[sh.k])] += float(sh.qty)
                    else:
                        value += end_k[sh.k] * sh.qty
                value += sum(pool["value"] * min(float(h), pool["cap"]) for pool, h in zip(self.pools, held))
            else:
                value += float(np.dot(end_stock, stock))
                value += sum(end_k[sh.k] * sh.qty for sh in state.pipeline)
            for fi, book in state.fab_wip.items():
                value += end_k[inst.nodes[inst.fabs[fi]].fab.product] * sum(book.values())
            for book in state.osat_wip.values():
                for lots in book.values():
                    value += sum(end_k[k] * q for k, q in lots.items())
        return sum(r.cost_cents for r in recs) - pkg("dynamics.state").cents(value)

    @classmethod
    def of(cls, task: str, entropy: int, n: int) -> "Episode":
        """Episode ``n`` of the root ``entropy`` with its whole future known."""
        return cls(*world(task, entropy, n), task, entropy, n)

    # ----- indices -------------------------------------------------------------------------------------------------
    def col(self, tag: str, t: int, *key) -> int:
        return (t - 1) * self.nc + self.tm[(tag, *key)]

    def has(self, tag: str, *key) -> bool:
        return (tag, *key) in self.tm

    def jlam(self, t: int, g: int) -> int:
        return self.n0 + (t - 1) * 2 * self.G + g

    def jrho(self, t: int, g: int) -> int:
        return self.n0 + (t - 1) * 2 * self.G + self.G + g

    def jdev(self, t: int, s: int) -> int:
        """The column of what slot ``s`` sends above the anchor in week ``t``; the next column is what it sends below."""
        return self.n1 + ((t - 1) * self.S + s) * 2

    def slot(self, node: int, k: int) -> int:
        return self.inst.slot_index[(node, k)]

    # ----- the simulator -------------------------------------------------------------------------------------------
    def validated(self, wire: list[dict]) -> list[tuple]:
        """Wire actions (one dict a week, as the environment takes them) as the simulator's (flows, overrides,
        holds)."""
        validate_action = pkg("dynamics.env").validate_action
        out = []
        for t, a in enumerate(wire, start=1):
            fl, ov, ho, _dropped = validate_action(self.inst, self.marks, t, a)
            out.append((dict(fl), ov, ho))
        return out

    def clean(self, acts: list[tuple]) -> list[tuple]:
        """``acts`` (one week each, from this window's first week) without the entries this window prohibits: a plan
        made last week on last week's forecast, as the simulator would take it now."""
        inst, out = self.inst, []
        for t, (fl, ov, ho) in enumerate(acts[: self.T], start=1):
            Z = self.marks.prohibited[t - 1]
            fl = {s: q for s, q in fl.items() if q > 0 and not Z[inst.action_slots[s][0], inst.action_slots[s][1]]}
            ov = {o: q for o, q in ov.items() if not Z[inst.override_slots[o][2], inst.override_slots[o][1]]}
            out.append((fl, ov, ho))
        return out

    @staticmethod
    def as_wire(act: tuple, week: int) -> dict:
        """One week's (flows, overrides, holds) as the environment's wire action of ``week``."""
        fl, ov, ho = act
        return {
            "week": week,
            "flows": {"slot": list(fl), "qty": list(fl.values())},
            "overrides": {"slot": list(ov), "qty": list(ov.values())} if ov else None,
            "hold": {"chokepoint": [c for c, _ in ho], "k": [k for _, k in ho]} if ho else None,
        }

    def simulate(self, acts: list[tuple]) -> tuple[list, int]:
        """The simulator's records of the episode under ``acts`` and its cost in cents."""
        sim = pkg("dynamics.sim")
        initial_state, step = sim.initial_state, sim.step
        st = initial_state(self.inst)
        recs = []
        for fl, ov, ho in acts:
            recs.append(step(self.inst, self.marks, st, fl, ov, ho))
            st.last = None
        return recs, self.cost(recs, st)

    def actions(self, x: np.ndarray) -> list[tuple]:
        """A solution's dispatches and tanker releases as the simulator's weekly (flows, overrides, holds), as the
        package maps a window's first week (``lp_common.week1_action``): every tanker release is an override, 0
        included; a strait's tanker cargo with no out-edge open is held."""
        inst, marks, tm, nc = self.inst, self.marks, self.tm, self.nc
        if not hasattr(self, "_first"):
            first: dict[tuple[int, int, int], int] = {}
            for o, (c, k, e, _lane) in enumerate(inst.override_slots):
                first.setdefault((c, k, e), o)
            self._first = [(c, k, e, o, tm.get(("x", e, k, None))) for (c, k, e), o in first.items()]
            self._slots = [(s, e, k, tm[("x", e, k, lane)]) for s, (e, k, lane) in enumerate(inst.action_slots)]
        acts = []
        for t in range(1, self.T + 1):
            Z = marks.prohibited[t - 1]
            w = x[(t - 1) * nc : t * nc]
            fl = {s: float(w[j]) for s, e, k, j in self._slots if w[j] > 0.0 and not Z[e, k]}
            ov, sendable = {}, {}
            for c, k, e, o, j in self._first:
                if j is None:
                    continue
                ok = not bool(Z[e, k])
                sendable[(c, k)] = sendable.get((c, k), False) or ok
                if ok:
                    ov[o] = max(float(w[j]), 0.0)
            acts.append((fl, dict(sorted(ov.items())), frozenset(ck for ck, ok in sendable.items() if not ok)))
        return acts

    def vector(self, recs: list, ref: dict) -> np.ndarray:
        """The records as the program's column vector (the package's replay mapping), with the load factors and the
        fabs' ratios of ``ref``."""
        z0, _extra = pkg("oracle.replay")._vector(self.m, types.SimpleNamespace(records=recs))
        z = np.concatenate([z0, np.zeros(self.N - self.n0)])
        for (t, g), v in ref["lam"].items():
            z[self.jlam(t, g)] = v
        for (t, g), v in ref["rho"].items():
            z[self.jrho(t, g)] = v
        for i, pool in enumerate(self.pools):
            z[self.n2 + i] = min(float(z[pool["cols"]].sum()), pool["cap"])
        return z

    # ----- regimes -------------------------------------------------------------------------------------------------
    def regimes(self, recs: list, hint: dict | None = None) -> tuple[dict, dict]:
        """The regime of every (week, element) in the simulator's records, and the reference point.

        A tie (both limits bind, so the trajectory lies in both cells) is read as ``hint`` says (``hints``: the side
        the last solution's duals push to); without a hint, as the regime that asks for more: the grid not short,
        the segment at its cap, the fab at capacity, everything packaged.
        """
        hint = hint or {}
        hg, hf, hb, ho = (hint.get(k, {}) for k in ("grid", "fuel", "fab", "osat"))
        inst, marks, psi = self.inst, self.marks, self.psi
        mode = {"grid": {}, "fuel": {}, "fab": {}, "osat": {}, "disp": {}, "lift": {}, "lane": {}}
        ref = {"lam": {}, "rho": {}, "av": {}, "onhand": {}, "W": {}, "raw": {}}
        prev = self.i0
        pool = inst.commodity_pool
        for t in range(1, self.T + 1):
            rec, ti = recs[t - 1], t - 1
            stock, disp = rec.stock, rec.disposal
            alpha, R = marks.alpha_bar[ti], marks.R[ti]
            W, phat, bmode = {}, {}, {}
            for fi, f in enumerate(inst.fabs):
                fa = inst.nodes[f].fab
                sw = self.slot(f, fa.input)
                capf = float(alpha[fi] * R[fi] * fa.cap0)
                W[fi] = float(stock[sw] + disp[sw] + rec.lots_started[fi])
                tolw = TOL * max(1.0, capf)
                bmode[fi] = 1 if W[fi] > capf + tolw else 0 if W[fi] < capf - tolw else hb.get((t, fi), 1)
                phat[fi] = min(capf, W[fi])
                mode["fab"][(t, fi)] = bmode[fi]
                ref["W"][(t, fi)] = W[fi]
            for gi, g in enumerate(inst.grids):
                ga = inst.nodes[g].grid
                Gbar, ybar = float(marks.G_bar[ti][gi]), float(marks.y_bar[ti][gi])
                gav = ga.shares.get(None, 0.0) * Gbar
                for k in ga.fuels:
                    s = self.slot(g, k)
                    cap = ga.shares[k] * Gbar
                    onhand = float(stock[s] + disp[s] + rec.segment[(gi, k)])
                    ration, want = 1.0, hf.get((t, gi, k), ())
                    rationed = False
                    if k == ga.rationed and psi * ga.ibar[k] > 0:
                        thr = psi * ga.ibar[k]
                        if prev[s] < thr * (1.0 - TOL):
                            ration, rationed = float(prev[s]) / thr, True
                        elif prev[s] <= thr * (1.0 + TOL) and "R" in want:  # at the threshold: the ration is about to bind
                            rationed = True
                    lim = cap * ration
                    tolf = TOL * max(1.0, cap)
                    if onhand < lim - tolf or (onhand <= lim + tolf and "S" in want):
                        fm, av = "S", min(onhand, lim)
                    else:
                        fm, av = ("R" if rationed else "F"), lim
                    mode["fuel"][(t, gi, k)] = fm
                    ref["av"][(t, gi, k)], ref["onhand"][(t, gi, k)] = av, onhand
                    gav += av
                members = [fi for fi in inst.grid_fabs[gi] if inst.nodes[inst.fabs[fi]].fab.e > 0]
                ehat = [inst.nodes[inst.fabs[fi]].fab.e * phat[fi] / R[fi] if R[fi] > 0 else 0.0 for fi in members]
                tot = float(sum(ehat))
                tol = TOL * max(1.0, ybar)
                y = min(ybar, gav)
                want = hg.get((t, gi))
                if gav < ybar - tol:
                    gm = "OFF"
                elif not members or tot <= tol:
                    gm = "OFF" if gav <= ybar + tol and want == "OFF" else "ON"
                elif gav <= ybar + tol:
                    gm = "OFF" if want == "OFF" else "MID"
                elif gav < ybar + tot - tol:
                    gm = "MID"
                elif gav <= ybar + tot + tol:
                    gm = "MID" if want == "MID" else "ON"
                else:
                    gm = "ON"
                mode["grid"][(t, gi)] = gm
                E = float(sum(rec.energy[fi] for fi in members))
                ref["lam"][(t, gi)] = 1.0 if gm != "ON" else (min(1.0, (E + y) / gav) if gav > 0 else 0.0)
                ref["rho"][(t, gi)] = 0.0 if gm == "OFF" else 1.0 if gm == "ON" else min(1.0, (gav - y) / tot)
            for oi, o in enumerate(inst.osats):
                pairs = sorted(inst.nodes[o].osat.packages.items())
                raw = [float(stock[self.slot(o, kr)] + disp[self.slot(o, kr)] + rec.packaged.get((oi, kp), 0.0)) for kr, kp in pairs]
                thr = self.osat_thr(t, oi)
                tolo = TOL * max(1.0, thr)
                mode["osat"][(t, oi)] = 1 if sum(raw) < thr - tolo else 0 if sum(raw) > thr + tolo else ho.get((t, oi), 1)
                ref["raw"][(t, oi)] = raw
            for s in self.disposal_slots:
                mode["disp"][(t, s)] = 1 if disp[s] > 0.0 else 0
            for s in self.supply_slots:
                avail = float(marks.supply[ti][s])
                mode["lift"][(t, s)] = "A" if rec.lift[s] >= avail - TOL * max(1.0, avail) else "S"
            # straits: the share of a lane's queue (with this week's arrivals) that the default release passed
            out_c, out_e = {}, {}
            for (e, k, _lane), q in rec.x.items():
                tail = inst.edges[e].tail
                if tail in inst.chokepoint_ordinal:
                    out_c[(tail, pool[k])] = out_c.get((tail, pool[k]), 0.0) + q
                    out_e[e] = out_e.get(e, 0.0) + q
            for c, k, lane in self.lanes:
                e = inst.lane_through[(lane, c)][1]
                rel = float(rec.x.get((e, k, lane), 0.0))
                left = float(rec.queue.get((c, k, lane), 0.0))
                if rel + left > 1e-9:
                    phi = rel / (rel + left)
                    phi = 1.0 if phi > 1.0 - 1e-9 else 0.0 if phi < 1e-9 else phi
                else:  # an empty lane: would a unit pass this week?
                    kap = float(marks.kappa[ti][inst.chokepoint_ordinal[c]][pool[k]])
                    room = kap - out_c.get((c, pool[k]), 0.0) > 1e-6 * max(1.0, kap)
                    edge = float(marks.u[ti][e]) - out_e.get(e, 0.0) > 1e-6 * max(1.0, float(marks.u[ti][e]))
                    phi = 1.0 if room and edge and not marks.prohibited[ti][e, k] else 0.0
                mode["lane"][(t, c, k, lane)] = phi
            prev = stock
        return mode, ref

    def osat_thr(self, t: int, oi: int) -> float:
        m = self.m
        pairs = self.inst.nodes[self.inst.osats[oi]].osat.packages
        if len(pairs) > 1:
            return float(m.b_ub[(t - 1) * self.nub + self.ubi[("osat", oi)]])
        (kp,) = pairs.values()
        return float(m.ub[self.col("xi", t, oi, kp)])

    # ----- the cell ------------------------------------------------------------------------------------------------
    def cell(self, mode: dict, ref: dict, anchor: list | None = None, price=None, bonus=None,
             gate: bool = False) -> Cell:
        """The oracle program's columns and rows restricted to the regimes ``mode``, linearised at ``ref``.

        A grid's week may also be "HULL", which is no regime of the simulator: the week stands for a mix of whole
        weeks (every segment at its cap, the fabs running) and of weeks without the scarce fuel, in the shares the
        fuel it burns allows. The fabs then run for the share of a full week's burn that the scarcest segment gets,
        so a unit of that fuel is worth its part of a whole week's lots and not only the base load it serves.
        ``_rounded`` turns such a solution into whole weeks. "SOFT" is a week asked to be whole: the regimes "OFF"
        and "MID" in one cell, with a price (``SOFT``) on its shed base load that no lot is worth, so that the fabs
        run only when the base load is served in full. The week is then closed when the fuel can be there and short
        as before when it cannot, the cell has a solution either way, and the solution is the simulator's.

        ``anchor`` (weekly (flows, overrides, holds), an ``anchored`` episode only) and ``price`` (USD per unit, per
        action slot): every unit a slot sends above or below the anchor's dispatch of the week costs its price. The
        plan then leaves the anchor only where that pays; the simulator's cost knows no such price.

        ``bonus`` (USD per lot by fab; the last week it is paid, or a weight per week): a lot started is worth that
        much beyond what the forecast lets it sell, the worth of a chip in stock should a way out open; with weights
        that fall week by week, a lot started sooner is worth more than the same lot later, which keeps a plan
        carried from week to week from putting its lots off for ever. Not the simulator's cost either.
        """
        inst, marks, m, psi = self.inst, self.marks, self.m, self.psi
        peak: dict = {}  # paradigm_lab E1a: (week, grid) of a "HULL" cell -> the week's peak above the base load
        ub = np.concatenate([m.ub, np.ones(self.n1 - self.n0), np.full(self.N - self.n1, INF)])
        C = Cell(np.concatenate([m.lb, np.zeros(self.N - self.n0)]), ub)
        if self.anchored or bonus is not None:
            C.cost = np.zeros(self.N)
        if self.anchored and anchor is None:
            C.ub[self.n1 : self.n2] = 0.0
        if bonus is not None:
            per_fab, weights = bonus[0], _weeks(bonus[1], self.T)
            for t in range(1, self.T + 1):
                if weights[t - 1] != 0.0:
                    for fi in range(len(inst.fabs)):
                        C.cost[self.col("p", t, fi)] = -float(per_fab[fi]) * weights[t - 1]
        col, has = self.col, self.has
        for t in range(1, self.T + 1):
            ti = t - 1
            alpha, R = marks.alpha_bar[ti], marks.R[ti]
            for gi, g in enumerate(inst.grids):
                ga = inst.nodes[g].grid
                Gbar, ybar = float(marks.G_bar[ti][gi]), float(marks.y_bar[ti][gi])
                gm = mode["grid"][(t, gi)]
                jl, jr = self.jlam(t, gi), self.jrho(t, gi)
                lam0, rho0 = ref["lam"][(t, gi)], ref["rho"][(t, gi)]
                dy = SLACK * TOL * max(1.0, ybar)
                if None in ga.shares:  # the fuel-free segment runs at the load factor
                    C.row([(col("G", t, gi, None), 1.0), (jl, -ga.shares[None] * Gbar)], 0.0, 0.0)
                else:
                    C.skip()
                rmax = 0.0
                if gm == "HULL":  # a short week as a mix of whole weeks and weeks without the scarce fuel
                    tot = sum(inst.nodes[inst.fabs[fi]].fab.e * float(m.ub[col("p", t, fi)]) / R[fi]
                              for fi in inst.grid_fabs[gi] if inst.nodes[inst.fabs[fi]].fab.e > 0 and R[fi] > 0)
                    rmax = min(1.0, max(0.0, (Gbar - ybar) / tot)) if tot > 0 else 0.0
                    peak[(t, gi)] = Gbar - ybar  # paradigm_lab E1a: a fuel is a gate when its burn exceeds this
                if gm == "OFF" or (gm == "HULL" and rmax <= 0.0):
                    C.fix(jl, 1.0)
                    C.fix(jr, 0.0)
                elif gm == "HULL":
                    C.fix(jl, 1.0)
                    C.hull[(t, gi)] = rmax
                elif gm == "SOFT":  # asked to be whole: closed when the fuel is there, short as before otherwise
                    C.fix(jl, 1.0)
                    if C.cost is None:
                        C.cost = np.zeros(self.N)
                    C.cost[col("ysh", t, gi)] += SOFT
                elif gm == "MID":
                    C.fix(jl, 1.0)
                    C.cap(col("ysh", t, gi), dy, ("grid", (t, gi), "mid_ysh"))
                else:
                    C.fix(jr, 1.0)
                    C.cap(col("ysh", t, gi), dy)
                for k in ga.fuels:
                    s = self.slot(g, k)
                    cap = ga.shares[k] * Gbar
                    jG, jI = col("G", t, gi, k), col("I", t, s)
                    jO = col("O", t, s) if has("O", s) else None
                    fm = mode["fuel"][(t, gi, k)]
                    key = (t, gi, k)
                    thr = psi * ga.ibar[k] if k == ga.rationed else 0.0
                    prevI = [(col("I", t - 1, s), 1.0)] if t > 1 else []
                    prev0 = float(self.i0[s]) if t == 1 else 0.0
                    onhand = [(jI, 1.0), (jG, 1.0)] + ([(jO, 1.0)] if jO is not None else [])
                    if cap <= 0.0:
                        C.fix(jG, 0.0)
                        C.skip(4)
                        continue
                    dk = SLACK * TOL * max(1.0, cap)
                    if gm != "ON":  # full load: the segment burns the smallest of its three limits
                        used = 0
                        if fm == "F":
                            C.floor(jG, cap - dk, ("fuel", key, "F1"))
                        elif fm == "R":
                            C.row([(jG, 1.0)] + [(j, -cap / thr * a) for j, a in prevI], cap / thr * prev0 - dk, cap / thr * prev0 + dk, ("fuel", key, "R1"))
                            used = 1
                        else:
                            C.cap(jI, dk, ("fuel", key, "S1"))
                            if jO is not None:
                                C.cap(jO, 0.0)
                        if (t, gi) in C.hull:  # the fabs run for the share of a week's burn the scarcest segment gets
                            # paradigm_lab E1a. A fuel whose weekly burn exceeds the peak above the base load is a
                            # "gate": without it the fabs get nothing (``FINDINGS.md``, "Energy and fabs"). The row
                            # below reads rho <= rmax G_k / cap, so the other fuels stay free above their share and
                            # the program may oversupply them in a short week - a pure loss, since a unit is worth
                            # 4.1 m USD of shed base load against 25 to 40 m in the peak. With ``gate`` a gate fuel
                            # is tied to the week's share exactly (G_k = cap y), which is the formal way of writing
                            # "do not spread the fuel thin". No new column and no new row: the row count of the cell
                            # is unchanged, so a carried basis, ``hints`` and ``shifted`` keep working.
                            tie = gate and cap > peak.get((t, gi), INF)
                            C.row([(jr, cap), (jG, -rmax)], -dk if tie else -INF, dk)
                            used += 1
                        C.skip(4 - used)
                        continue
                    # four rows a fuel: the regime's equation, the cap at the load factor, two rows of the regime's side
                    if fm == "F":  # G = lam cap; last week's stock at the threshold, a full week of fuel on hand
                        C.row([(jG, 1.0), (jl, -cap)], -dk, 0.0, ("fuel", key, "Fon"))
                        C.skip()
                        if thr > 0 and prevI:
                            C.row(prevI, thr * (1.0 - SLACK * TOL) - prev0, INF, ("fuel", key, "Fon_prev"))
                        else:
                            C.skip()
                        C.row(onhand, cap - dk, INF, ("fuel", key, "Fon_hand"))
                    elif fm == "R":  # G = lam cap I_prev / thr, linearised
                        a0 = ref["av"][(t, gi, k)]
                        C.row([(jG, 1.0), (jl, -a0)] + [(j, -lam0 * cap / thr * a) for j, a in prevI],
                              lam0 * cap / thr * prev0 - lam0 * a0 - dk, lam0 * cap / thr * prev0 - lam0 * a0 + dk, ("fuel", key, "Ron"))
                        C.row([(jG, 1.0), (jl, -cap)], -INF, 0.0)
                        if prevI:
                            C.row(prevI, -INF, thr * (1.0 + SLACK * TOL) - prev0, ("fuel", key, "Ron_prev"))
                        else:
                            C.skip()
                        C.row(onhand + [(j, -cap / thr * a) for j, a in prevI], cap / thr * prev0 - dk, INF, ("fuel", key, "Ron_hand"))
                    else:  # G = lam (fuel on hand), linearised
                        h0 = ref["onhand"][(t, gi, k)]
                        coefs = [(jG, 1.0 - lam0), (jI, -lam0), (jl, -h0)] + ([(jO, -lam0)] if jO is not None else [])
                        C.row(coefs, -lam0 * h0 - dk, -lam0 * h0 + dk, ("fuel", key, "Son"))
                        C.row([(jG, 1.0), (jl, -cap)], -INF, 0.0)
                        C.row(onhand, -INF, cap + dk, ("fuel", key, "Son_cap"))
                        if thr > 0:
                            C.row(onhand + [(j, -cap / thr * a) for j, a in prevI], -INF, cap / thr * prev0 + dk, ("fuel", key, "Son_rat"))
                        else:
                            C.skip()
                for fi in inst.grid_fabs[gi]:
                    fa = inst.nodes[inst.fabs[fi]].fab
                    if fa.e <= 0:
                        continue
                    if R[fi] <= 0:
                        C.cap(col("E", t, fi), 0.0)
                    n_rows = len(C.lo)
                    self._lots(C, mode, ref, t, fi, "MID" if (t, gi) in C.hull or gm == "SOFT" else "OFF" if gm == "HULL" else gm, jr, rho0, gi)
                    C.skip(2 - (len(C.lo) - n_rows))  # two rows a fab
            for fi, f in enumerate(inst.fabs):  # a fab without a grid or energy starts what it can
                fa = inst.nodes[f].fab
                if fa.grid is None or fa.e <= 0:
                    self._lots(C, mode, ref, t, fi, "ON", None, 1.0, None)
            for oi, o in enumerate(inst.osats):
                pairs = sorted(inst.nodes[o].osat.packages.items())
                thr = self.osat_thr(t, oi)
                key = (t, oi)
                if mode["osat"][(t, oi)] == 1 or thr <= 0:
                    for kr, _kp in pairs:
                        s = self.slot(o, kr)
                        C.cap(col("I", t, s), SLACK * TOL * max(1.0, thr), ("osat", key, "a1_I"))
                        if has("O", s):
                            C.cap(col("O", t, s), 0.0)
                    C.skip(len(pairs))
                elif len(pairs) == 1:
                    C.floor(col("xi", t, oi, pairs[0][1]), thr * (1.0 - SLACK * TOL), ("osat", key, "a0_xi"))
                    C.skip()
                else:  # the throughput split pro rata to the raw stock, linearised
                    raw0 = ref["raw"][(t, oi)]
                    tot0 = sum(raw0)
                    rawc = []
                    for kr, kp in pairs:
                        s = self.slot(o, kr)
                        rawc.append([(col("I", t, s), 1.0), (col("xi", t, oi, kp), 1.0)] + ([(col("O", t, s), 1.0)] if has("O", s) else []))
                    for n_, (kr, kp) in enumerate(pairs):
                        sh = raw0[n_] / tot0
                        coefs = {col("xi", t, oi, kp): 1.0}
                        for j, a in rawc[n_]:
                            coefs[j] = coefs.get(j, 0.0) - thr / tot0 * a
                        for part in rawc:
                            for j, a in part:
                                coefs[j] = coefs.get(j, 0.0) + thr / tot0 * sh * a
                        C.row(list(coefs.items()), thr * sh - SLACK * TOL * max(1.0, thr), thr * sh + SLACK * TOL * max(1.0, thr), ("osat", key, "a0_split"))
            for s in self.disposal_slots:
                if not has("O", s):
                    continue
                if mode["disp"][(t, s)] == 1:
                    C.floor(col("I", t, s), float(inst.stock_slots[s].storage), ("disp", (t, s), "d1"))
                else:
                    C.cap(col("O", t, s), 0.0, ("disp", (t, s), "d0"))
            for s in self.supply_slots:
                if not has("lift", s):
                    continue
                if mode["lift"][(t, s)] == "A":
                    C.floor(col("lift", t, s), float(m.ub[col("lift", t, s)]), ("lift", (t, s), "LA"))
                else:
                    C.floor(col("I", t, s), float(inst.stock_slots[s].storage), ("lift", (t, s), "LS"))
            for c, k, lane in self.lanes:
                e = inst.lane_through[(lane, c)][1]
                if not has("x", e, k, lane):
                    continue
                jx, jQ = col("x", t, e, k, lane), col("Q", t, c, k, lane)
                phi = mode["lane"][(t, c, k, lane)]
                key = (t, c, k, lane)
                if phi >= 1.0:
                    C.cap(jQ, 0.0, ("lane", key, "q1"))
                    C.skip()
                elif phi <= 0.0:
                    C.cap(jx, 0.0, ("lane", key, "q0"))
                    C.skip()
                else:
                    C.row([(jx, 1.0 - phi), (jQ, -phi)], 0.0, 0.0, ("lane", key, "qp"))
            if self.anchored:  # one row a slot: dispatch = the anchor's + above - below
                sent = anchor[t - 1][0] if anchor is not None and t <= len(anchor) else {}
                for s_, (e, k, lane) in enumerate(inst.action_slots):
                    j = self.jdev(t, s_)
                    w = 0.0 if anchor is None or price is None else float(price[s_])
                    if w > 0.0 and t <= len(anchor or ()):
                        q = float(sent.get(s_, 0.0))
                        C.row([(col("x", t, e, k, lane), 1.0), (j, -1.0), (j + 1, 1.0)], q, q)
                        C.cost[j] = C.cost[j + 1] = w
                    else:
                        C.skip()
                        C.ub[j] = C.ub[j + 1] = 0.0
        for i, pool in enumerate(self.pools):  # the fuel a pool is paid for: no more than it holds, nor than its cap
            C.ub[self.n2 + i] = pool["cap"]
            C.row([(self.n2 + i, 1.0)] + [(j, -1.0) for j in pool["cols"]], -INF, 0.0)
        return C

    def _lots(self, C: Cell, mode: dict, ref: dict, t: int, fi: int, gm: str, jr: int | None, rho0: float, gi: int | None) -> None:
        """Lot starts of fab ``fi``: p = rho min(capacity, wafers on hand)."""
        inst, col, has = self.inst, self.col, self.has
        fa = inst.nodes[inst.fabs[fi]].fab
        sw = self.slot(inst.fabs[fi], fa.input)
        jp, jI = col("p", t, fi), col("I", t, sw)
        jO = col("O", t, sw) if has("O", sw) else None
        capf = float(self.m.ub[jp])
        key = (t, fi)
        dw = SLACK * TOL * max(1.0, capf) + 1.0  # and one lot: the grid's tolerance in energy is a fraction of a lot
        wafers = [(jI, 1.0), (jp, 1.0)] + ([(jO, 1.0)] if jO is not None else [])
        if gm == "OFF":
            C.cap(jp, min(capf, dw), ("grid", (t, gi), "off_p"))
            return
        if mode["fab"][(t, fi)] == 1:  # at capacity: p = rho cap
            if gm == "ON":
                C.floor(jp, capf - dw, ("fab", key, "b1_p"))
                if gi is not None:
                    C.tags.append(("grid", (t, gi), "on_p", "col", jp))
            else:
                C.row([(jp, 1.0), (jr, -capf)], -dw, dw)
                C.row(wafers, capf - dw, INF, ("fab", key, "b1_W"))
        elif gm == "ON":  # every wafer on hand is started
            C.cap(jI, dw, ("fab", key, "b0_I"))
            if jO is not None:
                C.cap(jO, 0.0)
        else:  # p = rho W, linearised at (rho0, W0)
            W0 = ref["W"][(t, fi)]
            coefs = [(jp, 1.0 - rho0), (jI, -rho0), (jr, -W0)] + ([(jO, -rho0)] if jO is not None else [])
            C.row(coefs, -rho0 * W0 - dw, -rho0 * W0 + dw)
            C.row(wafers, -INF, capf + dw, ("fab", key, "b0_W"))

    def hints(self, C: Cell, sol: dict, mode: dict, eps: float = 10.0) -> dict:
        """Which side of every regime the solution's duals push to: how ``regimes`` should read a tie.

        A dual above ``eps`` (USD per unit) on a column or row says the cost would fall if its value were lower;
        below ``-eps``, if it were higher. Fuel: the set of regimes to prefer at a tie ("S" when the fuel on hand
        equals what the segment may burn, "R" when last week's stock is at the ration's threshold). Fab: 0 or 1
        (wafers on hand equal the capacity). Plant: 0 or 1. Grid: "OFF", "MID" or "ON". "value": for every short
        (week, grid), what the lots of a full week would be worth there at this solution's prices (USD; nothing reads
        it as a hint, ``share_features`` does: it is what the solve with the hull weighs and the state does not show).
        """
        cd, rd = sol["col_dual"], sol["row_dual"]
        out = {"grid": {}, "fuel": {}, "fab": {}, "osat": {}}
        less, more, value = set(), set(), {}
        for kind, key, role, where, idx in C.tags:
            d = float(cd[idx] if where == "col" else rd[idx])
            if kind == "fuel":
                if role in ("F1", "Fon", "Fon_prev", "Fon_hand", "R1", "Ron", "Ron_hand") and d > eps:
                    less.add(key)
                if role in ("R1", "Ron", "S1", "Ron_prev", "Son_cap", "Son_rat") and d < -eps:
                    more.add(key)
            elif kind == "fab":
                if role in ("b1_p", "b1_W"):
                    out["fab"][key] = 0 if d > eps else 1
                else:
                    out["fab"][key] = 1 if d < -eps else 0
            elif kind == "osat":
                if role == "a1_I":
                    out["osat"][key] = 0 if d < -eps or out["osat"].get(key) == 0 else 1
                elif role == "a0_xi":
                    out["osat"][key] = 1 if d > eps else 0
                else:
                    out["osat"][key] = 0
            elif kind == "grid":
                if role == "mid_ysh":
                    out["grid"][key] = "OFF" if d < -eps else "ON"
                elif role == "off_p":  # what the lots of a full week would be worth
                    value[key] = value.get(key, 0.0) - d * float(self.m.ub[idx] if self.m.ub[idx] > 0 else 0.0)
                elif role == "on_p":
                    value[key] = value.get(key, 0.0) + max(0.0, d) * float(self.m.ub[idx])
        for key, fm in mode["fuel"].items():
            if fm == "F":
                out["fuel"][key] = {"S", "R"} if key in less else set()
            elif fm == "R":
                out["fuel"][key] = ({"S"} if key in less else set()) | (set() if key in more else {"R"})
            else:
                out["fuel"][key] = set() if key in more else {"S"}
        for key, gm in mode["grid"].items():
            if gm == "OFF":
                out["grid"][key] = "MID" if value.get(key, 0.0) > 1e3 * eps else "OFF"
            elif gm == "ON":
                out["grid"][key] = "MID" if value.get(key, 0.0) > 1e3 * eps else "ON"
        out["value"] = {key: float(value.get(key, 0.0)) for key, gm in mode["grid"].items() if gm == "OFF"}
        return out

    # ----- solve ---------------------------------------------------------------------------------------------------
    def rows(self, C: Cell):
        """The cell's matrix and row bounds: the oracle's rows (energy and fab energy as equalities) and the extra."""
        m = self.m
        lo = np.concatenate([np.full(m.A_ub.shape[0], -INF), m.b_eq])
        hi = np.concatenate([m.b_ub, m.b_eq])
        for name, i in self.ubi.items():
            if name[0] in ("energy", "fab_energy"):  # all output is used; a lot takes exactly its energy
                idx = np.arange(self.T) * self.nub + i
                lo[idx] = hi[idx]
        ex = sp.csr_matrix((C.v, (C.r, C.c)), shape=(len(C.lo), self.N))
        return sp.vstack([self.base, ex], format="csc"), np.concatenate([lo, C.lo]), np.concatenate([hi, C.hi])

    def residual(self, C: Cell, z: np.ndarray) -> tuple[float, tuple]:
        """The worst relative violation of the cell by ``z`` and where: is a played trajectory inside its own cell?"""
        A, lo, hi = self.rows(C)
        act = A @ z
        Aabs = abs(A).tocsr()
        scale = np.maximum(1.0, np.maximum(Aabs @ np.abs(z), np.maximum(np.where(np.isfinite(lo), np.abs(lo), 0), np.where(np.isfinite(hi), np.abs(hi), 0))))
        viol = np.maximum(lo - act, act - hi) / scale
        i = int(np.argmax(viol))
        nb = self.base.shape[0]
        where = ("extra", i - nb) if i >= nb else (self.m.ub_name(i) if i < self.m.A_ub.shape[0] else self.m.eq_name(i - self.m.A_ub.shape[0]))
        bv = np.maximum(C.lb - z, z - C.ub) / np.maximum(1.0, np.abs(z))
        j = int(np.argmax(bv))
        if bv[j] > viol[i]:
            return float(bv[j]), ("bound", j if j >= self.n0 else self.m.key(j))
        return float(viol[i]), where

    def solve(self, C: Cell, method: str = "simplex", time_limit: float = 600.0, basis: tuple | None = None,
              crossover: bool = True, what: str = "", ipm_tol: float | None = None, big: str = "ipm") -> dict:
        """The cell's optimum: status, objective in USD (the simulator's J when the cell is exact), x, the duals and
        the basis. ``basis`` (the column and row statuses of another cell's optimum, ``shifted`` when that cell
        started a week earlier) is where the simplex starts: every cell has the same columns and rows. ``method``
        "auto": the simplex on a small program, the interior-point method on a large one (Full, a window of 26
        weeks: a steady 1.2 to 1.6 s a solve where the simplex, from a basis or not, takes 0.6 to 18 s).

        A cell that does not solve at once is run again, each way once:

        - cold, when the simplex started from a basis it could not use. The interior point reads no basis (with one
          set its run is the cold run to the last bit), so it is not run twice. With ``self.warm_limit`` (seconds)
          the simplex from a basis also gives way to the cold run when it takes longer: from last week's basis it now
          and then stalls for tens of seconds on a cell that solves cold in 0.1 s;
        - with ``self.tol_retry`` as the primal feasibility tolerance (ten times ``self.tol`` when that is already as
          loose), when it is left "Infeasible" or "Unknown".
          HiGHS judges with 1e-7 in the program's own units, the regimes are read with a relative ``TOL``, and a cell
          that holds its own trajectory to 1e-6 may fail by a rounding: such cells have an optimum at 1e-6 or 1e-5.

        With either set the runs share ``time_limit``. Every run is noted in ``self.solves``, ``what`` being the
        caller's name of the cell. ``ipm_tol``: the interior point's optimality tolerance (HiGHS's own is 1e-8), for a
        solution that is only read roughly: without the crossover a tolerance of 1e-4 ends a Full cell after 30
        iterations in place of 49.
        """
        hs, Highs = highs()
        if method == "auto":  # the simplex (from a basis, if any) on a small program, interior point on a large one
            method = "simplex" if self.N <= BIG else big
        # "rough": the interior point as it stops at ``self.rough_tol``, no crossover: a large program's exact cell in a
        # week whose clock leaves no room for a vertex (a heavy episode of Full on a slow server). Its x is no vertex
        # and its duals are approximate; the plan made of it is played on the model before it is taken
        if method == "rough":
            method, crossover, basis, ipm_tol = "ipm", False, None, self.rough_tol
        # "devex": the dual simplex with devex weights from a cold start, for a large program's exact cell: on 45 cells
        # of Full its median is 0.6 of the interior point's with the crossover, and its longest run no longer
        devex = method == "devex"
        if devex:
            method, basis = "simplex", None
        A, lo, hi = self.rows(C)
        n_row, n_col = A.shape
        inf = hs.kHighsInf
        lp = hs.HighsLp()
        lp.num_col_, lp.num_row_ = n_col, n_row
        lp.col_cost_, lp.col_lower_ = (self.obj if C.cost is None else self.obj + C.cost), C.lb
        lp.col_upper_ = np.where(np.isinf(C.ub), inf, C.ub)
        lp.row_lower_, lp.row_upper_ = np.where(np.isinf(lo), -inf, lo), np.where(np.isinf(hi), inf, hi)
        lp.offset_ = float(self.offset)
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n_col, n_row
        lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr.astype(np.int32), A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data.astype(np.float64)
        rough = method == "ipm" and not crossover  # plan_lab: the interior point as it stops, a fifth faster: its x is
        # feasible and optimal to the solver's tolerance, but there is no basis and the duals are not to be read
        start = basis if basis is not None and len(basis[0]) == n_col and len(basis[1]) == n_row else None
        done = ("Optimal", "Time limit reached")

        def run(start: tuple | None, limit: float, attempt: str, tol: float | None = None):
            h = Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("solver", method)
            if devex:
                h.setOptionValue("simplex_strategy", 1)
                h.setOptionValue("simplex_dual_edge_weight_strategy", 1)
            h.setOptionValue("time_limit", float(limit))
            if rough:
                h.setOptionValue("run_crossover", "off")
            if ipm_tol is not None and method == "ipm":
                h.setOptionValue("ipm_optimality_tolerance", float(ipm_tol))
            if tol is not None or self.tol is not None:
                h.setOptionValue("primal_feasibility_tolerance", float(self.tol if tol is None else tol))
            h.passModel(lp)
            if start is not None and method != "ipm":  # the interior point does not read it
                codes = _statuses(hs)
                b = hs.HighsBasis()
                b.col_status, b.row_status = [codes[i] for i in start[0]], [codes[i] for i in start[1]]
                b.valid, b.alien = True, True  # HiGHS completes it: a shifted or another cell's basis need not be square
                h.setBasis(b)
            t0 = time.process_time()
            h.run()
            status = h.modelStatusToString(h.getModelStatus())
            info = h.getInfo()
            if rough and status == "Unknown" and int(info.primal_solution_status) == 2:
                status = "Optimal"
            self.solves.append({
                "what": what, "attempt": attempt, "method": method, "status": status, "cpu": time.process_time() - t0,
                "seconds": float(h.getRunTime()), "limit": float(limit), "simplex": int(info.simplex_iteration_count),
                "ipm": int(info.ipm_iteration_count), "crossover": int(info.crossover_iteration_count),
                "rows": n_row, "cols": n_col, "end": time.process_time(),
            })  # fmt: skip
            return h, status

        warm = start is not None and method == "simplex"
        first = min(time_limit, self.warm_limit) if warm and self.warm_limit else time_limit
        if self.least > 0 and time_limit < self.least:  # no time for a run
            return {"status": "Time limit reached", "J": math.nan, "iterations": 0, "seconds": 0.0}
        h, status = run(start, first, "cold" if start is None else "warm")
        spent = float(h.getRunTime())
        again = self.least <= 0 or time_limit - spent >= self.least  # another run has the time to start
        if warm and again and (status not in done or (status == "Time limit reached" and first < time_limit)):
            h, status = run(None, max(0.05, time_limit - spent) if self.warm_limit else time_limit, "cold")
            spent += float(h.getRunTime())
            again = self.least <= 0 or time_limit - spent >= self.least
        if self.tol_retry and again and status not in done:  # looser than the run that failed, whatever its tolerance was
            looser = max(self.tol_retry, 10.0 * self.tol) if self.tol else self.tol_retry
            h, status = run(None, max(0.05, time_limit - spent), "tolerance", looser)
        out_seconds = float(h.getRunTime())
        info = h.getInfo()
        out = {"status": status, "J": float(info.objective_function_value) if status == "Optimal" else math.nan,
               "iterations": int(info.simplex_iteration_count), "seconds": out_seconds}
        if status == "Optimal":
            sol = h.getSolution()
            # the basis is read only where a simplex can start from it, in this program or in next week's, one week
            # shorter: reading it out of HiGHS takes 50 ms on a cell of Full
            got = h.getBasis() if not rough and self.N * (self.T - 1) <= 1.05 * BIG * self.T else None
            if C.cost is not None:  # J without the anchor's price and the lots' bonus: what the simulator will charge
                out["J"] -= float(C.cost @ np.asarray(sol.col_value, dtype=float))
            out.update(x=np.asarray(sol.col_value, dtype=float), col_dual=np.asarray(sol.col_dual),
                       row_dual=np.asarray(sol.row_dual)[self.base.shape[0] :],
                       basis=None if got is None else (np.array([int(v) for v in got.col_status], dtype=np.int8),
                                                       np.array([int(v) for v in got.row_status], dtype=np.int8)))
        return out

    def shifted(self, basis: tuple, weeks: int = 1) -> tuple | None:
        """The basis of a cell that started ``weeks`` earlier, without those weeks: a start for this window's cell."""
        if basis is None:
            return None
        col, row = basis
        P = len(self.pools)
        if P:  # the pools' columns and rows come last and move with the window as they are
            if len(col) <= P or len(row) <= P:
                return None
            col, col_end, row, row_end = col[:-P], col[-P:], row[:-P], row[-P:]
        nb, extra, dev = self.nc, 2 * self.G, 2 * self.S if self.anchored else 0
        T0 = len(col) // (nb + extra + dev)
        if T0 not in (self.T, self.T + weeks):
            return None
        n_ub, n_eq = self.nub * T0, self.neq * T0
        per = (len(row) - n_ub - n_eq) // T0  # extra rows a week
        if len(col) != (nb + extra + dev) * T0 or len(row) != n_ub + n_eq + per * T0:
            return None
        if T0 == self.T:  # a capped window moved on: its last week takes the statuses of the one before
            cut = lambda a, size: np.concatenate([a[size * weeks :], a[len(a) - size * weeks :]])  # noqa: E731
        else:
            cut = lambda a, size: a[size * weeks :]  # noqa: E731
        cols = [cut(col[: nb * T0], nb), cut(col[nb * T0 : (nb + extra) * T0], extra), cut(col[(nb + extra) * T0 :], dev)]
        rows = [cut(row[:n_ub], self.nub), cut(row[n_ub : n_ub + n_eq], self.neq), cut(row[n_ub + n_eq :], per)]
        return np.concatenate(cols + ([col_end] if P else [])), np.concatenate(rows + ([row_end] if P else []))


def _statuses(hs) -> dict:
    return {int(v): v for v in (hs.HighsBasisStatus.kLower, hs.HighsBasisStatus.kBasic, hs.HighsBasisStatus.kUpper,
                                hs.HighsBasisStatus.kZero, hs.HighsBasisStatus.kNonbasic)}


def descend(ep: "Episode", acts: list, iters: int = 60, min_gain: float = 1e6, patience: int = 2, hints: bool = True,
            tweak=None, basis: tuple | None = None, time_limit: float = 600.0, anchor: list | None = None,
            price=None, bonus=None, played: tuple | None = None, deadline: float | None = None,
            force_first: bool = False, method: str = "simplex", hint: dict | None = None, close: float = 0.0,
            close_until: int | None = None, close_rationed: bool = False, hull: bool | str = False,
            hull_rough: bool = False, hull_only: bool | str = False, marks: set | None = None,
            chain_deadline: float | None = None, record: bool = False, hull_tol: float | None = None,
            tilt: tuple | None = None,
            gate: bool = False,
            big_exact: str = "ipm",
            model: tuple | None = None, search: int = 0, hull_lean: float = 0.0, search_room: int = 1) -> dict:
    """The loop from ``acts`` (weekly (flows, overrides, holds)): the best played trajectory and how it was reached.

    Each pass reads the regimes of the trajectory the simulator played (a tie as the last solution's duals say, with
    ``hints``), solves the cell and plays the solution. It stops after ``patience`` passes in a row that gain less
    than ``min_gain`` cents, or when a played solution is worse than the trajectory it came from. ``tweak(cell)``
    may add bounds of its own to every cell before it is solved (a floor under some flows, say); with
    ``force_first`` the first solution is taken even when the start, which need not respect them, is cheaper.
    ``played``: the start's (records, cost) when it has been played already. ``deadline``: process time after which no
    pass starts, and which a solve may not pass. ``hint``: how to read the start's ties (last week's, moved on by a
    week: ``moved``); the last pass's hint comes back as ``hint``. ``chain_deadline``: process time after which a
    cell that did not solve is not written another way and solved again (without the whole weeks asked for, without
    the hint, without ``tweak``): the start stands. ``close``: in the first cell the weeks a
    grid nearly closes are gathered into whole weeks (``_whole_weeks``), up to week ``close_until``; when that cell
    has no solution the plain one is solved. ``hull``: the worth of a whole week inside the program. "round" (or
    True): before the first cell, the same cell with every short week of a grid with fabs written as "HULL" is
    solved, the shares of whole weeks it asks for are rounded into weeks (``_rounded``), and in the first cell those
    weeks are "SOFT": closed when the fuel can be gathered there, short as before when it cannot, so no choice of
    weeks leaves the cell without a solution (one more solve a call). "hard": the rounded weeks are written as
    closed instead, and the plain cell is solved when that has no solution. "burn": no solve before the first cell;
    the weeks are those ``_whole_weeks`` rounds the start's own burn into (``close``: the threshold, 1 when 0),
    "SOFT" too. "model": no solve before the first cell either; fitted trees name the weeks (``told_weeks``, ``model``:
    (``share_model``'s tuple, the episode's weeks past the window)); with ``record`` the solve with the hull is run
    beside them and its shares are kept, as they are when it decides (``shares``, ``share_features``), so that the
    trees can be fitted on the states their own play meets. ``anchor`` and ``price``
    (``Episode.cell``): the loop then minimises the played cost plus the price of leaving the anchor; ``bonus``:
    minus the lots' bonus.

    ``search``: after the first cell, up to so many other sets of whole weeks are tried, each in a cell of its own,
    and the cheapest plan as the simulator plays it is kept (``_other_weeks``: the rounding of the hull's shares is
    one of many sets its fuel could pay for, and neither the best placed nor always worth asking); no set is tried
    past ``deadline``, and none at all unless ``search_room`` of them fit before it, each taken to cost what the
    first cell did with its play (where a try is dear against the week, on a large network, the search stays out).
    ``hull_lean``: ``_rounded``'s ``lean``. ``search`` in the result: (sets tried, the moves taken with what each saved in bn USD);
    ``search_cpu``: the CPU seconds all of it took.

    Returns a dict: ``acts``, ``recs``, ``J`` (cents, played), ``J0`` (cents, the start as played), ``hist`` (per
    pass: the cell's claim in USD, the played cost in cents, CPU seconds of the pass), ``basis`` (of the last cell
    solved, a start for the next: ``basis`` here is such a start).
    """
    def away(a: list, r: list) -> int:
        return _beside(a, r, anchor, price, bonus)

    def late() -> bool:
        return chain_deadline is not None and time.process_time() >= chain_deadline

    def left() -> float:
        """Seconds a solve may take from now: the limit of one solve, and no more than is left of the week's clock.
        The solves that follow a cell left unsolved take it afresh: the limit reckoned before that cell would let
        them run past the clock by as long as the cell itself took."""
        return time_limit if deadline is None else min(time_limit, deadline - time.process_time())

    solved = ("Optimal", "Time limit reached")
    recs, J = ep.simulate(acts) if played is None else played
    out = {"J0": J, "hist": [], "basis": basis}
    J += away(acts, recs)  # from here on the cost compared includes these; ``out["J"]`` is the simulator's
    out["J0_compared"] = J
    best = (J, acts, recs)
    idle, spent = 0, 0.0
    for number in range(iters):
        t0 = time.process_time()
        if deadline is not None:  # a pass that is not likely to end in time does not start
            time_limit = min(time_limit, deadline - t0)
            if time_limit <= max(0.02, 1.25 * spent):
                break
        mode, ref = ep.regimes(recs, hint)
        closed, reuse, scout = 0, None, None
        if hull == "model" and number == 0:  # no solve with the hull: a fitted model tells its shares (``told_shares``)
            plain = dict(mode["grid"])
            worth = (hint or {}).get("value")  # of the last cell solved, in this window's weeks
            if record:  # the solve with the hull beside the model, for its answer in the states the model's play meets
                for key, gm in plain.items():
                    if gm == "OFF" and (close_until is None or key[0] <= close_until):
                        mode["grid"][key] = "HULL"
                C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
                if tweak is not None:
                    tweak(C)
                wide = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit, what="hull beside")
                mode["grid"] = dict(plain)
                if wide["status"] == "Optimal":  # kept for the record only: neither its weeks nor its basis are used
                    out["shares"] = {key: float(wide["x"][ep.jrho(*key)]) / rmax for key, rmax in C.hull.items()}
                    out["share_features"] = share_features(ep, recs, mode, ref, list(C.hull), worth)
            if len(model[0]) > 5 and model[0][5] is not None and not model[0][5].place:  # a network over all the plan's short weeks
                for key in told_net(ep, recs, mode, ref, close_until, model[0][5], model[1], worth):
                    mode["grid"][key] = "SOFT"
                    closed += 1
            elif model[0][3] is not None:  # the model tells the weeks themselves
                for key in told_weeks(ep, recs, mode, ref, close_until, *model, worth):
                    mode["grid"][key] = "SOFT"
                    closed += 1
            else:
                told, domain = told_shares(ep, recs, mode, ref, close_until, *model, worth)
                closed = _rounded_from(ep, told, domain, mode, write="SOFT")
            out["hull"], out["rounded"] = "model", closed
            out["marks"] = {key for key, gm in mode["grid"].items() if gm == "SOFT" and plain[key] != "SOFT"}
        elif hull == "burn" and number == 0:
            plain = dict(mode["grid"])
            closed = _whole_weeks(ep, recs, mode, close or 1.0, close_until, True, write="SOFT")
            out["hull"] = "burn"
        elif hull and number == 0:
            plain = dict(mode["grid"])
            tail = hull_only == "tail"  # plan_lab: this solution is the week's plan, so its first week stays exact
            for key, gm in plain.items():
                if gm == "OFF" and (close_until is None or key[0] <= close_until) and not (tail and key[0] == 1):
                    mode["grid"][key] = "HULL"
            C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
            if tweak is not None:
                tweak(C)
            # plan_lab: only this solution's x is read (the shares to round), so it may be the rough one
            # ``hull_tol``: nor need it be solved to the end: the interior point stops at this optimality tolerance
            loose = hull_tol if hull_tol and not tail else None
            _tilt(ep, C, tilt)  # next_lab: the earlier week of the hull cell is worth more
            wide = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit,
                            crossover=tail or not (hull_rough or loose), what="hull", ipm_tol=loose)
            mode["grid"] = dict(plain)
            out["hull"] = wide["status"]
            if record and wide["status"] == "Optimal":  # the shares the hull asked for, and what stood before them:
                # read before the rounding writes its weeks into ``mode``, as a model in its place would read it
                out["shares"] = {key: float(wide["x"][ep.jrho(*key)]) / rmax for key, rmax in C.hull.items()}
                out["share_features"] = share_features(ep, recs, mode, ref, list(C.hull), (hint or {}).get("value"))
            if wide["status"] == "Optimal":
                if search:  # the short weeks the hull covered and the share of a whole week it asked of each
                    scout = {key: float(wide["x"][ep.jrho(*key)]) / rmax for key, rmax in C.hull.items()}
                closed = _rounded(ep, C, wide["x"], mode, write="MID" if hull == "hard" else "SOFT", lean=hull_lean)
                ys = [float(wide["x"][ep.jrho(t, gi)]) / r for (t, gi), r in C.hull.items() if r > 0]
                out["fracy"] = (sum(1 for y in ys if 0.01 < y < 0.99), len(ys))  # paradigm_lab E1a's first number
                if wide["basis"] is not None:
                    out["basis"] = wide["basis"]
            out["rounded"] = closed
            # plan_lab: the weeks asked to be whole, for the cells of the weeks to come (``marks``)
            out["marks"] = {key for key, gm in mode["grid"].items() if gm == "SOFT" and plain[key] != "SOFT"}
            if tail and wide["status"] == "Optimal":  # one solve a week: this solution, relaxed after its first week
                reuse = wide
            elif hull_only:  # one solve a week: this week the start stands, the marked weeks go into next week's cell
                out["hist"].append((wide["status"], None, time.process_time() - t0))
                break
        elif marks and number == 0:  # plan_lab: whole weeks asked for by an earlier week's solve with the hull
            plain = dict(mode["grid"])
            for key in marks:
                if mode["grid"].get(key) == "OFF" and (close_until is None or key[0] <= close_until):
                    mode["grid"][key] = "SOFT"
                    closed += 1
        elif close > 0 and number == 0:
            plain = dict(mode["grid"])
            closed = _whole_weeks(ep, recs, mode, close, close_until, close_rationed)
        if reuse is None and deadline is not None:  # plan_lab: the solve with the hull took its part of the week
            time_limit = min(time_limit, deadline - time.process_time())
            if time_limit <= 0.05:
                out["hist"].append(("no time", None, time.process_time() - t0))
                break
        t_cell = time.process_time()  # from here to the played plan: what one more set of whole weeks would cost
        if reuse is not None:
            sol, closed = reuse, 0
        else:
            C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
            if tweak is not None:
                tweak(C)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit, what="exact", big=big_exact)
        if closed and sol["status"] not in solved and not late() and left() > 0.05:  # no fuel for all: the plain cell
            mode["grid"] = plain
            closed = 0
            C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
            if tweak is not None:
                tweak(C)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=left(), what="plain", big=big_exact)
        out["closed"] = closed
        # a tie read the other way left no room
        if sol["status"] not in solved and hint is not None and not late() and left() > 0.05:
            mode, ref = ep.regimes(recs)
            C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
            if tweak is not None:
                tweak(C)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=left(), what="no hint", big=big_exact)
        # the extra bounds left no room
        if sol["status"] not in solved and tweak is not None and not late() and left() > 0.05:
            C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=left(), what="no tweak", big=big_exact)
        if sol["status"] != "Optimal":
            out["hist"].append((sol["status"], None, time.process_time() - t0))
            break
        out["basis"] = sol["basis"]
        acts2 = ep.actions(sol["x"])
        recs2, J2 = ep.simulate(acts2)
        spent = time.process_time() - t0
        out["hist"].append((sol["J"], J2, spent, sol["iterations"]))
        J2 += away(acts2, recs2)
        first = force_first and len(out["hist"]) == 1
        if J2 < best[0] or first:
            best = (J2, acts2, recs2)
        roomy = deadline is None or time.process_time() + search_room * (time.process_time() - t_cell) <= deadline
        if search and number == 0 and scout is not None and reuse is None and roomy:  # other sets of whole weeks, the cheapest kept
            began = time.process_time()
            asked = frozenset(key for key, gm in mode["grid"].items() if gm == "SOFT" and plain[key] != "SOFT")
            cost, tried, taken = 0.0, {asked}, []

            def short(rs: list, weeks: frozenset) -> frozenset:  # the asked weeks that the plan does not close
                return frozenset((t, gi) for t, gi in weeks if rs[t - 1].shed[gi] > 1e-6 * max(1.0, float(ep.marks.y_bar[t - 1][gi])))

            queue = _other_weeks(scout, asked, short(recs2, asked))
            while queue and len(tried) <= search:
                kind, weeks = queue.pop(0)
                now = time.process_time()
                if weeks in tried:
                    continue
                if deadline is not None and now + 1.3 * cost > deadline:
                    break
                tried.add(weeks)
                trial = {**mode, "grid": dict(plain)}
                for key in weeks:
                    if plain.get(key) == "OFF":
                        trial["grid"][key] = "SOFT"
                Ct = ep.cell(trial, ref, anchor, price, bonus, gate=gate)
                if tweak is not None:
                    tweak(Ct)
                st = ep.solve(Ct, method=method, basis=sol["basis"], what="search " + kind, big=big_exact,
                              time_limit=time_limit if deadline is None else max(0.05, min(time_limit, deadline - now)))
                if st["status"] == "Optimal":
                    at = ep.actions(st["x"])
                    rt, Jt = ep.simulate(at)
                    Jt += away(at, rt)
                    if Jt < J2 - 1e6:  # cheaper by more than 10,000 USD: this set stands, the next ones start from it
                        taken.append(f"{kind}:{(J2 - Jt) / 1e11:.2f}")  # the move and what it saves, bn USD
                        J2, acts2, recs2, C, sol, mode, asked = Jt, at, rt, Ct, st, trial, weeks
                        queue = _other_weeks(scout, asked, short(recs2, asked))
                cost = max(cost, time.process_time() - now)
            out.update(search=(len(tried) - 1, ",".join(taken) or "0"), basis=sol["basis"], marks=set(asked), closed=len(asked),
                       search_cpu=time.process_time() - began)  # the sets' cells, solves and plays together
            if J2 < best[0]:
                best = (J2, acts2, recs2)
        idle = idle + 1 if J2 >= J - min_gain else 0
        if not first and (idle > patience or J2 > J + 100 * min_gain):
            break
        acts, recs, J = acts2, recs2, J2
        hint = ep.hints(C, sol, mode) if hints else None
    out.update(J=best[0] - away(best[1], best[2]), J_compared=best[0], acts=best[1], recs=best[2], hint=hint)
    return out


def _other_weeks(scout: dict, asked: frozenset, short: frozenset = frozenset()) -> list:
    """Sets of whole weeks next to ``asked``, the likeliest to pay first: [(the move's name, frozenset of (week, grid))].

    ``scout``: the share of a whole week the solve with the hull asked of every short (week, grid) it covered. The
    rounding gives each grid the whole weeks its shares sum to, at the weeks the sums are reached; here are its
    neighbours: when the plan leaves some asked weeks short (``short``), the set without them and the empty set; a
    grid's weeks one short week earlier; one more grid given a whole week, its last short week (the two grids whose
    shares were largest and rounded to nothing first); no week at all; a grid's weeks one short week later; the other
    grids' last weeks; one grid's weeks dropped; one more week after a grid's last; then a wider ring of the same
    moves.
    """
    weeks, own = {}, {}
    for t, gi in sorted(scout):
        weeks.setdefault(gi, []).append(t)
    for t, gi in sorted(asked):
        own.setdefault(gi, []).append(t)
    wish = {gi: sum(scout[(t, gi)] for t in ts) for gi, ts in weeks.items()}

    def moved(gi: int, by: int) -> frozenset:
        ts, out = weeks.get(gi, []), {key for key in asked if key[1] != gi}
        out.update((ts[ts.index(t) + by], gi) for t in own[gi] if t in ts and 0 <= ts.index(t) + by < len(ts))
        return frozenset(out)

    free = sorted((gi for gi in weeks if gi not in own and wish[gi] > 1e-6), key=lambda gi: -wish[gi])
    more = [("more", asked | {(weeks[gi][-1], gi)}) for gi in free]
    # the order is by what a try saved in the plays of 9 October on both networks. An asked week that the plan does not
    # close (``short``) is the sign of a plan bent by the price on its shed load: with one, the plan is dearer than
    # the plan without any asked week in half of the cases (in 3 % when all are closed), and dropping the weeks saves
    # three to seven times what another try does. So with such a week: first without the weeks left short, then
    # without any; else a week earlier, then one more grid's week, and no week at all only after those.
    lead = []
    if short:
        lead = [("unshort", asked - short)] + ([("none", frozenset())] if asked - short else [])
    out = lead + [("earlier", moved(gi, -1)) for gi in own] + more[:2] + ([("none", frozenset())] if asked else [])
    out += [("later", moved(gi, 1)) for gi in own] + more[2:]
    if len(own) > 1:
        out += [("drop", frozenset(key for key in asked if key[1] != gi)) for gi in own]
    for gi, mine in own.items():
        ts = weeks.get(gi, [])
        if mine[-1] in ts and ts.index(mine[-1]) + 1 < len(ts):
            out.append(("next", asked | {(ts[ts.index(mine[-1]) + 1], gi)}))
    # a wider ring, tried only when the count allows: a grid's weeks two short weeks away; one more grid's week where
    # its share was largest; one week of a grid moved or dropped alone; one more week before a grid's first
    out += [("earlier2", moved(gi, -2)) for gi in own] + [("later2", moved(gi, 2)) for gi in own]
    out += [("more_top", asked | {(max(weeks[gi], key=lambda t: scout[(t, gi)]), gi)}) for gi in free]
    for gi, mine in own.items():
        ts = weeks.get(gi, [])
        if len(mine) > 1:
            for t in mine:
                rest = asked - {(t, gi)}
                out.append(("drop_one", rest))
                out += [(f"one{by:+d}", rest | {(ts[ts.index(t) + by], gi)}) for by in (-1, 1) if t in ts and 0 <= ts.index(t) + by < len(ts)]
        if mine[0] in ts and ts.index(mine[0]) > 0:
            out.append(("before", asked | {(ts[ts.index(mine[0]) - 1], gi)}))
    for gi in weeks:  # and a grid the hull asked nothing of at all: its last short week
        if gi not in own and gi not in free:
            out.append(("more_zero", asked | {(weeks[gi][-1], gi)}))
    seen, kept = {asked}, []
    for kind, x in out:
        if x not in seen:
            seen.add(x)
            kept.append((kind, x))
    return kept


SHARE_FEATURES = (
    "week", "rmax", "fab_load", "fab_worth", "wafers", "shed", "burn_lng", "burn_crude", "burn_nuc", "burn_min",
    "burn_cum", "stock_lng", "stock_crude", "stock_nuc", "term_lng", "term_crude", "rationed", "ration_need",
    "lost_le", "lost_mat", "thrown", "others_min", "whole_prev", "whole_next", "lots_value", "has_value",
)  # fmt: skip


def share_features(ep: "Episode", recs: list, mode: dict, ref: dict, keys: list, value: dict | None = None) -> dict:
    """What a model standing in for the solve with the hull may know of a short (week, grid): {key: floats}.

    All of it is read from the trajectory the week's plan starts from (``recs``, with its regimes ``mode`` and
    ``ref``) and is relative, so that one model serves every network: the week as a share of the window; the fab
    ratio a whole week would give; the fabs' load as a share of the base load; the worth of a whole week's lots per
    USD of lost load their energy stands for; wafers on hand as a share of capacity; the shed share of the base
    load; each fuel segment's burn as a share of its cap, their smallest and its running sum over the grid's short
    weeks; the fuel at the grid and at its terminals in weeks of burn; whether the rationed fuel burns under its
    ration and the stock, in weeks of burn, that lifts it; the lost share of each chip's demand; chips thrown away
    as a share of all fabs' capacity; the other grids' smallest burn share; whether the weeks before and after
    are closed; what the lots of a full week would be worth at the prices of the last cell solved (``value``:
    ``hints``'s, moved to this window's weeks), per USD of lost load their energy stands for as the fabs' worth is,
    and whether there was such a price. The order is ``SHARE_FEATURES``'s.
    """
    inst, marks, psi, T = ep.inst, ep.marks, ep.psi, ep.T
    com = [c.id for c in inst.commodities]
    lng, crude, nuc = (com.index(n) if n in com else None for n in ("lng", "crude", "nucfuel"))
    supply, chk = set(inst.supply_nodes), set(inst.chokepoints)
    pi = np.zeros(len(com))
    for d in inst.demands:
        pi[d.k] = max(pi[d.k], d.pi)
    for o in inst.osats:
        for raw, packed in inst.nodes[o].osat.packages.items():
            pi[raw] = max(pi[raw], pi[packed])
    dk = np.array([d.k for d in inst.demands])
    chips = [(dk == com.index(n)) if n in com else np.zeros(len(dk), dtype=bool) for n in ("chip_le", "chip_mat")]
    chip_slots = [i for i, sl in enumerate(inst.stock_slots) if pi[sl.k] > 0 and sl.node not in supply]
    cap_all = sum(inst.nodes[f].fab.cap0 for f in inst.fabs) or 1.0
    with_fabs = [gi for gi in range(len(inst.grids)) if inst.grid_fabs[gi]]
    burn, low = {}, np.ones((T, len(inst.grids)))
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        for t in range(1, T + 1):
            for k in ga.fuels:
                cap = ga.shares[k] * float(marks.G_bar[t - 1][gi])
                burn[(t, gi, k)] = float(recs[t - 1].segment[(gi, k)]) / cap if cap > 0 else 1.0
                low[t - 1, gi] = min(low[t - 1, gi], burn[(t, gi, k)])
    out, total = {}, {}
    for t, gi in sorted(keys, key=lambda key: (key[1], key[0])):
        g = inst.grids[gi]
        ga, rec, ti = inst.nodes[g].grid, recs[t - 1], t - 1
        Gbar, ybar = float(marks.G_bar[ti][gi]), float(marks.y_bar[ti][gi])
        tails = dict.fromkeys(inst.edges[e].tail for e in inst.in_edges[g])
        terms = [x for x in tails if x not in supply and x not in chk and inst.nodes[x].grid is None
                 and all(inst.edges[e].head == g for e in inst.out_edges[x])]
        tot = worth = wafers = room = 0.0
        for fi in inst.grid_fabs[gi]:
            fa = inst.nodes[inst.fabs[fi]].fab
            capf, R = float(ep.m.ub[ep.col("p", t, fi)]), float(marks.R[ti][fi])
            if fa.e > 0 and R > 0:
                tot += fa.e * capf / R
            worth += capf * pi[fa.product]
            wafers += float(ref["W"].get((t, fi), 0.0))
            room += capf
        caps = {k: ga.shares[k] * Gbar for k in ga.fuels}

        def weeks(node, k):  # the stock of fuel ``k`` at ``node`` in weeks of this grid's burn
            if k is None or k not in caps or caps[k] <= 0 or (node, k) not in inst.slot_index:
                return 0.0
            return float(rec.stock[inst.slot_index[(node, k)]]) / caps[k]

        kr = ga.rationed if ga.rationed in ga.fuels else None
        rationed = kr is not None and mode["fuel"].get((t, gi, kr)) == "R"
        total[gi] = total.get(gi, 0.0) + low[ti, gi]
        others = [low[ti, x] for x in with_fabs if x != gi]
        out[(t, gi)] = (
            t / T,
            min(1.0, max(0.0, (Gbar - ybar) / tot)) if tot > 0 else 0.0,
            tot / ybar if ybar > 0 else 0.0,
            worth / (float(ga.voll) * tot) if tot > 0 else 0.0,
            wafers / room if room > 0 else 0.0,
            float(rec.shed[gi]) / ybar if ybar > 0 else 0.0,
            burn.get((t, gi, lng), 1.0), burn.get((t, gi, crude), 1.0), burn.get((t, gi, nuc), 1.0),
            float(low[ti, gi]), total[gi],
            weeks(g, lng), weeks(g, crude), weeks(g, nuc),
            sum(weeks(x, lng) for x in terms), sum(weeks(x, crude) for x in terms),
            1.0 if rationed else 0.0,
            psi * float(ga.ibar.get(kr, 0.0)) / caps[kr] if kr is not None and caps[kr] > 0 else 0.0,
            *(float(rec.lost[m].sum()) / max(1.0, float(rec.demand[m].sum())) for m in chips),
            float(rec.disposal[chip_slots].sum()) / cap_all,
            float(np.mean(others)) if others else 1.0,
            1.0 if t > 1 and mode["grid"].get((t - 1, gi)) != "OFF" else 0.0,
            1.0 if t < T and mode["grid"].get((t + 1, gi)) != "OFF" else 0.0,
            max(0.0, float(value[(t, gi)])) / (float(ga.voll) * tot) if value and (t, gi) in value and tot > 0 else 0.0,
            1.0 if value and (t, gi) in value else 0.0,
        )
    return out


PLAN_HORIZONS = (2, 4, 6, 8, 10, 14)
_PLAN_MEANS = ("burn_lng", "burn_crude", "shed", "wafers", "lost_le", "lost_mat", "thrown", "others_min")
_PLAN_FIRST = ("rmax", "fab_load", "fab_worth", "stock_lng", "stock_crude", "stock_nuc", "term_lng", "term_crude",
               "rationed", "whole_prev")  # fmt: skip
PLAN_FEATURES = (
    ["window", "after", "first_week", "weeks_short", "need"]
    + [f"burn_by_{h}" for h in PLAN_HORIZONS] + [f"short_by_{h}" for h in PLAN_HORIZONS]
    + [f"paid_by_{h}" for h in PLAN_HORIZONS] + [f"{n}_by_{h}" for h in (4, 14) for n in _PLAN_MEANS]
    + [f"{n}_first" for n in _PLAN_FIRST] + ["whole_next_mean", "term_lng_max", "term_crude_max", "wafers_max"]
    + ["lots_value_max", "lots_value_mean", "lots_value_last", "lots_value_week", "has_value"]
)  # fmt: skip
_RIVAL = ("lots_value_max", "fab_worth_first", "paid_by_14")
RIVAL_FEATURES = ("rivals",) + tuple(f"{n}_{kind}" for n in _RIVAL for kind in ("rank", "gap"))
MODEL_FEATURES = tuple(PLAN_FEATURES) + RIVAL_FEATURES  # what the fitted trees read, in this order


def plan_features(t: np.ndarray, x: np.ndarray, window: float, after: float) -> np.ndarray:
    """One grid's short weeks of a plan as one vector (``PLAN_FEATURES``): ``t`` their weeks in the window, rising,
    ``x`` their rows of ``SHARE_FEATURES``. The scarce fuel's burn summed up to each horizon (the whole weeks the
    start's own fuel would pay for by then), as it is and in units of the price of a first whole week; the count of
    short weeks by horizon; means of the rest over the first 4 and the first 14 weeks; the first short week's state;
    the worth of a full week's lots at the last prices: its largest value, its mean, its value in the last short
    week, the week of the largest, and whether the prices were there (rows without these columns: zeros).
    ``after``: the episode's weeks past the window, no more than 26, so that the length of the episode says nothing."""
    c = {n: i for i, n in enumerate(SHARE_FEATURES)}
    burn, first = x[:, c["burn_min"]], x[0]
    worth = x[:, c["lots_value"]] if x.shape[1] > c["lots_value"] else np.zeros(len(t))
    priced = float(x[:, c["has_value"]].max()) if x.shape[1] > c["has_value"] else 0.0
    need = max(1.0, float(first[c["ration_need"]])) if first[c["rationed"]] > 0.5 else 1.0
    by = [float(burn[t <= h].sum()) for h in PLAN_HORIZONS]
    means = [float(x[t <= h, c[n]].mean()) if (t <= h).any() else 0.0 for h in (4, 14) for n in _PLAN_MEANS]
    return np.array(
        [window, min(float(after), 26.0), float(t[0]), float(len(t)), need, *by, *(float((t <= h).sum()) for h in PLAN_HORIZONS),
         *(v / need for v in by), *means, *(float(first[c[n]]) for n in _PLAN_FIRST), float(x[:, c["whole_next"]].mean()),
         float(x[:, c["term_lng"]].max()), float(x[:, c["term_crude"]].max()), float(x[:, c["wafers"]].max()),
         float(worth.max()), float(worth.mean()), float(worth[-1]), float(t[int(np.argmax(worth))]) if worth.max() > 0 else 0.0, priced]
    )  # fmt: skip


def with_rivals(plans: dict) -> dict:
    """{grid: its plan's features (``plan_features``) followed by where it stands among the other grids that have
    short weeks in the same plan (``RIVAL_FEATURES``)}: their number, and for the worth of a week's lots (at the last
    prices and at the list prices) and the fuel the start pays for, the share of the others below it and its distance
    to the best of them. The grids draw on the same fuel, so a whole week goes to the one that is worth more, not to
    each that is worth something."""
    idx = [PLAN_FEATURES.index(n) for n in _RIVAL]
    out = {}
    for gi, v in plans.items():
        others = [w for g, w in plans.items() if g != gi]
        extra = [float(len(others))]
        for i in idx:
            rest = [float(w[i]) for w in others]
            extra += [float(np.mean([r < v[i] for r in rest])) if rest else 0.5, float(v[i] - max(rest)) if rest else 0.0]
        out[gi] = np.r_[v, extra]
    return out


class Trees:
    """Boosted trees read from arrays (``shares.py fit`` writes them): numpy alone, as the scoring image has."""

    def __init__(self, data, name: str) -> None:
        self.feature, self.threshold = data[f"{name}_feature"], data[f"{name}_threshold"]
        self.left, self.right, self.value = data[f"{name}_left"], data[f"{name}_right"], data[f"{name}_value"]
        self.start, self.base = data[f"{name}_start"], float(data[f"{name}_base"])

    def raw(self, v: np.ndarray) -> float:
        total = self.base
        for root in self.start:
            i = int(root)
            while self.left[i] >= 0:
                i = int(self.left[i] if v[self.feature[i]] <= self.threshold[i] else self.right[i]) + int(root)
            total += float(self.value[i])
        return total


def share_model(path) -> tuple:
    """(trees for "a week is marked", trees for the whole weeks asked, the probability to act on, trees for the first
    whole week's place in the window or None when the file has none, the number of ``PLAN_FEATURES`` the trees were
    fitted on: features added since stand after them)."""
    data = np.load(path)
    when = Trees(data, "when") if "when_base" in data.files else None
    net = Net(data) if "net_in_w" in data.files else None
    return Trees(data, "mark"), Trees(data, "total"), float(data["threshold"]), when, len(data["features"]), net


NET_SHARE = 24  # the first so many of ``SHARE_FEATURES`` are a token's own features (the ones every record has)
NET_EXTRA = ("place", "place_back", "weeks_short", "after", "last")


NET_PLAN = 53  # with ``window``, a token also carries the first so many of its grid's ``PLAN_FEATURES``


def net_tokens(rows: dict, after: float, window: float | None = None) -> tuple:
    """A plan's short weeks as tokens for ``Net``: ([(week, grid)], features (n, ``NET_SHARE`` + 5), same (n, n)).

    ``rows``: ``_short_weeks``'s {grid: [(week, its ``SHARE_FEATURES``)]}. Beside its own features a token carries
    its place among its grid's short weeks from the start and from the end, their count, the episode's weeks past
    the window and whether it is the grid's last short week; ``same`` is 1 for two tokens of one grid: the grids
    have no names, so that one network serves any number of them. With ``window`` (the window's length) every token
    also carries its grid's sums over its short weeks, the ones the trees read (``plan_features``).
    """
    keys, feats, grid = [], [], []
    for gi in sorted(rows):
        own = rows[gi]
        n = len(own)
        wide = []
        if window is not None:
            wide = plan_features(np.array([r[0] for r in own], dtype=float), np.array([r[1] for r in own], dtype=float),
                                 float(window), after)[:NET_PLAN].tolist()
        for i, (t, f) in enumerate(own):
            keys.append((int(t), gi))
            feats.append([*f[:NET_SHARE], i / n, (n - 1 - i) / n, n / 14.0, min(float(after), 26.0) / 26.0, 1.0 if i == n - 1 else 0.0, *wide])
            grid.append(gi)
    g = np.array(grid)
    width = NET_SHARE + len(NET_EXTRA) + (NET_PLAN if window is not None else 0)
    return keys, np.array(feats, dtype=float).reshape(len(keys), width), (g[:, None] == g[None, :]).astype(float)


class Net:
    """A small attention network over a plan's tokens, read from arrays (``net.py`` fits and writes them): numpy
    alone. Every token looks at every other one, with a learnt leaning to the tokens of its own grid; out come, per
    token, the probability that the solve with the hull asks that week to be whole and the share it asks of it."""

    def __init__(self, data) -> None:
        self.p = {k[4:]: np.asarray(data[k], dtype=float) for k in data.files if k.startswith("net_")}
        self.layers, self.heads = int(self.p["layers"]), int(self.p["heads"])
        self.threshold = float(self.p["threshold"])  # a grid is asked a week when its likeliest week reaches this
        self.more = float(self.p.get("more", 2.0))  # and its other weeks are asked too when they reach this
        self.place = bool(self.p.get("place", 0.0))  # the trees say which grid and how many weeks, the network which week
        self.wide = self.p["in_w"].shape[1] > NET_SHARE + len(NET_EXTRA)  # fitted on tokens with their grid's sums

    @staticmethod
    def _norm(h: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = h.mean(-1, keepdims=True)
        return (h - m) / np.sqrt(((h - m) ** 2).mean(-1, keepdims=True) + 1e-5) * w + b

    def out(self, x: np.ndarray, same: np.ndarray) -> np.ndarray:
        """(n, 2): the mark's logit and the share, for tokens ``x`` (n, features) with ``same`` (n, n)."""
        p = self.p
        h = ((x - p["mean"]) / p["std"]) @ p["in_w"].T + p["in_b"]
        n, d = h.shape
        H = self.heads
        for i in range(self.layers):
            a = self._norm(h, p[f"l{i}_ln1_w"], p[f"l{i}_ln1_b"])
            q, k, v = ((a @ p[f"l{i}_{c}_w"].T + p[f"l{i}_{c}_b"]).reshape(n, H, d // H).transpose(1, 0, 2) for c in "qkv")
            att = q @ k.transpose(0, 2, 1) / math.sqrt(d // H) + p[f"l{i}_same"][:, None, None] * same[None]
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            h = h + (att @ v).transpose(1, 0, 2).reshape(n, d) @ p[f"l{i}_o_w"].T + p[f"l{i}_o_b"]
            f = self._norm(h, p[f"l{i}_ln2_w"], p[f"l{i}_ln2_b"])
            h = h + np.maximum(f @ p[f"l{i}_f1_w"].T + p[f"l{i}_f1_b"], 0.0) @ p[f"l{i}_f2_w"].T + p[f"l{i}_f2_b"]
        return self._norm(h, p["ln_w"], p["ln_b"]) @ p["out_w"].T + p["out_b"]


def told_net(ep: "Episode", recs: list, mode: dict, ref: dict, until: int | None, net: "Net", after: float,
             value: dict | None = None) -> set:
    """The (week, grid) a fitted network asks to be whole in place of the solve with the hull: all of a plan's short
    weeks are read together (``net_tokens``); a grid whose likeliest week reaches the network's threshold is asked
    that week, and any other week of it that reaches the second threshold (``Net.more``)."""
    rows = _short_weeks(ep, recs, mode, ref, until, value)
    if not rows:
        return set()
    keys, x, same = net_tokens(rows, after, float(ep.T) if net.wide else None)
    prob = 1.0 / (1.0 + np.exp(-net.out(x, same)[:, 0]))
    weeks = set()
    for gi in rows:
        own = [i for i, key in enumerate(keys) if key[1] == gi]
        top = max(own, key=lambda i: prob[i])
        if prob[top] >= net.threshold:
            weeks.add(keys[top])
            weeks.update(keys[i] for i in own if prob[i] >= net.more)
    return weeks


def told_shares(ep: "Episode", recs: list, mode: dict, ref: dict, until: int | None, model: tuple, after: float,
                value: dict | None = None) -> tuple:
    """The shares of whole weeks a fitted model tells in place of the solve with the hull: ({(week, grid): share},
    the (week, grid) the hull would have covered). For every grid the model reads the state the plan starts from
    (``plan_features``): where it finds a whole week likely, the whole weeks it tells (at least the price of one)
    are spread over the grid's short weeks as the start's scarce fuel is burned."""
    mark, total, threshold = model[:3]
    i_burn = SHARE_FEATURES.index("burn_min")
    rows, told = _short_weeks(ep, recs, mode, ref, until, value), {}
    plans = _plans(ep, rows, after)
    for gi, own in rows.items():
        t, x = np.array([r[0] for r in own], dtype=float), np.array([r[1] for r in own], dtype=float)
        v = plans[gi]
        if 1.0 / (1.0 + math.exp(-mark.raw(v))) < threshold:
            continue
        need = v[PLAN_FEATURES.index("need")]
        whole = max(total.raw(v), need + 1e-3)
        burn = x[:, i_burn]
        spread = burn / burn.sum() if burn.sum() > 0 else np.full(len(t), 1.0 / len(t))
        for (week, _f), share in zip(own, whole * spread):
            told[(int(week), gi)] = float(share)
    return told, {(int(t), gi) for gi, own in rows.items() for t, _f in own}


def _short_weeks(ep: "Episode", recs: list, mode: dict, ref: dict, until: int | None, value: dict | None = None) -> dict:
    """{grid: [(week, its ``SHARE_FEATURES``)] in the order of the weeks}: the short weeks the hull would cover."""
    inst = ep.inst
    keys = [key for key, gm in mode["grid"].items() if gm == "OFF" and (until is None or key[0] <= until) and inst.grid_fabs[key[1]]]
    i_rmax, rows = SHARE_FEATURES.index("rmax"), {}
    for (t, gi), f in share_features(ep, recs, mode, ref, keys, value).items():
        if f[i_rmax] > 0.0:
            rows.setdefault(gi, []).append((t, f))
    for own in rows.values():
        own.sort(key=lambda r: r[0])
    return rows


def _plans(ep: "Episode", rows: dict, after: float) -> dict:
    """{grid: what the fitted trees read of it (``MODEL_FEATURES``)} from ``_short_weeks``'s rows."""
    return with_rivals({gi: plan_features(np.array([r[0] for r in own], dtype=float), np.array([r[1] for r in own], dtype=float),
                                          float(ep.T), after) for gi, own in rows.items()})


def told_weeks(ep: "Episode", recs: list, mode: dict, ref: dict, until: int | None, model: tuple, after: float,
               value: dict | None = None) -> set:
    """The (week, grid) a fitted model asks to be whole in place of the solve with the hull, when it also tells
    where the first whole week falls (``share_model``'s fourth): for every grid it finds a whole week likely in, the
    short week nearest to the week told, and after it as many of the grid's next short weeks as the whole weeks told
    pay for beyond the price of the first. No shares are rounded: where the hull puts a week follows from the fuel
    it re-routes, which the start's own burn does not show."""
    mark, total, threshold, when, fitted = model[:5]
    net = model[5] if len(model) > 5 and model[5] is not None and model[5].place else None
    weeks = set()
    rows = _short_weeks(ep, recs, mode, ref, until, value)
    plans = _plans(ep, rows, after)
    likely = {}
    if net is not None and rows:  # the network reads all the plan's short weeks together: its likeliest week of a grid
        keys, x, same = net_tokens(rows, after, float(ep.T) if net.wide else None)
        likely = dict(zip(keys, net.out(x, same)[:, 0]))
    for gi, own in rows.items():
        t, v = np.array([r[0] for r in own], dtype=float), plans[gi]
        if 1.0 / (1.0 + math.exp(-mark.raw(v))) < threshold:
            continue
        if net is not None:
            first = max(range(len(t)), key=lambda i: likely[(int(t[i]), gi)])
        else:
            first = int(np.argmin(np.abs(t - when.raw(np.r_[v[:fitted], t[-1], float(len(t))]))))
        count = 1 + max(0, int(math.floor(total.raw(v) - v[PLAN_FEATURES.index("need")])))
        weeks.update((int(week), gi) for week in t[first : first + count])
    return weeks


def _rounded_from(ep: "Episode", shares: dict, domain: set, mode: dict, write: str = "SOFT") -> int:
    """``_rounded`` for shares that come as a table: ``domain`` are the (week, grid) the hull would have covered."""
    inst, marks, psi = ep.inst, ep.marks, ep.psi
    count = 0
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        k = ga.rationed
        have, after_whole = 0.0, False
        for t in range(1, ep.T + 1):
            if (t, gi) not in domain:
                after_whole = mode["grid"][(t, gi)] != "OFF"
                continue
            share = float(shares.get((t, gi), 0.0))
            cap = ga.shares[k] * float(marks.G_bar[t - 1][gi]) if k in ga.fuels else 0.0
            if not after_whole and cap > 0 and psi * ga.ibar[k] > 0 and mode["fuel"][(t, gi, k)] == "R":
                need, ready = max(1.0, psi * ga.ibar[k] / cap), have
            else:
                need, ready = 1.0, have + share
            have += share
            after_whole = ready >= need - 1e-6
            if after_whole:
                have -= need
                mode["grid"][(t, gi)] = write
                count += 1
    return count


def moved(hint: dict | None, weeks: int = 1) -> dict | None:
    """A hint of a window that started ``weeks`` earlier, in this window's weeks."""
    if hint is None:
        return None
    return {kind: {(key[0] - weeks, *key[1:]): v for key, v in table.items() if key[0] > weeks}
            for kind, table in hint.items()}


def _tilt(ep: "Episode", C: Cell, tilt: tuple | None) -> int:
    """next_lab: the earlier week of the hull cell is worth more, so that scarce fuel is brought forward.

    ``tilt`` is ``(worth, span)``: ``worth`` USD a unit of a grid's fab ratio is worth, by grid, and ``span`` the
    weeks over which that falls to zero. Written on the hull cell only, on ``jrho`` - the share with which a grid
    feeds its fabs in a week the program may make whole. Returns the cells priced.

    Why on ``jrho`` and not on the lots. Without it two schedules of the same whole weeks cost the program the same,
    so it takes the one easier to reach, which is the later one (fuel arrives over 2 to 6 weeks and a terminal holds
    more of it later). ``_rounded`` then marks a whole week late in the window, only the first week of the plan is
    played, and next week the same choice is made again: the full week is always "in a few weeks". Measured on Small
    111, episode 62, grid KR: the base holds ``lots 0.00`` for fourteen weeks in a row and writes "whole 1 of 1"
    every week; with the tilt weeks 15, 16 and 20 give 0.99, 0.23 and 1.00, and the episode goes 0.9007 to 0.9665.

    Not ``lot_tilt``, which paid per lot and was measured to lose (0.1/0.25/0.5 gave +0.033/+0.030/+0.025 against
    +0.038 without it): a bonus on lots is earned by starting more lots, including ones that never sell, while a
    share cannot be raised without fuel actually being there, so this one is earned only by moving fuel earlier.

    Nor is it a price for keeping a dated whole week: that was measured a nil by construction (the output matched to
    the cent with a price ten times the largest coefficient of the objective), because ``_rounded`` scans the weeks
    from the first and marks where the banked fuel first pays for a whole week - the date follows the fuel's arrival
    and is not a variable of the objective (``hub/tried/mpc.md``, next_lab).

    Measured, paired on the tuning sets against ``anastasiia_plan_hull``: Small 111 x64 +0.0029 (+0.0010..+0.0051),
    cheaper in 38 of 64, at no CPU; Full 111 x32 +0.0008 with the interval holding zero. The grid of the number is a
    plateau (0.02 the same, 0.04 worse). It does not carry to the one-solve form (-0.0007 Small / -0.0035 Full), and
    ``hull_every`` 1 there is worse still: with ``hull_only`` "tail" the relaxation is played as the action.
    """
    if not tilt:
        return 0
    worth, span = tilt
    if worth is None:
        return 0
    if C.cost is None:
        C.cost = np.zeros(ep.N)
    reach = max(1.0, float(span))
    n = 0
    for t, gi in C.hull:
        w = max(0.0, 1.0 - (t - 1) / reach) * (float(worth[gi]) if np.ndim(worth) else float(worth))
        if w > 0.0:
            C.cost[ep.jrho(t, gi)] -= w
            n += 1
    return n


def _rounded(ep: "Episode", C: Cell, x: np.ndarray, mode: dict, write: str = "MID", lean: float = 0.0) -> int:
    """Whole weeks from a solution with "HULL" weeks: ``mode["grid"]`` changed in place, the count of weeks written.

    Along each grid's weeks the shares of a whole week the solution asked for are summed, and each time the sum
    reaches the price of one more whole week that week is written as closed ("MID": the base load served in full,
    the fabs on what is above it). The price is one week's burn, or, for a week whose rationed fuel burns under its
    ration and that does not follow a whole week, the stock the ration asks for the week before (more than a week's
    burn), counted from the fuel of the weeks before it: a ration is lifted by last week's stock. The number of whole
    weeks is then no more than the solution's fuel pays for, each no earlier than its fuel was there. ``lean``: a
    week is written already when the sum is this share of a whole week short of the price (0.5: rounding to the
    nearest instead of down); what was missing is owed by the grid's later shares.
    """
    inst, marks, psi = ep.inst, ep.marks, ep.psi
    count = 0
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        k = ga.rationed
        have, after_whole = 0.0, False
        for t in range(1, ep.T + 1):
            rmax = C.hull.get((t, gi))
            if rmax is None:
                after_whole = mode["grid"][(t, gi)] != "OFF"
                continue
            share = float(x[ep.jrho(t, gi)]) / rmax
            cap = ga.shares[k] * float(marks.G_bar[t - 1][gi]) if k in ga.fuels else 0.0
            if not after_whole and cap > 0 and psi * ga.ibar[k] > 0 and mode["fuel"][(t, gi, k)] == "R":
                need, ready = max(1.0, psi * ga.ibar[k] / cap), have  # the stock must be there the week before
            else:
                need, ready = 1.0, have + share
            have += share
            after_whole = ready >= need - lean - 1e-6
            if after_whole:
                have -= need
                mode["grid"][(t, gi)] = write
                count += 1
    return count


def _whole_weeks(ep: "Episode", recs: list, mode: dict, close: float, until: int | None, rationed: bool = False,
                 write: str = "MID") -> int:
    """Write a grid's nearly closed weeks as whole weeks: ``mode["grid"]`` changed in place, the count of weeks closed.

    A fab runs only on what its grid delivers above the base load, so a fuel a grid receives a little of every week
    closes no week, while the same fuel in whole weeks closes some. The shed base load is the same either way, which
    is why no regime read off the trajectory makes the program gather it. Here the trajectory's own schedule is
    rounded: over the weeks a grid sheds less than ``close`` of its base load, the smallest share of a week's burn
    that a stock-limited fuel segment got is summed, and each time the sum passes one more whole week that week is
    written as closed (the base load served in full); the rest stay as they are and give their fuel up. The total
    burned is the trajectory's, week by week no earlier than it was there. ``rationed``: a segment that burns less
    than its cap because last week's stock was under the ration's threshold counts as well (gas gathered into whole
    weeks needs the stock back above the threshold the week before, which the program may or may not manage).
    """
    inst, marks = ep.inst, ep.marks
    count = 0
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        have = 0.0
        for t in range(1, min(until or ep.T, ep.T) + 1):
            if mode["grid"][(t, gi)] != "OFF":
                continue
            rec = recs[t - 1]
            if rec.shed[gi] >= close * float(marks.y_bar[t - 1][gi]):
                continue
            share = 1.0
            for k in ga.fuels:  # the scarcest stock-limited segment: what part of a whole week it had
                cap = ga.shares[k] * float(marks.G_bar[t - 1][gi])
                if cap > 0 and (mode["fuel"][(t, gi, k)] == "S" or (rationed and mode["fuel"][(t, gi, k)] == "R")):
                    share = min(share, float(rec.segment[(gi, k)]) / cap)
            if share >= 1.0:  # short for another reason (a ration): not a week to gather into
                continue
            have += share
            if have >= 1.0 - 1e-9:
                have -= 1.0
                mode["grid"][(t, gi)] = write
                count += 1
    return count


def _beside(acts: list, recs: list, anchor, price, bonus) -> int:
    """In cents: the price of what a plan sends above and below the anchor, minus its lots' bonus: what the loop
    compares beside the simulator's cost."""
    total = 0.0
    if anchor is not None and price is not None:
        for (fl, _o, _h), (base, _o2, _h2) in zip(acts, anchor):
            total += sum(price[s] * abs(fl.get(s, 0.0) - base.get(s, 0.0)) for s in set(fl) | set(base))
    if bonus is not None:
        per_fab, weights = bonus[0], _weeks(bonus[1], len(recs))
        total -= sum(w * float(np.dot(per_fab, rec.lots_started)) for w, rec in zip(weights, recs) if w != 0.0)
    return int(round(total * 100))


def _weeks(weights, T: int) -> np.ndarray:
    """A bonus's weight per week of a window of ``T`` weeks: 1 up to a last week given as a number, else as given."""
    if np.isscalar(weights):
        return (np.arange(1, T + 1) <= int(weights)).astype(float)
    out = np.zeros(T)
    w = np.asarray(weights, dtype=float)[:T]
    out[: len(w)] = w
    return out


def switch_step(ep: "Episode", d: dict, tries: int = 4, tweak=None, bonus=None, deadline: float | None = None,
                gate: bool = False,
                last_week: int | None = None, min_gain: float = 1e8, anchor: list | None = None, price=None,
                time_limit: float = 600.0, method: str = "simplex") -> dict:
    """A few grid-weeks tried for a switch from "sheds base load" to "runs its fabs", on top of ``descend``'s result
    ``d`` (``switch_on`` is the offline loop; this is one bounded step for an agent's week).

    The cell of ``d``'s trajectory is solved once for its duals; the ``tries`` shedding grid-weeks whose lots are worth
    most there are tried one by one (the grid serves its base load in full, every fuel segment at its cap; if that
    leaves no room, the week before on its ration); a cheaper claim is played and kept when the simulator agrees.
    Returns ``d`` updated (``switched``: the (week, grid) kept).
    """
    inst = ep.inst
    acts, recs, J = d["acts"], d["recs"], d["J_compared"]
    d.setdefault("switched", [])

    def left() -> float:
        return time_limit if deadline is None else min(time_limit, deadline - time.process_time())

    def cell_of(mode, ref):
        C = ep.cell(mode, ref, anchor, price, bonus, gate=gate)
        if tweak is not None:
            tweak(C)
        return C

    spent = max([h[2] for h in d.get("hist", []) if h[1] is not None], default=0.0)  # what a solve takes here
    if left() <= max(0.05, 1.25 * spent):
        return d
    mode, ref = ep.regimes(recs)
    sol = ep.solve(cell_of(mode, ref), method=method, basis=d.get("basis"), time_limit=left())
    if sol["status"] != "Optimal":
        return d
    basis, cd = sol["basis"], sol["col_dual"]
    worth = []
    for (t, gi), gm in mode["grid"].items():
        if gm != "OFF" or (last_week is not None and t > last_week):
            continue
        v = sum(-cd[ep.col("p", t, fi)] * float(ep.m.ub[ep.col("p", t, fi)]) for fi in inst.grid_fabs[gi])
        if v > min_gain / 100:
            worth.append((v, t, gi))
    for _v, t, gi in sorted(worth, reverse=True)[:tries]:
        if mode["grid"][(t, gi)] != "OFF":
            continue
        ga = inst.nodes[inst.grids[gi]].grid
        for prime in (False, True):
            if left() <= max(0.05, 1.25 * spent):
                break
            trial = {k: dict(m) if k in ("grid", "fuel") else m for k, m in mode.items()}
            trial["grid"][(t, gi)] = "MID"
            for k in ga.fuels:
                trial["fuel"][(t, gi, k)] = "F"
            if prime:
                if ga.rationed is None or t == 1:
                    break
                trial["fuel"][(t - 1, gi, ga.rationed)] = "R"
            s2 = ep.solve(cell_of(trial, ref), method=method, basis=basis, time_limit=left())
            if s2["status"] != "Optimal":
                continue
            acts2 = ep.actions(s2["x"])
            recs2, J2 = ep.simulate(acts2)
            J2 += _beside(acts2, recs2, anchor, price, bonus)
            if J2 < J - min_gain:
                acts, recs, J, basis = acts2, recs2, J2, s2["basis"]
                d["switched"].append((t, gi))
                mode, ref = ep.regimes(recs)
            break
    d.update(acts=acts, recs=recs, J_compared=J, J=J - _beside(acts, recs, anchor, price, bonus), basis=basis)
    return d


def switch_on(ep: "Episode", acts: list, sweeps: int = 3, tries: int = 40, min_gain: float = 1e8, passes: int = 60,
              last_week: int | None = None, log=None) -> dict:
    """``descend``, then grid-weeks switched on away from a border, while that pays.

    ``descend`` changes a regime only where the trajectory already lies on the border of two cells. A grid that sheds
    in a week stays off there even when a week of lots is worth more than the base load it would take from other
    weeks, because the fuel must first be gathered. A trial names such a (week, grid), by the worth of its lots in the
    last solution's duals: the grid serves its base load in full that week and every fuel segment runs at its cap
    (the week before may run on its ration, so that the stock is there). The cell is solved; a cheaper claim is
    played, and kept when the simulator agrees. ``tries`` trials a sweep, ``sweeps`` sweeps, a ``descend`` after each
    sweep that kept something.

    Returns ``descend``'s dict, with ``switched``: the (week, grid) kept.
    """
    d = descend(ep, acts, iters=passes)
    d["switched"] = []
    inst = ep.inst
    for _ in range(sweeps):
        recs, J, kept = d["recs"], d["J"], 0
        mode, ref = ep.regimes(recs)
        sol = ep.solve(ep.cell(mode, ref), basis=d.get("basis"))
        if sol["status"] != "Optimal":
            break
        basis = sol["basis"]
        cd = sol["col_dual"]
        worth = []
        for (t, gi), gm in mode["grid"].items():
            if gm != "OFF" or (last_week is not None and t > last_week):
                continue
            v = sum(-cd[ep.col("p", t, fi)] * float(ep.m.ub[ep.col("p", t, fi)]) for fi in inst.grid_fabs[gi])
            if v > min_gain / 100:
                worth.append((v, t, gi))
        for v, t, gi in sorted(worth, reverse=True)[:tries]:
            if mode["grid"][(t, gi)] != "OFF":
                continue
            ga = inst.nodes[inst.grids[gi]].grid
            done = False
            for prime in (False, True):
                trial = {k: dict(m) if k in ("grid", "fuel") else m for k, m in mode.items()}
                trial["grid"][(t, gi)] = "MID"
                for k in ga.fuels:
                    trial["fuel"][(t, gi, k)] = "F"
                if prime:
                    if ga.rationed is None or t == 1:
                        break
                    trial["fuel"][(t - 1, gi, ga.rationed)] = "R"
                s2 = ep.solve(ep.cell(trial, ref), basis=basis)
                if s2["status"] != "Optimal":
                    continue
                if s2["J"] * 100 < J - min_gain:
                    acts2 = ep.actions(s2["x"])
                    recs2, J2 = ep.simulate(acts2)
                    if J2 < J - min_gain:
                        if log:
                            log(f"    week {t} grid {gi}: {J / 1e11:.2f} -> {J2 / 1e11:.2f} (claim {s2['J'] / 1e9:.2f}){' primed' if prime else ''}")
                        acts, recs, J, basis, kept = acts2, recs2, J2, s2["basis"], kept + 1
                        d["switched"].append((t, gi))
                        mode, ref = ep.regimes(recs, ep.hints(ep.cell(trial, ref), s2, trial))
                        done = True
                break
            del done
        if not kept:
            break
        switched = d["switched"]
        d = descend(ep, acts, iters=passes, basis=basis)
        d["switched"] = switched
    return d
