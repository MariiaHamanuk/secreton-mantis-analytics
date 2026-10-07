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
                pools.append({"k": k, "value": float(end_k[k]), "cap": float(ga.ibar.get(k, 0.0)) + weeks * burn, "cols": []})
        for k in fuels:
            burn = sum(inst.nodes[g].grid.shares.get(k, 0.0) * inst.nodes[g].grid.deliverable for g in inst.grids)
            rest[k] = len(pools)
            pools.append({"k": k, "value": float(end_k[k]), "cap": weeks * burn, "cols": []})
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
    def cell(self, mode: dict, ref: dict, anchor: list | None = None, price=None, bonus=None) -> Cell:
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
                            C.row([(jr, cap), (jG, -rmax)], -INF, dk)
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
        (wafers on hand equal the capacity). Plant: 0 or 1. Grid: "OFF", "MID" or "ON".
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
              crossover: bool = True) -> dict:
        """The cell's optimum: status, objective in USD (the simulator's J when the cell is exact), x, the duals and
        the basis. ``basis`` (the column and row statuses of another cell's optimum, ``shifted`` when that cell
        started a week earlier) is where the simplex starts: every cell has the same columns and rows. ``method``
        "auto": the simplex on a small program, the interior-point method on a large one (Full, a window of 26
        weeks: a steady 1.2 to 1.6 s a solve where the simplex, from a basis or not, takes 0.6 to 18 s)."""
        hs, Highs = highs()
        if method == "auto":  # the simplex (from a basis, if any) on a small program, interior point on a large one
            method = "simplex" if self.N <= BIG else "ipm"
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
        h = Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("solver", method)
        h.setOptionValue("time_limit", float(time_limit))
        rough = method == "ipm" and not crossover  # plan_lab: the interior point as it stops, a fifth faster: its x is
        if rough:  # feasible and optimal to the solver's tolerance, but there is no basis and the duals are not to be read
            h.setOptionValue("run_crossover", "off")
        h.passModel(lp)
        if basis is not None and len(basis[0]) == n_col and len(basis[1]) == n_row:
            codes = _statuses(hs)
            b = hs.HighsBasis()
            b.col_status, b.row_status = [codes[i] for i in basis[0]], [codes[i] for i in basis[1]]
            b.valid, b.alien = True, True  # HiGHS completes it: a shifted or another cell's basis need not be square
            h.setBasis(b)
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        if rough and status == "Unknown" and int(h.getInfo().primal_solution_status) == 2:
            status = "Optimal"
        if status not in ("Optimal", "Time limit reached") and basis is not None:  # a start HiGHS could not use: cold
            return self.solve(C, method, time_limit)
        out_seconds = float(h.getRunTime())
        info = h.getInfo()
        out = {"status": status, "J": float(info.objective_function_value) if status == "Optimal" else math.nan,
               "iterations": int(info.simplex_iteration_count), "seconds": out_seconds}
        if status == "Optimal":
            sol, got = h.getSolution(), h.getBasis()
            if C.cost is not None:  # J without the anchor's price and the lots' bonus: what the simulator will charge
                out["J"] -= float(C.cost @ np.asarray(sol.col_value, dtype=float))
            out.update(x=np.asarray(sol.col_value, dtype=float), col_dual=np.asarray(sol.col_dual),
                       row_dual=np.asarray(sol.row_dual)[self.base.shape[0] :],
                       basis=None if rough else (np.array([int(v) for v in got.col_status], dtype=np.int8),
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
            hull_rough: bool = False, hull_only: bool | str = False, marks: set | None = None) -> dict:
    """The loop from ``acts`` (weekly (flows, overrides, holds)): the best played trajectory and how it was reached.

    Each pass reads the regimes of the trajectory the simulator played (a tie as the last solution's duals say, with
    ``hints``), solves the cell and plays the solution. It stops after ``patience`` passes in a row that gain less
    than ``min_gain`` cents, or when a played solution is worse than the trajectory it came from. ``tweak(cell)``
    may add bounds of its own to every cell before it is solved (a floor under some flows, say); with
    ``force_first`` the first solution is taken even when the start, which need not respect them, is cheaper.
    ``played``: the start's (records, cost) when it has been played already. ``deadline``: process time after which no
    pass starts, and which a solve may not pass. ``hint``: how to read the start's ties (last week's, moved on by a
    week: ``moved``); the last pass's hint comes back as ``hint``. ``close``: in the first cell the weeks a
    grid nearly closes are gathered into whole weeks (``_whole_weeks``), up to week ``close_until``; when that cell
    has no solution the plain one is solved. ``hull``: the worth of a whole week inside the program. "round" (or
    True): before the first cell, the same cell with every short week of a grid with fabs written as "HULL" is
    solved, the shares of whole weeks it asks for are rounded into weeks (``_rounded``), and in the first cell those
    weeks are "SOFT": closed when the fuel can be gathered there, short as before when it cannot, so no choice of
    weeks leaves the cell without a solution (one more solve a call). "hard": the rounded weeks are written as
    closed instead, and the plain cell is solved when that has no solution. "burn": no solve before the first cell;
    the weeks are those ``_whole_weeks`` rounds the start's own burn into (``close``: the threshold, 1 when 0),
    "SOFT" too. ``anchor`` and ``price``
    (``Episode.cell``): the loop then minimises the played cost plus the price of leaving the anchor; ``bonus``:
    minus the lots' bonus.

    Returns a dict: ``acts``, ``recs``, ``J`` (cents, played), ``J0`` (cents, the start as played), ``hist`` (per
    pass: the cell's claim in USD, the played cost in cents, CPU seconds of the pass), ``basis`` (of the last cell
    solved, a start for the next: ``basis`` here is such a start).
    """
    def away(a: list, r: list) -> int:
        return _beside(a, r, anchor, price, bonus)

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
        closed, reuse = 0, None
        if hull == "burn" and number == 0:
            plain = dict(mode["grid"])
            closed = _whole_weeks(ep, recs, mode, close or 1.0, close_until, True, write="SOFT")
            out["hull"] = "burn"
        elif hull and number == 0:
            plain = dict(mode["grid"])
            tail = hull_only == "tail"  # plan_lab: this solution is the week's plan, so its first week stays exact
            for key, gm in plain.items():
                if gm == "OFF" and (close_until is None or key[0] <= close_until) and not (tail and key[0] == 1):
                    mode["grid"][key] = "HULL"
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None:
                tweak(C)
            # plan_lab: only this solution's x is read (the shares to round), so it may be the rough one
            wide = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit,
                            crossover=tail or not hull_rough)
            mode["grid"] = dict(plain)
            out["hull"] = wide["status"]
            if wide["status"] == "Optimal":
                closed = _rounded(ep, C, wide["x"], mode, write="MID" if hull == "hard" else "SOFT")
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
        if reuse is not None:
            sol, closed = reuse, 0
        else:
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None:
                tweak(C)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit)
        if closed and sol["status"] not in ("Optimal", "Time limit reached"):  # no fuel for all of them: the plain cell
            mode["grid"] = plain
            closed = 0
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None:
                tweak(C)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit)
        out["closed"] = closed
        if sol["status"] not in ("Optimal", "Time limit reached") and hint is not None:  # a tie read the other way left no room
            mode, ref = ep.regimes(recs)
            C = ep.cell(mode, ref, anchor, price, bonus)
            if tweak is not None:
                tweak(C)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit)
        if sol["status"] not in ("Optimal", "Time limit reached") and tweak is not None:  # the extra bounds left no room
            C = ep.cell(mode, ref, anchor, price, bonus)
            sol = ep.solve(C, method=method, basis=out["basis"], time_limit=time_limit)
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
        idle = idle + 1 if J2 >= J - min_gain else 0
        if not first and (idle > patience or J2 > J + 100 * min_gain):
            break
        acts, recs, J = acts2, recs2, J2
        hint = ep.hints(C, sol, mode) if hints else None
    out.update(J=best[0] - away(best[1], best[2]), J_compared=best[0], acts=best[1], recs=best[2], hint=hint)
    return out


def moved(hint: dict | None, weeks: int = 1) -> dict | None:
    """A hint of a window that started ``weeks`` earlier, in this window's weeks."""
    if hint is None:
        return None
    return {kind: {(key[0] - weeks, *key[1:]): v for key, v in table.items() if key[0] > weeks}
            for kind, table in hint.items()}


def _rounded(ep: "Episode", C: Cell, x: np.ndarray, mode: dict, write: str = "MID") -> int:
    """Whole weeks from a solution with "HULL" weeks: ``mode["grid"]`` changed in place, the count of weeks written.

    Along each grid's weeks the shares of a whole week the solution asked for are summed, and each time the sum
    reaches the price of one more whole week that week is written as closed ("MID": the base load served in full,
    the fabs on what is above it). The price is one week's burn, or, for a week whose rationed fuel burns under its
    ration and that does not follow a whole week, the stock the ration asks for the week before (more than a week's
    burn), counted from the fuel of the weeks before it: a ration is lifted by last week's stock. The number of whole
    weeks is then no more than the solution's fuel pays for, each no earlier than its fuel was there.
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
            after_whole = ready >= need - 1e-6
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
        C = ep.cell(mode, ref, anchor, price, bonus)
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
