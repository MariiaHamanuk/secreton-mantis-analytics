"""The public generator as a filtering model, and the events still to come drawn from what an agent has seen.

``Tables`` holds what of a task's generator a filter reads: the branching matrix of the two-block Hawkes process,
the type and target rules, the laws of duration and severity, the lead laws and decoy shares of the announcement
channels, the regime's laws. Everything is taken from the installed package (``shockbench_flow.disruption``,
``task_generator(task)``); nothing is fitted.

``Futures`` draws, for each of K scenarios, the events that start after an instant ``s`` given what the
observations up to ``s`` show (``lab/anastasiia/frontier_lab/notes/s_information.md``). One scenario is one joint
draw of:

- the regime of every region (conflict state, tension) from its law given the region's warning score and the
  conflicts seen there;
- the cluster process itself, by the generator's own cluster representation: immigrants of every week at the rate
  that regime and the straits' warning scores imply, the children of the events already seen, the children of what
  ran before the episode (at the stationary excitation), and their descendants. An event of the draw that falls at
  or before ``s`` stays only if an agent could not have seen it (an event without a graph operation); a visible one
  is dropped with its descendants, since the episode's own events stand in its place;
- of the events of an announced type still to come, only those whose announcement would not be out by ``s`` (each
  draws its own lead): the announced ones are the threads below;
- every announcement thread alive at ``s``: real with the share of real threads of its channel, its effect in the
  stated week, or at an instant drawn from the channel's lead law given the thread's age and that a shown thread
  takes effect inside the episode. A real thread's event is a parent like any other.

The draws are kept from week to week: every random number is keyed by the scenario and by the object it belongs to
(a week's immigrants, a parent's children, a thread), so a scenario's future changes only where the observations
changed its law, and a thread's draw stands until the observations contradict it (its date passed and it is still
alive), which keeps its law exact.
"""

import math

import numpy as np
from scipy.special import ndtr


TARIFF, SANCTION, OUTAGE, CLOSURE, CONFLICT, PIRACY, ENERGY, WEATHER, STRIKE = range(9)
NOOP = -1  # an event of the cluster process without a graph operation: never seen, it still excites
BLOCK = (0, 0, 0, 1, 1, 1, 1, -1, -1)  # the block of an event type: policy 0, militarised 1, no block -1
K_CHOKEPOINT, K_EDGE, K_NODE, K_REGION = range(4)  # the target kinds of an event
CHANNEL_TYPE = {"tariff_formal": TARIFF, "tariff_informal": TARIFF, "tariff_final": TARIFF,
                "sanction_legal": SANCTION, "ties_threat": SANCTION, "mid_threat": CLOSURE}  # fmt: skip
UNDATED = {SANCTION: "ties_threat", CLOSURE: "mid_threat"}  # the channel of a thread that states no date
NEW_ROW = 1_000_000  # the row of an event that is not one of the episode's
LATE = 4  # a threat announced up to this many weeks after an event of its kind on its target is that event's
RECENT = 14.0  # weeks after which a seen event's own children are no longer drawn (exp(-14) of them are left)


class Tables:
    """What of a task's generator the filter reads, built once per task (``Tables.of``)."""

    _kept: dict = {}

    @classmethod
    def of(cls, task: str) -> "Tables":
        if task not in cls._kept:
            cls._kept[task] = cls(task)
        return cls._kept[task]

    def __init__(self, task: str) -> None:
        from shockbench_flow import marks as M
        from shockbench_flow.disruption import hawkes, regime, targets
        from shockbench_flow.hosting.tasks import task_generator
        from shockbench_flow.information.theta import Standard
        from shockbench_flow.omega import codes

        inst, p = task_generator(task)
        self.task, self.inst, self.p, self.M, self.names = task, inst, p, M, codes.EVENT_TYPES
        self.T, self.R, self.C = inst.T, len(inst.regions), len(inst.chokepoints)
        R = self.R
        self.G = hawkes.branching_matrix(inst, p)  # (2R, 2R): rows the child's stream, columns the parent's
        self.kids = [np.flatnonzero(self.G[:, s]) for s in range(2 * R)]
        self.beta = 1.0 / float(p.hawkes.beta_inv)
        self.base = hawkes.baseline_vector(inst, p)
        self.multM = np.array(p.hawkes.militarised_by_conflict, dtype=float)
        self.multP = np.array(p.hawkes.policy_by_tension, dtype=float)
        self.Pi = regime.weekly_conflict_matrix(p.regime.P_yr)
        self._pi_pow = {0: np.eye(3)}
        self.rules = targets.TypeRules(inst, p, None)
        self.dyads = [(inst.region_index[x], inst.region_index[y]) for x, y in p.regime.dyads]
        self.D = len(self.dyads)
        theta = Standard()
        self.a = float(theta.a_region)  # the share of the latent risk in the warning score
        self.b = math.sqrt(1.0 - self.a**2)
        self.lat_r, self.lat_c = p.latent_region, p.latent_chokepoint
        self.c_c = regime.normaliser(self.lat_c)
        self.chk_ord = dict(inst.chokepoint_ordinal)
        self.reg_chk = {m: [self.chk_ord[c] for c in inst.chokepoints if m in inst.chokepoint_adjacency.get(c, ())]
                        for m in range(R)}  # fmt: skip
        self.duration, self.severity = dict(p.laws.duration), dict(p.laws.severity)
        self.straits = targets.strait_chokepoints(inst, p)  # {the dyad's regions: the strait's node}
        self.restoration = M.restoration_regions(inst)
        info = p.information
        self.lead = {ch: law for ch, law in info.lead}
        self.real = {ch: 1.0 - phi for ch, phi in info.phi_bar}  # the share of real threads of a channel
        self.informal = float(info.informal_share)
        q = regime.tension_entry(p.regime)
        self.p_tension = q / (q + 1.0 / float(p.regime.tension_spell))  # by the conflict state
        # the law of a unit's conflict chain given its score: the chain's stationary law at X, X | S ~ N(a S, b^2)
        xs = np.linspace(-6.0, 6.0, 241)
        at_x = regime.conflict_stationary(xs, 1.0, self.Pi, self.lat_r)
        self._S = np.linspace(-5.0, 5.0, 201)
        w = np.exp(-0.5 * ((xs[None, :] - self.a * self._S[:, None]) / self.b) ** 2)
        self._pz_S = (w / w.sum(axis=1, keepdims=True)) @ at_x
        # the stationary excitation of every stream, at the stationary law of the regimes
        prior = np.exp(-0.5 * xs**2)
        pz = np.tile((prior / prior.sum()) @ at_x, (R, 1))
        for x, y in self.dyads:
            for m in (x, y):
                war = 1.0 - (1.0 - pz[m, 2]) ** 2  # its own chain's war, or the dyad's
                calm = np.array([pz[m, 0], pz[m, 1], 0.0]) * (1.0 - war) / max(1.0 - pz[m, 2], 1e-12)
                pz[m] = calm + np.array([0.0, 0.0, war])
        mu = np.zeros(2 * R)
        mu[:R] = self.base[:R] * (1.0 + (self.multP[1] - 1.0) * (pz @ self.p_tension))
        mu[R:] = self.base[R:] * (pz @ self.multM)  # the parts of a militarised week sum to 1 at the mean tilt
        self.psi_bar = np.linalg.solve(np.eye(2 * R) - self.G, self.G @ mu)
        # what an event cuts, for the coverage tables
        geo = M._Geo(inst)
        self.geo = geo
        self.units = sorted({t.index for m in range(R) for t, _ in self.rules.candidates(SANCTION, m)})
        self.ff = {j: self._friendly_fire(j) for j in self.units}
        self.conflict_edges = {m: geo.between(m, None) for m in range(R)}
        fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
        self.fuel_edge = np.array([bool(fuels.intersection(e.K)) for e in inst.edges])

    def _friendly_fire(self, j: int) -> list:
        """The other edges of the dyad that a sanction on edge ``j`` cuts (``marks._sanction``)."""
        inst, geo = self.inst, self.geo
        edge = inst.edges[j]
        a, b = geo.region[edge.tail], geo.region[edge.head]
        if a == b:
            return []
        return [e for e in geo.between(a, b) if inst.edges[e].K and not set(edge.K).intersection(inst.edges[e].K)]

    def pi_pow(self, k: int) -> np.ndarray:
        """The conflict chain's matrix of ``k`` weeks (untilted)."""
        k = int(min(max(k, 0), 520))
        if k not in self._pi_pow:
            self._pi_pow[k] = np.linalg.matrix_power(self.Pi, k)
        return self._pi_pow[k]

    def pz_from_S(self, S) -> np.ndarray:
        """(len(S), 3): the law of a unit's conflict state given its warning score."""
        S = np.clip(np.asarray(S, dtype=float), self._S[0], self._S[-1])
        return np.stack([np.interp(S, self._S, self._pz_S[:, k]) for k in range(3)], axis=-1)

    def tilt(self, S, lag) -> np.ndarray:
        """E[g(X) / c] of a strait ``lag`` weeks after the week its score ``S`` shows: the factor of the rate of
        the closures that are immigrants there (``g = exp(k min(X, cap))``; X | S is normal, an AR(1) step a week)."""
        lat, rho = self.lat_c, self.lat_c.rho
        mu = rho**lag * self.a * S
        sd = np.sqrt(np.maximum(1.0 - rho ** (2 * lag) * self.a**2, 1e-18))
        k, cap = lat.k, lat.x_cap
        return (np.exp(k * mu + 0.5 * k * k * sd * sd) * ndtr((cap - mu - k * sd * sd) / sd)
                + np.exp(k * cap) * (1.0 - ndtr((cap - mu) / sd))) / self.c_c  # fmt: skip

    def cuts(self, q) -> dict:
        """The elements a new event cuts, by item of the coverage tables: {"strait": {...}, "grid": {...},
        "proh": {...}, "edge": {...}} (a militarised closure's strait, an energy shock's grid, a sanction's own edge
        and the dyad's other edges, a conflict's edges)."""
        out = {"strait": set(), "grid": set(), "proh": set(), "edge": set()}
        if q.type == CLOSURE:
            out["strait"].add(self.chk_ord[q.target])
        elif q.type == ENERGY:
            out["grid"].add(self.inst.grid_ordinal[q.target])
        elif q.type == SANCTION and q.target_kind == K_EDGE:
            out["proh"].add(q.target)
            out["edge"].update(self.ff[q.target] if q.target in self.ff else self._friendly_fire(q.target))
        elif q.type == CONFLICT and q.counterpart >= 0:
            out["edge"].update(self.conflict_edges[q.region])
        return out


def _pick(u: float, weights) -> int:
    """The index a uniform picks among ``weights`` (inversion)."""
    c = np.cumsum(np.asarray(weights, dtype=float))
    return int(min(np.searchsorted(c, u * c[-1], side="right"), len(c) - 1))


class Futures:
    """The events still to come of K scenarios of one episode, each one joint draw (the module docstring).

    ``root`` is the root of the draws (never the episode's own), ``seed`` the agent's seed, ``episode`` its number.
    """

    def __init__(self, tables: Tables, K: int, seed: int, episode: int, root: int) -> None:
        self.t, self.K = tables, K
        self.seed = (int(root), int(seed) % (1 << 32), int(episode))
        t = tables
        self.u_imm = [self._rng(j, (0,)).random((2, t.R, t.T + 1)) for j in range(K)]  # a week's immigrants
        self.u_reg = [self._rng(j, (1,)).random((2, t.R + t.D)) for j in range(K)]  # the regimes
        self._us: list[dict] = [{} for _ in range(K)]
        self._kids: list[dict] = [{} for _ in range(K)]
        self.thread: list[dict] = [{} for _ in range(K)]  # thread -> [draws so far, real, effect instant]
        self.count = {"immigrant": 0, "child": 0, "thread": 0, "announced": 0, "redrawn": 0}

    # ----- the keyed random numbers -----------------------------------------------------------------------------------
    def _rng(self, j: int, key: tuple) -> np.random.Generator:
        return np.random.default_rng([*self.seed, j, *key])

    def _u(self, j: int, key: tuple) -> np.ndarray:
        """The uniforms of one object of scenario ``j`` (an event, a thread), drawn once."""
        got = self._us[j].get(key)
        if got is None:
            got = self._us[j][key] = np.clip(self._rng(j, (2, *key)).random(16), 1e-12, 1.0 - 1e-12)
        return got

    def _children(self, j: int, key: tuple, stream: int) -> list:
        """The children of one event of ``stream``: [(the child's stream, its rank, its delay in weeks)]."""
        got = self._kids[j].get(key)
        if got is None:
            t, rng, got = self.t, self._rng(j, (3, *key)), []
            for child in t.kids[stream]:
                for r in range(int(rng.poisson(t.G[child, stream]))):
                    got.append((int(child), r, float(rng.exponential(1.0 / t.beta))))
            self._kids[j][key] = got
        return got

    def _before(self, j: int) -> list:
        """What the events of before the episode still excite at the instant 0, at the stationary excitation:
        [(stream, rank, instant)] of their children."""
        got = self._kids[j].get((9,))
        if got is None:
            t, rng, got = self.t, self._rng(j, (4,)), []
            for stream in np.flatnonzero(t.psi_bar):
                for r in range(int(rng.poisson(t.psi_bar[stream] / t.beta))):
                    got.append((int(stream), r, float(rng.exponential(1.0 / t.beta))))
            self._kids[j][(9,)] = got
        return got

    # ----- one scenario -----------------------------------------------------------------------------------------------
    def draw(self, j: int, s: float, started: list, conflicts: list, messages: dict, scores) -> list:
        """The events of scenario ``j`` that start after the instant ``s`` (``marks.Event``, by onset).

        ``started``: the episode's events seen by ``s`` that started inside it, [(row, type, region, target, the
        instant it was first seen, whether it excites)]; ``conflicts``: the conflicts running at ``s``, [(region, the
        instant it was first seen or None if it was carried in)]; ``messages``: the columns of the messages shown by
        ``s``; ``scores``: the warning scores shown at ``s`` (regions, dyads, straits).
        """
        t, R, T = self.t, self.t.R, self.t.T
        S = np.asarray(scores, dtype=float)
        # the scenario's regimes
        pz = t.pz_from_S(S[:R])
        for m, first in conflicts:
            war = t.pi_pow(int(s - first))[2] if first is not None else t.pi_pow(int(80 + s))[2]
            pz[m] = war if first is not None else 0.5 * pz[m] + 0.5 * war
        u = self.u_reg[j]
        z = (u[0, :R] > pz[:, 0]).astype(int) + (u[0, :R] > pz[:, 0] + pz[:, 1])
        partners: dict = {m: () for m in range(R)}
        for d, (x, y) in enumerate(t.dyads):
            if u[0, R + d] > 1.0 - t.pz_from_S(S[R + d : R + d + 1])[0, 2]:  # the dyad at war: both regions are
                z[x] = z[y] = 2
                partners[x] += (y,)
                partners[y] += (x,)
        zp = (u[1, :R] < t.p_tension[z]).astype(int)
        weeks = np.arange(T + 1)
        tilt = np.stack([t.tilt(S[R + t.D + c], np.maximum(weeks - s, 1.0)) for c in range(t.C)], axis=1)  # (T + 1, C)
        self._state = (j, s, z, zp, partners, tilt)
        self._out: list = []
        # immigrants of every week: the rate of the week's regime, the closures at a strait times its tilt
        rate = np.zeros((2, R, T + 1))
        rate[0] = (t.base[:R] * t.multP[zp])[:, None]
        for m in np.flatnonzero(t.base[R:]):
            parts = t.rules.parts(1, int(m), int(z[m]))
            total = sum(w * tilt[:, t.chk_ord[g.index]] if code == CLOSURE else w for code, g, w in parts)
            rate[1, m] = t.base[R + m] * t.multM[z[m]] * total
        rate[:, :, 0] = 0.0
        for b, m, wk in np.argwhere(self.u_imm[j] > np.exp(-rate)):
            lam, uu = float(rate[b, m, wk]), float(self.u_imm[j][b, m, wk])
            n, term = 0, math.exp(-lam)
            cdf = term
            while cdf < uu and n < 50:
                n += 1
                term *= lam / n
                cdf += term
            for r in range(n):
                key = (5, int(b), int(m), int(wk), r)
                self._visit(key, int(b) * R + int(m), wk - 1.0 + float(self._u(j, key)[0]), int(wk))
        for stream, r, at in self._before(j):
            self._visit((6, stream, r), stream, at, None)
        for row, ty, m, _target, first, excites in started:
            if excites and s - first <= RECENT:
                key = (8, int(row))
                self._descend(key, BLOCK[ty] * R + m, first - float(self._u(j, key)[0]))
        self._threads(messages, started)
        return sorted(self._out, key=lambda q: q.onset)

    def _parts(self, block: int, m: int, week: int) -> list:
        """The typed parts of an immigrant of a week: [(type, fixed target or None, share)], the closures at a strait
        times the strait's tilt of that week (``targets.TypeRules.week_parts`` with the scenario's regime)."""
        t, (_j, _s, z, _zp, _partners, tilt) = self.t, self._state
        return [(code, tgt, w * tilt[week, t.chk_ord[tgt.index]] if code == CLOSURE else w)
                for code, tgt, w in t.rules.parts(block, m, int(z[m]))]  # fmt: skip

    def _descend(self, key: tuple, stream: int, at: float) -> None:
        j = self._state[0]
        for child, r, delay in self._children(j, key, stream):
            self._visit((*key, child, r), child, at + delay, None)

    def _visit(self, key: tuple, stream: int, at: float, week: int | None) -> None:
        """One event of the draw at the instant ``at`` (an immigrant of ``week``, or a child): typed, kept by the
        rules of the module docstring, and its children after it."""
        t = self.t
        j, s, z, _zp, _partners, _tilt = self._state
        if at >= t.T:
            return
        block, m = divmod(stream, t.R)
        u = self._u(j, key)
        if week is not None:
            parts = self._parts(block, m, week)
        else:
            parts = [(code, None, w) for code, w in t.rules.type_weights(block, m, int(z[m]))] or [(NOOP, None, 1.0)]
        code, target, _w = parts[_pick(float(u[1]), [p[2] for p in parts])]
        if code != NOOP:
            if at <= s:  # an agent would have seen it: the episode's own events stand in its place
                return
            events = self._event(u, code, target, m, at)
            if events is None:  # its announcement would be out by now
                self.count["announced"] += 1
                return
            self.count["immigrant" if week is not None else "child"] += 1
            self._out += events
        self._descend(key, stream, at)

    def _marks(self, u: np.ndarray, code: int, m: int, at: float, target_kind: int, target: int, commodity: int = -1,
               counterpart: int = -1) -> list:  # fmt: skip
        """The event of a type at a target with its marks drawn from the generator's laws (a conflict of a strait's
        dyad with the closure of that strait it brings)."""
        t = self.t
        zp = self._state[3]
        name = t.names[code]
        law = t.duration[name]
        duration = float(law.ppf(u[3 : 3 + law.uniforms] if law.uniforms > 1 else float(u[3])))
        severity = float(t.severity[name].ppf(float(u[5])))
        rate, T0, tau = math.nan, math.nan, math.nan
        if code == TARIFF:
            rate = float(t.p.laws.tariff_rate_by_tension[int(zp[m])])
        elif code == PIRACY:
            rate = severity
        if code == CONFLICT and m in t.restoration:
            T0 = float(t.p.laws.conflict_dead_time.ppf(float(u[6])))
            tau = float(t.p.laws.conflict_tau_rho.ppf(float(u[7])))
        Event = t.M.Event
        out = [
            Event(NEW_ROW, code, m, counterpart, target_kind, target, commodity, at, duration, severity, rate, T0, tau)
        ]
        strait = t.straits.get(frozenset((m, counterpart))) if code == CONFLICT and counterpart >= 0 else None
        if strait is not None:
            out.append(Event(NEW_ROW, CLOSURE, m, -1, K_CHOKEPOINT, strait, -1, at,
                             max(duration, float(t.p.laws.strait_closure_weeks)),
                             float(t.p.laws.strait_closure_severity), math.nan, math.nan, math.nan))  # fmt: skip
        return out

    def _event(self, u: np.ndarray, code: int, target, m: int, at: float) -> list | None:
        """A typed event of the cluster process at ``at`` > s with its target and marks; None if its announcement
        would be out by ``s`` (an announced event is a thread, not a draw)."""
        t = self.t
        _j, s, _z, _zp, partners, _tilt = self._state
        ahead = at - s
        if code == TARIFF:
            channel = "tariff_informal" if u[8] < t.informal else "tariff_formal"
            if t.lead[channel].ppf(float(u[9])) >= ahead:
                return None
        elif code == SANCTION:
            if max(t.lead["ties_threat"].ppf(float(u[9])), t.lead["sanction_legal"].ppf(float(u[10]))) >= ahead:
                return None
        elif code == CLOSURE and t.lead["mid_threat"].ppf(float(u[9])) >= ahead:
            return None
        counterpart = -1
        if code == CONFLICT:
            cands = t.rules.candidates(CONFLICT, m, partners[m])
            if cands[0][0].counterpart >= 0:
                counterpart = cands[_pick(float(u[2]), [w for _c, w in cands])][0].counterpart
            if frozenset((m, counterpart)) in t.straits and t.lead["mid_threat"].ppf(float(u[10])) >= ahead:
                return None  # the closure of the dyad's strait is announced like any other
            return self._marks(u, code, m, at, K_REGION, m, counterpart=counterpart)
        if target is None:
            cands = t.rules.candidates(code, m)
            target = cands[_pick(float(u[2]), [w for _c, w in cands])][0]
        if code == TARIFF:
            return self._marks(u, code, m, at, K_REGION, m, int(target.commodity), int(target.counterpart))
        return self._marks(u, code, m, at, int(target.kind), int(target.index), int(target.commodity))

    # ----- announcement threads ---------------------------------------------------------------------------------------
    def alive(self, messages: dict, started: list, s: float) -> dict:
        """The threads alive at ``s`` as the messages shown by then tell them: {thread: its type, region, target,
        commodity, the instant it was first shown, the stated week or None, its decoy-bearing channel}.

        A thread is over when it is withdrawn, when its stated week has come, or, with no date stated, when an event
        of its type was seen on its target: at or up to ``LATE`` weeks before the thread was first shown (an
        announcement that came with or after its event), else after it (the oldest such thread).
        """
        threads: dict = {}
        cols = [messages[name] for name in ("msg_id", "channel", "kind", "region", "target", "k", "announced_week",
                                             "stated_effective_week")]  # fmt: skip
        for tid, channel, kind, region, target, k, announced, stated in zip(*cols):
            fresh = {
                "type": CHANNEL_TYPE[channel],
                "region": int(region),
                "target": int(target),
                "dated": None,
                "k": -1 if k is None else int(k),
                "first": int(announced) - 1,
                "over": False,
                "channel": None,
            }
            th = threads.setdefault(int(tid), fresh)
            th["first"] = min(th["first"], int(announced) - 1)
            if kind == "withdrawal":
                th["over"] = True
            elif stated is not None:
                th["dated"] = int(stated)
            if channel in self.t.real:
                th["channel"] = channel
        for th in threads.values():
            if th["dated"] is not None and th["dated"] <= s + 1:
                th["over"] = True
        open_ = {tid: th for tid, th in threads.items() if not th["over"] and th["dated"] is None}
        for _row, ty, _m, target, first, _excites in sorted(started, key=lambda e: (e[4], e[0])):
            same = [tid for tid, th in open_.items() if th["type"] == ty and th["target"] == target and not th["over"]]
            late = [tid for tid in same if first <= open_[tid]["first"] <= first + LATE]
            early = [tid for tid in same if open_[tid]["first"] < first]
            pick = late or early
            if pick:
                open_[min(pick, key=lambda tid: (open_[tid]["first"], tid))]["over"] = True
        return {tid: th for tid, th in threads.items() if not th["over"]}

    def _threads(self, messages: dict, started: list) -> None:
        """The events of the threads alive at ``s`` that are real in this scenario, and their descendants."""
        t = self.t
        j, s, _z, _zp, _partners, _tilt = self._state
        kept = self.thread[j]
        for tid, th in sorted(self.alive(messages, started, s).items()):
            state = kept.get(tid)
            n = state[0] if state else 0
            if th["dated"] is not None:  # the effect falls in the week before the stated one
                u = self._u(j, (7, tid, n))
                at = th["dated"] - 2.0 + float(u[11])
            elif state is not None and state[2] > s and state[3] is None:
                u, at = self._u(j, (7, tid, n)), state[2]
            else:  # first seen, or the draw's date has passed with the thread still alive: drawn (again)
                n += state is not None
                self.count["redrawn"] += state is not None
                u = self._u(j, (7, tid, n))
                law = t.lead[UNDATED[th["type"]]] if th["type"] in UNDATED else None
                shown = th["first"] - float(u[12])  # the instant of the announcement, within its week
                lo, hi = (law.cdf(max(s - shown, 0.0)), law.cdf(t.T - shown)) if law is not None else (0.0, 0.0)
                if hi - lo > 1e-9:
                    at = shown + float(law.ppf(min(max(lo + float(u[11]) * (hi - lo), 1e-12), 1.0 - 1e-12)))
                else:
                    at = s + float(u[11]) * (t.T - s)
                at = min(max(at, s + 1e-6), t.T - 1e-9)
            channel = th["channel"] or UNDATED.get(th["type"], "tariff_formal")
            real = bool(u[13] < t.real[channel])
            kept[tid] = [n, real, at, th["dated"]]
            if not real or at <= s or at >= t.T:
                continue
            code, m = th["type"], th["region"]
            if code == TARIFF:  # the message names the importer and the good, not the exporter
                cands = [(c, w) for c, w in t.rules.candidates(TARIFF, m) if c.commodity == th["k"]]
                if not cands:
                    continue
                c = cands[_pick(float(u[2]), [w for _c, w in cands])][0]
                events = self._marks(u, code, m, at, K_REGION, m, th["k"], int(c.counterpart))
            else:
                events = self._marks(u, code, m, at, K_EDGE if code == SANCTION else K_CHOKEPOINT, th["target"])
            self.count["thread"] += 1
            self._out += events
            self._descend((7, tid, n), BLOCK[code] * t.R + m, at)
