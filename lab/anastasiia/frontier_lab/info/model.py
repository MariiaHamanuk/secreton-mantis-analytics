"""The generator as a filtering model: what each element's chance of a new cut is, given what is known at an instant.

Stage 2 of the study (root 1001 draws of sample.py). Everything here is formulas of the generator's own code
(shockbench_flow.disruption: hawkes, regime, targets, announce); nothing is fitted except the stationary means.
"""

import math

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr
from shockbench_flow import marks as M
from shockbench_flow.disruption import hawkes, regime, targets
from shockbench_flow.hosting.tasks import task_generator
from shockbench_flow.information.theta import Standard
from shockbench_flow.omega import codes


TARIFF, SANCTION, OUTAGE, CLOSURE, CONFLICT, PIRACY, ENERGY, WEATHER, STRIKE = range(9)
NB = 13  # weekly bins of the horizon
HS = (1, 4, 13)
CH = {n: i for i, n in enumerate(codes.CHANNELS)}


class Gen:
    def __init__(self, task):
        self.task = task
        inst, p = task_generator(task)
        self.inst, self.p = inst, p
        self.T, self.R = inst.T, len(inst.regions)
        R = self.R
        self.G = hawkes.branching_matrix(inst, p)
        self.beta = 1.0 / p.hawkes.beta_inv
        A = self.beta * (self.G - np.eye(2 * R))
        Ei = [expm(A * i) for i in range(NB + 1)]
        Ainv = np.linalg.inv(A)
        self.K = np.stack([Ainv @ (Ei[i + 1] - Ei[i]) for i in range(NB)])  # (NB, 2R, 2R)
        self.IG = np.linalg.solve(np.eye(2 * R) - self.G, self.G)  # psi_inf = IG @ mu
        self.base = hawkes.baseline_vector(inst, p)
        self.multM = np.array(p.hawkes.militarised_by_conflict)
        self.multP = np.array(p.hawkes.policy_by_tension)
        self.Pi = regime.weekly_conflict_matrix(p.regime.P_yr)
        self.rules = targets.TypeRules(inst, p, None)
        self.geo = M._Geo(inst)
        self.adj = targets.adjacency_dict(inst, p)
        th = Standard()
        self.a = th.a_region
        self.b = math.sqrt(1 - self.a**2)
        self.lat_r, self.lat_c, self.lat_d = p.latent_region, p.latent_chokepoint, p.latent_dyad
        self.dyads = [(inst.region_index[x], inst.region_index[y]) for x, y in p.regime.dyads]
        self.C = len(inst.chokepoints)
        self.chk_ord = inst.chokepoint_ordinal
        self.chk_region = {}  # chokepoint ordinal -> adjacent regions
        for c in inst.chokepoints:
            self.chk_region[self.chk_ord[c]] = tuple(inst.chokepoint_adjacency.get(c, ()))
        self.reg_chk = {
            m: [self.chk_ord[c] for c in inst.chokepoints if m in inst.chokepoint_adjacency.get(c, ())]
            for m in range(R)
        }
        self.reg_grid = {m: [inst.grid_ordinal[g] for g in inst.grids if inst.nodes[g].region == m] for m in range(R)}
        self.NG = len(inst.grids)
        # M-block type weights by region and conflict state: closure, conflict, piracy, energy; total feasible flag
        self.wM = np.zeros((R, 3, 4))
        self.noopM = np.zeros((R, 3), dtype=bool)
        order = {CLOSURE: 0, CONFLICT: 1, PIRACY: 2, ENERGY: 3}
        for m in range(R):
            for z in range(3):
                tw = self.rules.type_weights(1, m, z)
                if not tw:
                    self.noopM[m, z] = True
                for code, w in tw:
                    self.wM[m, z, order[code]] = w
        # P-block: the sanction share (V2: raw shares, the infeasible part is a no-op)
        self.cand = {m: [t.index for t, _ in self.rules.candidates(SANCTION, m)] for m in range(R)}
        self.wS = np.zeros(R)
        for m in range(R):
            for code, w in self.rules.type_weights(0, m, 0):
                if code == SANCTION:
                    self.wS[m] = w
        # prohibition units: every candidate edge of a sanction
        self.units = sorted({j for js in self.cand.values() for j in js})
        self.unit_of = {j: i for i, j in enumerate(self.units)}
        self.unit_region = np.array([self.geo.region[inst.edges[j].tail] for j in self.units])
        self.ncand = np.array([max(len(self.cand[m]), 1) for m in range(R)], dtype=float)
        E = len(inst.edges)
        self.E = E
        self.ff = {j: self._ff(j) for j in self.units}
        self.FFR = np.zeros((E, R))
        for m, js in self.cand.items():
            for j in js:
                for e in self.ff[j]:
                    self.FFR[e, m] += 1.0 / len(js)
        self.CR = np.zeros((E, R))
        for m in range(R):
            has_partner = bool(self.adj.get(m)) or any(m in d for d in self.dyads)
            if has_partner:
                for e in self.geo.between(m, None):
                    self.CR[e, m] = 1.0
        self.conf_edges = {m: self.geo.between(m, None) for m in range(R)}
        fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
        self.fuel_edge = np.array([bool(fuels.intersection(e.K)) for e in inst.edges])
        # derived strait closures: {frozenset(regions): chokepoint ordinal}
        self.straits = {k: self.chk_ord[c] for k, c in targets.strait_chokepoints(inst, p).items()}
        self.kstar = None
        info = p.information
        self.lead = {codes.CHANNELS.index(ch): law for ch, law in info.lead}
        self.phi = dict(info.phi_bar)
        grid = np.arange(0.0, 420.0, 0.25)
        self.lead_grid = grid
        self.leadF = {c: np.array([law.cdf(x) for x in grid]) for c, law in self.lead.items()}
        self.c_r, self.c_c = regime.normaliser(self.lat_r), regime.normaliser(self.lat_c)
        # P(z_c | S) of a region or dyad unit: the quasi-static law of the tilted chain at X, X | S ~ N(aS, b^2)
        xs = np.linspace(-6, 6, 241)
        st = regime.conflict_stationary(xs, 1.0, self.Pi, self.lat_r)  # (241, 3)
        self.S_grid = np.linspace(-5, 5, 201)
        w = np.exp(-0.5 * ((xs[None, :] - self.a * self.S_grid[:, None]) / self.b) ** 2)
        w /= w.sum(1, keepdims=True)
        self.pz_S = w @ st  # (201, 3)
        self.st_x, self.xs = st, xs
        tq = regime.tension_entry(p.regime)
        self.p_tension = tq / (tq + 1.0 / p.regime.tension_spell)  # by z_c

    def _ff(self, j):
        inst, geo = self.inst, self.geo
        edge = inst.edges[j]
        a, b = geo.region[edge.tail], geo.region[edge.head]
        if a == b:
            return []
        s = set(edge.K)
        return [e for e in geo.between(a, b) if inst.edges[e].K and not s.intersection(inst.edges[e].K)]

    def F(self, ch, x):
        """P(lead <= x weeks) of a channel, x >= 0 (vectorised, from the law's own cdf on a quarter-week grid)."""
        x = np.clip(np.asarray(x, dtype=float), 0.0, self.lead_grid[-1])
        return np.interp(x, self.lead_grid, self.leadF[ch])

    def tilt(self, mu, sd, lat, c):
        """E[g(X)] / c for X ~ N(mu, sd^2), g(x) = exp(k min(x, cap))."""
        k, cap = lat.k, lat.x_cap
        mu = np.asarray(mu, dtype=float)
        if np.all(np.asarray(sd) <= 1e-9):
            return np.exp(k * np.minimum(mu, cap)) / c
        sd = np.maximum(np.asarray(sd, dtype=float), 1e-9)
        return (
            np.exp(k * mu + 0.5 * k * k * sd * sd) * ndtr((cap - mu - k * sd * sd) / sd)
            + np.exp(k * cap) * (1 - ndtr((cap - mu) / sd))
        ) / c

    def pz_from_S(self, S):
        """(len(S), 3) the quasi-static law of a unit's chain given its score."""
        S = np.clip(np.asarray(S, dtype=float), self.S_grid[0], self.S_grid[-1])
        return np.stack([np.interp(S, self.S_grid, self.pz_S[:, k]) for k in range(3)], axis=-1)
