"""Stage 2: the chance of a new cut per element and instant under seven states of knowledge, as histograms of
the scores.

    python predict.py <task> <the draws of sample.py, a glob> <the file of the histograms>
"""

import glob
import math
import pickle
import sys
import time

import numpy as np
from model import CH, CLOSURE, CONFLICT, ENERGY, HS, NB, SANCTION, STRIKE, WEATHER, Gen


VARS = ("clim", "warn", "events", "threads", "filter", "latent", "oracle_id")
ITEMS = ("strait", "grid", "proh", "edge", "edge_fuel")
NBINS = 900
TRUNC = True


def hbin(p):
    return np.clip(((np.log10(np.maximum(p, 1e-10)) + 10.0) / 10.0 * (NBINS - 1)).astype(np.int64), 0, NBINS - 1)


class Acc:
    def __init__(self):
        self.h = {}

    def add(self, key, p, y):
        p, y = np.asarray(p, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
        assert np.isfinite(p).all() and p.min() >= -1e-12 and p.max() <= 1 + 1e-12, key
        idx = hbin(p)
        if key not in self.h:
            self.h[key] = np.zeros((3, NBINS))
        a = self.h[key]
        a[0] += np.bincount(idx, minlength=NBINS)
        a[1] += np.bincount(idx, weights=y, minlength=NBINS)
        a[2] += np.bincount(idx, weights=p, minlength=NBINS)


def rates(g, pz, EmP, tilt0):
    """Stream rates of immigrants under a law ``pz`` (R, 3) of the regions' conflict state."""
    R = g.R
    bP, bM = g.base[:R], g.base[R:]
    mz = pz * g.multM[None, :]  # (R, 3)
    wcl = bM * (mz * g.wM[:, :, 0]).sum(1)
    wcf = bM * (mz * g.wM[:, :, 1]).sum(1)
    wpi = bM * (mz * g.wM[:, :, 2]).sum(1)
    wen = bM * (mz * g.wM[:, :, 3]).sum(1)
    noop = bM * (mz * g.noopM).sum(1)
    mt = np.array([np.mean(tilt0[g.reg_chk[m]]) if g.reg_chk[m] else 0.0 for m in range(R)])
    muM = wcf + wpi + wen + noop + wcl * mt
    muP = bP * EmP
    ch = dict(cl=(pz * g.wM[:, :, 0]).sum(1), cf=(pz * g.wM[:, :, 1]).sum(1), en=(pz * g.wM[:, :, 3]).sum(1))
    return dict(muP=muP, muM=muM, wcl=wcl, wcf=wcf, wen=wen, ch=ch, mu=np.concatenate([muP, muM]))


def lam(g, r, psi, tilt, qd):
    """Expected counts per weekly bin: the cascade from ``psi`` and the immigrants of ``r``; per stream and item."""
    R = g.R
    psi_inf = g.IG @ r["mu"]
    off = np.einsum("ijk,k->ij", g.K, psi - psi_inf) + psi_inf[None, :]  # (NB, 2R)
    offP, offM = off[:, :R], off[:, R:]
    S = g.wS[None, :] * (r["muP"][None, :] + offP)  # sanctions per region and bin
    Lf = r["wcf"][None, :] + r["ch"]["cf"][None, :] * offM  # conflicts per region and bin
    Lc = np.zeros((NB, g.C))
    Lg = np.zeros((NB, g.NG))
    for c, ms in g.chk_region.items():
        for m in ms:
            Lc[:, c] += (r["wcl"][m] * tilt[:, c] + r["ch"]["cl"][m] * offM[:, m]) / len(g.reg_chk[m])
    for m in range(R):
        for gi in g.reg_grid[m]:
            Lg[:, gi] = (r["wen"][m] + r["ch"]["en"][m] * offM[:, m]) / len(g.reg_grid[m])
    Ld = np.zeros((NB, g.C))  # the strait a conflict of its dyad shuts
    for pair, c in g.straits.items():
        for m in pair:
            (other,) = pair - {m}
            nadj = len(g.adj.get(m, {}))
            pa = (1.0 / nadj) if other in g.adj.get(m, {}) else 0.0
            Ld[:, c] += Lf[:, m] * (qd + (1 - qd) * pa)
    return dict(S=S, Lf=Lf, Lc=Lc, Lg=Lg, Ld=Ld, off=off, psi_inf=psi_inf, r=r, tilt=tilt)


def main(task, pattern, out):
    g = Gen(task)
    R, C, T, E = g.R, g.C, g.T, g.E
    eps = []
    for path in sorted(glob.glob(pattern)):
        eps += pickle.load(open(path, "rb"))
    D = len(g.dyads)
    beta = g.beta
    decay, fresh = math.exp(-beta), 1.0 - math.exp(-beta)
    nU = len(g.units)
    rho_c = g.lat_c.rho

    def ci(c):  # the row of a strait's signal unit
        return R + D + c

    # ---- pass 0: stationary means under the truth -------------------------------------------------------------
    zc_all = np.concatenate([e["z_c"][:, 2:] for e in eps], axis=1)
    pz0 = np.stack([(zc_all == k).mean(1) for k in range(3)], axis=1)  # (R, 3)
    zd_all = np.concatenate([e["z_dyad"][:, 2:] for e in eps], axis=1)
    pd0 = float((zd_all == 2).mean()) if D else 0.0
    zp_all = np.concatenate([e["z_p"][:, 2:] for e in eps], axis=1)
    EmP0 = 1.0 + (g.multP[1] - 1.0) * zp_all.mean(1)
    r0 = rates(g, pz0, EmP0, np.ones(C))
    mus = []
    for e in eps[:: max(1, len(eps) // 100)]:
        for s in range(0, T, 4):
            zc = e["z_c"][:, s + 2]
            pz = np.eye(3)[zc]
            Xc = e["X"][R + D :, s + 2].astype(float)
            mus.append(rates(g, pz, g.multP[e["z_p"][:, s + 2]], g.tilt(Xc, 0.0, g.lat_c, g.c_c))["mu"])
    mu_bar = np.mean(mus, axis=0)
    psi_bar = g.IG @ mu_bar
    print(
        "stationary: P(z_c) by region\n",
        np.round(pz0, 4),
        "\nP(dyad war)",
        round(pd0, 4),
        "E mult P",
        np.round(EmP0, 3),
    )
    print("mu_bar P", np.round(mu_bar[:R], 5), "\nmu_bar M", np.round(mu_bar[R:], 5))
    print(
        "stationary rate per week P",
        np.round(mu_bar[:R] + psi_bar[:R], 5),
        "\nM",
        np.round(mu_bar[R:] + psi_bar[R:], 5),
    )
    thS = np.mean(
        [
            g.F(CH["ties_threat"], np.arange(NB) + u) * g.F(CH["sanction_legal"], np.arange(NB) + u)
            for u in (0.125, 0.375, 0.625, 0.875)
        ],
        axis=0,
    )
    thC = np.mean([g.F(CH["mid_threat"], np.arange(NB) + u) for u in (0.125, 0.375, 0.625, 0.875)], axis=0)
    print(
        "share of events of a bin with no message before the instant: sanction",
        np.round(thS, 3),
        "closure",
        np.round(thC, 3),
    )
    piS, piC = 1.0 - g.phi["ties_threat"], 1.0 - g.phi["mid_threat"]
    risk = dict(
        strait=np.array([bool(g.chk_region[c]) for c in range(C)]),
        grid=np.array([g.base[R + g.inst.nodes[gn].region] > 0 for gn in g.inst.grids]),
        proh=np.ones(nU, dtype=bool),
        edge=(g.FFR.sum(1) + g.CR.sum(1)) > 0,
    )
    risk["edge_fuel"] = risk["edge"] & g.fuel_edge
    acc, thr_rows, stats = Acc(), [], []
    PiPow = [np.linalg.matrix_power(g.Pi, k) for k in range(0, 400)]
    t0 = time.process_time()
    for n_ep, e in enumerate(eps):
        X, W = e["X"].astype(float), e["W"].astype(float)
        cl, st, sh, shr = e["cluster"], e["stored"], e["shadows"], e["shadow_raw"]
        kstar = float(e["shadow_scale"])
        # ---- labels by absolute weekly bin b: onset in (b, b + 1] ------------------------------------------------
        Y = dict(strait=np.zeros((T, C)), grid=np.zeros((T, g.NG)), proh=np.zeros((T, nU)), edge=np.zeros((T, E)))
        obsK = np.zeros((T + 1, 2 * R))  # what the events first seen at each instant add to the excitation
        conf_seen = {}  # region -> instant a conflict of the episode was first seen there
        n_new = np.zeros(9)
        ew = dict(new=0.0, new_fuel=0.0, carried=0.0, carried_fuel=0.0)
        for row in cl:
            blk, m, on, ty = int(row[0]), int(row[1]), row[2], int(row[3])
            if on <= 0 or ty < 0:
                continue
            b = int(math.ceil(on)) - 1
            if b >= T:
                continue
            obsK[b + 1] += g.G[:, blk * R + m]
            n_new[ty] += 1
            tgt, cp = int(row[8]), int(row[10])
            if ty == CLOSURE:
                Y["strait"][b, g.chk_ord[tgt]] = 1
            elif ty == ENERGY:
                Y["grid"][b, g.inst.grid_ordinal[tgt]] = 1
            elif ty == SANCTION:
                Y["proh"][b, g.unit_of[tgt]] = 1
                Y["edge"][b, g.ff[tgt]] = 1
            elif ty == CONFLICT:
                conf_seen.setdefault(m, b + 1)
                if cp >= 0:
                    Y["edge"][b, g.conf_edges[m]] = 1
        carried_conf = set()
        for row in st:
            ty, m, cp, tgt, on, dur, der = (
                int(row[0]),
                int(row[2]),
                int(row[3]),
                int(row[5]),
                row[7],
                row[8],
                int(row[11]),
            )
            if der and 0 < on < T:
                Y["strait"][int(math.ceil(on)) - 1, g.chk_ord[tgt]] = 1
            if ty == CONFLICT and on <= 0:
                carried_conf.add(m)
            # edge-weeks of capacity cuts inside the episode, new and carried
            if ty == SANCTION:
                es, end = g.ff[tgt] if tgt in g.ff else g._ff(tgt), on + dur
            elif ty == CONFLICT and cp >= 0:
                es, end = g.conf_edges[m], on + max(dur, 52.0) + 52.0
            else:
                continue
            wk = max(0.0, min(end, T) - max(on, 0.0))
            key = "new" if on > 0 else "carried"
            ew[key] += wk * len(es)
            ew[key + "_fuel"] += wk * int(g.fuel_edge[es].sum()) if len(es) else 0.0
        pois_new = [int(((st[:, 0] == k) & (st[:, 7] > 0)).sum()) for k in (WEATHER, STRIKE)]
        stats.append(
            np.concatenate(
                [
                    n_new,
                    pois_new,
                    [ew["new"], ew["new_fuel"], ew["carried"], ew["carried_fuel"]],
                    [(st[:, 7] <= 0).sum(), (st[:, 7] > 0).sum()],
                ]
            )
        )
        cumY = {k: np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)]) for k, v in Y.items()}
        # ---- threads ---------------------------------------------------------------------------------------------
        threads = []
        for row, off0, decoy in [(r_, 14, 0) for r_ in st] + [(r_, 9, 1) for r_ in sh]:
            ty = int(row[0])
            on = row[7] if not decoy else row[6]
            tgt = int(row[5]) if not decoy else int(row[4])
            m = int(row[2])
            if ty not in (SANCTION, CLOSURE) or not 0 < on < T:
                continue
            if ty == SANCTION:
                l_und, l_dat, chn, pi = (
                    row[off0 + CH["ties_threat"]],
                    row[off0 + CH["sanction_legal"]],
                    CH["ties_threat"],
                    piS,
                )
                if tgt not in g.unit_of:
                    continue
                elem = g.unit_of[tgt]
            else:
                l_und, l_dat, chn, pi = row[off0 + CH["mid_threat"]], np.nan, CH["mid_threat"], piC
                elem = g.chk_ord[tgt]
            a_und = on - l_und if (l_und == l_und and l_und > 0) else math.inf
            a_dat = on - l_dat if (l_dat == l_dat and l_dat > 0) else math.inf
            if min(a_und, a_dat) >= on:
                continue
            threads.append((ty, elem, m, on, decoy, a_und, a_dat, chn, pi))
        # ---- instants ----------------------------------------------------------------------------------------------
        P = {
            (it, v, h): np.zeros((T, Y["edge" if it == "edge_fuel" else it].shape[1]))
            for it in ITEMS[:4]
            for v in VARS
            for h in HS
        }
        psi_obs = psi_bar.copy()
        for s in range(T):
            if s > 0:
                psi_obs = psi_obs * decay  # the part of before the episode fades; what was seen is added below
            psi_obs = psi_obs + fresh * obsK[s] if s > 0 else psi_obs
            S_now = g.a * X[:, s + 1] + g.b * W[:, s + 1]
            # the state of knowledge of each base
            tilt1 = np.ones((NB, C))
            lag = np.arange(1, NB + 1)
            for c in range(C):
                tilt1[:, c] = g.tilt(
                    rho_c**lag * g.a * S_now[ci(c)], np.sqrt(1 - rho_c ** (2 * lag) * g.a**2), g.lat_c, g.c_c
                )
            pzS = g.pz_from_S(S_now[:R])
            pdS = float(g.pz_from_S(S_now[R : R + D])[0, 2]) if D else 0.0
            for d, (x, y) in enumerate(g.dyads):  # a war of the dyad is a war of both regions
                for m in (x, y):
                    w = 1 - (1 - pzS[m, 2]) * (1 - pdS)
                    pzS[m] = np.array(
                        [
                            pzS[m, 0] * (1 - w) / max(1 - pzS[m, 2], 1e-12),
                            pzS[m, 1] * (1 - w) / max(1 - pzS[m, 2], 1e-12),
                            w,
                        ]
                    )
            pzE = pzS.copy()
            for m, first in conf_seen.items():
                if first <= s:
                    pzE[m] = PiPow[min(s - first, 399)][2]
            for m in carried_conf:
                if m not in conf_seen or conf_seen[m] > s:
                    pzE[m] = 0.5 * pzE[m] + 0.5 * PiPow[min(80 + s, 399)][2]

            def EmP(pz):  # the mean multiplier of the policy block under a law of the conflict state
                return 1.0 + (g.multP[1] - 1.0) * (pz @ g.p_tension)

            zc, zp = e["z_c"][:, s + 2], e["z_p"][:, s + 2]
            Xc_true = X[R + D :, s + 2]
            tiltB = np.ones((NB, C))
            for c in range(C):
                tiltB[:, c] = g.tilt(
                    rho_c ** np.arange(NB) * Xc_true[c], np.sqrt(1 - rho_c ** (2 * np.arange(NB))), g.lat_c, g.c_c
                )
            psi_true = (
                g.G[:, (cl[:, 0] * R + cl[:, 1]).astype(int)]
                * (beta * np.exp(-beta * np.clip(s - cl[:, 2], 0, None)) * (cl[:, 2] <= s))[None, :]
            ).sum(1)
            psiY = (
                g.G[:, (shr[:, 0] * R + shr[:, 1]).astype(int)]
                * (beta * np.exp(-beta * np.clip(s - shr[:, 2], 0, None)) * (shr[:, 2] <= s))[None, :]
            ).sum(1)
            dwar = float(e["z_dyad"][0, s + 2] == 2) if D else 0.0
            qdS = min(1.0, pdS / max(pzS[g.dyads[0][0], 2], 1e-9)) if D else 0.0
            base = {
                "clim": lam(
                    g,
                    r0,
                    g.IG @ r0["mu"],
                    np.ones((NB, C)),
                    min(1.0, pd0 / max(pz0[g.dyads[0][0], 2], 1e-9)) if D else 0.0,
                ),
                "warn": None,
                "events": None,
                "latent": None,
            }
            rW = rates(g, pzS, EmP(pzS), tilt1[0])
            base["warn"] = lam(g, rW, g.IG @ rW["mu"], tilt1, qdS)
            rE = rates(g, pzE, EmP(pzE), tilt1[0])
            base["events"] = lam(g, rE, psi_obs, tilt1, qdS)
            rB = rates(g, np.eye(3)[zc], g.multP[zp], tiltB[0])
            base["latent"] = lam(g, rB, psi_true, tiltB, dwar)
            psiY_inf = kstar * base["latent"]["psi_inf"]
            offY = np.einsum("ijk,k->ij", g.K, psiY - psiY_inf) + psiY_inf[None, :]
            # thread chances per bin, per variant
            surv = {
                v: {it: np.ones((len(HS), n)) for it, n in (("strait", C), ("proh", nU), ("edge", E))}
                for v in ("threads", "filter", "latent", "oracle_id")
            }
            for ty, elem, m, on, decoy, a_und, a_dat, chn, pi in threads:
                if not (min(a_und, a_dat) <= s < on):
                    continue
                if a_dat <= s:
                    f = np.zeros(NB + 1)
                    f[min(int(math.ceil(on)) - 1 - s, NB)] = 1.0
                    dated = 1
                else:
                    age = max(s - math.ceil(a_und), 0) + 0.5
                    Fa = g.F(chn, age + np.arange(NB + 1))
                    f = np.append(np.diff(Fa), 1.0 - Fa[-1])
                    if TRUNC:  # a thread is shown only if its effect falls inside the episode: it is over by T
                        left = T - s
                        f[min(left, NB) : NB] = 0.0
                        f[NB] = max(float(g.F(chn, age + left)) - Fa[-1], 0.0) if left > NB else 0.0
                    dated = 0
                tot = f.sum()
                if tot <= 1e-12:
                    continue
                rr_by = {}
                for v, bname in (("filter", "events"), ("latent", "latent")):
                    B_ = base[bname]
                    if ty == SANCTION:
                        num = B_["r"]["muP"][m] + B_["off"][:, m]
                        den = B_["r"]["muP"][m] + B_["psi_inf"][m]
                        numY = kstar * B_["r"]["muP"][m] + offY[:, m]
                    else:
                        num = B_["r"]["wcl"][m] * B_["tilt"][:, elem] + B_["r"]["ch"]["cl"][m] * B_["off"][:, R + m]
                        den = B_["r"]["wcl"][m] * B_["tilt"][0, elem] + B_["r"]["ch"]["cl"][m] * B_["psi_inf"][R + m]
                        numY = kstar * B_["r"]["wcl"][m] * B_["tilt"][:, elem] + B_["r"]["ch"]["cl"][m] * offY[:, R + m]
                    rhoY = np.ones(NB + 1)
                    if den <= 1e-12:  # a strait shut by a conflict of its dyad: no rate of its own to compare with
                        rhoR = np.ones(NB + 1)
                    else:
                        rhoR = np.append(num / den, 1.0)
                        rhoR[-1] = rhoR[-2]
                        if v == "latent":
                            rhoY = np.append(numY / (kstar * den), 1.0)
                            rhoY[-1] = rhoY[-2]
                    wR, wD = pi * f * rhoR, (1 - pi) * f * rhoY
                    z = wR.sum() + wD.sum()
                    rr_by[v] = wR[:NB] / z
                    if v == "latent":
                        wR1 = f * rhoR
                        rr_by["oracle_id"] = (wR1[:NB] / wR1.sum()) if not decoy else np.zeros(NB)
                        post_lat = wR.sum() / z
                    else:
                        post_fil = wR.sum() / z
                rr_by["threads"] = pi * f[:NB] / tot
                thr_rows.append((ty, decoy, dated, on - s, pi, post_fil, post_lat, s - min(a_und, a_dat)))
                for v, rr in rr_by.items():
                    cr = np.cumsum(rr)
                    q = np.array([min(cr[h - 1], 1.0) for h in HS])
                    if ty == SANCTION:
                        surv[v]["proh"][:, elem] *= 1 - q
                        ffe = g.ff[g.units[elem]]
                        if ffe:
                            surv[v]["edge"][:, ffe] *= (1 - q)[:, None]
                    else:
                        surv[v]["strait"][:, elem] *= 1 - q
            for v in VARS:
                bname = {
                    "clim": "clim",
                    "warn": "warn",
                    "events": "events",
                    "threads": "events",
                    "filter": "events",
                    "latent": "latent",
                    "oracle_id": "latent",
                }[v]
                B_ = base[bname]
                thin = v in surv
                tS = thS if thin else np.ones(NB)
                tC = thC if thin else np.ones(NB)
                Ls = np.cumsum(tC[:, None] * B_["Lc"] + B_["Ld"], axis=0)
                Lg = np.cumsum(B_["Lg"], axis=0)
                Sreg = tS[:, None] * B_["S"]
                Lp = np.cumsum(Sreg[:, g.unit_region] / g.ncand[g.unit_region][None, :], axis=0)
                Le = np.cumsum(Sreg @ g.FFR.T + B_["Lf"] @ g.CR.T, axis=0)
                for hi, h in enumerate(HS):
                    sv = surv[v] if thin else None
                    P[("strait", v, h)][s] = 1 - (sv["strait"][hi] if thin else 1.0) * np.exp(-Ls[h - 1])
                    P[("grid", v, h)][s] = 1 - np.exp(-Lg[h - 1])
                    P[("proh", v, h)][s] = 1 - (sv["proh"][hi] if thin else 1.0) * np.exp(-Lp[h - 1])
                    P[("edge", v, h)][s] = 1 - (sv["edge"][hi] if thin else 1.0) * np.exp(-Le[h - 1])
        for it in ITEMS:
            src = "edge" if it == "edge_fuel" else it
            for h in HS:
                valid = T - h + 1  # instants s with s + h <= T
                y = (cumY[src][h : h + valid] - cumY[src][:valid]) > 0
                for v in VARS:
                    acc.add((it, v, h), P[(src, v, h)][:valid][:, risk[it]], y[:, risk[it]])
                c0 = np.maximum(P[(src, "clim", h)][:valid][:, risk[it]], 1e-12)
                for v in (
                    "warn",
                    "events",
                    "filter",
                ):  # the score over the element's own base rate: what is added inside an element
                    acc.add(
                        (it, v + "_rel", h),
                        np.clip(P[(src, v, h)][:valid][:, risk[it]] / c0 * 1e-4, 0, 1),
                        y[:, risk[it]],
                    )
        if n_ep % 25 == 0:
            print("episode", n_ep, "cpu", round(time.process_time() - t0, 1), flush=True)
    pickle.dump(
        dict(
            task=task,
            n=len(eps),
            hist=acc.h,
            threads=np.array(thr_rows),
            stats=np.array(stats),
            pz0=pz0,
            mu_bar=mu_bar,
            psi_bar=psi_bar,
            thS=thS,
            thC=thC,
            risk={k: int(v.sum()) for k, v in risk.items()},
        ),
        open(out, "wb"),
    )
    print("done", task, len(eps), "episodes, cpu", round(time.process_time() - t0, 1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
