"""Persistence plan: the full-episode MILP of planner_H built from what a week-1 observer knows, played blind.

    uv run python lab/anastasiia/mpc_lab/planners/persist_plan.py run --episodes=8 --entropy=444 --tl=60 --n_jobs=2
    uv run python lab/anastasiia/mpc_lab/planners/persist_plan.py diag --n=3 --entropy=444   # one episode, by week/component

Question: planner_H's full-horizon plan (rules base,fuel,lots, lazy constraint generation, tl 60 s a round, gap 1e-3)
built with the episode's TRUE marks and played open loop beats the rules agents. How much of that survives when the
plan is built from the week-1 observation extended by persistence (what an honest agent knows at week 1)?

Per episode three played costs (the TRUE simulator, the same omega, policy seed and fallback as planner_H.play):
  hybrid   agents/anastasiia_hybrid_chiplp as played through gym by planner_LSF (cache outputs/planner_LSF/cache, the
           Env's stored wire actions replayed here through the core Env to get per-week records; J checked to the cent)
  true     planner_H.solve_lazy on the true marks, played blind (planner_H.play)
  persist  the same lazy MILP on PERSISTENCE marks, played blind in the true simulator

Persistence marks = ``dataclasses.replace(true_marks, **arrays, fab_hits=())`` where ``arrays`` is the package's own
week-1 point forecast of mpc_det, ``lp_common.persistence_arrays(inst, obs1, memory, H=T)``: every graph field
(u, c, open, kappa, supply, G-bar, y-bar, R, alpha-bar, R_osat, prohibitions, tariffs, war-risk class -> h^Q, c^wr) at
its instant-0 value from week 1's ``graph_now`` repeated for all T weeks; pending prohibitions announced in week 1
switched on from their effective week; sigma_scr 0 and no fab hits (future scrap is unknowable); demand = the shown
forecast for h <= 7 and d-bar m_sea(t) (the seasonal mean) beyond. This is slightly MORE honest than "week 1's true
week-averages repeated": the observer sees instants at t-1 = 0, not the week-1 average; ``run`` logs how far the two
differ (they are the same unless an event starts inside week 1).

The model is ``build_lp(inst, persistence marks, planning_rules=True)``, so it has the true model's columns (checked)
and ``lp_common.week1_action`` maps each week's x to a wire action; prohibitions are masked with the TRUE week's Z_t,
so every action is valid in the true world. The claimed J of the persistence plan is its cost under persistence marks.
"""

import dataclasses
import os
import pickle
import sys
import time

import fire
import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
os.environ.setdefault("SBF_CACHE_DIR", os.path.join(ROOT, "hub", "refcache"))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import planner_H as H  # noqa: E402

TASK = "small"
OUT = os.path.join(ROOT, "outputs", "persist_plan")
LOG = os.path.join(HERE, "persist_run.log")
RULES = ("base", "fuel", "lots")
BN = 1e11  # cents per bn USD


def configure(entropy):
    H.ENTROPY, H.TASK = int(entropy), TASK


def logline(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def cache_path(entropy, n, tag):
    d = os.path.join(OUT, f"cache_{TASK}_{entropy}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"ep{n}_{tag}.pkl")


# ----- the week-1 observation and the persistence marks ---------------------------------------------------------------
def week1_obs(n, w):
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = w
    env = Env(fallback=fb)
    obs, _info = env.reset(inst, "standard", omega, _policy_seed(H.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="planH")
    return obs


def persistence_marks(n, w):
    """(persistence WeeklyMarks, arrays dict) from week 1's observation (module docstring)."""
    from shockbench_flow.policies import lp_common as L

    inst, omega, marks, fb = w
    obs = week1_obs(n, w)
    mem = L.ObservedGraph.nominal(inst)
    mem.update(inst, obs)
    arrays = L.persistence_arrays(inst, obs, mem, inst.T)
    kw = {k: np.asarray(v) for k, v in arrays.items() if k in {f.name for f in dataclasses.fields(marks)}}
    pm = dataclasses.replace(marks, fab_hits=(), **kw)
    return pm, kw


def mark_drift(marks, pm):
    """How the persistence marks differ from the true ones: week-1 gap (instant 0 vs week-1 average) and later weeks."""
    out = {}
    for name in ("u", "supply", "G_bar", "y_bar", "R", "alpha_bar", "R_osat", "kappa", "c", "demand"):
        a, b = np.asarray(getattr(marks, name), float), np.asarray(getattr(pm, name), float)
        fin = np.isfinite(a) & np.isfinite(b)
        a, b = np.where(fin, a, 0.0), np.where(fin, b, 0.0)
        ax = tuple(range(1, a.ndim))
        tot = np.abs(a).sum(axis=ax) + 1e-12
        rel = np.abs(a - b).sum(axis=ax) / tot  # per week, relative L1
        out[name] = (float(rel[0]), float(rel[1:].mean()), float(rel.max()), int(np.argmax(rel)) + 1)
    pa, pb = np.asarray(marks.prohibited), np.asarray(pm.prohibited)
    out["prohibited"] = (int((pa[0] != pb[0]).sum()), float((pa != pb).sum(axis=(1, 2)).mean()), int((pa != pb).sum(axis=(1, 2)).max()), -1)
    ta, tb = np.asarray(marks.tariff), np.asarray(pm.tariff)
    out["tariff"] = (float(np.abs(ta[0] - tb[0]).sum()), float(np.abs(ta - tb).sum(axis=(1, 2)).mean()), float(np.abs(ta - tb).sum(axis=(1, 2)).max()), -1)
    out["sigma_scr_true_sum"] = float(np.asarray(marks.sigma_scr).sum())
    out["fab_hits_true"] = len(marks.fab_hits)
    return out


# ----- the lazy MILP on given marks (planner_H.solve_lazy with the world passed in) ----------------------------------
def solve_lazy_on(n, w, rules=RULES, tl=60, gap=1e-3, rounds=8, label=""):
    from shockbench_flow.dynamics.sim import initial_stock

    inst = w[0]
    i0 = initial_stock(inst)
    gw, ow, hist, pref = set(), set(), [], None
    t0 = time.time()
    out, sol = None, None
    for it in range(rounds + 1):
        out, sol = H.solve_full(n, rules, tl, gap, only_gw=gw, only_ow=ow, world_=w, phat_ref=pref)
        if sol is None:
            break
        m, x = sol
        g2, o2, kinds, pref = H.violations(inst, m, x, i0)
        hist.append((it, len(gw), len(ow), len(g2 - gw), len(o2 - ow), round(out["secs"]), out["gap"], out["J"]))
        print(f"  [{label}] ep {n} round {it}: viol {len(g2)}/{len(o2)} new {len(g2 - gw)}/{len(o2 - ow)} gap {out['gap']} J {out['J']} {out['secs']:.0f}s",
              file=sys.stderr, flush=True)
        if not (g2 - gw) and not (o2 - ow):
            break
        gw |= g2
        ow |= o2
    out = dict(out or {}, hist=hist, rounds=len(hist) - 1, secs_total=time.time() - t0)
    return out, sol


def play_on(n, w, m, x):
    """planner_H.play with the world passed in (true marks): open loop in the true Env."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow.policies import lp_common as L
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = w
    nc = m.meta["nc"]
    env = Env(fallback=fb)
    obs, _ = env.reset(inst, "standard", omega, _policy_seed(H.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="planH")
    done, t = False, 0
    while not done:
        xt = np.zeros_like(m.lb)
        xt[:nc] = x[t * nc : (t + 1) * nc]
        obs, _r, done, _tr, _inf = env.step(L.week1_action(inst, m, xt, obs, np.asarray(marks.prohibited[t])))
        t += 1
    return int(env.trajectory.J_cents), summarize(env.trajectory)


def replay_wire(n, w, actions):
    """Replay stored wire actions in the core Env (hybrid agent)."""
    from shockbench_flow.dynamics.env import Env
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed

    inst, omega, marks, fb = w
    env = Env(fallback=fb)
    env.reset(inst, "standard", omega, _policy_seed(H.ENTROPY, n, NO_ZIP_SHA256), marks=marks, policy_name="hybrid")
    for a in actions:
        _o, _r, done, _tr, _inf = env.step(a)
        if done:
            break
    return int(env.trajectory.J_cents), summarize(env.trajectory)


def summarize(traj):
    """Per-week cost components (USD), shed by grid, lots and served, and the salvage (USD)."""
    from shockbench_flow.dynamics.state import COST_COMPONENTS

    R = traj.records
    comp = np.array([[r.costs.as_dict()[c] for c in COST_COMPONENTS] for r in R], dtype=float)
    return dict(
        comp_names=tuple(COST_COMPONENTS),
        comp=comp,
        shed=np.array([r.shed for r in R], float),
        lots=np.array([r.lots_started for r in R], float),
        served=np.array([r.served for r in R], float),
        demand=np.array([r.demand for r in R], float),
        energy=np.array([r.energy for r in R], float),
        invalid=sum(len(r.invalid) for r in R),
        J=int(traj.J_cents),
        salvage_usd=float(comp.sum()) - traj.J_cents / 100.0,
    )


def hybrid(n, w):
    path = os.path.join(ROOT, "outputs", "planner_LSF", "cache" if H.ENTROPY == 444 else f"cache_{TASK}_{H.ENTROPY}", f"ep{n}.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        d = pickle.load(f)
    if d["omega_hash"] != w[2].omega_hash:
        raise ValueError(f"hybrid cache ep{n}: omega hash differs from the planner world")
    J, S = replay_wire(n, w, d["wire"])
    if J != d["J"]:
        raise ValueError(f"hybrid replay ep{n}: J {J} != cached {d['J']}")
    return J, S


def episode(n, entropy, tl, gap, rounds):
    configure(entropy)
    path = cache_path(entropy, n, f"tl{tl}")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    w = H.world(n)
    inst, omega, marks, fb = w
    pm, _ = persistence_marks(n, w)
    drift = mark_drift(marks, pm)
    res = dict(n=n, entropy=entropy, tl=tl, drift=drift)
    res["J_hyb"], res["S_hyb"] = hybrid(n, w)
    models = {}
    for tag, wm in (("true", w), ("persist", (inst, omega, pm, fb))):
        out, sol = solve_lazy_on(n, wm, RULES, tl, gap, rounds, label=tag)
        res[f"out_{tag}"] = {k: v for k, v in out.items() if k != "x"}
        if sol is None:
            res[f"J_{tag}"], res[f"S_{tag}"], res[f"claim_{tag}"] = None, None, None
            continue
        m, x = sol
        models[tag] = m
        res[f"claim_{tag}"] = out["J"]
        res[f"J_{tag}"], res[f"S_{tag}"] = play_on(n, w, m, x)  # always the TRUE world
        res[f"x_{tag}"] = x
    if len(models) == 2:
        mt, mp = models["true"], models["persist"]
        res["same_columns"] = bool(mt.meta["template"] == mp.meta["template"] and mt.lb.shape == mp.lb.shape)
        if not res["same_columns"]:
            raise ValueError(f"ep{n}: persistence model columns differ from the true model's")
    res["secs"] = time.time() - t0
    with open(path, "wb") as f:
        pickle.dump(res, f)
    f = lambda v: "   n/a" if v is None else f"{v / BN:9.1f}"
    logline(f"ep {n:2d}  J hybrid {f(res['J_hyb'])}  true-plan played {f(res['J_true'])} (claimed {f(res['claim_true'])}, "
            f"rounds {res['out_true'].get('rounds')}, gap {res['out_true'].get('gap')})  persist-plan played {f(res['J_persist'])} "
            f"(claimed {f(res['claim_persist'])}, rounds {res['out_persist'].get('rounds')}, gap {res['out_persist'].get('gap')})  "
            f"{res['secs']:.0f}s")
    return res


def references(entropy, episodes):
    from sbf_starter import scoring

    return list(scoring.episode_set(TASK, episodes, entropy=entropy, verbose=False).references)


def run(episodes=8, start=0, entropy=444, tl=60, gap=1e-3, rounds=8, n_jobs=2, rss_refs=True):
    configure(entropy)
    logline(f"\n=== persist_plan run {time.strftime('%Y-%m-%d %H:%M:%S')}: {TASK} root {entropy} eps {start}..{start + episodes - 1} "
            f"rules {RULES} tl {tl} gap {gap} rounds {rounds} n_jobs {n_jobs}")
    res = Parallel(n_jobs=n_jobs)(delayed(episode)(n, entropy, tl, gap, rounds) for n in range(start, start + episodes))
    table(res, entropy, rss_refs, start)
    return None


def table(res, entropy, rss_refs=True, start=0):
    logline("\n| ep | J hybrid | J true-plan played | J persist-plan played | persist - true | persist - hybrid | claimed true | claimed persist |")
    logline("| --- | --- | --- | --- | --- | --- | --- | --- |")
    ok = [r for r in res if None not in (r["J_hyb"], r["J_true"], r["J_persist"])]
    for r in res:
        g = lambda k: r.get(k)
        cells = [g("J_hyb"), g("J_true"), g("J_persist")]
        d1 = None if None in (g("J_true"), g("J_persist")) else g("J_persist") - g("J_true")
        d2 = None if None in (g("J_hyb"), g("J_persist")) else g("J_persist") - g("J_hyb")
        fmt = lambda v: "n/a" if v is None else f"{v / BN:.1f}"
        logline(f"| {r['n']} | " + " | ".join(fmt(v) for v in cells + [d1, d2, g("claim_true"), g("claim_persist")]) + " |")
    if ok:
        m = lambda k: np.mean([r[k] for r in ok]) / BN
        dpt = np.array([r["J_persist"] - r["J_true"] for r in ok]) / BN
        dph = np.array([r["J_persist"] - r["J_hyb"] for r in ok]) / BN
        dth = np.array([r["J_true"] - r["J_hyb"] for r in ok]) / BN
        logline(f"| mean ({len(ok)}) | {m('J_hyb'):.1f} | {m('J_true'):.1f} | {m('J_persist'):.1f} | {dpt.mean():.1f} | {dph.mean():.1f} | "
                f"{m('claim_true'):.1f} | {m('claim_persist'):.1f} |")
        se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        logline(f"paired (bn USD/ep, mean +- se; negative = first is cheaper): persist-true {dpt.mean():+.1f} +- {se(dpt):.1f} "
                f"(median {np.median(dpt):+.1f}); persist-hybrid {dph.mean():+.1f} +- {se(dph):.1f}; true-hybrid {dth.mean():+.1f} +- {se(dth):.1f}")
        if rss_refs:
            try:
                from package_baselines import rss

                n_max = max(r["n"] for r in ok) + 1
                refs = references(entropy, n_max)
                idx = [r["n"] for r in ok]
                rr = [refs[i] for i in idx]
                for k in ("J_hyb", "J_true", "J_persist", "claim_true"):
                    sc, _lv = rss(rr, [r[k] for r in ok])
                    logline(f"RSS {k:12s} {sc:.4f}")
            except Exception as e:  # noqa: BLE001
                logline(f"RSS not computed: {e!r}")
    drift_keys = ("u", "supply", "G_bar", "y_bar", "R", "kappa", "demand", "prohibited", "tariff")
    logline("\nmark drift (persistence vs true): rel. L1 per week -> week 1 / mean of weeks 2..T / max (week); prohibited: count of (e,k) differing")
    for r in res:
        d = r["drift"]
        logline(f"  ep {r['n']}: " + "  ".join(f"{k} {d[k][0]:.3f}/{d[k][1]:.3f}/{d[k][2]:.3f}" + (f"@{d[k][3]}" if d[k][3] > 0 else "") for k in drift_keys)
                + f"  true scrap sum {d['sigma_scr_true_sum']:.3f} hits {d['fab_hits_true']}")


def diag(n=None, entropy=444, tl=60, episodes=8, start=0):
    """One episode: per-component and per-band (1-13, 14-26, 27-39, 40-52) cost of persist vs true plan vs hybrid (bn)."""
    configure(entropy)
    res = []
    for k in range(start, start + episodes):
        p = cache_path(entropy, k, f"tl{tl}")
        if os.path.exists(p):
            with open(p, "rb") as f:
                res.append(pickle.load(f))
    if n is None:  # the episode where persistence loses most against the true plan
        r = max((r for r in res if r["J_persist"] is not None and r["J_true"] is not None), key=lambda r: r["J_persist"] - r["J_true"])
    else:
        r = next(r for r in res if r["n"] == n)
    names = r["S_true"]["comp_names"]
    logline(f"\n--- diag ep {r['n']} root {entropy}: bn USD, persist - true (and persist - hybrid) ---")
    bands = [(1, 13), (14, 26), (27, 39), (40, 52)]
    for tag_b in ("true", "hyb"):
        Sb, Sp = r[f"S_{tag_b}"], r["S_persist"]
        if Sb is None:
            continue
        d = (Sp["comp"] - Sb["comp"]) / 1e9
        line = "  vs " + tag_b + ": total " + f"{d.sum() - (Sp['salvage_usd'] - Sb['salvage_usd']) / 1e9:+.1f}" + " = "
        line += ", ".join(f"{c} {d[:, i].sum():+.1f}" for i, c in enumerate(names) if abs(d[:, i].sum()) > 0.05)
        line += f", salvage {-(Sp['salvage_usd'] - Sb['salvage_usd']) / 1e9:+.1f}"
        logline(line)
        for a, b in bands:
            logline(f"     weeks {a:2d}-{b:2d}: " + ", ".join(f"{c} {d[a - 1:b, i].sum():+.1f}" for i, c in enumerate(names) if abs(d[a - 1:b, i].sum()) > 0.5))
    St, Sp = r["S_true"], r["S_persist"]
    logline(f"  shed by grid (energy units, persist - true): {np.round(Sp['shed'].sum(0) - St['shed'].sum(0), 0).tolist()}")
    logline(f"  lots by fab (persist - true): {np.round(Sp['lots'].sum(0) - St['lots'].sum(0), 0).tolist()}")
    logline(f"  served by sink (persist - true): {np.round(Sp['served'].sum(0) - St['served'].sum(0), 0).tolist()}")
    wk = (Sp["comp"].sum(1) - St["comp"].sum(1)) / 1e9
    worst = np.argsort(-wk)[:8]
    logline("  worst weeks (persist - true, bn): " + ", ".join(f"w{t + 1} {wk[t]:+.1f}" for t in worst))
    logline(f"  drift: {r['drift']}")
    return None


GROUPS = {
    "straits": ("o", "o_now", "kappa", "kappa_now", "wr_class", "h_queue", "c_wr"),
    "edges": ("u", "u_now", "c"),
    "bans": ("prohibited", "tariff"),
    "demand": ("demand",),
    "plants": ("G_bar", "G_bar_now", "y_bar", "y_bar_now", "supply", "supply_now", "R", "R_now", "alpha_bar", "alpha_now",
               "sigma_scr", "R_osat"),
}


def attrib_one(n, entropy, rounds, tl, gap):
    """Played J of plans on mixed marks: persistence with ONE group made true (+g), and true with one group persisted (-g)."""
    configure(entropy)
    w = H.world(n)
    inst, omega, marks, fb = w
    pm, _ = persistence_marks(n, w)
    out = {}
    variants = [("true", marks), ("persist", pm)]
    for g, fields in GROUPS.items():
        variants.append((f"+{g}", dataclasses.replace(pm, **{f: getattr(marks, f) for f in fields}, **({"fab_hits": marks.fab_hits} if g == "plants" else {}))))
        variants.append((f"-{g}", dataclasses.replace(marks, **{f: getattr(pm, f) for f in fields}, **({"fab_hits": ()} if g == "plants" else {}))))
    for tag, mk in variants:
        o, sol = solve_lazy_on(n, (inst, omega, mk, fb), RULES, tl, gap, rounds, label=tag)
        out[tag] = None if sol is None else play_on(n, w, *sol)[0]
    return n, out


def attrib(episodes=8, start=0, entropy=444, rounds=0, tl=20, gap=1e-3, n_jobs=1):
    """Which group of marks the persistence loss comes from (cheap: by default round 0 = the plain LP, no binaries)."""
    configure(entropy)
    res = Parallel(n_jobs=n_jobs)(delayed(attrib_one)(n, entropy, rounds, tl, gap) for n in range(start, start + episodes))
    tags = list(res[0][1])
    logline(f"\n--- attribution (rounds {rounds}, tl {tl}): played J, bn USD; +g = persistence with group g true, -g = true with g persisted ---")
    logline("| ep | " + " | ".join(tags) + " |")
    logline("| --- " * (len(tags) + 1) + "|")
    for n, o in res:
        logline(f"| {n} | " + " | ".join("n/a" if o[t] is None else f"{o[t] / BN:.1f}" for t in tags) + " |")
    ok = [o for _, o in res if all(v is not None for v in o.values())]
    mean = {t: np.mean([o[t] for o in ok]) / BN for t in tags}
    logline("| mean | " + " | ".join(f"{mean[t]:.1f}" for t in tags) + " |")
    gapm = mean["persist"] - mean["true"]
    logline(f"persistence loss {gapm:+.1f} bn/ep; recovered by making one group true (persist - (+g)): "
            + ", ".join(f"{g} {mean['persist'] - mean['+' + g]:+.1f}" for g in GROUPS)
            + "; caused by persisting one group alone ((-g) - true): "
            + ", ".join(f"{g} {mean['-' + g] - mean['true']:+.1f}" for g in GROUPS))


def show(episodes=8, start=0, entropy=444, tl=60):
    configure(entropy)
    res = []
    for k in range(start, start + episodes):
        p = cache_path(entropy, k, f"tl{tl}")
        if os.path.exists(p):
            with open(p, "rb") as f:
                res.append(pickle.load(f))
    table(res, entropy)


if __name__ == "__main__":
    fire.Fire({"run": run, "diag": diag, "show": show, "attrib": attrib})
