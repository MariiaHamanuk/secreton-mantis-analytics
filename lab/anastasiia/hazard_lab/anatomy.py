"""Anatomy of the generator's capacity cuts: what each event type changes, how long it lasts, what the agent sees.

    uv run python lab/anastasiia/hazard_lab/anatomy.py --small=300 --full=100 --n_jobs=3

Root 333 only (the statistics root). For every event of every episode that overlaps the episode window it records the
type, the target, the onset, the duration, the severity, the marks the event alone changes (via
``marks.graph_marks`` of that single event) and the edges it cuts. From these it writes, per task:

  a. events per episode by type (all / carried in, onset < 0 / ending inside the episode);
  b. quantiles of duration and severity per type, share carried in;
  c. P(the cut is over within h weeks | it is running and its observed age is in a bucket), empirical, split into
     all / started during the episode / carried in, and the analytic value for the started ones where a law exists;
  d. the regional conflict: duration, share with duration <= 52, carried-in onsets, the computability of W0's end;
  e. cut edge-weeks by event type, fuel-lane edges (lng, crude, nucfuel) against the others, and the part of them
     that lies in spells ending inside the episode;
  f. depth signature: the surviving share u_now / u0 of the cut edges per type, and which mark fields each type moves;
  g. the duration laws of the profile with their closed-form moments.

The observed age of an event running at week t is t - first_seen + 1, first_seen = max(1, ceil(onset) + 1): for an event
carried in, weeks since week 1. "Over within h" means the true end of the effect window is at or before the instant
t - 1 + h (uncensored: the generator knows the end even beyond the episode). The markdown goes to
outputs/hazard_lab/anatomy/anatomy_<date_time>.md; the raw sums to a pickle beside it.
"""

import math
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import fire
import numpy as np

HORIZONS = (1, 2, 4, 8, 13, 26)
BUCKETS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 13), (14, 26), (27, 10**6))
BUCKET_NAMES = ("1", "2", "3-4", "5-8", "9-13", "14-26", "27+")
VARIANTS = ("all", "new", "carried")
FUEL = (0, 1, 2)  # commodity indices of lng, crude, nucfuel (instance order, checked in ``episode``)
TOL = 1e-9
# pseudo types the hazard tables are cut into (an event can enter two of them: conflict -> W0 and W1 ends)
LABELS = (
    "tariff", "sanction", "sanction_ucut", "material_outage", "militarised_closure", "strait_closure_derived",
    "regional_conflict_W1end", "regional_conflict_W0end", "piracy", "energy_shock", "weather_closure",
    "port_strike_stoppage", "port_strike_slowdown",
)
FIELDS = ("u", "o", "supply", "G_bar", "tariff", "prohibited", "c", "R_fab_osat", "war_risk")


def bucket_of(age: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= age <= hi:
            return i
    raise AssertionError(age)


def episode(task: str, entropy: int, n: int) -> dict:
    """Events and hazard sums of one episode (a worker process: imports the package itself)."""
    from shockbench_flow import marks as M
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.omega import codes

    inst, params = task_generator(task)
    om = sample_omega(inst, params, entropy, n, "train")
    T = inst.T
    mp = params.marks
    assert [c.id for c in inst.commodities[:3]] == ["lng", "crude", "nucfuel"]
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
    c0 = np.array([e.c0 for e in inst.edges])
    fuel_edge = np.array([any(k in FUEL for k in e.K) for e in inst.edges])
    sup0 = np.array([s.supply for s in inst.stock_slots])
    Gb0 = np.array([inst.nodes[g].grid.deliverable for g in inst.grids])
    Z0 = np.zeros((T, len(inst.edges), len(inst.commodities)), dtype=bool)
    for e, k in inst.prohibitions_at_reset:
        Z0[:, e, k] = True
    weeks = np.arange(1, T + 1, dtype=np.float64)
    finite = np.isfinite(u0)
    haz = np.zeros((len(LABELS), len(VARIANTS), len(BUCKETS), 1 + len(HORIZONS)), dtype=np.int64)
    depth: dict[str, Counter] = {}
    events = []
    for q in M.read_events(inst, om):
        name = codes.EVENT_TYPES[q.type]
        derived = name == "militarised_closure" and q.severity == 1.0
        # the effect window end: a conflict lasts W0 + W1 = max(d, 52) + 52, the others onset + duration
        w0_end, w1_end = M.war_windows(q.onset, q.duration, mp.war_profile_window)
        end = w1_end if name == "regional_conflict" else q.end
        gm = M.graph_marks(inst, (q,), mp)
        ucut = (gm["u"] < u0 * (1 - TOL)) & finite  # (T, E) week-average cut
        cut_edges = np.flatnonzero(ucut.any(axis=0))
        changed = {
            "u": bool(cut_edges.size),
            "o": bool((gm["o"] < 1 - TOL).any()),
            "supply": bool((gm["supply"] < sup0 * (1 - TOL)).any()),
            "G_bar": bool((gm["G_bar"] < Gb0 * (1 - TOL)).any()),
            "tariff": bool((gm["tariff"] > 0).any()),
            "prohibited": bool((gm["prohibited"] & ~Z0).any()),
            "c": bool((gm["c"] > c0 * (1 + TOL)).any()),
            "R_fab_osat": False,
            "war_risk": bool((M._war_risk_class(inst, (q,), mp, weeks) > 0).any()),
        }
        if name == "regional_conflict":
            Rf, Ro = M.restoration_factors(inst, (q,), mp)
            changed["R_fab_osat"] = bool((Rf < 1 - TOL).any() or (Ro < 1 - TOL).any())
        # edges the closure touches: every edge into or out of its chokepoint
        o_edges = []
        o_weeks = 0
        if changed["o"]:
            ci = [i for i, c in enumerate(inst.chokepoints) if (c == q.target)]
            o_edges = sorted(set(inst.out_edges[q.target]) | set(inst.in_edges[q.target]))
            o_weeks = int((gm["o"][:, ci[0]] < 1 - TOL).sum())
        # depth: the instantaneous surviving share u_now / u0 of the cut edges, in the weeks it is cut
        if cut_edges.size:
            f = (gm["u_now"][:, cut_edges] / u0[cut_edges])[ucut[:, cut_edges]]
            depth.setdefault(name + ("_derived" if derived else ""), Counter()).update(np.round(f, 2).tolist())
        # cut edge-weeks, fuel against other, and those in spells that end before the last week
        ew = {"fuel": 0, "other": 0, "fuel_end": 0, "other_end": 0}
        for e in cut_edges:
            col = np.flatnonzero(ucut[:, e])
            k = "fuel" if fuel_edge[e] else "other"
            ew[k] += col.size
            if col[-1] < T - 1:
                ew[k + "_end"] += col.size
        ow = {"fuel": 0, "other": 0, "fuel_end": 0, "other_end": 0}  # edge-weeks behind a closed chokepoint
        if o_weeks:
            colw = np.flatnonzero(gm["o"][:, [i for i, c in enumerate(inst.chokepoints) if c == q.target][0]] < 1 - TOL)
            for e in o_edges:
                k = "fuel" if fuel_edge[e] else "other"
                ow[k] += colw.size
                if colw[-1] < T - 1:
                    ow[k + "_end"] += colw.size
        n_fuel_cut = int(fuel_edge[cut_edges].sum())
        n_o_fuel = int(fuel_edge[o_edges].sum()) if o_edges else 0
        events.append({
            "type": name, "derived": derived, "persistent": None, "kind": codes.TARGET_KINDS[q.target_kind],
            "region": q.region, "cp": q.counterpart, "onset": q.onset, "duration": q.duration,
            "severity": q.severity, "rate": q.rate, "T0": q.T0, "tau": q.tau_rho, "end": end, "w0_end": w0_end,
            "w1_end": w1_end, "commodity": q.commodity, "n_cut": int(cut_edges.size), "n_fuel_cut": n_fuel_cut,
            "n_o_edges": len(o_edges), "n_o_fuel": n_o_fuel, "changed": changed, "ew": ew, "ow": ow,
            "carried": q.onset < 0,
        })
        # hazard sums
        labs = []
        if name == "port_strike":
            labs.append(("port_strike_stoppage" if q.severity > 0.9 else "port_strike_slowdown", end))
        elif name == "militarised_closure" and derived:
            labs.append(("strait_closure_derived", end))
        elif name == "regional_conflict":
            if q.counterpart >= 0:
                labs += [("regional_conflict_W1end", w1_end), ("regional_conflict_W0end", w0_end)]
        else:
            labs.append((name, end))
            if name == "sanction" and changed["u"]:
                labs.append(("sanction_ucut", end))
        first = max(1, math.ceil(q.onset) + 1)
        for lab, lend in labs:
            li = LABELS.index(lab)
            for t in range(1, T + 1):
                s = t - 1
                if not (q.onset <= s < lend):
                    continue
                b = bucket_of(t - first + 1)
                for vi in ((0, 2) if q.onset < 0 else (0, 1)):
                    haz[li, vi, b, 0] += 1
                    for hi, h in enumerate(HORIZONS):
                        haz[li, vi, b, 1 + hi] += lend <= s + h
    return {"T": T, "events": events, "haz": haz, "depth": depth}


def quant(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> str:
    if len(x) == 0:
        return "-"
    return " / ".join(f"{v:.1f}" for v in np.quantile(np.asarray(x, dtype=float), qs))


def md_table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def laws_table() -> str:
    from shockbench_flow.hosting.tasks import task_generator

    rows = []
    for task in ("small", "full"):
        _, p = task_generator(task)
        d = dict(p.laws.duration)
        sev = dict(p.laws.severity)
        pp = p.poisson
        items = list(d.items()) + [("port_strike stoppage", pp.stoppage_duration),
                                   ("port_strike slowdown", pp.slowdown_duration)]
        for name, law in items:
            rows.append([task, name, f"{law.kind}{tuple(round(x, 4) for x in law.params)}",
                         f"{law.quantile(0.5):.1f}", f"{law.mean():.1f}"]
                        + [f"{law.cdf(x):.2f}" for x in (4, 13, 26, 52, 104)]
                        + [str(sev.get(name, pp.stoppage_severity if "stoppage" in name else pp.slowdown_severity))
                           .replace("Law(kind=", "").replace(")", "")])
    return md_table(["task", "type", "duration law (days or weeks per kind)", "median wk", "mean wk",
                     "F(4w)", "F(13w)", "F(26w)", "F(52w)", "F(104w)", "severity law"], rows)


def analytic(label: str, task: str, e: float, h: int) -> float | None:
    """P(over within h | elapsed e weeks since onset) from the duration law, for the new events (None: no closed form)."""
    from shockbench_flow.disruption.laws import Law
    from shockbench_flow.hosting.tasks import task_generator

    _, p = task_generator(task)
    d = dict(p.laws.duration)
    if label in d and label not in ("regional_conflict",):
        parts = [(1.0, d[label])]
    elif label == "sanction_ucut":
        parts = [(1.0, d["sanction"])]
    elif label == "port_strike_stoppage":
        parts = [(1.0, p.poisson.stoppage_duration)]
    elif label == "port_strike_slowdown":
        parts = [(1.0, p.poisson.slowdown_duration)]
    else:
        return None
    F = lambda x: sum(w * law.cdf(x) for w, law in parts)  # noqa: E731
    s = 1 - F(e)
    return None if s <= 1e-12 else (F(e + h) - F(e)) / s


def main(small: int = 300, full: int = 100, entropy: int = 333, n_jobs: int = 3, out: str = "") -> None:
    from joblib import Parallel, delayed

    assert entropy == 333, "the statistics root is 333"
    assert n_jobs <= 3
    out_dir = Path(out) if out else Path(__file__).resolve().parents[3] / "outputs" / "hazard_lab" / "anatomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    md = [f"# Generator anatomy, root {entropy}, Small episodes 0..{small - 1}, Full episodes 0..{full - 1}\n",
          "## g. Duration laws of the profile (closed-form moments)\n", laws_table()]
    raw = {}
    for task, count in (("small", small), ("full", full)):
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs)(delayed(episode)(task, entropy, n) for n in range(count))
        print(f"{task}: {count} episodes in {time.time() - t0:.0f} s", file=sys.stderr)
        raw[task] = res
        md.append(f"\n# Task {task} (T = {res[0]['T']}), {count} episodes\n")
        md += report(task, res, count)
    path = out_dir / f"anatomy_{stamp}.md"
    path.write_text("\n".join(md))
    with open(out_dir / f"anatomy_{stamp}.pkl", "wb") as f:
        pickle.dump(raw, f)
    print("\n".join(md))
    print(f"written {path}", file=sys.stderr)


def report(task: str, res: list[dict], count: int) -> list[str]:
    md = []
    T = res[0]["T"]
    evs = [dict(e, ep=i) for i, r in enumerate(res) for e in r["events"]]

    def tname(e):
        if e["derived"]:
            return "strait_closure_derived"
        if e["type"] == "port_strike":
            return "port_strike_stoppage" if e["severity"] > 0.9 else "port_strike_slowdown"
        return e["type"]

    types = sorted({tname(e) for e in evs})
    # a
    rows = []
    for ty in types:
        s = [e for e in evs if tname(e) == ty]
        rows.append([ty, f"{len(s) / count:.2f}", f"{sum(e['carried'] for e in s) / count:.2f}",
                     f"{sum(e['end'] <= T for e in s) / count:.2f}",
                     f"{sum(e['end'] <= T and not e['carried'] for e in s) / count:.2f}"])
    rows.append(["all", f"{len(evs) / count:.2f}", f"{sum(e['carried'] for e in evs) / count:.2f}",
                 f"{sum(e['end'] <= T for e in evs) / count:.2f}",
                 f"{sum(e['end'] <= T and not e['carried'] for e in evs) / count:.2f}"])
    md += ["## a. Events per episode (stored: effect window reaches past week 0 and onset < T)\n",
           "'ends inside' = effect-window end <= T (conflict: end of W1).\n",
           md_table(["type", "all", "carried in (onset<0)", "ends inside", "ends inside, started in episode"], rows)]
    # b
    rows = []
    for ty in types:
        s = [e for e in evs if tname(e) == ty]
        rows.append([ty, len(s), quant([e["duration"] for e in s]), quant([e["severity"] for e in s]),
                     f"{np.mean([e['carried'] for e in s]):.2f}",
                     f"{np.mean([e['duration'] <= 26 for e in s]):.2f}",
                     f"{np.mean([e['duration'] <= 52 for e in s]):.2f}"])
    md += ["## b. Duration and severity per type (quantiles 10/25/50/75/90, weeks; stored events)\n",
           "Stored events are length-biased (a long event is more likely to overlap the episode), so these are not the "
           "laws' own quantiles (see g).\n",
           md_table(["type", "n", "duration wk", "severity", "share onset<0", "dur<=26", "dur<=52"], rows)]
    # f: fields changed, depth
    rows = []
    for ty in types:
        s = [e for e in evs if tname(e) == ty]
        rows.append([ty, len(s)] + [f"{np.mean([e['changed'][f] for e in s]):.2f}" for f in FIELDS]
                    + [f"{np.mean([e['n_cut'] for e in s]):.1f}", f"{np.mean([e['n_fuel_cut'] for e in s]):.1f}",
                       f"{np.mean([e['n_o_edges'] for e in s]):.1f}", f"{np.mean([e['n_o_fuel'] for e in s]):.1f}"])
    md += ["## f1. Which marks each type moves (share of events whose single-event marks differ from nominal)\n",
           "n_cut: edges with a cut u; fuel: of them with lng/crude/nucfuel; o_edges: edges into/out of a closed "
           "chokepoint, o_fuel: of them with fuel.\n",
           md_table(["type", "n"] + list(FIELDS) + ["n_cut", "n_fuel_cut", "n_o_edges", "n_o_fuel"], rows)]
    dep: dict[str, Counter] = {}
    for r in res:
        for k, c in r["depth"].items():
            dep.setdefault(k, Counter()).update(c)
    rows = []
    for k, c in sorted(dep.items()):
        tot = sum(c.values())
        top = ", ".join(f"{v:.2f} ({100 * n / tot:.0f}%)" for v, n in c.most_common(6))
        vals = np.repeat(list(c.keys()), list(c.values()))
        rows.append([k, tot, f"{vals.min():.2f}", f"{np.median(vals):.2f}", f"{vals.max():.2f}", top])
    md += ["## f2. Depth signature: u_now / u0 of the cut edges, per cut edge-week (the observed graph_now.u / u0)\n",
           md_table(["type", "cut edge-weeks", "min", "median", "max", "most common values"], rows)]
    # c
    haz = sum(r["haz"] for r in res)
    md.append("## c. P(the cut is over within h weeks | running, observed age in bucket)\n")
    md.append("Cells: empirical P; n = event-weeks in the bucket. For 'new' the analytic value of the duration law "
              "(elapsed = bucket mid - 0.5) is in square brackets where the law is closed-form.\n")
    for li, lab in enumerate(LABELS):
        if haz[li, 0, :, 0].sum() == 0:
            continue
        for vi, var in enumerate(VARIANTS):
            if haz[li, vi, :, 0].sum() == 0:
                continue
            rows = []
            for b, bn in enumerate(BUCKET_NAMES):
                n = int(haz[li, vi, b, 0])
                if n == 0:
                    continue
                lo, hi = BUCKETS[b]
                e = (lo + min(hi, 40)) / 2 - 0.5
                cells = []
                for hi_, h in enumerate(HORIZONS):
                    c = f"{haz[li, vi, b, 1 + hi_] / n:.2f}"
                    if var == "new":
                        a = analytic(lab, task, e, h)
                        c += f" [{a:.2f}]" if a is not None else ""
                    cells.append(c)
                rows.append([bn, n] + cells)
            md += [f"### {lab} - {var}\n", md_table(["age wk", "n"] + [f"<= {h}w" for h in HORIZONS], rows)]
    # d
    cf = [e for e in evs if e["type"] == "regional_conflict"]
    cp = [e for e in cf if e["cp"] >= 0]
    new_cp = [e for e in cp if not e["carried"]]
    car_cp = [e for e in cp if e["carried"]]
    md.append("## d. Regional conflict\n")
    rows = [
        ["stored conflicts per episode (all / with counterpart)", f"{len(cf) / count:.2f} / {len(cp) / count:.2f}"],
        ["with fab/OSAT restoration (T0, tau drawn)", f"{np.mean([e['T0'] == e['T0'] for e in cf]):.2f}"],
        ["duration d, weeks, 10/25/50/75/90 (with counterpart)", quant([e["duration"] for e in cp])],
        ["share d <= 52 (W0 = onset+52 exactly)", f"{np.mean([e['duration'] <= 52 for e in cp]):.2f}"],
        ["share carried in (onset<0)", f"{np.mean([e['carried'] for e in cp]):.2f}"],
        ["carried in: weeks before week 1, 10/25/50/75/90", quant([-e["onset"] for e in car_cp])],
        ["carried in: still in W0 at week 1", f"{np.mean([e['w0_end'] > 0 for e in car_cp]):.2f}" if car_cp else "-"],
        ["carried in: W0 ends inside the episode", f"{np.mean([0 < e['w0_end'] <= T for e in car_cp]):.2f}" if car_cp else "-"],
        ["carried in: W1 ends inside the episode", f"{np.mean([e['w1_end'] <= T for e in car_cp]):.2f}" if car_cp else "-"],
        ["started in episode: n", len(new_cp)],
        ["started in episode: W0 end inside the episode", f"{np.mean([e['w0_end'] <= T for e in new_cp]):.2f}" if new_cp else "-"],
        ["started in episode: W1 end inside the episode", f"{np.mean([e['w1_end'] <= T for e in new_cp]):.2f}" if new_cp else "-"],
        ["started in episode: d <= 52 (W0 end exactly onset+52)", f"{np.mean([e['duration'] <= 52 for e in new_cp]):.2f}" if new_cp else "-"],
        ["started in episode, d <= 52, onset+52 <= T", f"{np.mean([e['duration'] <= 52 and e['onset'] + 52 <= T for e in new_cp]):.2f}" if new_cp else "-"],
        ["conflict with counterpart: n_cut edges 10/50/90", quant([e["n_cut"] for e in cp], (0.1, 0.5, 0.9))],
        ["T0 (dead time of fab/OSAT hit), weeks, 10/50/90", quant([e["T0"] for e in cf if e["T0"] == e["T0"]], (0.1, 0.5, 0.9))],
        ["tau_rho weeks, 10/50/90", quant([e["tau"] for e in cf if e["tau"] == e["tau"]], (0.1, 0.5, 0.9))],
    ]
    md += [md_table(["quantity", "value"], rows)]
    # e
    rows = []
    tot = {k: 0 for k in ("fuel", "other", "fuel_end", "other_end")}
    for ty in types:
        s = [e for e in evs if tname(e) == ty]
        row = {k: sum(e["ew"][k] for e in s) for k in tot}
        orow = {k: sum(e["ow"][k] for e in s) for k in tot}
        rows.append([ty, f"{row['fuel'] / count:.1f}", f"{row['other'] / count:.1f}", f"{row['fuel_end'] / count:.1f}",
                     f"{row['other_end'] / count:.1f}", f"{orow['fuel'] / count:.1f}", f"{orow['other'] / count:.1f}",
                     f"{orow['fuel_end'] / count:.1f}", f"{orow['other_end'] / count:.1f}"])
        for k in tot:
            tot[k] += row[k]
    allrows = [[r[0]] + r[1:] for r in rows]
    md += ["## e. Cut edge-weeks per episode by event type (an edge-week cut by two events counts for both)\n",
           "u-cut = week-average u below nominal. 'ends' = in cut spells (event, edge) that end before the last week. "
           "'via o' = edge-weeks of edges into/out of a chokepoint with o < 1 (closures).\n",
           md_table(["type", "u fuel", "u other", "u fuel ends", "u other ends", "via o fuel", "via o other",
                     "via o fuel ends", "via o other ends"], allrows)]
    shares = []
    for key, name in (("fuel", "u fuel"), ("other", "u other"), ("fuel_end", "u fuel ends"), ("other_end", "u other ends")):
        total = sum(sum(e["ew"][key] for e in evs if tname(e) == ty) for ty in types)
        if total:
            shares.append([name] + [f"{100 * sum(e['ew'][key] for e in evs if tname(e) == ty) / total:.0f}%" for ty in types])
    md += ["Share of the (double-counted) total by type:\n", md_table(["column"] + types, shares)]
    return md


if __name__ == "__main__":
    fire.Fire(main)
