"""Aggregate statistics of planner_LSF: what the improved fuel schedules do differently from the hybrid agent's rules.

    uv run python lab/anastasiia/mpc_lab/planners/lsf_stats.py run --episodes=12

For every episode cache (outputs/planner_LSF/cache/ep<n>.pkl: the hybrid agent's played actions; lsf_ep<n>.pkl: the
improved flows after the fuel local search) both trajectories are replayed once with the fast simulator
(planner_LSF.FReplay) and the step records give, per week:
  per (grid, fuel): fuel arriving at the grid, closing stock at the grid and at its feeder terminal, segment burn,
                    rationing state (rationed fuel: last week's closing stock under psi x ibar), shed (per grid);
  per (destination, fuel): fuel ordered at the sources (executed dispatches whose tail is a supply node) by dispatch
                    week, destination = the last node of the order's lane.
After - before deltas are summed over week bands (quarters of the horizon) and aggregated over the episodes: mean,
median and sign consistency (episodes with delta > 0 / < 0 / ~ 0). Raw per-episode numbers go to
outputs/mpc_research/lsf_signal/lsf_signal_raw.csv, the tables to reports/lsf_signal.md (text after the
"<!-- INTERPRETATION -->" marker of an existing report is kept).
"""

import csv
import os
import pickle
import sys

import fire
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import planner_LSF as F  # noqa: E402

CACHE = F.CACHE
OUT = os.path.join(ROOT, "outputs", "mpc_research", "lsf_signal")
REPORT = os.path.join(os.path.dirname(HERE), "reports", "lsf_signal.md")
MARK = "<!-- INTERPRETATION -->"


# ----- per-episode series ---------------------------------------------------------------------------------------------
def series(inst, R):
    """Weekly series (arrays of length T, index week-1) of one replayed trajectory, keyed by tuples."""
    T = inst.T
    node_id = [n.id for n in inst.nodes]
    kname = [c.id for c in inst.commodities]
    sidx = inst.slot_index
    psi = inst.params.psi
    s0 = np.asarray(R.states[0].stock, dtype=float)
    out = {}

    def put(key, t, v):
        out.setdefault(key, np.zeros(T))[t - 1] += v

    feeder = {}  # (grid, k) -> terminal node
    for s, (nm, ty) in F.fuel_slots(inst).items():
        e, k, lane = inst.action_slots[s]
        ed = inst.edges[e]
        if ty == "valve":
            feeder[(ed.head, k)] = ed.tail
    grids = {g: [k for k in inst.nodes[g].grid.fuels if k is not None] for g in inst.grids}
    supply = {i for i, n in enumerate(inst.nodes) if node_id[i].startswith("src_")}
    fuels = {inst.commodity_index[f] for f in F.FUELS}
    for t in range(1, T + 1):
        rec = R.records[t]
        prev = s0 if t == 1 else R.records[t - 1].stock
        for (e, k, lane), q in rec.x.items():  # every executed dispatch, incl. releases; arrival at the edge's head
            if k not in fuels or q <= 0:
                continue
            ed = inst.edges[e]
            ta = t + ed.tau
            if ta <= T:
                put(("arr_node", ed.head, k), ta, q)
            if ed.tail in supply and ed.id.startswith(("sea.", "pipe.", "bypass.")):  # an order at a source
                dest = inst.edges[inst.lanes[lane].edges[-1]].head if lane is not None else ed.head
                put(("ord", dest, k), t, q)
                put(("ordsrc", ed.tail, dest, k), t, q)
        for gi, g in enumerate(inst.grids):
            grid = inst.nodes[g].grid
            put(("shed", g), t, float(rec.shed[gi]))
            for k in grids[g]:
                put(("burn", g, k), t, float(rec.segment[(gi, k)]))
                put(("gstock", g, k), t, float(rec.stock[sidx[(g, k)]]))
                if (g, k) in feeder:
                    put(("tstock", g, k), t, float(rec.stock[sidx[(feeder[(g, k)], k)]]))
                if k == grid.rationed and grid.ibar.get(k, 0) > 0:
                    thr = psi * grid.ibar[k]
                    put(("under", g, k), t, float(prev[sidx[(g, k)]] < thr))
                    put(("short", g, k), t, max(0.0, 1 - prev[sidx[(g, k)]] / thr))  # ration shortfall factor
    # arrivals at grids and terminals as the replay saw them
    for g, ks in grids.items():
        for k in ks:
            out[("arr", g, k)] = out.get(("arr_node", g, k), np.zeros(T)).copy()
            if (g, k) in feeder:
                out[("tin", g, k)] = out.get(("arr_node", feeder[(g, k)], k), np.zeros(T)).copy()
    out["_names"] = (node_id, kname)
    return out


def replay_pair(n):
    d = F.play(n)
    inst, _omega, marks = F.world(n)
    acts, bad = F.wire_actions(d, inst, marks)
    assert not bad
    Rb = F.FReplay(inst, marks, acts)
    assert Rb.J == d["J"], (Rb.J, d["J"])
    with open(os.path.join(CACHE, f"lsf_ep{n}.pkl"), "rb") as f:
        L = pickle.load(f)
    assert L["before"] == Rb.J
    Ra = F.FReplay(inst, marks, acts)
    Ra.flows = [dict(x) for x in L["flows"]]
    Ra.commit(1)
    assert Ra.J == L["after"], (Ra.J, L["after"])
    return inst, series(inst, Rb), series(inst, Ra), L, Rb, Ra


# ----- aggregation ----------------------------------------------------------------------------------------------------
def bands_of(T):
    q = T // 4
    return [(1 + i * q, (i + 1) * q if i < 3 else T) for i in range(4)]


def cell(vals, tol):
    """'mean / median (+a -b)' of per-episode deltas."""
    v = np.asarray(vals, float)
    pos, neg = int((v > tol).sum()), int((v < -tol).sum())
    return v.mean(), np.median(v), pos, neg


def fmt(vals, tol, scale=1.0, nd=0):
    m, md, p, ng = cell(vals, tol)
    return f"{m / scale:+.{nd}f} / {md / scale:+.{nd}f} (+{p} -{ng})"


def run(episodes=12):
    os.makedirs(OUT, exist_ok=True)
    data = []
    for n in range(episodes):
        inst, sb, sa, L, Rb, Ra = replay_pair(n)
        data.append((sb, sa, L))
        print(f"ep {n}: J {Rb.J / 1e11:.3f} -> {Ra.J / 1e11:.3f} bn, accepted {L['accepted']}", flush=True)
    T = inst.T
    names = data[0][0]["_names"]
    allk = sorted({k for sb, sa, _ in data for d in (sb, sa) for k in d if k != "_names"}, key=str)
    for sb, sa, _ in data:  # a key missing from a trajectory is an all-zero series
        for d in (sb, sa):
            for k in allk:
                d.setdefault(k, np.zeros(T))
    node_id, kname = names
    bands = bands_of(T)
    keys = allk
    # raw csv: per episode, key, week
    with open(os.path.join(OUT, "lsf_signal_raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "metric", "node", "node2", "fuel", "band", "before", "after", "delta"])
        for n, (sb, sa, _L) in enumerate(data):
            for key in keys:
                tag = key[0]
                if tag == "ordsrc":
                    nodes, fuel = (node_id[key[1]], node_id[key[2]]), kname[key[3]]
                elif tag in ("arr_node",):
                    continue
                else:
                    nodes, fuel = (node_id[key[1]], ""), (kname[key[2]] if len(key) > 2 else "")
                for lo, hi in bands:
                    b, a = sb[key][lo - 1 : hi].sum(), sa[key][lo - 1 : hi].sum()
                    w.writerow([n, tag, nodes[0], nodes[1], fuel, f"{lo}-{hi}", f"{b:.4f}", f"{a:.4f}", f"{a - b:.4f}"])
    with open(os.path.join(OUT, "lsf_signal_weekly_mean_delta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "node", "node2", "fuel"] + [f"w{t}" for t in range(1, T + 1)])
        for key in keys:
            if key[0] == "arr_node":
                continue
            tag = key[0]
            nodes = (node_id[key[1]], node_id[key[2]]) if tag == "ordsrc" else (node_id[key[1]], "")
            fuel = kname[key[3]] if tag == "ordsrc" else (kname[key[2]] if len(key) > 2 else "")
            dm = np.mean([sa[key] - sb[key] for sb, sa, _ in data], axis=0)
            w.writerow([tag, nodes[0], nodes[1], fuel] + [f"{x:.3f}" for x in dm])

    md = ["# LSF: системний сигнал паливного локального пошуку (lsf_stats.py)", "",
          f"Агент `anastasiia_hybrid_chiplp`, small, root {F.ENTROPY}, епізоди 0..{episodes - 1}; траєкторії «до» (гра агента) "
          "і «після» (planner_LSF) відтворено швидким replay (J збігається до цента з кешем). Усі дельти = після − до, "
          "суми за тижневими смугами. Клітинка: `середнє / медіана (+a −b)`, де a/b — кількість епізодів із дельтою "
          "> 0 / < 0 за порогом (шум нижче порогу не рахується). Одиниці: GWh (паливо, шед) або тижні (під нормуванням). "
          "Перевірено одним прогоном, 12 епізодів одного root.", ""]
    dJ = np.array([(L["before"] - L["after"]) / 1e11 for _, _, L in data])
    md += [f"Виграш J: середнє {dJ.mean():.2f} млрд USD, медіана {np.median(dJ):.2f}, мін {dJ.min():.2f}, макс {dJ.max():.2f}.", ""]
    comps = data[0][2]["comp0"].keys()
    md += ["Компоненти вартості, середня зміна (млрд USD/епізод): " + ", ".join(
        f"{c} {np.mean([L['comp1'][c] - L['comp0'][c] for _, _, L in data]) / 1e9:+.2f}" for c in comps), ""]

    bl = [f"{lo}-{hi}" for lo, hi in bands]

    def table(title, tag, tol, scale=1.0, nd=0, rows=None, level="gf", note=""):
        md.append(f"## {title}")
        if note:
            md.append(note)
        md.append("")
        md.append("| ряд | " + " | ".join(bl) + " | усі |")
        md.append("|---|" + "---|" * (len(bl) + 1))
        for key in keys:
            if key[0] != tag:
                continue
            if tag == "ordsrc":
                label = f"{node_id[key[1]]}→{node_id[key[2]]} {kname[key[3]]}"
            elif tag in ("ord", "shed"):
                label = node_id[key[1]] + ("" if tag == "shed" else f" {kname[key[2]]}")
            else:
                label = f"{node_id[key[1]]} {kname[key[2]]}"
            cells = []
            for lo, hi in bands + [(1, T)]:
                dv = [sa[key][lo - 1 : hi].sum() - sb[key][lo - 1 : hi].sum() for sb, sa, _ in data]
                cells.append(fmt(dv, tol, scale, nd))
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")

    table("Шед, GWh", "shed", 10.0, 1.0, 0, note="Шед за системою (усі палива).")
    table("Тижні під нормуванням (lng): стан = закриття минулого тижня < psi·ibar", "under", 0.5, 1.0, 1)
    table("Паливо, що надходить у систему (валв + прямі поставки), GWh", "arr", 10.0, 1.0, 0)
    table("Паливо, що надходить на фідерний термінал системи, GWh", "tin", 10.0, 1.0, 0)
    table("Спалене паливо (сегмент), GWh", "burn", 10.0, 1.0, 0)
    table("Замовлення на джерелах за призначенням (термінал або система), за тижнем відправки, GWh", "ord", 10.0, 1.0, 0)
    table("Замовлення на джерелах за джерелом→призначення", "ordsrc", 10.0, 1.0, 0)

    # stocks: mean closing stock per band, delta of the band mean
    for tag, title in (("gstock", "Закриття запасу в системі (середнє за смугу), GWh"),
                       ("tstock", "Закриття запасу на фідерному терміналі (середнє за смугу), GWh")):
        md.append(f"## {title}")
        md.append("")
        md.append("| ряд | " + " | ".join(bl) + " |")
        md.append("|---|" + "---|" * len(bl))
        for key in keys:
            if key[0] != tag:
                continue
            cells = []
            for lo, hi in bands:
                dv = [(sa[key][lo - 1 : hi] - sb[key][lo - 1 : hi]).mean() for sb, sa, _ in data]
                cells.append(fmt(dv, 10.0))
            md.append(f"| {node_id[key[1]]} {kname[key[2]]} | " + " | ".join(cells) + " |")
        md.append("")

    # centroid (flow-weighted mean week) of valve flows and orders: timing shift, per episode
    md += ["## Центроїд відправок (середньозважений тиждень), зсув після − до, тижні", "",
           "| ряд | зсув: середнє / медіана (+a −b) | до: середнє | після: середнє |", "|---|---|---|---|"]
    wk = np.arange(1, T + 1)
    for tag in ("arr", "ord"):
        for key in keys:
            if key[0] != tag:
                continue
            cb, ca = [], []
            for sb, sa, _ in data:
                if sb[key].sum() > 100 and sa[key].sum() > 100:
                    cb.append((sb[key] * wk).sum() / sb[key].sum())
                    ca.append((sa[key] * wk).sum() / sa[key].sum())
            if len(cb) < 3:
                continue
            dv = np.array(ca) - np.array(cb)
            label = (node_id[key[1]] + " " + kname[key[2]]) + (" (надх.)" if tag == "arr" else " (замовл.)")
            md.append(f"| {label} | {fmt(dv, 0.25, 1.0, 2)} | {np.mean(cb):.1f} | {np.mean(ca):.1f} |")
    md.append("")

    # orders in the last weeks and last order week
    md += ["## Пізні замовлення на джерелах (останні 8 тижнів відправки), GWh: до, після", "",
           "| призначення паливо | до | після | зміна: середнє / медіана (+a −b) | останній тиждень замовлення до → після (медіана) |",
           "|---|---|---|---|---|"]
    for key in keys:
        if key[0] != "ord":
            continue
        b = [sb[key][-8:].sum() for sb, sa, _ in data]
        a = [sa[key][-8:].sum() for sb, sa, _ in data]
        lw = lambda x: (np.nonzero(x > 1)[0].max() + 1) if (x > 1).any() else 0  # noqa: E731
        lb = [lw(sb[key]) for sb, sa, _ in data]
        la = [lw(sa[key]) for sb, sa, _ in data]
        md.append(f"| {node_id[key[1]]} {kname[key[2]]} | {np.mean(b):.0f} | {np.mean(a):.0f} | {fmt(np.array(a) - np.array(b), 10.0)} "
                  f"| {np.median(lb):.0f} → {np.median(la):.0f} |")
    md.append("")

    # accepted moves
    md += ["## Прийняті ходи пошуку (12 епізодів)", "", "| тип | паливо | хід | к-сть | виграш, млрд | епізодів із ходом |", "|---|---|---|---|---|---|"]
    agg = {}
    for n, (_, _, L) in enumerate(data):
        for m in L["moves"]:
            k = (m["type"], m["fuel"], m["move"])
            a = agg.setdefault(k, [0, 0.0, set()])
            a[0] += 1
            a[1] += m["gain"] / 1e11
            a[2].add(n)
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {v[0]} | {v[1]:.2f} | {len(v[2])} |")
    md.append("")
    # direction of shifts by band and type
    md += ["### Напрям зсувів за (тип, паливо, смуга тижня відправки): виграш млрд USD", "",
           "| тип | паливо | смуга | зсув раніше (к-сть / виграш) | зсув пізніше | зменшення (drop, half) | збільшення (x1.5, x2) |", "|---|---|---|---|---|---|---|"]
    dirs = {}
    for _, _, L in data:
        for m in L["moves"]:
            b = next(f"{lo}-{hi}" for lo, hi in bands if lo <= m["week"] <= hi)
            nm = m["move"]
            cls = "early" if nm.endswith("-1") or nm.endswith("-2") else "late" if nm.endswith("+1") or nm.endswith("+2") else (
                "down" if nm in ("drop", "half") else "up")
            a = dirs.setdefault((m["type"], m["fuel"], b), {}).setdefault(cls, [0, 0.0])
            a[0] += 1
            a[1] += m["gain"] / 1e11
    for k in sorted(dirs):
        c = dirs[k]
        g = lambda x: f"{c[x][0]} / {c[x][1]:.2f}" if x in c else "0"  # noqa: E731
        md.append(f"| {k[0]} | {k[1]} | {k[2]} | {g('early')} | {g('late')} | {g('down')} | {g('up')} |")
    md.append("")


    # order up-moves by lane group
    def group(m):
        nm = m["name"]
        lane = nm[nm.index("[") + 1 : -1] if "[" in nm else nm
        kind = "alt" if any(x in lane for x in ("lombok", "east", "cape", "bypass")) else "main"
        return m["fuel"], kind, lane.replace("lane.", "")

    md += ["### Збільшення замовлень (x1.5, x2) за паливом і типом ланки (main/alt = lombok, east, cape, bypass) і смугою", "",
           "| паливо | ланка | " + " | ".join(bl) + " | усього |", "|---|---|" + "---|" * (len(bl) + 1)]
    ag = {}
    for n, (_, _, L) in enumerate(data):
        for m in L["moves"]:
            if m["type"] != "order" or m["move"] not in ("x1.5", "x2"):
                continue
            f, kd, lane = group(m)
            b = next(i for i, (lo, hi) in enumerate(bands) if lo <= m["week"] <= hi)
            a = ag.setdefault((f, kd, lane), [[0, 0.0, set()] for _ in bands])
            a[b][0] += 1
            a[b][1] += m["gain"] / 1e11
            a[b][2].add(n)
    for k, v in sorted(ag.items(), key=lambda kv: -sum(x[1] for x in kv[1]))[:14]:
        md.append(f"| {k[0]} | {k[2]} ({k[1]}) | " + " | ".join(f"{x[0]} / {x[1]:.1f} bn / {len(x[2])} ep" for x in v) + f" | {sum(x[1] for x in v):.1f} |")
    md.append("")
    # concentration of the gain
    md += ["### Концентрація виграшу", "", "| епізод | виграш, млрд | частка топ-5 ходів | частка ходів тижнів 1-6 |", "|---|---|---|---|"]
    for n, (_, _, L) in enumerate(data):
        g = np.array(sorted((m["gain"] for m in L["moves"]), reverse=True)) / 1e11
        e = sum(m["gain"] for m in L["moves"] if m["week"] <= 6) / 1e11
        md.append(f"| {n} | {g.sum():.1f} | {g[:5].sum() / g.sum():.2f} | {e / g.sum():.2f} |")
    md.append("")
    text = "\n".join(md) + "\n"
    old = ""
    if os.path.exists(REPORT):
        with open(REPORT) as f:
            s = f.read()
        if MARK in s:
            old = s[s.index(MARK) :]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(text + ("\n" + old if old else ""))
    print(text)


# ----- open-loop probes: fixed rule-like transforms of the hybrid agent's played flows -------------------------------------
def probe_transforms(inst):
    """Name -> function(flows list[dict], T) -> flows list[dict]; transforms of fuel slots by kind / fuel / week band."""
    fs = F.fuel_slots(inst)
    lead = {}
    for s in fs:
        e, k, lane = inst.action_slots[s]
        lead[s] = sum(inst.edges[x].tau for x in (inst.lanes[lane].edges if lane is not None else [e]))

    def scale(kind, fuel, lo, hi, f):
        def tr(flows, T):
            out = []
            for t, fl in enumerate(flows, start=1):
                if lo <= t <= hi:
                    fl = {s: (q * f if s in fs and fs[s] == (fuel, kind) else q) for s, q in fl.items()}
                out.append(fl)
            return out
        return tr

    def cutoff(fuel, margin):  # drop orders that cannot arrive by T - margin
        def tr(flows, T):
            return [{s: q for s, q in fl.items() if not (s in fs and fs[s][1] == "order" and fs[s][0] in fuel and t + lead[s] > T - margin)}
                    for t, fl in enumerate(flows, start=1)]
        return tr

    def dump(fuel, k):  # endgame: every terminal -> grid valve of the fuel requests "everything" in the last k weeks
        def tr(flows, T):
            return [{**fl, **{s: 1e9 for s in fs if fs[s] == (fuel, "valve")}} if t > T - k else fl for t, fl in enumerate(flows, start=1)]
        return tr

    def stop(fuel, k):  # endgame: no orders of the fuel in the last k weeks
        def tr(flows, T):
            return [{s: q for s, q in fl.items() if not (fs.get(s) == (fuel, "order"))} if t > T - k else fl for t, fl in enumerate(flows, start=1)]
        return tr

    T = inst.T
    q = T // 4
    bands = [(1, q), (q + 1, 2 * q), (2 * q + 1, 3 * q), (3 * q + 1, T)]
    tr = {}
    for kind in ("order", "valve"):
        for fuel in ("lng", "crude"):
            for lo, hi in bands + [(1, 3), (1, 6), (1, T)]:
                for f in (0.5, 1.25, 1.5, 2.0):
                    tr[f"{kind} {fuel} w{lo}-{hi} x{f}"] = scale(kind, fuel, lo, hi, f)
    for fuel in ("lng", "crude"):
        for k in (1, 2, 3, 4, 6, 8, 13):
            tr[f"endgame dump {fuel} valves last {k}w"] = dump(fuel, k)
            tr[f"endgame stop {fuel} orders last {k}w"] = stop(fuel, k)
    for fuel, nm in ((("lng",), "lng"), (("crude",), "crude"), (("lng", "crude"), "lng+crude")):
        for m in (0, 2, 4, 6, 8):
            tr[f"cutoff {nm} arrival>T-{m}"] = cutoff(fuel, m)
    return tr


def probe(episodes=12):
    """Replays of fixed transforms of the hybrid agent's flows on every episode; mean J change and sign consistency."""
    import time

    t0 = time.time()
    worlds = []
    for n in range(episodes):
        d = F.play(n)
        inst, _omega, marks = F.world(n)
        acts, _ = F.wire_actions(d, inst, marks)
        worlds.append(F.FReplay(inst, marks, acts))
    trs = probe_transforms(worlds[0].inst)
    rows = []
    for name, f in trs.items():
        dj = np.array([(R._run(1, f(R.flows, R.T), keep=False) - R.J) / 1e11 for R in worlds])  # + is worse (cost up)
        rows.append((name, -dj.mean(), -np.median(dj), int((dj < -1e-6).sum()), int((dj > 1e-6).sum())))
    rows.sort(key=lambda r: -r[1])
    rows = rows[:60] + [r for r in rows[60:] if r[0].startswith(("endgame", "cutoff"))]
    md = ["## Відкриті проби: фіксовані перетворення зіграних потоків агента (replay, без реакції агента)", "",
          f"Виграш = зменшення J, млрд USD/епізод (середнє по {episodes} епізодах); + / − = к-сть епізодів, де J впав / зріс. "
          "Перетворення множать наявні відправки (не створюють нових), тож це перший порядок ефекту правила, "
          "без реакції закритого циклу.", "", "| перетворення | виграш середнє | медіана | + | − |", "|---|---|---|---|---|"]
    md += [f"| {n} | {m:+.2f} | {md_:+.2f} | {p} | {ng} |" for n, m, md_, p, ng in rows]
    text = "\n".join(md) + "\n"
    with open(os.path.join(OUT, "lsf_probe.md"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, "lsf_probe.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transform", "gain_mean_bn", "gain_median_bn", "n_better", "n_worse"])
        w.writerows(rows)
    print(text)
    print(f"[{time.time() - t0:.0f}s, {len(trs)} transforms x {episodes} episodes]")


if __name__ == "__main__":
    fire.Fire({"run": run, "probe": probe})
