"""Statistics of the disruption generator, for rules that read the observation: what happens, how often, for how long,
and how far the three early signals (messages, pending prohibitions, warning scores) can be trusted.

    uv run python lab/anastasiia/stats_lab/event_stats.py
    uv run python lab/anastasiia/stats_lab/event_stats.py --task=small --episodes=2000

It draws scenarios of a root of your own from the public generator and reads them on the trusted side (the events, the
decoy flag of every announcement thread, the weekly marks), so nothing is played and no agent is involved. What an
agent sees of this is the ``standard`` regime's: the warning score of week t, the threads shown by week t, and
``graph_now.*`` at the instant t - 1.

Writes ``summary.md`` and the raw tables (``events.csv``, ``messages.csv``, ``spells.csv``, ``weekly.npz``) under
``outputs/event_stats/<date_time>/``.
"""

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


CLOSED, OPEN = 0.01, 0.999  # a strait's week counts as closed at o <= CLOSED and as disrupted at o < OPEN
AGES = (0, 1, 2, 4, 8, 13, 26)  # weeks since a thread's first message
HORIZONS = (1, 4, 8, 13)  # weeks ahead a warning score is judged on
SCORE_BINS = (-np.inf, -1.0, 0.0, 1.0, 2.0, np.inf)
SPELL_DAYS = (1, 2, 3, 4, 6, 8, 12, 20)
QS = (0.1, 0.25, 0.5, 0.75, 0.9)
RAW = ("events", "messages", "units")  # an episode's entries that are not weekly arrays


_SETUP: dict = {}  # per process: a cached function would not pickle into the workers


def _setup(task: str):
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.information.theta import resolve_regime
    from shockbench_flow.marks import event_free_marks

    if task not in _SETUP:
        inst, params = task_generator(task)
        _SETUP[task] = (inst, params, resolve_regime("standard"), event_free_marks(inst))
    return _SETUP[task]


def episode(task: str, entropy: int, n: int) -> dict:
    """Everything the statistics read of episode n: its events, its shown message threads and its weekly marks."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.information import messages, warning
    from shockbench_flow.information.theta import a_by_kind
    from shockbench_flow.marks import compute_marks, read_events

    inst, params, theta, calm = _setup(task)
    omega = sample_omega(inst, params, entropy, n, "train")
    arrays = omega.arrays
    events = read_events(inst, omega)
    marks = compute_marks(inst, omega)
    lead = np.asarray(arrays["ev_lead"])
    ev_rows = [
        (n, e.type, e.target_kind, e.target, e.commodity, e.region, e.counterpart, e.onset, e.duration, e.severity)
        + (e.rate, *lead[e.row])
        for e in events
    ]
    phi_bar = dict(json.loads(str(arrays["meta_information_params"]))["phi_bar"])
    shown = messages.shown(messages.announcements(inst, omega, events), theta, omega, phi_bar)
    msg_rows = [
        (n, a.thread, int(a.decoy), a.event.type, a.target_kind, a.target, a.region, a.channel, a.kind)
        + (a.instant, a.effect)
        for a in shown
    ]
    units = warning.unit_table(inst, omega)
    a = np.array([a_by_kind(theta)[u.kind] for u in units])
    score = warning.warning_scores(np.asarray(arrays["X"]), np.asarray(arrays["W"]), a, theta.L, omega.burn_in, inst.T)
    finite = np.isfinite(calm.u[0])
    return {
        "events": ev_rows,
        "messages": msg_rows,
        "o": marks.o.astype(np.float32),
        "o_now": marks.o_now.astype(np.float32),
        "wr": marks.wr_class.astype(np.int8),
        "banned": (marks.prohibited & ~calm.prohibited).any(axis=2),  # (T, E): a prohibition the calm plan lacks
        "tariff": (marks.tariff - calm.tariff).max(axis=2).astype(np.float32),  # (T, E): the largest added rate
        "u": np.where(finite, marks.u / np.where(finite, calm.u, 1.0), 1.0).astype(np.float32),
        "c": (marks.c / np.where(calm.c > 0, calm.c, 1.0)).astype(np.float32),
        "fab": marks.R.astype(np.float32),
        "osat": marks.R_osat.astype(np.float32),
        "grid": (marks.G_bar / calm.G_bar).astype(np.float32),
        "score": np.asarray(score, dtype=np.float32),
        "units": [(u.kind, u.unit) for u in units],
        # (T, R): the region's own conflict chain is not at peace in week t
        "conflict": np.asarray(arrays["z_c_own"])[:, omega.burn_in + 1 : omega.burn_in + inst.T + 1].T > 0,
    }


# ----- helpers ----------------------------------------------------------------------------------------------------
def table(header: list[str], rows: list[list]) -> str:
    def cell(x) -> str:
        if isinstance(x, float):
            return "-" if math.isnan(x) else f"{x:.2f}"
        return str(x)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]) + "\n"


def quantiles(x) -> list[float]:
    x = np.asarray(x, dtype=float)
    return [float("nan")] * len(QS) if x.size == 0 else [float(q) for q in np.quantile(x, QS)]


def spells(bad: np.ndarray) -> list[tuple[int, int, bool]]:
    """The runs of True of a (T,) series as (first week, 1-based; length; censored: it still runs in the last week)."""
    out, start = [], None
    for t, b in enumerate(bad):
        if b and start is None:
            start = t
        if not b and start is not None:
            out.append((start + 1, t - start, False))
            start = None
    if start is not None:
        out.append((start + 1, len(bad) - start, True))
    return out


def survival(lengths: np.ndarray, censored: np.ndarray, d: int) -> float:
    """P(a spell that has lasted d weeks lasts one more), over the spells seen for at least d weeks."""
    at_risk = (lengths > d) | ((lengths == d) & ~censored)
    return float("nan") if at_risk.sum() < 20 else float((lengths > d)[at_risk].mean())


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="stable")
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def ahead(flag: np.ndarray, h: int) -> np.ndarray:
    """(N, T, ...) -> True at week t where ``flag`` holds in any of the weeks t .. t + h - 1."""
    out = np.zeros_like(flag)
    for s in range(h):
        out[:, : flag.shape[1] - s] |= flag[:, s:]
    return out


def spell_section(name: str, units: list[str], bad: np.ndarray, depth: np.ndarray) -> str:
    """One table per unit of a (N, T, U) disruption flag: how often, how long, how deep, and how it persists."""
    n_ep = bad.shape[0]
    rows, every_len, every_cens = [], [], []
    for j, unit in enumerate(units):
        runs = [s for i in range(n_ep) for s in spells(bad[i, :, j])]
        if not runs:
            continue
        lengths = np.array([s[1] for s in runs])
        cens = np.array([s[2] for s in runs])
        every_len.append(lengths)
        every_cens.append(cens)
        rows.append(
            [unit, 100 * float(bad[:, :, j].any(axis=1).mean()), 100 * float(bad[:, :, j].mean())]
            + [100 * float(bad[:, 0, j].mean()), len(runs) / n_ep, *quantiles(lengths)[1:4]]
            + [100 * float(cens.mean()), float(np.median(depth[:, :, j][bad[:, :, j]]))]
        )
    rows.sort(key=lambda r: -r[1])
    head = ["unit", "episodes hit %", "weeks hit %", "hit in week 1 %", "spells / episode", "weeks p25", "p50", "p75"]
    text = f"### {name}\n\n" + table(head + ["runs to the end %", "median level when hit"], rows[:25])
    if every_len:
        lengths, cens = np.concatenate(every_len), np.concatenate(every_cens)
        keep = [[d, survival(lengths, cens, d), int((lengths >= d).sum())] for d in SPELL_DAYS]
        text += "\nPersistence, all units pooled:\n\n"
        text += table(["hit for d weeks", "P(still hit next week)", "spells"], keep)
    return text + "\n"


# ----- the report ----------------------------------------------------------------------------------------------------
def report(task: str, entropy: int, eps: list[dict]) -> tuple[str, list, list, list]:
    from shockbench_flow.information import messages as msg
    from shockbench_flow.omega import codes

    inst, _, theta, _ = _setup(task)
    n_ep, T = len(eps), inst.T
    stack = {k: np.stack([e[k] for e in eps]) for k in eps[0] if k not in RAW}
    events = [r for e in eps for r in e["events"]]
    messages = [r for e in eps for r in e["messages"]]
    node = [x.id for x in inst.nodes]
    edge = [x.id for x in inst.edges]
    straits = [node[c] for c in inst.chokepoints]

    def target(kind: int, i: int) -> str:
        return (node, edge, node, inst.regions)[kind][i]

    out = [f"# Disruption statistics: {task}, {n_ep} episodes of root {entropy}, {T} weeks each\n"]
    out.append(
        "Drawn from the public generator and read on the trusted side; the signals are the `standard` regime's.\n"
        "Weeks are 1-based as `observation['week']`. Written by `lab/anastasiia/stats_lab/event_stats.py`.\n"
    )

    # 1. events
    out.append("## 1. Events by type\n")
    out.append(
        "`carried in` started before week 1 and still acts in the episode; `new` starts inside it. Durations of new\n"
        "events are the generator's, in weeks, not cut at the episode's end.\n"
    )
    rows = []
    for ty, name in enumerate(codes.EVENT_TYPES):
        mine = [r for r in events if r[1] == ty]
        new = [r for r in mine if r[7] >= 0]
        per_ep = Counter(r[0] for r in new)
        dur = quantiles([r[8] for r in new])
        rows.append(
            [name, (len(mine) - len(new)) / n_ep, len(new) / n_ep, 100 * len(per_ep) / n_ep]
            + [dur[1], dur[2], dur[3], dur[4], float(np.median([r[9] for r in new])) if new else float("nan")]
        )
    head = ["type", "carried in / episode", "new / episode", "episodes with a new one %"]
    out.append(table(head + ["weeks p25", "p50", "p75", "p90", "median severity"], rows))

    out.append("### Where new events land\n")
    rows = []
    for ty, name in enumerate(codes.EVENT_TYPES):
        hits = Counter((r[2], r[3]) for r in events if r[1] == ty and r[7] >= 0)
        total = sum(hits.values())
        top = ", ".join(f"{target(*key)} {100 * c / total:.0f}%" for key, c in hits.most_common(6))
        if total:
            rows.append([name, len(hits), top])
    out.append(table(["type", "distinct targets", "most frequent (share of the type's new events)"], rows))

    out.append("### When new events start\n")
    edges_w = np.linspace(0, T, 5)
    rows = []
    for ty, name in enumerate(codes.EVENT_TYPES):
        onset = np.array([r[7] for r in events if r[1] == ty and r[7] >= 0])
        if onset.size:
            rows.append([name] + [100 * float(x) for x in np.histogram(onset, edges_w)[0] / onset.size])
    quarters = [f"weeks {int(a) + 1}-{int(b)} %" for a, b in zip(edges_w[:-1], edges_w[1:])]
    out.append(table(["type"] + quarters, rows))

    # 2. the network's weekly state
    out.append("## 2. What the network looks like week by week\n")
    out.append(
        "`hit` means the week differs from the calm plan. `runs to the end` is the share of spells still running in\n"
        "the last week. The level is the open fraction, the capacity ratio or the restoration factor (1 is normal).\n"
    )
    out.append(spell_section("Straits (`graph_now.open` < 1)", straits, stack["o"] < OPEN, stack["o"]))
    full = (stack["o"] <= CLOSED).sum() / max((stack["o"] < OPEN).sum(), 1)
    out.append(f"Of the strait-weeks below 1, {100 * full:.0f}% are fully closed (open fraction 0).\n")
    n_closed = (stack["o"] < OPEN).sum(axis=2).ravel()
    share = ", ".join(f"{k}: {100 * float((n_closed == k).mean()):.1f}%" for k in range(4))
    out.append(f"Straits disrupted in the same week (share of weeks): {share}.\n")
    risk = ", ".join(f"class {k}: {100 * float((stack['wr'] == k).mean()):.1f}%" for k in (1, 2))
    out.append(f"War-risk surcharge in force (share of strait-weeks): {risk}.\n")
    out.append(
        spell_section(
            "New prohibitions per edge (`graph_now.prohibited`)", edge, stack["banned"], stack["banned"] * 1.0
        )
    )
    out.append(spell_section("Capacity cuts per edge (`graph_now.u` < nominal)", edge, stack["u"] < OPEN, stack["u"]))
    levels = [(1.0, "1"), (0.25, "0.25"), (0.0625, "0.06"), (0.0, "below 0.06")]
    cut_u = stack["u"][stack["u"] < OPEN]
    parts, top = [], 1.0
    for level, name in levels[1:]:
        parts.append(f"{name}: {100 * float(((cut_u < top - 1e-3) & (cut_u >= level - 1e-3)).mean()):.0f}%")
        top = level
    out.append(
        f"Capacity is below nominal in {100 * float((stack['u'] < OPEN).mean()):.0f}% of edge-weeks; of those, by "
        + "level: "
        + ", ".join(parts)
        + ". A sanction on one commodity cuts the other edges between the same two regions to 0.25 of their\n"
        "capacity, and the cuts multiply (0.25, 0.06, 0.016, ...). A closed strait cuts its lanes too.\n"
    )
    costly = stack["c"] > 1.001
    out.append(spell_section("Freight surcharges per edge (`graph_now.c` > nominal)", edge, costly, stack["c"]))
    taxed = stack["tariff"] > 1e-9
    out.append(spell_section("Added tariffs per edge (`graph_now.tariff`)", edge, taxed, stack["tariff"]))
    fabs = [node[i] for i in inst.fabs]
    out.append(spell_section("Fabs (`graph_now.fab.R` < 1)", fabs, stack["fab"] < OPEN, stack["fab"]))
    osats = [node[i] for i in inst.osats]
    out.append(spell_section("OSATs (`graph_now.osat.R` < 1)", osats, stack["osat"] < OPEN, stack["osat"]))
    grids = [node[i] for i in inst.grids]
    out.append(spell_section("Grids (`graph_now.grid.G_bar` < nominal)", grids, stack["grid"] < OPEN, stack["grid"]))

    # 3. messages
    out.append("## 3. Messages: how often a thread is real, and how much time it gives\n")
    out.append(
        "A thread is one event's announcements. A decoy is withdrawn at the moment it would have taken effect, so\n"
        "until then it looks like a real one. Only threads whose effect falls inside the episode are ever shown.\n"
        f"The regime publishes its decoy shares in `config['regime']['phi']`: {theta_phi(theta)}.\n"
    )
    threads: dict[tuple, dict] = {}
    for n, thread, decoy, ty, tk, tg, region, channel, kind, instant, effect in messages:
        th = threads.setdefault((n, thread), {"decoy": bool(decoy), "type": ty, "effect": effect, "msgs": []})
        if kind != msg.WITHDRAWAL:
            th["msgs"].append((instant, channel))
    for th in threads.values():
        th["msgs"].sort()
        th["first"], th["channel"] = th["msgs"][0]
    by_channel = defaultdict(list)
    for th in threads.values():
        by_channel[th["channel"]].append(th)
    rows = []
    for channel, group in sorted(by_channel.items()):
        real = [th for th in group if not th["decoy"]]
        fake = [th for th in group if th["decoy"]]
        lead = quantiles([msg.week_of(th["effect"]) - msg.week_of(th["first"]) for th in real])
        wait = quantiles([msg.week_of(th["effect"]) - msg.week_of(th["first"]) for th in fake])
        rows.append(
            [codes.CHANNELS[channel], len(group) / n_ep, 100 * len(fake) / len(group)]
            + [lead[0], lead[1], lead[2], lead[3], lead[4], wait[2]]
        )
    head = ["first message", "threads / episode", "decoys %", "real: weeks to effect p10", "p25", "p50", "p75", "p90"]
    out.append(table(head + ["decoy: weeks to withdrawal p50"], rows))

    out.append("### P(real) of a thread still open a weeks after its first message\n")
    rows = []
    for channel, group in sorted(by_channel.items()):
        row = [codes.CHANNELS[channel]]
        for age in AGES:
            alive = [th for th in group if th["effect"] > th["first"] + age]
            row.append(float("nan") if len(alive) < 30 else 100 * sum(not th["decoy"] for th in alive) / len(alive))
        rows.append(row)
    out.append(table(["first message"] + [f"a = {a} %" for a in AGES], rows))

    out.append("### Follow-up messages (a dated notice after the first message)\n")
    rows = []
    for name, first, follow in (("sanction: legal publication", msg.TIES, msg.LEGAL), ("tariff: final notice", -1, 2)):
        group = [th for th in threads.values() if (th["channel"] == first or (first < 0 and th["channel"] in (0, 1)))]
        dated = [th for th in group if any(ch == follow for _, ch in th["msgs"])]
        solo = [th for th in threads.values() if th["channel"] == follow]
        gap = [
            msg.week_of(th["effect"]) - msg.week_of(min(i for i, ch in th["msgs"] if ch == follow))
            for th in dated + solo
            if not th["decoy"]
        ]
        q = quantiles(gap)
        rows.append(
            [name, 100 * len(dated) / max(len(group), 1), 100 * sum(th["decoy"] for th in dated) / max(len(dated), 1)]
            + [len(solo) / n_ep, 100 * sum(th["decoy"] for th in solo) / max(len(solo), 1), q[1], q[2], q[3]]
        )
    head = ["follow-up", "threads that get one %", "decoys among them %", "threads opening with it / episode"]
    out.append(table(head + ["decoys among those %", "real: weeks from it to effect p25", "p50", "p75"], rows))

    out.append("### Pending prohibitions (`pending_prohibitions.*`)\n")
    out.append(
        "An entry is a sanction's legal publication dated at least a week ahead. Most real publications come out\n"
        "the week the sanction starts, so they never show up here.\n"
    )
    pending = [
        (th["decoy"], msg.week_of(th["effect"]) - msg.week_of(instant))
        for th in threads.values()
        for instant, channel in th["msgs"]
        if channel == msg.LEGAL and msg.week_of(th["effect"]) > msg.week_of(instant)
    ]
    rows = []
    for name, lo, hi in (
        ("all", 1, 10**6),
        ("1 week", 1, 1),
        ("2-3 weeks", 2, 3),
        ("4-8 weeks", 4, 8),
        ("9+", 9, 10**6),
    ):
        group = [d for d, lead in pending if lo <= lead <= hi]
        rows.append([name, len(group) / n_ep, 100 * sum(group) / max(len(group), 1)])
    out.append(table(["weeks ahead", "publications / episode", "decoys %"], rows))
    q = quantiles([lead for d, lead in pending if not d])
    out.append(f"Weeks ahead of a real one: p25 {q[1]:.0f}, p50 {q[2]:.0f}, p75 {q[3]:.0f}, p90 {q[4]:.0f}.\n")

    out.append("### Share of new events an agent hears of in advance\n")
    announced = defaultdict(lambda: [0, 0, 0])
    for r in events:
        if r[7] < 0:
            continue
        leads = np.array(r[11:], dtype=float)
        weeks = msg.week_of(r[7]) - min([msg.week_of(r[7] - x) for x in leads[np.isfinite(leads)]], default=10**6)
        announced[r[1]][0] += 1
        announced[r[1]][1] += weeks >= 1
        announced[r[1]][2] += weeks >= 4
    rows = [
        [codes.EVENT_TYPES[ty], c[0] / n_ep, 100 * c[1] / c[0], 100 * c[2] / c[0]]
        for ty, c in sorted(announced.items())
    ]
    out.append(table(["type", "new / episode", "announced >= 1 week ahead %", ">= 4 weeks ahead %"], rows))

    # 4. warning scores
    out.append("## 4. Warning scores: what a high score is worth\n")
    out.append(
        f"`warning.score` is {theta.a_chokepoint:g} x the unit's latent risk of {theta.L} week(s) ago plus noise, so\n"
        "it moves slowly and never names an event. Each cell is the share of unit-weeks with that score in which the\n"
        "thing happens within the next h weeks; the last row is the AUROC of the score for it (0.5: no information).\n"
    )
    units = eps[0]["units"]

    def onsets(types: tuple[int, ...], column: int, ids: list[int]) -> np.ndarray:
        """(N, T, len(ids)) True in the week a new event of ``types`` starts, by the id in ``column`` of its row."""
        flag = np.zeros((n_ep, T, len(ids)), dtype=bool)
        where = {x: i for i, x in enumerate(ids)}
        for r in events:
            if r[1] in types and 0 <= r[7] < T and r[column] in where:
                flag[r[0], int(r[7]), where[r[column]]] = True
        return flag

    def score_table(title: str, what: str, score: np.ndarray, label: np.ndarray, mask: np.ndarray) -> str:
        rows = []
        for lo, hi in zip(SCORE_BINS[:-1], SCORE_BINS[1:]):
            inside = mask & (score >= lo) & (score < hi)
            rows.append(
                [f"{lo:g} to {hi:g}", 100 * float(inside.sum() / mask.sum())]
                + [100 * float(ahead(label, h)[inside].mean()) if inside.any() else float("nan") for h in HORIZONS]
            )
        rows.append(["AUROC", ""] + [auroc(score[mask], ahead(label, h)[mask]) for h in HORIZONS])
        head = ["score", "share of unit-weeks %"] + [f"{what} within {h} w %" for h in HORIZONS]
        return f"### {title}\n\n" + table(head, rows)

    hit = stack["o"] < OPEN
    is_open = stack["o_now"] >= OPEN
    cols = [j for j, (kind, _) in enumerate(units) if kind == "chokepoint"]
    score_c = stack["score"][:, :, cols]
    military = onsets((codes.EVENT_TYPES.index("militarised_closure"),), 3, list(inst.chokepoints))
    out.append(score_table("Straits: any disruption, while the strait is open", "disrupted", score_c, hit, is_open))
    out.append(score_table("Straits: a militarised closure starts", "starts", score_c, military, is_open))
    rows = []
    for c, name in enumerate(straits):
        m = is_open[:, :, c]
        high = m & (score_c[:, :, c] >= 1.0)
        row = [name]
        for label in (hit, military):
            far = ahead(label, 13)[:, :, c]
            row += [100 * float(far[m].mean()), 100 * float(far[high].mean()) if high.any() else float("nan")]
            row.append(auroc(score_c[:, :, c][m], far[m]))
        rows.append(row)
    head = ["strait", "disrupted within 13 w: any score %", "score >= 1 %", "AUROC"]
    out.append(table(head + ["militarised closure within 13 w: any score %", "score >= 1 %", "AUROC"], rows))

    cols_r = [j for j, (kind, _) in enumerate(units) if kind == "region"]
    score_r = stack["score"][:, :, cols_r]
    everywhere = np.ones_like(score_r, dtype=bool)
    regions = list(range(len(inst.regions)))
    block_m = tuple(ty for ty, b in enumerate(codes.EVENT_BLOCK) if b == 1)
    block_p = tuple(ty for ty, b in enumerate(codes.EVENT_BLOCK) if b == 0)
    names_m = ", ".join(codes.EVENT_TYPES[ty] for ty in block_m)
    names_p = ", ".join(codes.EVENT_TYPES[ty] for ty in block_p)
    peace = ~stack["conflict"]
    out.append(
        score_table("Regions: the region enters a conflict state", "in conflict", score_r, stack["conflict"], peace)
    )
    out.append(score_table(f"Regions: {names_m}", "new event", score_r, onsets(block_m, 5, regions), everywhere))
    out.append(score_table(f"Regions: {names_p}", "new event", score_r, onsets(block_p, 5, regions), everywhere))

    spell_rows = [
        (i, straits[j], *s, float(stack["o"][i, s[0] - 1 : s[0] - 1 + s[1], j].min()))
        for i in range(n_ep)
        for j in range(len(straits))
        for s in spells(stack["o"][i, :, j] < OPEN)
    ]
    return "\n".join(out), events, messages, spell_rows


def theta_phi(theta) -> str:
    from shockbench_flow.information.theta import phi_by_channel

    return ", ".join(f"{k} {v:g}" for k, v in phi_by_channel(theta).items())


def main(task: str = "tiny", episodes: int = 200, entropy: int = 333, n_jobs: int = -1, out: str | None = None):
    """Draw ``episodes`` scenarios of ``task`` and write the statistics.

    Args:
        task: tiny, small or full.
        episodes: how many scenarios to draw (episodes 0 .. episodes - 1 of the root).
        entropy: the scenarios' root; not 0, the dev episodes' (keep those for confirmation).
        n_jobs: workers (-1: all cores).
        out: the folder to write to (default: outputs/event_stats/<date_time>/).

    """
    from sbf_starter import ROOT, check_task

    check_task(task)
    if entropy == 0:
        raise ValueError("entropy 0 is the dev root: draw the statistics from a root of your own")
    folder = Path(out) if out else ROOT / "outputs" / "event_stats" / time.strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    eps = Parallel(n_jobs=n_jobs)(delayed(episode)(task, entropy, n) for n in range(episodes))
    print(f"{episodes} episodes of {task} drawn in {time.perf_counter() - start:.0f} s")
    text, events, messages, spell_rows = report(task, entropy, eps)
    (folder / "summary.md").write_text(text)
    channels = ("tariff_formal", "tariff_informal", "tariff_final", "sanction_legal", "ties_threat", "mid_threat")
    heads = {
        "events": ("episode", "type", "target_kind", "target", "commodity", "region", "counterpart", "onset_week")
        + ("duration_weeks", "severity", "rate")
        + tuple(f"lead_{c}" for c in channels),
        "messages": ("episode", "thread", "decoy", "type", "target_kind", "target", "region", "channel", "kind")
        + ("instant_week", "effect_week"),
        "spells": ("episode", "strait", "first_week", "weeks", "runs_to_end", "min_open"),
    }
    for name, rows in (("events", events), ("messages", messages), ("spells", spell_rows)):
        with (folder / f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(heads[name])
            writer.writerows(rows)
    keys = [k for k in eps[0] if k not in RAW]
    np.savez_compressed(folder / "weekly.npz", **{k: np.stack([e[k] for e in eps]) for k in keys})
    print(f"written {folder}")


if __name__ == "__main__":
    fire.Fire(main)
