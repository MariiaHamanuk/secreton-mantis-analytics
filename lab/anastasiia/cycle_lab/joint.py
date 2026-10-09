"""What the hull's tilt and the search over sets of whole weeks do to each other, read from play records.

Nothing is played here: ``play.py run`` wrote the records, this reads their week notes ("whole 2 of 1 ... search 8
took earlier:0.42") and the runs of the solver kept beside them.

    uv run python lab/anastasiia/regime_lab/joint.py asked j_base j_tilt j_search j_both --task=small
    uv run python lab/anastasiia/regime_lab/joint.py moves j_search j_both --task=small
    uv run python lab/anastasiia/regime_lab/joint.py same j_base j_tilt --task=small
    uv run python lab/anastasiia/regime_lab/joint.py played j_base j_tilt --task=full
    uv run python lab/anastasiia/regime_lab/joint.py cells j_base j_tilt j_search j_both --task=small
    uv run python lab/anastasiia/regime_lab/joint.py acted j_base j_search --task=full --episodes=32
"""

from __future__ import annotations

import pickle
import re
from collections import Counter

import fire
import numpy as np
import play


NOTE = re.compile(r"whole (\d+) of (\d+)(?: \S+)?(?: search (\d+) took (\S+))?")


def _load(tag: str, task: str, entropy: int) -> dict:
    return pickle.loads(play._path(tag, task, entropy).read_bytes())


def _weeks(ep: dict) -> list[dict]:
    """A record's weeks: whole weeks asked by the rounding, those of the played plan, the search's tries and moves."""
    out = []
    for line, detail in zip(ep["log"], ep["detail"]):
        m = NOTE.search(str(line[-1]))
        tries = [s["what"][7:] for s in detail["solves"] if s["what"].startswith("search ")]
        week = {"hull": any(s["what"] == "hull" for s in detail["solves"]), "tries": tries, "took": []}
        if m:
            week.update(closed=int(m.group(1)), asked=int(m.group(2)), searched=m.group(3) is not None)
            if m.group(4) and m.group(4) != "0":
                week["took"] = [
                    (kind, float(gain)) for kind, gain in (move.split(":") for move in m.group(4).split(","))
                ]
        out.append(week)
    return out


def _common(tags: tuple, task: str, entropy: int, episodes: int) -> tuple[dict, list]:
    data = {tag: _load(tag, task, entropy) for tag in tags}
    ns = [n for n in range(episodes) if all(n in d for d in data.values())]
    return data, ns


def _parted(a: dict, b: dict) -> int:
    """The first week (from 0) two records of an episode send different flows; the number of weeks when none."""
    scale = max(1.0, float(np.abs(a["sent"]).max()))
    apart = np.abs(a["sent"] - b["sent"]).max(axis=1) > 1e-6 * scale
    return int(np.argmax(apart)) if apart.any() else len(apart)


def asked(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 64) -> None:
    """Whole weeks the rounding asks for a week, and how often the plan left one of them short (seen only where the
    search ran: its first try is then "unshort")."""
    data, ns = _common(tags, task, entropy, episodes)
    print(f"{task}, root {entropy}, {len(ns)} episodes; weeks with the hull solved")
    for tag in tags:
        weeks = [w for n in ns for w in _weeks(data[tag][n])]
        hull = [w for w in weeks if w["hull"] and "asked" in w]
        a = np.array([w["asked"] for w in hull])
        line = (
            f"{tag:10s} hull in {len(hull) / len(weeks):5.1%} of weeks; asked a week {a.mean():.3f}, "
            f"none in {np.mean(a == 0):5.1%}, 3 or more in {np.mean(a >= 3):5.1%}"
        )
        ran = [w for w in hull if w["tries"]]
        if ran:
            with_asked = [w for w in ran if w["asked"] > 0]
            short = [w for w in with_asked if w["tries"][0] == "unshort"]
            c = np.array([w["closed"] for w in ran])
            line += (
                f"; search ran in {len(ran) / len(weeks):5.1%} of weeks, an asked week left short in "
                f"{len(short) / max(1, len(with_asked)):5.1%} of those with one; "
                f"whole weeks after the search {c.mean():.3f}"
            )
        print(line)


def moves(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 64) -> None:
    """The moves the search takes: how often by kind, and what the plans say they save (bn USD an episode)."""
    data, ns = _common(tags, task, entropy, episodes)
    print(f"{task}, root {entropy}, {len(ns)} episodes; moves taken, and the plan's own saving in bn USD an episode")
    table, kinds = {}, Counter()
    for tag in tags:
        weeks = [w for n in ns for w in _weeks(data[tag][n])]
        ran = [w for w in weeks if w["tries"]]
        count, gain, tried = Counter(), Counter(), Counter()
        for w in ran:
            tried.update(set(w["tries"]))
            for kind, g in w["took"]:
                count[kind] += 1
                gain[kind] += g
        kinds.update(gain)
        table[tag] = (count, gain, tried, len(ran), len(weeks))
        took = sum(1 for w in ran if w["took"])
        tries = np.mean([len(w["tries"]) for w in ran]) if ran else 0.0
        print(
            f"{tag:10s} search ran in {len(ran)} of {len(weeks)} weeks, tries a week {tries:.1f}, "
            f"took a move in {took / max(1, len(ran)):5.1%}, saving {sum(gain.values()) / len(ns):.2f} bn an episode"
        )
    for kind, _ in kinds.most_common():
        cols = []
        for tag in tags:
            count, gain, tried, ran, _weeks_all = table[tag]
            cols.append(
                f"{tag}: tried {tried[kind] / max(1, ran):5.1%}, taken {count[kind] / max(1, ran):5.1%}, "
                f"{gain[kind] / len(ns):6.2f} bn"
            )
        print(f"  {kind:10s} " + "   ".join(cols))


def same(base: str, other: str, task: str = "small", entropy: int = 111, episodes: int = 64) -> None:
    """Does ``other`` ask other whole weeks than ``base`` from the same state? Until the first week the two send
    different flows the state is the same, so there the counts of asked weeks can be set side by side."""
    data, ns = _common((base, other), task, entropy, episodes)
    first, differ, total, more, fewer, week1 = [], 0, 0, 0, 0, 0
    for n in ns:
        a, b = data[base][n], data[other][n]
        wa, wb = _weeks(a), _weeks(b)
        split = _parted(a, b)
        first.append(split)
        for t in range(min(split + 1, len(wa))):  # week ``split`` is still planned from the same state
            if "asked" not in wa[t] or "asked" not in wb[t] or not (wa[t]["hull"] and wb[t]["hull"]):
                continue
            total += 1
            differ += wa[t]["asked"] != wb[t]["asked"]
            more += wb[t]["asked"] > wa[t]["asked"]
            fewer += wb[t]["asked"] < wa[t]["asked"]
            week1 += t == 0 and wa[t]["asked"] != wb[t]["asked"]
    first = np.array(first)
    weeks = len(data[base][ns[0]]["sent"])
    print(f"{task}, root {entropy}, {len(ns)} episodes: {other} against {base}")
    print(
        f"  the flows part in week (1 = at once): median {np.median(first) + 1:.0f}, quartiles "
        f"{np.percentile(first, 25) + 1:.0f} and {np.percentile(first, 75) + 1:.0f}; "
        f"never in {int((first >= weeks).sum())} episodes, in week 1 in {int((first == 0).sum())}"
    )
    print(
        f"  weeks planned from the same state: {total}; the count of asked weeks differs in {differ} "
        f"({differ / max(1, total):.1%}): more in {more}, fewer in {fewer}; "
        f"in week 1 it differs in {week1} of {len(ns)} episodes"
    )
    J = np.array([[data[tag][n]["J"] for n in ns] for tag in (base, other)], dtype=float)
    print(f"  episodes with the same cost to the cent: {int((J[0] == J[1]).sum())} of {len(ns)}")


def played(*tags: str, task: str = "small", entropy: int = 111, episodes: int = 64) -> None:
    """What was played: weeks of a grid with its base load short, and when the lots were started."""
    data, ns = _common(tags, task, entropy, episodes)
    print(f"{task}, root {entropy}, {len(ns)} episodes; played weeks")
    for tag in tags:
        short = shed = lots = when = 0.0
        for n in ns:
            ep = data[tag][n]
            s, made = np.asarray(ep["shed"], dtype=float), np.asarray(ep["lots"], dtype=float)
            short += (s > 1e-6 * max(1.0, s.max())).sum()
            shed += s.sum()
            lots += made.sum()
            when += (made.sum(axis=tuple(range(1, made.ndim))) * np.arange(1, len(made) + 1)).sum()
        print(
            f"{tag:10s} grid-weeks with base load shed {short / len(ns):6.1f} an episode, shed {shed / len(ns):10.1f}, "
            f"lots {lots / len(ns):9.1f}, mean week of a lot {when / max(lots, 1e-9):5.2f}"
        )


def cells(
    base: str,
    one: str,
    two: str,
    both: str,
    task: str = "small",
    entropy: int = 111,
    episodes: int = 64,
    draws: int = 4000,
) -> None:
    """Two levers in a 2x2: what each adds alone, what each adds on top of the other, and whether they add up.

    The score is the board's (levels weighing 50/30/15/5 % when all four are there), the 90 % intervals come from
    drawing episodes again inside their harm levels, the same draw for every cell."""
    import plan
    from package_baselines import WEIGHTS

    tags = (base, one, two, both)
    data, ns = _common(tags, task, entropy, episodes)
    refs = plan.references(task, entropy, max(ns) + 1)
    ns = [n for n in ns if refs[n]["J_oracle_cents"] is not None]
    level = np.array([refs[n]["stratum"] for n in ns])
    naive = np.array([refs[n]["J_naive_cents"] for n in ns], dtype=float)
    room = naive - np.array([refs[n]["J_oracle_cents"] for n in ns], dtype=float)
    saved = {tag: naive - np.array([data[tag][n]["J"] for n in ns], dtype=float) for tag in tags}
    levels = sorted(set(level))
    weight = {s: WEIGHTS[s] for s in levels} if set(levels) == set(WEIGHTS) else dict.fromkeys(levels, 1.0)
    where = {s: np.flatnonzero(level == s) for s in levels}

    def scores(pick: dict) -> np.ndarray:
        under = sum(weight[s] * room[pick[s]].mean() for s in levels)
        return np.array([sum(weight[s] * saved[tag][pick[s]].mean() for s in levels) / under for tag in tags])

    def lines(s: np.ndarray) -> np.ndarray:  # the cells, then the differences read from them
        return np.concatenate(
            [s, [s[1] - s[0], s[2] - s[0], s[3] - s[2], s[3] - s[1], s[3] - s[0], s[3] - s[1] - s[2] + s[0]]]
        )

    rng = np.random.default_rng(0)
    here = lines(scores(where))
    drawn = np.array([lines(scores({s: rng.choice(where[s], len(where[s])) for s in levels})) for _ in range(draws)])
    lo, hi = np.percentile(drawn, [5, 95], axis=0)
    names = [
        *tags,
        f"{one} over {base}",
        f"{two} over {base}",
        f"{both} over {two}",
        f"{both} over {one}",
        f"{both} over {base}",
        "the two together less their sum",
    ]
    print(f"{task}, root {entropy}, {len(ns)} episodes ({', '.join(f'level {s}: {len(where[s])}' for s in levels)})")
    for i, name in enumerate(names):
        sign = "" if i < 4 else "+"
        print(f"  {name:32s} {here[i]:{sign}.4f} ({lo[i]:{sign}.4f} to {hi[i]:{sign}.4f})")
    d1, d2 = saved[one] - saved[base], saved[two] - saved[base]
    print(
        f"  by episode: {one} cheaper than {base} in {int((d1 > 0).sum())}, {two} in {int((d2 > 0).sum())}, both in "
        f"{int(((d1 > 0) & (d2 > 0)).sum())}; correlation of the two savings {np.corrcoef(d1, d2)[0, 1]:+.2f}"
    )
    for s in levels:
        print(
            f"  level {s}: "
            + "  ".join(f"{tag} {saved[tag][where[s]].sum() / room[where[s]].sum():.4f}" for tag in tags)
        )


def acted(
    base: str, search: str, task: str = "full", entropy: int = 111, episodes: int = 32, draws: int = 4000
) -> None:
    """Is a difference between a run with the search and one without it the search's doing?

    With the clock two runs of one version may part by themselves (a solve stopped by its limit in one and not in
    the other). Where the flows part before the search ran for the first time, the search cannot be the cause: those
    episodes show the clock's own scatter, the rest what the search did. Saved cents over the room between the naive
    rule and the clairvoyant plan, the levels not weighed; 90 % intervals from drawing the group's episodes again."""
    import plan

    data, ns = _common((base, search), task, entropy, episodes)
    refs = plan.references(task, entropy, max(ns) + 1)
    ns = [n for n in ns if refs[n]["J_oracle_cents"] is not None]
    room = np.array([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns], dtype=float)
    saved = np.array([data[base][n]["J"] - data[search][n]["J"] for n in ns], dtype=float)
    parted = np.array([_parted(data[base][n], data[search][n]) for n in ns])
    weeks = [_weeks(data[search][n]) for n in ns]
    first = np.array([next((t for t, w in enumerate(ws) if w["tries"]), len(ws)) for ws in weeks])
    told = np.array([sum(gain for w in ws for _, gain in w["took"]) for ws in weeks]) * 1e11
    clock = parted < first
    rng = np.random.default_rng(0)
    print(f"{task}, root {entropy}, {len(ns)} episodes: {search} against {base}")
    for name, group in (
        ("parted before the search ran", clock),
        ("the search parted them", ~clock),
        ("all", clock | ~clock),
    ):
        idx = np.flatnonzero(group)
        if not len(idx):
            continue
        drawn = [saved[i].sum() / room[i].sum() for i in (rng.choice(idx, len(idx)) for _ in range(draws))]
        lo, hi = np.percentile(drawn, [5, 95])
        print(
            f"  {name:28s} {len(idx):3d} episodes, {saved[idx].sum() / room[idx].sum():+.4f} ({lo:+.4f} to {hi:+.4f}) "
            f"of their room, {saved[idx].sum() / room.sum():+.4f} of the set's; "
            f"cheaper in {int((saved[idx] > 0).sum())}; "
            f"{saved[idx].sum() / 1e11:+.1f} bn against {told[idx].sum() / 1e11:.1f} bn the plans told"
        )
    print(
        "  parted before the search ran (episode: week parted, week of the first search): "
        + ", ".join(f"{ns[i]}: {parted[i] + 1}, {first[i] + 1}" for i in np.flatnonzero(clock))
    )


if __name__ == "__main__":
    fire.Fire({"asked": asked, "moves": moves, "same": same, "played": played, "cells": cells, "acted": acted})
