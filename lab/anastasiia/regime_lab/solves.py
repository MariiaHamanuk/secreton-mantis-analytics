"""Where a recorded agent's week goes: every run of the solver, the weeks over budget and the server's meter.

    uv run python lab/anastasiia/regime_lab/solves.py same L_hre1 hull_a --task=full --entropy=111 --episodes=32
    uv run python lab/anastasiia/regime_lab/solves.py table hull_a hull_abc --task=full --entropy=111 --episodes=32
    uv run python lab/anastasiia/regime_lab/solves.py meter L_hre1 hull_a F_t2 --task=full --entropy=111 --episodes=32
    uv run python lab/anastasiia/regime_lab/solves.py claims hull_abc --task=full --entropy=111 --episodes=32

Reads ``play.py``'s records (``outputs/regime_lab/play/<tag>_<task>_<entropy>.pkl``); plays nothing. ``same``: the
episodes two tags played at the same cost to the cent, and the others. ``table``: the runs of the solver by cell,
attempt and status (a record made after 8 October keeps them, ``detail``), the weeks by the number of runs, the weeks
that kept the carried plan because no cell solved. ``meter``: the weeks the server would hand to the naive rule. Its
meter charges what a week takes over the budget to the next week (``shockbench_flow_agent.scoring._metered_shim``),
so a week far over costs several, and near the budget the count runs away; ``speeds`` scales the recorded seconds
(a slower server). An estimate of the count only: after a substituted week the later weeks would differ, here they
are taken as recorded. ``claims``: the cell's claimed cost beside the cost its solution played at, by how the cell
was solved. The seconds are this machine's process time, not the container's.
"""

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import plan  # noqa: E402

BUDGET_S = {"small": 2.0, "full": 4.0}  # the boards' CPU seconds per week
SOLVED = ("Optimal", "Time limit reached")


def _load(tag: str, task: str, entropy: int, episodes: int, first: int) -> dict:
    kept = pickle.loads((plan.OUT / "play" / f"{tag}_{task}_{entropy}.pkl").read_bytes())
    return {n: kept[n] for n in range(first, first + episodes) if n in kept}


def _dist(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"median {np.median(x):5.2f}  p95 {np.percentile(x, 95):5.2f}  p99 {np.percentile(x, 99):5.2f}  max {x.max():6.2f}"


def same(a: str, b: str, task: str = "small", entropy: int = 111, episodes: int = 64, first: int = 0) -> None:
    A, B = _load(a, task, entropy, episodes, first), _load(b, task, entropy, episodes, first)
    ns = sorted(set(A) & set(B))
    other = [n for n in ns if A[n]["J"] != B[n]["J"]]
    print(f"{task}, root {entropy}: {a} and {b} played {len(ns) - len(other)} of {len(ns)} episodes at the same cost to the cent")
    for n in other:
        week = next((w + 1 for w, (x, y) in enumerate(zip(A[n]["costs"], B[n]["costs"])) if not np.array_equal(x, y)), None)
        print(f"  episode {n:3d}: {b} minus {a} {(B[n]['J'] - A[n]['J']) / 1e11:+9.3f} bn USD, the weeks differ from week {week}")


def meter(*tags: str, task: str = "full", entropy: int = 111, episodes: int = 32, first: int = 0,
          speeds: tuple = (0.82, 1.0, 1.15, 1.33, 1.5), to_median: float = 0.0) -> None:
    """``to_median``: every record's seconds are first scaled so that its median week takes this long: records made
    under different loads of the machine side by side (versions whose usual week is the same work)."""
    B = BUDGET_S[task]
    print(f"{task}, root {entropy}: weeks the server's meter would hand to naive, % of weeks (weeks over {B:g} s, %), by the speed of the server against this record's machine")
    print(f"  {'tag':10s} " + "  ".join(f"x{s:4.2f}           " for s in speeds))
    for tag in tags:
        d = _load(tag, task, entropy, episodes, first)
        total = sum(len(e["cpu"]) for e in d.values())
        median = float(np.median(np.concatenate([e["cpu"] for e in d.values()])))
        scale = to_median / median if to_median else 1.0
        cells = []
        for s in speeds:
            handed = over = 0
            for e in d.values():
                carry = 0.0
                for c in np.asarray(e["cpu"]) * s * scale:
                    over += c > B
                    used = carry + c
                    carry = used - B if used > B else 0.0
                    handed += used > B
            cells.append(f"{100 * handed / total:6.2f} ({100 * over / total:5.2f})")
        print(f"  {tag:10s} " + "  ".join(cells) + f"   {len(d)} episodes, {total} weeks, median week {median:.2f} s" + (f" scaled to {to_median:g}" if to_median else ""))


def table(*tags: str, task: str = "full", entropy: int = 111, episodes: int = 32, first: int = 0, longest: int = 0) -> None:
    B = BUDGET_S[task]
    for tag in tags:
        d = _load(tag, task, entropy, episodes, first)
        cpu = np.concatenate([e["cpu"] for e in d.values()])
        print(f"\n{tag}: {task}, root {entropy}, {len(d)} episodes, {len(cpu)} weeks; CPU seconds a week: {_dist(cpu)}; over {B:g} s: {int((cpu > B).sum())} ({100 * (cpu > B).mean():.2f} %)")
        if not all(e.get("detail") for e in d.values()):
            print("  the record keeps no runs of the solver (made before 8 October)")
            continue
        runs, per_week, lost, over = defaultdict(list), Counter(), Counter(), Counter()
        weeks = []
        for n, e in d.items():
            for w, (c, det, row) in enumerate(zip(e["cpu"], e["detail"], e["log"]), start=1):
                sv = det["solves"]
                for s in sv:
                    runs[(s["what"], s["attempt"], s["status"])].append(s["cpu"])
                failed = [s for s in sv if s["status"] not in SOLVED]
                kind = "no run failed" if not failed else "a run failed"
                per_week[(len(sv), kind)] += 1
                over[(len(sv), kind)] += c > B
                note = str(row[-1])
                if failed and "kept" in note.split(" ")[0]:
                    lost[n] += 1
                weeks.append((c, n, w, len(sv), note, sum(s["cpu"] for s in sv)))
        print("  runs of the solver: cell, attempt, status: count; CPU seconds")
        for key, x in sorted(runs.items(), key=lambda kv: -len(kv[1])):
            print(f"    {key[0]:9s} {key[1]:10s} {key[2]:20s} {len(x):6d}   {_dist(x)}")
        print("  weeks by the number of runs: weeks, of them over budget")
        for key in sorted(per_week):
            print(f"    {key[0]} runs, {key[1]:14s} {per_week[key]:6d} ({100 * per_week[key] / len(cpu):5.2f} %)   over budget {over[key]:4d}")
        inside = np.array([x[5] for x in weeks])
        print(f"  CPU seconds a week inside the solver: {_dist(inside)}; outside it: median {np.median(cpu - inside):.2f}")
        runs_lost = sorted(lost.values(), reverse=True)
        print(f"  weeks that kept the carried plan after a run that failed: {sum(runs_lost)} ({100 * sum(runs_lost) / len(cpu):.2f} %), in {len(lost)} episodes; most in one episode: {runs_lost[:8]}")
        for c, n, w, k, note, _i in sorted(weeks, reverse=True)[:longest]:
            sv = d[n]["detail"][w - 1]["solves"]
            print(f"    {c:6.2f} s  episode {n:2d} week {w:3d}  {note} | " + "; ".join(f"{s['what']} {s['attempt']} {s['status']} {s['cpu']:.2f}" for s in sv))


def claims(*tags: str, task: str = "full", entropy: int = 111, episodes: int = 32, first: int = 0) -> None:
    """A week's first pass: the exact cell's claim against what its solution played at, by how the cell was solved."""
    for tag in tags:
        d = _load(tag, task, entropy, episodes, first)
        gaps = defaultdict(list)
        for e in d.values():
            for det in e.get("detail", []):
                done = [s for s in det["solves"] if s["what"] != "hull"]
                if not det["passes"] or not done or det["passes"][0][1] is None:
                    continue
                claim, played = det["passes"][0][0], det["passes"][0][1] / 100.0
                last = done[-1]
                how = "at the tolerance of the second run" if last["attempt"] == "tolerance" else f"{last['what']}, {last['attempt']}"
                gaps[how].append(abs(played - claim) / 1e9)
        print(f"\n{tag}: {task}, root {entropy}: |played - claimed| of the week's plan, bn USD (a window costs thousands of bn)")
        for how, x in sorted(gaps.items(), key=lambda kv: -len(kv[1])):
            x = np.array(x)
            print(f"  {how:36s} {len(x):6d} weeks   median {np.median(x):.4f}  p95 {np.percentile(x, 95):.4f}  max {x.max():.3f}  over 1 bn: {int((x > 1).sum())}")


if __name__ == "__main__":
    fire.Fire({"same": same, "meter": meter, "table": table, "claims": claims})
