"""Is there room in the rounding at all? The greedy ``_rounded`` plays against the optimum of its own problem.

    uv run python lab/anastasiia/cycle_lab/probe_rounding.py --episodes=8 --n_jobs=4
    uv run python lab/anastasiia/cycle_lab/probe_rounding.py --which=6,2,4 --n_jobs=3

This decides whether paradigm_lab P17's dynamic program is worth building before any of it is built. It reaches the
state ``plan.py starts --hull=1`` reaches - a descent from the agent's own played trajectory, then one solve with
the hull - reads the whole-week schedule problem of every grid off that solve (``cycles.problems``), and prints,
per grid, what ``core._rounded`` marks against what the problem's optimum holds.

Two numbers decide it:

- **identical**: ``cycles.greedy`` must mark exactly what ``core._rounded`` marks. If it does not, the problem
  written here is not the problem the model solves, and nothing below means anything.
- **room**: whole weeks at the optimum minus whole weeks greedily. Zero everywhere means the greedy is already
  optimal on the problems that actually arise and P17's dynamic program is dead - one run, and the direction is
  closed. Above zero means the bank is being spent early on isolated weeks instead of buying runs.

Nothing is decided here: a whole week more is not a cent less until the simulator replays it. This only says
whether there is anything to propose.
"""

import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab"), str(HERE.parent / "stats_lab")]
import core  # noqa: E402
import cycles  # noqa: E402
import plan  # noqa: E402

import sbf_starter  # noqa: E402, F401 - points the package at the team's reference cache


def hull_state(ep, acts: list, iters: int = 60, method: str = "simplex") -> tuple:
    """(the hull cell, its solution, the regimes it was built from) at the state ``plan.py --hull=1`` reaches.

    The same two steps that function takes: a plain descent from the start, then the cell of the trajectory it
    reached with every short week of a grid with fabs written as "HULL", solved. ``close_until`` is ``T - 12`` there,
    so it is here too.
    """
    d = core.descend(ep, acts, iters=iters)
    recs, _J = ep.simulate(d["acts"])
    mode, ref = ep.regimes(recs)
    plain = dict(mode["grid"])
    for key, gm in plain.items():
        if gm == "OFF" and key[0] <= ep.T - 12:
            mode["grid"][key] = "HULL"
    C = ep.cell(mode, ref)
    wide = ep.solve(C, method=method, time_limit=600.0, what="hull")
    mode["grid"] = dict(plain)  # as ``descend`` does before it rounds: the rounding writes into the plain regimes
    return C, wide, mode, d


def one(task: str, entropy: int, n: int, start_agent: str, iters: int) -> dict:
    ep = core.Episode.of(task, entropy, n)
    method = "ipm" if ep.N > 4 * core.BIG else "simplex"
    acts = plan.start_hybrid(ep, agent=Path(start_agent).name)
    t0 = time.process_time()
    C, wide, mode, d = hull_state(ep, acts, iters=iters, method=method)
    if wide["status"] != "Optimal":
        return {"n": n, "status": wide["status"]}
    probs = cycles.problems(ep, C, wide["x"], mode)

    # the gate of identity: ``cycles.greedy`` must mark exactly what ``core._rounded`` marks
    check = {"grid": dict(mode["grid"]), "fuel": mode["fuel"]}
    core._rounded(ep, C, wide["x"], check, write="SOFT")
    # only the (week, grid) the hull covered can be written by the rounding; a week already whole in the regimes is
    # not its doing and must not be counted as if it were
    theirs = {gi: sorted(t for (t, g) in C.hull if g == gi and check["grid"][(t, g)] == "SOFT") for gi in probs}
    mine = {gi: sorted(cycles.greedy(p)) for gi, p in probs.items()}
    identical = all(theirs[gi] == mine[gi] for gi in probs)

    # what the search already has: ``_other_weeks``' ring around the greedy's set, as ``descend`` builds it
    asked = frozenset((t, gi) for gi, ts in theirs.items() for t in ts)
    scout = {key: float(wide["x"][ep.jrho(*key)]) / r for key, r in C.hull.items() if r > 0}
    ring = {w for _kind, w in core._other_weeks(scout, asked)}
    fresh = [(name, cand) for name, cand in cycles.proposals(probs, asked) if cand not in ring]

    rows = []
    for gi, p in probs.items():
        g = ep.inst.grids[gi]
        best = cycles.solve(p, cycles.values(p, "count"), kept=1)
        opt = best[0] if best else []
        rows.append({
            "grid": gi, "name": str(ep.inst.nodes[g].name if hasattr(ep.inst.nodes[g], "name") else g),
            "weeks": len(p), "bank": float(p.share.sum()), "prime_max": float(p.prime.max()),
            "greedy": mine[gi], "opt_n": len(opt), "opt": opt,
            "room": len(opt) - len(mine[gi]),
            "greedy_same_as_rounded": theirs[gi] == mine[gi],
            # placement: the schedules of the same count the program also holds, which the greedy cannot reach
            "others": [w for name in ("late", "peak", "ratio", "share")
                       for w in cycles.solve(p, cycles.values(p, name), kept=2)
                       if len(w) == len(mine[gi]) and w != mine[gi]],
        })
    frac = sum(1 for (t, gi), r in C.hull.items() if r > 0 and 0.01 < float(wide["x"][ep.jrho(t, gi)]) / r < 0.99)
    zero = sum(1 for (t, gi), r in C.hull.items() if r > 0 and float(wide["x"][ep.jrho(t, gi)]) / r <= 0.01)
    return {"n": n, "status": "Optimal", "identical": identical, "rows": rows, "J_descend": d["J"], "J0": d["J0"],
            "hull_cells": len(C.hull), "fractional": frac, "zero": zero, "seconds": time.process_time() - t0,
            "ring": len(ring), "cycle": len(cycles.proposals(probs, asked)), "fresh": len(fresh),
            "fresh_names": [name for name, _ in fresh][:8]}


def main(task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0, which: str | tuple = "",
         n_jobs: int = 4, iters: int = 60, start_agent: str = "agents/anastasiia_plan_hull3") -> None:
    """``--which=6,2,4``: just those episodes (the tail P7 measured as carrying 68 % of the gap)."""
    if which:  # fire hands "6,2,4" over as a tuple already, and a single number as an int
        ns = [int(x) for x in (which if isinstance(which, (list, tuple)) else str(which).split(","))]
    else:
        ns = list(range(first, first + episodes))
    res = Parallel(n_jobs=n_jobs)(delayed(one)(task, entropy, n, start_agent, iters) for n in ns)
    bad = [r for r in res if r.get("status") != "Optimal"]
    for r in bad:
        print(f"ep {r['n']}: hull {r['status']}")
    res = [r for r in res if r.get("status") == "Optimal"]
    print(f"\n{'ep':>3} {'grid':>4} {'weeks':>5} {'bank':>6} {'prime':>6} {'greedy':>6} {'opt':>4} {'room':>5}  weeks greedy -> optimum")
    total_room, grids_with_room = 0, 0
    for r in res:
        for row in r["rows"]:
            if row["weeks"] == 0:
                continue
            total_room += max(0, row["room"])
            grids_with_room += 1 if row["room"] > 0 else 0
            mark = " !" if not row["greedy_same_as_rounded"] else ""
            print(f"{r['n']:>3} {row['grid']:>4} {row['weeks']:>5} {row['bank']:>6.2f} {row['prime_max']:>6.2f}"
                  f" {len(row['greedy']):>6} {row['opt_n']:>4} {row['room']:>+5}  {row['greedy']} -> {row['opt']}{mark}")
    print(f"\nidentity to ``core._rounded``: {'OK on every grid' if all(r['identical'] for r in res) else 'BROKEN - see the ! rows'}")
    print(f"hull cells {sum(r['hull_cells'] for r in res)}, of them fractional {sum(r['fractional'] for r in res)}"
          f", asking zero {sum(r['zero'] for r in res)}")
    print(f"room in the count: {total_room} whole week(s) over {grids_with_room} grid-episode(s) of"
          f" {sum(len([x for x in r['rows'] if x['weeks']]) for r in res)}")
    moved = sum(1 for r in res for x in r["rows"] if x["others"])
    print(f"room in the placement: {moved} grid-episode(s) hold a schedule of the same count the greedy cannot reach")
    for r in res:
        for x in r["rows"]:
            for w in {tuple(o) for o in x["others"]}:
                print(f"   ep {r['n']} grid {x['grid']}: {x['greedy']} -> {list(w)}")
    print(f"proposals: the ring ``_other_weeks`` already offers {sum(r['ring'] for r in res)} set(s);"
          f" the cycle program offers {sum(r['cycle'] for r in res)}, of them {sum(r['fresh'] for r in res)} outside the ring")
    print("a whole week more is not a cent less: the judge is the replay. This says only whether there is anything to propose.")


if __name__ == "__main__":
    fire.Fire(main)
