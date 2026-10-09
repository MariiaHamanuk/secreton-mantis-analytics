"""The ceiling of the whole-week family: how much is in the choice of whole weeks at all, judged by the replay.

    uv run --python 3.13.11 python lab/anastasiia/cycle_lab/ceiling.py --which=6,2,4 --budget=400 --n_jobs=3
    uv run --python 3.13.11 python lab/anastasiia/cycle_lab/ceiling.py --episodes=8 --budget=300 --n_jobs=4

Why this and not another proposer. ``descend``'s ``search`` tries sets of whole weeks next to the rounding's own set
and keeps the cheapest **as the simulator plays it**. Measured on Small 444 episodes 0-7, eight tries give 0.9230
and sixty give 0.9234, so the ring is exhausted; and the rounding is already optimal in the **count** of whole weeks
(``probe_rounding.py``: room 0 on all ten grid-episodes). What is unmeasured is the whole family: of every set of
whole weeks this state admits, how good is the best one? That is the upper bound of P17 and of every proposer
anyone could write, including the cycle program - the same argument P13 was closed by (a tighter relaxation of a
choice cannot beat the exact solution of that choice).

The state is held **fixed**: one trajectory, one reading of its regimes, one hull cell. Only the set of weeks
written whole varies, and every candidate is costed the way the model costs it in the game - its own exact cell,
solved, played by the simulator. So the number this prints is comparable to ``--search=K``'s number on the same
episodes and nothing else moved.

The search is a hill climb from the rounding's set over single moves (add a week, drop a week, slide a week by one
to three short weeks, swap two) with random restarts, plus the cycle program's own proposals as starts. It is not
exact - the exact problem is a mixed-integer program over 60 to 100 binaries whose objective is a simulator and so
not writable - but with a budget of a few hundred plays it is a much better upper bound than eight tries.

Reported per episode: the cost the rounding's set gives, the best cost the search found, and the gap in bn USD;
then the RSS of each over the set, which is what the gate compares.
"""

import pickle
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

OUT = ROOT / "outputs" / "cycle_lab" / "ceiling"


class Judge:
    """One fixed state; a set of whole weeks in, the cost the simulator pays out. Costs are remembered."""

    def __init__(self, ep, mode: dict, ref: dict, plain: dict, basis=None, method: str = "simplex") -> None:
        self.ep, self.ref, self.plain, self.method = ep, ref, plain, method
        self.mode, self.basis, self.seen, self.solves = mode, basis, {}, 0

    def __call__(self, weeks: frozenset) -> int | None:
        """The played cost of writing exactly ``weeks`` whole, or None when that cell has no solution."""
        if weeks in self.seen:
            return self.seen[weeks]
        ep = self.ep
        trial = {**self.mode, "grid": dict(self.plain)}
        for key in weeks:
            if self.plain.get(key) == "OFF":
                trial["grid"][key] = "SOFT"
        C = ep.cell(trial, self.ref)
        st = ep.solve(C, method=self.method, basis=self.basis, time_limit=600.0, what="ceiling")
        self.solves += 1
        out = None
        if st["status"] == "Optimal":
            self.basis = st["basis"]
            _recs, J = ep.simulate(ep.actions(st["x"]))
            out = int(J)
        self.seen[weeks] = out
        return out


def moves(weeks: frozenset, domain: list, rng) -> list:
    """Single changes to a set of whole weeks: drop one, add one, slide one along its grid, swap one for another.

    ``domain`` is the (week, grid) the hull covered, in order, which is the only place a whole week may go. A slide
    is by one to three **short weeks of the same grid**, not by one calendar week: between two short weeks of a grid
    there may be weeks it already runs whole, and those are not the rounding's to move.
    """
    by_grid = {}
    for t, gi in domain:
        by_grid.setdefault(gi, []).append(t)
    out = []
    for key in weeks:  # drop one
        out.append(weeks - {key})
    for key in domain:  # add one
        if key not in weeks:
            out.append(weeks | {key})
    for t, gi in weeks:  # slide one along its grid
        ts = by_grid[gi]
        at = ts.index(t)
        for by in (-3, -2, -1, 1, 2, 3):
            if 0 <= at + by < len(ts) and (ts[at + by], gi) not in weeks:
                out.append((weeks - {(t, gi)}) | {(ts[at + by], gi)})
    for key in weeks:  # swap one for a week of the same grid anywhere
        ts = by_grid[key[1]]
        for t in rng.permutation(ts)[:6]:
            cand = (weeks - {key}) | {(int(t), key[1])}
            if cand != weeks:
                out.append(cand)
    return [frozenset(w) for w in out]


def climb(judge: Judge, start: frozenset, domain: list, budget: int, rng, best: tuple) -> tuple:
    """A hill climb from ``start`` while the budget of plays lasts: (cost, set). ``best`` carries the run's best."""
    here = judge(start)
    if here is None:
        return best
    if here < best[0]:
        best = (here, start)
    while judge.solves < budget:
        cands = moves(start, domain, rng)
        rng.shuffle(cands)
        improved = False
        for cand in cands:
            if judge.solves >= budget:
                break
            J = judge(cand)
            if J is not None and J < here - 1e6:  # cheaper by more than 10,000 USD, as ``search`` reckons it
                here, start, improved = J, cand, True
                if J < best[0]:
                    best = (J, cand)
                break
        if not improved:
            break
    return best


def sweep_moves(weeks: frozenset, domain: list) -> list:
    """The cheap single moves only: drop one asked week, or slide one by a single short week of its own grid.

    For Full a play costs about 14 CPU seconds and the candidate domain runs to 284 weeks, so the hill climb cannot
    finish even one step inside any budget this machine affords. This sweep answers the narrower question that one
    sweep **can** answer: is the rounding's set a local optimum of the cheap moves? It is about 3 |W| plays rather
    than |domain| + 13 |W|.
    """
    by_grid = {}
    for t, gi in domain:
        by_grid.setdefault(gi, []).append(t)
    out = []
    for key in weeks:
        out.append(weeks - {key})
    for t, gi in weeks:
        ts = by_grid[gi]
        at = ts.index(t)
        for by in (-1, 1):
            if 0 <= at + by < len(ts) and (ts[at + by], gi) not in weeks:
                out.append((weeks - {(t, gi)}) | {(ts[at + by], gi)})
    seen, kept = {weeks}, []
    for w in out:
        fw = frozenset(w)
        if fw not in seen:
            seen.add(fw)
            kept.append(fw)
    return kept


def one(task: str, entropy: int, n: int, start_agent: str, iters: int, budget: int, restarts: int,
        seed: int, sweep: bool = False) -> dict:
    ep = core.Episode.of(task, entropy, n)
    method = "ipm" if ep.N > 4 * core.BIG else "simplex"
    acts = plan.start_hybrid(ep, agent=start_agent)
    t0 = time.process_time()
    d = core.descend(ep, acts, iters=iters)  # the same first step ``plan.py --hull=N`` takes
    recs, _J = ep.simulate(d["acts"])
    mode, ref = ep.regimes(recs)
    plain = dict(mode["grid"])
    for key, gm in plain.items():
        if gm == "OFF" and key[0] <= ep.T - 12:
            mode["grid"][key] = "HULL"
    C = ep.cell(mode, ref)
    wide = ep.solve(C, method=method, time_limit=600.0, what="hull")
    mode["grid"] = dict(plain)
    if wide["status"] != "Optimal":
        return {"n": n, "status": wide["status"]}

    probs = cycles.problems(ep, C, wide["x"], mode)
    marked = {"grid": dict(plain), "fuel": mode["fuel"]}
    core._rounded(ep, C, wide["x"], marked, write="SOFT")
    asked = frozenset(key for key in C.hull if marked["grid"][key] == "SOFT")
    domain = sorted(C.hull)

    judge = Judge(ep, mode, ref, plain, basis=wide["basis"], method=method)
    base = judge(asked)  # the rounding's own set: the number ``--search=0`` would give from this state
    if base is None:
        return {"n": n, "status": "rounding infeasible"}
    best = (base, asked)
    rng = np.random.default_rng(seed + n)

    if sweep:  # one exhaustive sweep of the cheap single moves: is the rounding's set a local optimum?
        taken = []
        for cand in sweep_moves(asked, domain):
            J = judge(cand)
            if J is not None and J < base - 1e6:
                taken.append(((base - J) / 1e11, sorted(cand)))
                if J < best[0]:
                    best = (J, cand)
        taken.sort(reverse=True)
        return {"n": n, "status": "Optimal", "J_base": base, "J_best": best[0],
                "gain_bn": (base - best[0]) / 1e11, "asked": sorted(asked), "best": sorted(best[1]),
                "solves": judge.solves, "domain": len(domain), "seconds": time.process_time() - t0,
                "improving": len(taken), "swept": judge.solves - 1,
                "top": [f"{g:.2f}" for g, _ in taken[:3]]}

    # the starts: the rounding's set, then the cycle program's proposals, then random sets of the same size
    starts = [asked] + [cand for _name, cand in cycles.proposals(probs, asked, limit=12)]
    while len(starts) < 1 + restarts:
        k = len(asked) + int(rng.integers(-1, 2))
        if 0 < k <= len(domain):
            starts.append(frozenset(tuple(domain[i]) for i in rng.choice(len(domain), size=k, replace=False)))
        else:
            starts.append(frozenset())
    for st in starts:
        if judge.solves >= budget:
            break
        best = climb(judge, st, domain, budget, rng, best)

    gain = (base - best[0]) / 1e11
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{task}_{entropy}_{n}.pkl").write_bytes(pickle.dumps(
        {"asked": asked, "best": best[1], "J_base": base, "J_best": best[0], "domain": domain,
         "solves": judge.solves, "J_descend": d["J"]}))
    return {"n": n, "status": "Optimal", "J_base": base, "J_best": best[0], "gain_bn": gain,
            "asked": sorted(asked), "best": sorted(best[1]), "solves": judge.solves,
            "domain": len(domain), "seconds": time.process_time() - t0}


def main(task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0, which: str | tuple = "",
         n_jobs: int = 4, iters: int = 60, budget: int = 300, restarts: int = 8, seed: int = 20261009,
         start_agent: str = "anastasiia_plan_hull3", sweep: bool = False) -> None:
    """``--budget``: plays a episode may spend. ``--restarts``: random sets tried besides the cycle proposals."""
    if which:
        ns = [int(x) for x in (which if isinstance(which, (list, tuple)) else str(which).split(","))]
    else:
        ns = list(range(first, first + episodes))
    refs = plan.references(task, entropy, max(ns) + 1)
    res = Parallel(n_jobs=n_jobs)(
        delayed(one)(task, entropy, n, start_agent, iters, budget, restarts, seed, sweep) for n in ns)
    ok = [r for r in res if r.get("status") == "Optimal"]
    for r in res:
        if r.get("status") != "Optimal":
            print(f"ep {r['n']}: {r['status']}")
    print(f"\n{'ep':>3} {'rounding':>10} {'best':>10} {'gain bn':>8} {'plays':>6} {'dom':>4}  set")
    for r in ok:
        print(f"{r['n']:>3} {r['J_base'] / 1e11:>10.1f} {r['J_best'] / 1e11:>10.1f} {r['gain_bn']:>8.2f}"
              f" {r['solves']:>6} {r['domain']:>4}  {[(t, g) for t, g in r['asked']]} -> {[(t, g) for t, g in r['best']]}")
    if sweep:
        for r in ok:
            print(f"  ep {r['n']}: swept {r['swept']} cheap moves of a domain of {r['domain']},"
                  f" improving {r['improving']}, best {r['top'] or 'none'}")
    sub = [refs[r["n"]] for r in ok]
    print(f"\nrounding's set   {plan.both(sub, [r['J_base'] for r in ok])}")
    print(f"the family's best {plan.both(sub, [r['J_best'] for r in ok])}")
    print(f"gain {sum(r['gain_bn'] for r in ok):.2f} bn USD over {len(ok)} episode(s),"
          f" {sum(r['solves'] for r in ok)} plays")
    print("this is an upper bound on every proposer of whole weeks, the cycle program included: the state is fixed,")
    print("only the set of whole weeks varies, and every candidate is costed by its own exact cell and the replay.")


if __name__ == "__main__":
    fire.Fire(main)
