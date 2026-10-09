"""Full-knowledge plans from the cell program: descent from a start, and the starts themselves.

    uv run python lab/anastasiia/regime_lab/plan.py starts --episodes=8 --n_jobs=3
    uv run python lab/anastasiia/regime_lab/plan.py starts --episodes=8 --which=hybrid,basefirst --time_limit=60

``descend``: read the regimes of the trajectory the simulator played, solve the cell, play the solution, repeat while
the played cost falls. Every cost printed is the simulator's (``Episode.simulate``), never the program's claim.

Starts: ``hybrid`` (the weekly actions an agent played, kept by ``../search_lab/search.py``), ``oracle`` (the board's
clairvoyant plan), ``basefirst`` (the plan with base load first as a mixed-integer program,
``../stats_lab/plan_stats.py``; kept in ``outputs/regime_lab/basefirst/``).
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

import sbf_starter  # noqa: E402, F401 - points the package at the team's reference cache


OUT = ROOT / "outputs" / "cycle_lab"
PLAYED = ROOT / "outputs" / "search_lab" / "played"


descend = core.descend


# ----- starts ---------------------------------------------------------------------------------------------------------
def start_hybrid(ep: core.Episode, agent: str = "anastasiia_hybrid_chiplp") -> list:
    path = PLAYED / f"{agent}_{ep.task}_{ep.entropy}_{ep.n}.pkl"
    return ep.validated(pickle.loads(path.read_bytes())["actions"])


def start_oracle(ep: core.Episode) -> list:
    import highspy
    from shockbench_flow.policies.lp_common import to_highs_lp

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(to_highs_lp(ep.m))
    h.run()
    return ep.actions(np.asarray(h.getSolution().col_value))


def start_basefirst(ep: core.Episode, time_limit: float = 60.0, gap: float = 1e-3) -> list:
    """The plan with base load first (a mixed-integer program over every grid and week); its solution is kept."""
    from plan_stats import base_first_plan

    path = ROOT / "outputs" / "regime_lab" / "basefirst" / f"{ep.task}_{ep.entropy}_{ep.n}_tl{int(time_limit)}.pkl"
    if path.is_file():
        x = pickle.loads(path.read_bytes())["x"]
    else:
        t0 = time.time()
        x, status, mip_gap = base_first_plan(ep.inst, ep.marks, ep.m, time_limit, gap)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"x": x, "status": status, "gap": mip_gap, "seconds": time.time() - t0}))
    return ep.actions(x)


STARTS = {"hybrid": start_hybrid, "oracle": start_oracle, "basefirst": start_basefirst}


def references(task: str, entropy: int, episodes: int) -> list[dict]:
    from sbf_starter import scoring

    return list(scoring.episode_set(task, episodes, entropy=entropy, verbose=False).references)


def score(refs: list[dict], costs: list[int], levels_equal: bool = False) -> float:
    """The board's score of per-episode costs. A set without one of the four harm levels is scored as ``sbf
    evaluate`` scores it: all that was saved over all that could be (``levels_equal``: the levels present weigh the
    same, the reading of the numbers of 6 and 7 October in ``hub/tried/mpc.md``, 0.01 to 0.02 higher on root 444's
    first 8 episodes)."""
    from package_baselines import rss

    ok = [(r, j) for r, j in zip(refs, costs) if r["J_oracle_cents"] is not None]
    if levels_equal or {r["stratum"] for r, _ in ok} == {1, 2, 3, 4}:
        return rss(refs, costs)[0]
    return float(sum(r["J_naive_cents"] - j for r, j in ok) / sum(r["J_naive_cents"] - r["J_oracle_cents"] for r, _ in ok))


def both(refs: list[dict], costs: list[int]) -> str:
    return f"{score(refs, costs):.4f} [{score(refs, costs, True):.4f}]"


def _one(task, entropy, n, which, time_limit, iters, switch, hull=0, search=0, gate: bool = False,
         start_agent: str = "anastasiia_hybrid_chiplp", cycle: int = 0, cycle_after: int = 0):
    ep = core.Episode.of(task, entropy, n)
    out = {}
    method = "ipm" if ep.N > 4 * core.BIG else "simplex"
    for name in which:
        # ``start_agent``: the trajectory a descent starts from, so a new model can be the start too (P7)
        acts = (STARTS[name](ep, time_limit=time_limit) if name == "basefirst"
                else STARTS[name](ep, agent=start_agent) if name == "hybrid" else STARTS[name](ep))
        t0 = time.process_time()
        d = core.switch_on(ep, acts, passes=iters, last_week=ep.T - 12) if switch else descend(ep, acts, iters)
        for _ in range(hull):  # rounds of whole weeks asked for by the hull of the short weeks, each followed by a descent
            d2 = core.descend(ep, d["acts"], iters=iters, hull="round", close_until=ep.T - 12, method=method, search=search, gate=gate, cycle=cycle, cycle_after=cycle_after)
            if d2["J"] > d["J"] - 1e8:
                break
            d = {**d2, "J0": d["J0"], "hist": d["hist"] + d2["hist"]}
        out[name] = (d["J0"], d["J"], len(d.get("switched", d["hist"])), time.process_time() - t0)
        if d.get("fracy"):  # paradigm_lab E1a's first number: fractional shares of a whole week in the hull solve
            print(f"  ep {n} {name}: fractional y {d['fracy'][0]} of {d['fracy'][1]}", flush=True)
        if d.get("search"):  # paradigm_lab P17: which moves the replay actually took, and what each saved
            print(f"  ep {n} {name}: search tried {d['search'][0]}, took {d['search'][1]}", flush=True)
        path = OUT / ("hull" if hull else "switch" if switch else "descend") / f"{task}_{entropy}_{n}_{name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"acts": d["acts"], "J": d["J"], "J0": d["J0"], "hist": d["hist"]}))
    return n, out


def starts(task: str = "small", entropy: int = 444, episodes: int = 8, which: str | tuple = "hybrid,oracle",
           time_limit: float = 60.0, iters: int = 60, n_jobs: int = 3, first: int = 0, switch: bool = False,
           hull: int = 0, search: int = 0, gate: bool = False,
           start_agent: str = "anastasiia_hybrid_chiplp", cycle: int = 0, cycle_after: int = 0) -> None:
    """Descent from each start on episodes ``first .. first + episodes - 1``: played cost before and after, and RSS.
    ``--switch``: with grid-weeks switched on away from a border (``core.switch_on``); the count printed is theirs.
    ``--hull=N``: after the descent, up to N rounds of whole weeks asked for by the hull of the short weeks
    (``core.descend``'s ``hull``), each with a descent of its own; kept under ``outputs/regime_lab/hull``.
    ``--search=K``: in each of those rounds, up to K other sets of whole weeks are tried too (``descend``'s ``search``).
    ``--cycle=M``: M of those sets come from the grid's cycle schedule (paradigm_lab P17).
    ``--cycle_after=R``: R of the ring's moves are tried before those M, so the cycles do not starve the ring's
    first moves, which are the ones measured to pay."""
    which = tuple(which.split(",")) if isinstance(which, str) else tuple(which)
    refs = references(task, entropy, first + episodes)[first:]
    res = Parallel(n_jobs=n_jobs)(delayed(_one)(task, entropy, n, which, time_limit, iters, switch, hull, search, gate, start_agent, cycle, cycle_after) for n in range(first, first + episodes))
    for n, out in res:
        print(f"ep {n}: " + " | ".join(f"{k} {v[0] / 1e11:8.1f} -> {v[1] / 1e11:8.1f} ({v[2]} it, {v[3]:.1f} s)" for k, v in out.items()), flush=True)
    for name in which:
        print(f"{name:10s} start {both(refs, [o[name][0] for _, o in res])}  after descent {both(refs, [o[name][1] for _, o in res])}")
    if len(which) > 1:
        print(f"best of {which}: {both(refs, [min(o[k][1] for k in which) for _, o in res])}")
    print("score: as `sbf evaluate`; in brackets: harm levels weighing the same (the reading of hub/tried/mpc.md)")


if __name__ == "__main__":
    fire.Fire({"starts": starts})
