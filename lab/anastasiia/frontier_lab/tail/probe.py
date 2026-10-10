"""One episode of a built agent with a shadow planner at chosen weeks: the same state, another window.

    uv run python lab/anastasiia/frontier_lab/tail/probe.py run outputs/hazard_lab/agents/h3_f --n=3 --every=4
    uv run python lab/anastasiia/frontier_lab/tail/probe.py times outputs/hazard_lab/agents/h3_f --n=3 --weeks=30

``run`` plays the agent on the environment. Every ``every`` weeks a deep copy of it (the same memory, the same carried
plan) plans the same observation with the settings ``shadow`` (the window of 20 weeks by default), and both plans of
the window are kept: per week of the window the shed load by grid, the lots by fab, the lost sales by market, the
stock by slot and what every slot sent. Kept in ``outputs/tail_lab/probe/<name>_<task>_<entropy>_<n>.pkl``.

``times`` plays the first ``weeks`` weeks and prints where the week's CPU goes: the solve with the hull, the exact
cell, the rules' rollout, the rest.
"""

import copy
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "outputs" / "tail_lab" / "probe"


def _start(agent: str, task: str, entropy: int, n: int):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load, resolve

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = load(str(resolve(agent).resolve()))(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    return env, u, obs, ag


def plan_of(ag) -> dict | None:
    """The window's plan the agent's last ``act`` ended with, as the model played it."""
    last = getattr(ag, "last", None)
    if last is None:
        return None
    ep, d, *_rest, H = last
    recs = d["recs"]
    S = len(ep.inst.action_slots)
    sent = np.zeros((len(recs), S))
    for t, r in enumerate(recs):
        for slot, q in r.executed.items():
            sent[t, slot] = q
    return {
        "H": H, "J": d["J"], "J0": d["J0"],
        "shed": np.array([r.shed for r in recs]), "lots": np.array([r.lots_started for r in recs]),
        "lost": np.array([r.lost for r in recs]), "stock": np.array([r.stock for r in recs]), "sent": sent,
        "costs": np.array([r.cost_cents for r in recs]) / 100.0,
    }  # fmt: skip


def prices(ag, at: int = 20) -> dict | None:
    """What a unit left in every stock slot at the end of week ``at`` of the window is worth to the window's program,
    USD: minus the dual of the slot's balance row of the week after, in the cell of the plan the agent ended with (its
    regimes read again from that plan, solved once more here). None: the window is no longer than ``at`` weeks or the
    cell did not solve."""
    last = getattr(ag, "last", None)
    if last is None:
        return None
    ep, d, tweak, bonus, anchor, price, H = last
    if at + 1 > H:
        return None
    core = sys.modules[type(ep).__module__]
    mode, ref = ep.regimes(d["recs"])
    C = ep.cell(mode, ref, anchor, price, bonus)
    if tweak is not None:
        tweak(C)
    hs, Highs = core.highs()
    A, lo, hi = ep.rows(C)
    n_row, n_col = A.shape
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = n_col, n_row
    lp.col_cost_, lp.col_lower_ = (ep.obj if C.cost is None else ep.obj + C.cost), C.lb
    lp.col_upper_ = np.where(np.isinf(C.ub), hs.kHighsInf, C.ub)
    lp.row_lower_, lp.row_upper_ = np.where(np.isinf(lo), -hs.kHighsInf, lo), np.where(np.isinf(hi), hs.kHighsInf, hi)
    lp.offset_ = float(ep.offset)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n_col, n_row
    lp.a_matrix_.start_, lp.a_matrix_.index_ = A.indptr.astype(np.int32), A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    h = Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("simplex_strategy", 1)
    h.setOptionValue("simplex_dual_edge_weight_strategy", 1)
    h.setOptionValue("time_limit", 30.0)
    h.passModel(lp)
    h.run()
    if h.modelStatusToString(h.getModelStatus()) != "Optimal":
        return None
    sol = h.getSolution()
    y, x = np.asarray(sol.row_dual, dtype=float), np.asarray(sol.col_value, dtype=float)
    eq0 = ep.nub * ep.T
    worth = np.zeros(len(ep.inst.stock_slots))
    for s in ep.plain:
        worth[s] = -y[eq0 + at * ep.neq + ep.eqi[("balance", s)]]  # the row of week ``at`` + 1
    held = np.array([x[ep.col("I", at, s)] if ep.has("I", s) else 0.0 for s in range(len(ep.inst.stock_slots))])
    lots = np.array([[x[ep.col("p", t, fi)] for fi in range(len(ep.inst.fabs))] for t in range(1, ep.T + 1)])
    return {"worth": worth, "held": held, "lots": lots}


def run(agent: str, n: int, task: str = "full", entropy: int = 444, every: int = 4, shadow: dict | None = None,
        name: str = "w20", weeks: int = 0, duals: bool = False, since: int = 1, until: int = 0) -> None:
    """``since``, ``until``: the weeks a shadow plans in (``until`` 0: to the end); ``every`` counts from ``since``."""
    shadow = {"horizon": 20, "hull_until": 6} if shadow is None else dict(shadow)
    env, u, obs, ag = _start(agent, task, entropy, n)
    out, done, week = {"n": n, "shadow": shadow, "weeks": {}}, False, 0
    while not done:
        week += 1
        row = {}
        if every and week >= since and (not until or week <= until) and (week - since) % every == 0:
            sh = copy.deepcopy(ag)
            sh.p = sh.p | shadow
            t0 = time.process_time()
            a2 = sh.act(obs)
            row["shadow_cpu"], row["shadow_flows"], row["shadow_plan"] = time.process_time() - t0, np.array(a2["flows"]), plan_of(sh)
            row["shadow_log"] = sh.log[-1]
            del sh
        t0 = time.process_time()
        action = ag.act(obs)
        row["cpu"], row["log"] = time.process_time() - t0, ag.log[-1]
        row["solves"] = [(s["what"], s["attempt"], s["status"], s["cpu"], s["rows"], s["cols"]) for s in ag.week_solves]
        row["rolled"] = ag.rolled
        if "shadow_cpu" in row:
            row["flows"], row["plan"] = np.array(action["flows"]), plan_of(ag)
            if duals:
                row["prices"] = prices(ag, int(shadow.get("horizon", 20)))
        out["weeks"][week] = row
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc or (weeks and week >= weeks)
    out["J"] = int(u.core._ep.traj.J_cents) if not weeks else None  # the kept play's when the agent's play repeats
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}_{task}_{entropy}_{n}.pkl").write_bytes(pickle.dumps(out))
    print(f"{name} episode {n}: {week} weeks kept, J {out['J']}")


def times(agent: str, n: int = 3, task: str = "full", entropy: int = 444, weeks: int = 30, **numbers) -> None:
    """Where the week's CPU goes, the agent's settings changed by ``numbers`` (``--horizon=20 --hull_until=6``)."""
    env, _u, obs, ag = _start(agent, task, entropy, n)
    ag.p = ag.p | numbers
    rows = []
    for _week in range(weeks):
        t0 = time.process_time()
        action = ag.act(obs)
        total = time.process_time() - t0
        by = {}
        for s in ag.week_solves:
            by[s["what"]] = by.get(s["what"], 0.0) + s["cpu"]
        rows.append((total, by.get("hull", 0.0), by.get("exact", 0.0), ag.rolled,
                     sum(v for k, v in by.items() if k not in ("hull", "exact"))))
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    a = np.array(rows)
    rest = a[:, 0] - a[:, 1:].sum(1)
    print(f"{Path(agent).name} {numbers or ''} episode {n}, {len(a)} weeks, CPU seconds a week, median (mean):")
    for label, v in (("week", a[:, 0]), ("hull", a[:, 1]), ("exact", a[:, 2]), ("rules' rollout", a[:, 3]),
                     ("other solves", a[:, 4]), ("the rest", rest)):
        print(f"  {label:15s} {np.median(v):.3f} ({v.mean():.3f})")


def smoke(agent: str, n: int = 3, task: str = "full", entropy: int = 444, weeks: int = 6, **numbers) -> None:
    """The first ``weeks`` weeks of an episode: the cost so far and every week's note (an error shows there)."""
    env, u, obs, ag = _start(agent, task, entropy, n)
    ag.p = ag.p | numbers
    for week in range(1, weeks + 1):
        t0 = time.process_time()
        action = ag.act(obs)
        print(f"  week {week:3d} cpu {time.process_time() - t0:.2f}", *ag.log[-1][1:])
        obs, _r, term, trunc, _i = env.step(action)
        if term or trunc:
            break
    print(f"{Path(agent).name} {numbers or ''} episode {n}: the cost of {week} weeks, cents: {sum(r.cost_cents for r in u.core._ep.traj.records)}")


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    fire.Fire({"run": run, "times": times, "smoke": smoke})
