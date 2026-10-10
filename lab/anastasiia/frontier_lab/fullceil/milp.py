"""The integer program "base load first" of one Full episode, started from the best executed plan on disk.

    uv run python lab/anastasiia/frontier_lab/fullceil/milp.py --only=2,3,9 --time_limit=900

The program: the oracle's linear program (the simulator's other automatic rules left free) with one binary a (grid,
week): either no base load is shed or the grid's fabs draw nothing (``stats_lab/plan_stats.py``'s rows on
``regime_lab``'s program, as ``frontier_lab/audit/milp_start.py`` on Small). Every trajectory of the simulator is
feasible for it at its played cost, so its dual bound is a bound for any agent. Three numbers an episode, never to be
mixed: the incumbent (the program's OWN value: a relaxation, not a played cost), the dual bound, and what the
incumbent's actions cost when the simulator plays them blind. Kept in ``outputs/frontier_lab/fullceil/milp/``.
"""

import pickle
import time

import common as K
import fire
import numpy as np
import scipy.sparse as sp
from descend import milp_path


def best_run(task: str, entropy: int, n: int, among: str = "") -> tuple[str, dict] | None:
    """The cheapest executed plan kept for the episode: (label, its run)."""
    runs = {}
    for f in (K.OUT / "runs").glob(f"*_{task}_{entropy}_{n}.pkl"):
        label = f.name[: -len(f"_{task}_{entropy}_{n}.pkl")]
        if among and label not in among.split(","):
            continue
        d = pickle.loads(f.read_bytes())
        if "J" in d:
            runs[label] = d
    if not runs:
        return None
    label = min(runs, key=lambda k: runs[k]["J"])
    return label, runs[label]


def program(ep) -> tuple:
    """(cost, lb, ub, A, row_lb, row_ub, offset, number of the program's own columns) of the integer program."""
    from shockbench_flow.policies import lp_common as L

    inst, marks, m = ep.inst, ep.marks, ep.m
    T, nc, tmpl, G = m.T, m.meta["nc"], m.meta["template"], len(inst.grids)
    ncol = T * nc
    rows, cols, vals, rhs = [], [], [], []
    for t in range(T):
        for go in range(G):
            z, rr = ncol + t * G + go, len(rhs)
            y_bar = float(marks.y_bar[t, go])
            rows += [rr, rr]  # shed <= base load (1 - z)
            cols += [t * nc + tmpl[("ysh", go)], z]
            vals += [1.0, y_bar]
            rhs.append(y_bar)
            for fo in inst.grid_fabs[go]:
                if ("E", fo) not in tmpl:
                    continue
                fab = inst.nodes[inst.fabs[fo]].fab
                most = fab.e * float(marks.alpha_bar[t, fo]) * fab.cap0 * (1 + 1e-6) + 1e-9
                rr = len(rhs)
                rows += [rr, rr]  # the fab's energy <= its largest draw x z
                cols += [t * nc + tmpl[("E", fo)], z]
                vals += [1.0, -most]
                rhs.append(0.0)
    extra = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), ncol + T * G))
    A, row_lb, row_ub = L._stacked(m)
    A = sp.vstack([sp.hstack([A, sp.csr_matrix((A.shape[0], T * G))], format="csr"), extra], format="csr")
    rl = np.concatenate([row_lb, np.full(len(rhs), -np.inf)])
    ru = np.concatenate([row_ub, np.array(rhs)])
    c = np.concatenate([m.objective(), np.zeros(T * G)])
    lb = np.concatenate([m.lb, np.zeros(T * G)])
    ub = np.concatenate([m.ub, np.ones(T * G)])
    return c, lb, ub, A, rl, ru, L.model_offset(m), ncol


def lifted(ep, P: tuple, x: np.ndarray) -> tuple:
    """A trajectory's columns with the binaries its shed load implies: (vector, the program's value in USD, the
    largest row violation, the same relative to the row's scale)."""
    c, _lb, _ub, A, rl, ru, off, _ncol = P
    m, marks = ep.m, ep.marks
    T, nc, tmpl, G = m.T, m.meta["nc"], m.meta["template"], len(ep.inst.grids)
    x = np.asarray(x, dtype=float)
    zb = np.array([[1.0 if x[t * nc + tmpl[("ysh", go)]] <= 1e-7 * max(1.0, float(marks.y_bar[t, go])) else 0.0
                    for go in range(G)] for t in range(T)]).ravel()  # fmt: skip
    full = np.concatenate([x, zb])
    act = A @ full
    scale = 1.0 + np.abs(A).dot(np.abs(full))
    viol = np.maximum(np.maximum(rl - act, act - ru), 0.0)
    return full, float(c @ full + off), float(viol.max()), float((viol / scale).max())


def plan_weeks(ep, x: np.ndarray) -> dict:
    """What the program's solution does by week in its own columns: shed load, fab energy, lots, by grid and fab."""
    m, inst = ep.m, ep.inst
    T, nc, tmpl, G = m.T, m.meta["nc"], m.meta["template"], len(inst.grids)
    F = len(inst.fabs)
    w = np.asarray(x[: T * nc]).reshape(T, nc)
    return {"shed": np.array([[w[t, tmpl[("ysh", go)]] for go in range(G)] for t in range(T)]),
            "energy": np.array([[w[t, tmpl[("E", fo)]] if ("E", fo) in tmpl else 0.0 for fo in range(F)] for t in range(T)]),
            "lots": np.array([[w[t, tmpl[("p", fo)]] if ("p", fo) in tmpl else 0.0 for fo in range(F)] for t in range(T)])}  # fmt: skip


def main(only: str | tuple | int = "", task: str = "full", entropy: int = 444, first: int = 0, episodes: int = 16,
         time_limit: float = 900.0, among: str = "", name: str = "milp") -> None:
    import core  # regime_lab's
    import highspy
    from shockbench_flow.oracle.lp import lp_cents
    from shockbench_flow.policies import lp_common as L

    refs = K.references(task, entropy, 64)
    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else range(first, first + episodes)
    for n in ns:
        path = milp_path(task, entropy, n, name)
        if path.is_file():
            continue
        got = best_run(task, entropy, n, among)
        if got is None:
            print(f"ep {n}: no executed plan kept yet", flush=True)
            continue
        label, run = got
        w0 = time.time()
        ep = core.Episode.of(task, entropy, n)
        J_or, J_na = refs[n]["J_oracle_cents"] / 100, refs[n]["J_naive_cents"] / 100
        recs, J_exec = ep.simulate(run["acts"])
        _mode, ref = ep.regimes(recs)
        z_exec = ep.vector(recs, ref)[: ep.n0]
        P = program(ep)
        c, lb, ub, A, rl, ru, off, ncol = P
        full, J_lift, viol, rel = lifted(ep, P, z_exec)
        print(f"ep {n} level {refs[n]['stratum']}: naive {J_na / 1e9:.1f}, clairvoyant {J_or / 1e9:.1f} | best executed '{label}' "
              f"played {J_exec / K.BN:.2f} bn, as the program's columns {lp_cents(ep.m, z_exec) / K.BN:.2f} bn (with binaries "
              f"{J_lift / 1e9:.2f}), largest row violation {viol:.3g} (relative {rel:.2g}); rows {A.shape[0]}, columns {A.shape[1]}", flush=True)
        h = highspy.Highs()
        log = path.with_suffix(".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.unlink(missing_ok=True)
        for key, val in (("log_to_console", False), ("log_file", str(log)), ("threads", 1), ("time_limit", float(time_limit)),
                         ("mip_rel_gap", 1e-4)):
            h.setOptionValue(key, val)
        h.passModel(L.highs_lp(c, lb, ub, A, rl, ru, off))
        idx = np.arange(ncol, len(c), dtype=np.int32)
        h.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx)))
        sol = highspy.HighsSolution()
        sol.col_value = full
        took = h.setSolution(sol)
        t0 = time.time()
        h.run()
        info = h.getInfo()
        J, Jb = float(info.objective_function_value), float(info.mip_dual_bound)
        status = h.modelStatusToString(h.getModelStatus())
        x = np.asarray(h.getSolution().col_value, dtype=float)
        room = J_na - J_or
        acts = ep.actions(x[:ncol])
        recs_p, J_played = ep.simulate(acts)
        print(f"ep {n}: MILP {time.time() - t0:.0f} s wall (setSolution {took}), {status}, nodes {info.mip_node_count}; "
              f"incumbent (program's own value) {J / 1e9:.2f} bn, dual bound {Jb / 1e9:.2f} bn, gap {100 * info.mip_gap:.2f} %; "
              f"its actions played blind {J_played / K.BN:.2f} bn", flush=True)
        print(f"ep {n}: episode score: best executed {(J_na - J_exec / 100) / room:.4f}, incumbent {(J_na - J) / room:.4f}, "
              f"bound {(J_na - Jb) / room:.4f}; executed - incumbent {(J_exec / 100 - J) / 1e9:.1f} bn, incumbent - bound "
              f"{(J - Jb) / 1e9:.1f} bn, executed - bound {(J_exec / 100 - Jb) / 1e9:.1f} bn; peak {K.rss_mb():.0f} MB, "
              f"total {time.time() - w0:.0f} s wall", flush=True)
        path.write_bytes(pickle.dumps({
            "x": x[:ncol], "z": x[ncol:], "J": J, "bound": Jb, "gap": float(info.mip_gap), "status": status,
            "seconds": time.time() - t0, "time_limit": time_limit, "start": label, "J_exec": int(J_exec), "acts": acts,
            "J_played": int(J_played), "plan": plan_weeks(ep, x), "weeks": K.weeks_of(recs_p), "sent": K.sent_of(ep, recs_p),
            "nodes": int(info.mip_node_count),
        }))  # fmt: skip


if __name__ == "__main__":
    fire.Fire(main)
