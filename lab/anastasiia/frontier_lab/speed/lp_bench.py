"""Time HiGHS on the programs ``cells.py`` kept, one way of solving against the model's own.

    uv run python lab/anastasiia/frontier_lab/speed/lp_bench.py log outputs/speed_lab/cells/full_444_3/w005_10_exact.npz
    uv run python lab/anastasiia/frontier_lab/speed/lp_bench.py run outputs/speed_lab/cells/full_444_3 --what=exact \
        --options='{"dual_feasibility_tolerance": 1e-6}'

``log`` solves one program as the model does with the solver's log on (presolve, the sizes after it, the phases).
``run`` solves every kept program of a kind twice, as the model does and with ``options`` over that, in turn, and
prints for each the CPU seconds (``time.process_time``), the iterations, the objective's difference and whether x is
the same to the bit as the kept one. ``strip`` does the same for the program handed over without its unbounded
rows and its columns fixed at zero, the two ways interleaved. One process, one program at a time.
"""

import time
from pathlib import Path

import fire
import numpy as np


def _hs():
    import scipy.optimize._highspy._core as hs

    return hs


def lp_of(d):
    hs = _hs()
    inf = hs.kHighsInf
    n_row, n_col = (int(v) for v in d["shape"])
    lp = hs.HighsLp()
    lp.num_col_, lp.num_row_ = n_col, n_row
    lp.col_cost_, lp.col_lower_ = d["cost"], d["lb"]
    lp.col_upper_ = np.where(np.isinf(d["ub"]), inf, d["ub"])
    lp.row_lower_, lp.row_upper_ = np.where(np.isinf(d["lo"]), -inf, d["lo"]), np.where(np.isinf(d["hi"]), inf, d["hi"])
    lp.offset_ = float(d["offset"])
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n_col, n_row
    lp.a_matrix_.start_, lp.a_matrix_.index_ = d["indptr"].astype(np.int32), d["indices"].astype(np.int32)
    lp.a_matrix_.value_ = d["data"].astype(np.float64)
    return lp


def solver(d, options: dict | None = None, quiet: bool = True):
    """A solver set as ``plan_core.Episode.solve`` sets it for this program, ``options`` over that; not run."""
    hs = _hs()
    h = hs._Highs()
    h.setOptionValue("output_flag", not quiet)
    method = str(d["method"])
    devex = method == "devex"
    h.setOptionValue("solver", "simplex" if devex else method)
    if devex:
        h.setOptionValue("simplex_strategy", 1)
        h.setOptionValue("simplex_dual_edge_weight_strategy", 1)
    h.setOptionValue("time_limit", 600.0)
    if method == "ipm" and not bool(d["crossover"]):
        h.setOptionValue("run_crossover", "off")
    if method == "ipm" and not np.isnan(float(d["ipm_tol"])):
        h.setOptionValue("ipm_optimality_tolerance", float(d["ipm_tol"]))
    for key, value in (options or {}).items():
        h.setOptionValue(key, value)
    return h


def solve(d, options: dict | None = None, quiet: bool = True) -> dict:
    h = solver(d, options, quiet)
    h.passModel(lp_of(d))
    t0 = time.process_time()
    h.run()
    cpu = time.process_time() - t0
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    if status == "Unknown" and int(info.primal_solution_status) == 2:
        status = "Optimal"
    x = np.asarray(h.getSolution().col_value, dtype=float)
    return {"cpu": cpu, "status": status, "J": float(info.objective_function_value), "x": x,
            "simplex": int(info.simplex_iteration_count), "ipm": int(info.ipm_iteration_count)}  # fmt: skip


def log(path: str, options: dict | None = None) -> None:
    d = np.load(path)
    out = solve(d, options, quiet=False)
    print(f"{Path(path).name}: {out['status']} J {out['J']:.6e} CPU {out['cpu']:.3f} s, simplex {out['simplex']} "
          f"ipm {out['ipm']}, x the same as kept: {np.array_equal(out['x'], d['x'])}")  # fmt: skip


def run(folder: str, what: str = "exact", options: dict | None = None, first: int = 0, count: int = 99,
        rounds: int = 1) -> None:
    files = sorted(Path(folder).glob(f"*_{what}.npz"))[first : first + count]
    rows = []
    for path in files:
        d = np.load(path)
        base = min((solve(d) for _ in range(rounds)), key=lambda r: r["cpu"])
        new = min((solve(d, options) for _ in range(rounds)), key=lambda r: r["cpu"])
        gap = (new["J"] - base["J"]) / max(1.0, abs(base["J"]))
        rows.append((base["cpu"], new["cpu"]))
        moved = float(np.abs(new["x"] - base["x"]).max()) if len(new["x"]) == len(base["x"]) else float("nan")
        print(f"{path.name:22s} model {base['cpu']:.3f} s {base['simplex'] + base['ipm']:6d} it {base['status']:8s} | "
              f"new {new['cpu']:.3f} s {new['simplex'] + new['ipm']:6d} it {new['status']:8s} J {gap:+.2e} "
              f"x same as model's {np.array_equal(new['x'], base['x'])}, largest move {moved:.3g}; "
              f"model's x as kept {np.array_equal(base['x'], d['x'])}, "
              f"new x as kept {np.array_equal(new['x'], d['x'])}")  # fmt: skip
    a = np.array(rows)
    print(f"{what} x{len(a)} {options or ''}: model median {np.median(a[:, 0]):.3f} mean {a[:, 0].mean():.3f} | new "
          f"median {np.median(a[:, 1]):.3f} mean {a[:, 1].mean():.3f} "
          f"({a[:, 1].sum() / a[:, 0].sum() - 1:+.1%})")  # fmt: skip


def stripped(d) -> tuple[dict, np.ndarray]:
    """The program without its unbounded rows and without the columns fixed at zero; the mask of the columns kept.
    Nothing is computed: rows and columns are only left out, so the bounds and the offset stay as they are."""
    import scipy.sparse as sp

    n_row, n_col = (int(v) for v in d["shape"])
    A = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=(n_row, n_col))
    keep_c = ~((d["lb"] == d["ub"]) & (d["lb"] == 0.0))
    keep_r = ~(np.isinf(d["lo"]) & np.isinf(d["hi"]))
    A2 = A[:, keep_c].tocsr()[keep_r].tocsc()
    A2.sort_indices()
    out = {k: d[k] for k in ("method", "crossover", "ipm_tol", "offset")}
    out.update(indptr=A2.indptr, indices=A2.indices, data=A2.data, shape=np.array(A2.shape), lo=d["lo"][keep_r],
               hi=d["hi"][keep_r], lb=d["lb"][keep_c], ub=d["ub"][keep_c], cost=d["cost"][keep_c])  # fmt: skip
    return out, keep_c


def strip(folder: str, what: str = "exact", first: int = 0, count: int = 99) -> None:
    """Every kept program of a kind as the model solves it and without its unbounded rows and zero columns, in the
    order model, stripped, stripped, model, so that neither is always the second. Not the same x: HiGHS's presolve
    takes the same rows and columns out itself, in another order, and the simplex ends at another vertex."""
    files = sorted(Path(folder).glob(f"*_{what}.npz"))[first : first + count]
    rows = []
    for path in files:
        d = np.load(path)
        d = {k: d[k] for k in d.files}
        small, keep_c = stripped(d)
        a1, b1, b2, a2 = solve(d), solve(small), solve(small), solve(d)
        x = np.zeros(len(keep_c))
        x[keep_c] = b1["x"]
        gap = (b1["J"] - a1["J"]) / max(1.0, abs(a1["J"]))
        rows.append((a1["cpu"], a2["cpu"], b1["cpu"], b2["cpu"], a1["simplex"] + a1["ipm"], b1["simplex"] + b1["ipm"]))
        print(f"{path.name:22s} {small['shape'][0]} x {small['shape'][1]} | model {a1['cpu']:.3f} {a2['cpu']:.3f} s "
              f"{rows[-1][4]:6d} it | stripped {b1['cpu']:.3f} {b2['cpu']:.3f} s {rows[-1][5]:6d} it {b1['status']} "
              f"J {gap:+.1e} x the same {np.array_equal(x, a1['x'])}", flush=True)  # fmt: skip
    a = np.array(rows)
    model, new = a[:, 0] + a[:, 1], a[:, 2] + a[:, 3]
    won = int((new < model).sum())
    print(f"{what} x{len(a)}: model {model.sum() / 2:.2f} s, stripped {new.sum() / 2:.2f} s "
          f"({new.sum() / model.sum() - 1:+.1%}), "
          f"cheaper in {won} of {len(a)}; iterations {int(a[:, 4].sum())} and {int(a[:, 5].sum())} "
          f"({a[:, 5].sum() / a[:, 4].sum() - 1:+.1%}); first runs {a[:, 2].sum() / a[:, 0].sum() - 1:+.1%}, "
          f"second runs {a[:, 3].sum() / a[:, 1].sum() - 1:+.1%}")  # fmt: skip


if __name__ == "__main__":
    fire.Fire({"log": log, "run": run, "strip": strip})
