"""The exact cell from last week's basis, on the whole program and on one cut down by hand, against the cold solve.

    uv run python lab/anastasiia/frontier_lab/speed/warm_bench.py outputs/speed_lab/cells/full_444_3

From a basis HiGHS runs without its presolve, on all the rows and columns the model hands it (31.7 thousand by 50
thousand at 26 weeks, of which 19 thousand columns are fixed and 11 thousand rows have no bound), and an iteration
costs 0.25 ms against 0.07 ms on the presolved program: the reason a warm start was slower. Here the fixed columns
and the unbounded rows are taken out before the program is handed over (``cut``; the fixed columns' part goes into
the row bounds and the offset), the basis is cut the same way, and the solution is put back.

For every pair of consecutive weeks of the kept exact cells: the cold solve of the second as the model does it, and
the solve from the optimal basis of the first, moved on by a week (``plan_core.Episode.shifted``'s rule), on the
whole program and on the cut one; CPU seconds, simplex iterations, the objective against the cold one.
"""

import time
from pathlib import Path

import fire
import lp_bench as B
import numpy as np
import scipy.sparse as sp


def cut(d: dict, basis: tuple | None = None):
    """The program without its fixed columns and its unbounded rows; the masks of what is kept."""
    n_row, n_col = (int(v) for v in d["shape"])
    A = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=(n_row, n_col))
    lb, ub = d["lb"], d["ub"]
    keep_c = lb != ub
    keep_r = ~(np.isinf(d["lo"]) & np.isinf(d["hi"]))
    xf = np.where(keep_c, 0.0, lb)
    shift = A @ xf
    A2 = A[:, keep_c].tocsr()[keep_r].tocsc()
    A2.sort_indices()
    out = dict(d)
    out.update(indptr=A2.indptr, indices=A2.indices, data=A2.data, shape=np.array(A2.shape),
               lo=(d["lo"] - shift)[keep_r], hi=(d["hi"] - shift)[keep_r], lb=lb[keep_c], ub=ub[keep_c],
               cost=d["cost"][keep_c], offset=float(d["offset"]) + float(d["cost"] @ xf))  # fmt: skip
    small = None if basis is None else (basis[0][keep_c], basis[1][keep_r])
    return out, keep_c, keep_r, small


def shifted(d: dict, basis: tuple) -> tuple:
    """``plan_core.Episode.shifted`` for a capped window of the same length: every week takes the next week's
    statuses, the last week keeps its own; the pools' columns and rows stay."""
    col, row = basis
    P = int(d["N"]) - int(d["n2"])
    T, nb, extra, dev = int(d["T"]), int(d["nc"]), 2 * int(d["G"]), (int(d["n2"]) - int(d["n1"])) // int(d["T"])
    col, col_end, row, row_end = (col[:-P], col[-P:], row[:-P], row[-P:]) if P else (col, col[:0], row, row[:0])
    n_ub, n_eq = int(d["nub"]) * T, int(d["neq"]) * T
    per = (len(row) - n_ub - n_eq) // T
    move = lambda a, size: np.concatenate([a[size:], a[len(a) - size :]])  # noqa: E731
    cols = [move(col[: nb * T], nb), move(col[nb * T : (nb + extra) * T], extra), move(col[(nb + extra) * T :], dev)]
    rows = [move(row[:n_ub], int(d["nub"])), move(row[n_ub : n_ub + n_eq], int(d["neq"])),
            move(row[n_ub + n_eq :], per)]  # fmt: skip
    return np.concatenate(cols + [col_end]), np.concatenate(rows + [row_end])


def run_from(d: dict, basis: tuple | None, options: dict | None = None, limit: float = 600.0) -> dict:
    hs = B._hs()
    h = B.solver(d, {"time_limit": float(limit)} | (options or {}))
    h.passModel(B.lp_of(d))
    if basis is not None:
        codes = {int(v): v for v in (hs.HighsBasisStatus.kLower, hs.HighsBasisStatus.kBasic, hs.HighsBasisStatus.kUpper,
                                     hs.HighsBasisStatus.kZero, hs.HighsBasisStatus.kNonbasic)}  # fmt: skip
        b = hs.HighsBasis()
        b.col_status, b.row_status = [codes[int(i)] for i in basis[0]], [codes[int(i)] for i in basis[1]]
        b.valid, b.alien = True, True
        h.setBasis(b)
    t0 = time.process_time()
    h.run()
    cpu = time.process_time() - t0
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    out = {"cpu": cpu, "status": status, "J": float(info.objective_function_value),
           "it": int(info.simplex_iteration_count)}  # fmt: skip
    if status == "Optimal":
        got = h.getBasis()
        out["basis"] = (np.array([int(v) for v in got.col_status], dtype=np.int8),
                        np.array([int(v) for v in got.row_status], dtype=np.int8))  # fmt: skip
        out["x"] = np.asarray(h.getSolution().col_value, dtype=float)
    return out


def whole(keep_c, keep_r, small: tuple) -> tuple:
    """A cut program's basis as the whole program's: a fixed column at its bound, an unbounded row basic."""
    col, row = np.zeros(len(keep_c), dtype=np.int8), np.ones(len(keep_r), dtype=np.int8)
    col[keep_c], row[keep_r] = small
    return col, row


def main(folder: str, first: int = 0, count: int = 99, limit: float = 5.0, full: bool = True) -> None:
    files = sorted(Path(folder).glob("*_exact.npz"))[first : first + count]
    prev, rows = None, []
    for path in files:
        d = np.load(path)
        d = {k: d[k] for k in d.files}
        cold = run_from(d, None)
        line = f"{path.name:20s} cold {cold['cpu']:.3f} s {cold['it']:6d} it"
        if prev is not None and cold["status"] == "Optimal":
            start = shifted(d, prev)
            small, keep_c, keep_r, start_small = cut(d, start)
            warm_small = run_from(small, start_small, limit=limit)
            gap = (warm_small["J"] - cold["J"]) / abs(cold["J"]) if warm_small["status"] == "Optimal" else float("nan")
            line += (f" | from last week's basis, cut to {small['shape'][0]} x {small['shape'][1]}: "
                     f"{warm_small['cpu']:.3f} s {warm_small['it']:6d} it {warm_small['status']} J {gap:+.1e}")
            row = [cold["cpu"], warm_small["cpu"], float(warm_small["status"] == "Optimal")]
            if full:
                warm_full = run_from(d, start, limit=limit)
                line += f" | whole program: {warm_full['cpu']:.3f} s {warm_full['it']:6d} it {warm_full['status']}"
                row.append(warm_full["cpu"])
            rows.append(row)
        print(line, flush=True)
        prev = cold.get("basis")
    a = np.array(rows)
    print(f"{len(a)} pairs: cold median {np.median(a[:, 0]):.3f} mean {a[:, 0].mean():.3f} s; cut and warm median "
          f"{np.median(a[:, 1]):.3f} mean {a[:, 1].mean():.3f} s, solved {int(a[:, 2].sum())} of {len(a)} within "
          f"{limit} s"
          + (f"; whole and warm median {np.median(a[:, 3]):.3f} mean {a[:, 3].mean():.3f} s" if full else ""))


if __name__ == "__main__":
    fire.Fire(main)
