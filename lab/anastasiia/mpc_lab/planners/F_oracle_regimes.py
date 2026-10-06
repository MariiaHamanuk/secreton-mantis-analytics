"""Solve D's full-horizon base-rule MILP and save the regimes s (grid, week) per episode: python F_oracle_regimes.py --episodes=4"""
import pickle, sys, time
import fire, numpy as np
from joblib import Parallel, delayed
import planner_D as D


def one(n, tl):
    from scipy import sparse
    from scipy.optimize import LinearConstraint, Bounds, milp
    inst, marks, m, A_eq, b_eq, obj, lb, ub, ex, nb = D.build(n, ("base",))
    n0 = len(obj); N = ex.ncol
    pad = lambda A: sparse.hstack([A, sparse.csr_matrix((A.shape[0], N - n0))]).tocsr()
    Ex = sparse.coo_matrix((ex.v, (ex.r, ex.c)), shape=(ex.nrow, N)).tocsr()
    cons = [LinearConstraint(pad(A_eq), b_eq, b_eq), LinearConstraint(pad(m.A_ub), -np.inf, m.b_ub), LinearConstraint(Ex, np.array(ex.rlo), np.array(ex.rhi))]
    c = np.concatenate([obj, np.zeros(N - n0)])
    res = milp(c, constraints=cons, integrality=np.concatenate([np.zeros(n0), ex.integer]),
               bounds=Bounds(np.concatenate([lb, ex.lb]), np.concatenate([ub, ex.ub])), options=dict(time_limit=tl, mip_rel_gap=1e-3))
    x = np.asarray(res.x)
    s = np.round(x[n0:]).astype(int)  # in creation order: for t, for go (base_first grids)
    inst_grids = [go for go, g in enumerate(inst.grids) if inst.nodes[g].grid.priority == "base_first"]
    T = m.T
    S = {}
    i = 0
    for t in range(1, T + 1):
        for go in inst_grids:
            S[(go, t)] = int(s[i]); i += 1
    print(n, "done", res.status, getattr(res, "mip_gap", None), len(S), flush=True)
    pickle.dump(S, open(f"oracle_regimes_{n}.pkl", "wb"))
    return n


def main(episodes=4, tl=200, n_jobs=3):
    Parallel(n_jobs=n_jobs)(delayed(one)(n, tl) for n in range(episodes))

if __name__ == "__main__":
    fire.Fire(main)
