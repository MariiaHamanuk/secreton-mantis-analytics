"""Base-load-first MILP of one Small 444 episode, started from the best executed plan (lifted) or the kept 60 s plan.
Usage: milp_start.py <episode> <time_limit_s>. Writes only to the scratchpad."""
import pickle, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

R = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
SP = Path(__file__).resolve().parent
sys.path[:0] = [str(R / "lab/anastasiia/frontier_lab")]
import opening as O  # noqa: E402  (adds regime_lab, mpc_lab, stats_lab to the path)
import core  # noqa: E402
import highspy  # noqa: E402
from shockbench_flow.oracle.lp import lp_cents  # noqa: E402
from shockbench_flow.policies import lp_common as L  # noqa: E402

n, tl = int(sys.argv[1]), float(sys.argv[2])
bn = 1e9  # USD
t_all = time.time()
ep = core.Episode.of("small", 444, n)
inst, marks, m = ep.inst, ep.marks, ep.m
T, nc, tmpl, G = m.T, m.meta["nc"], m.meta["template"], len(inst.grids)
ncol = T * nc
refs = O._refs("small", 444, 64)
J_or, J_na = refs[n]["J_oracle_cents"] / 100, refs[n]["J_naive_cents"] / 100
zs = np.load(R / "outputs/plan_stats/20261006_040041/episodes.npz")
J_inc0, gap0 = float(zs["J_plan"][n]), float(zs["mip_gap"][n])
print(f"ep {n} level {refs[n]['stratum']}: naive {J_na/bn:.1f} oracle {J_or/bn:.1f} | kept MILP incumbent {J_inc0/bn:.1f} bound {J_inc0*(1-gap0)/bn:.1f} (gap {100*gap0:.2f} %)", flush=True)

# ----- the best executed plan of the opening runs, replayed here to have its records -----
start = pickle.loads(O._start_path("h3_s", "small", 444, n).read_bytes())
base = ep.validated(start["actions"])
tb = O._tables(ep)
data = pickle.loads(O._run_path("h3_s", "small", 444, n).read_bytes())
cands = {k: (data["base"]["J"] if "same_as" in r else r["J"]) for k, r in data.items() if not k.startswith("_")}
label = min(cands, key=cands.get)
r = data[label]
if label == "base" or "same_as" in r:
    acts = base
elif label.startswith("plan_"):
    acts = O.pattern(base, tb, {(t, tb["names"].index(g)) for t, g in r["cells"]})
else:
    acts = O.opening(base, tb, [tb["names"].index(g) for g in r["grids"]], r["K"], "v" in r["parts"], "w" in r["parts"], r.get("offset", 0))
t0 = time.process_time()
d = O._descent(core, ep, acts, 60, 3, 8)
print(f"best start '{label}': recorded {cands[label]/1e11:.1f} bn, replayed here {d['J']/1e11:.1f} bn ({time.process_time()-t0:.0f} s CPU)", flush=True)
recs = d["recs"]
mode, ref = ep.regimes(recs)
z_exec = ep.vector(recs, ref)[: ep.n0]
print(f"lifted trajectory: LP cost {lp_cents(m, z_exec)/1e11:.2f} bn, played {d['J']/1e11:.2f} bn", flush=True)
pickle.dump({"acts": d["acts"], "J": d["J"], "label": label}, open(SP / f"best_exec_{n}.pkl", "wb"))

# ----- the MILP of plan_stats.base_first_plan, as a HiGHS model -----
rows, cols, vals, rhs = [], [], [], []
for t in range(T):
    for go in range(G):
        z, rr = ncol + t * G + go, len(rhs)
        y_bar = float(marks.y_bar[t, go])
        rows += [rr, rr]; cols += [t * nc + tmpl[("ysh", go)], z]; vals += [1.0, y_bar]; rhs.append(y_bar)
        for fo in inst.grid_fabs[go]:
            if ("E", fo) not in tmpl:
                continue
            fab = inst.nodes[inst.fabs[fo]].fab
            most = fab.e * float(marks.alpha_bar[t, fo]) * fab.cap0 * (1 + 1e-6) + 1e-9
            rr = len(rhs)
            rows += [rr, rr]; cols += [t * nc + tmpl[("E", fo)], z]; vals += [1.0, -most]; rhs.append(0.0)
extra = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), ncol + T * G))
A, row_lb, row_ub = L._stacked(m)
A = sp.vstack([sp.hstack([A, sp.csr_matrix((A.shape[0], T * G))], format="csr"), extra], format="csr")
rl = np.concatenate([row_lb, np.full(len(rhs), -np.inf)]); ru = np.concatenate([row_ub, np.array(rhs)])
c = np.concatenate([m.objective(), np.zeros(T * G)])
lb = np.concatenate([m.lb, np.zeros(T * G)]); ub = np.concatenate([m.ub, np.ones(T * G)])
off = L.model_offset(m)

def lift(x):
    x = np.asarray(x, float)
    zb = np.array([[1.0 if x[t * nc + tmpl[("ysh", go)]] <= 1e-7 * max(1.0, float(marks.y_bar[t, go])) else 0.0 for go in range(G)] for t in range(T)]).ravel()
    full = np.concatenate([x, zb])
    act = A @ full
    scale = 1.0 + np.abs(A).dot(np.abs(full))
    viol = np.maximum(np.maximum(rl - act, act - ru), 0.0)
    return full, float(c @ full + off), float(viol.max()), float((viol / scale).max())

cand = {"executed": lift(z_exec)}
kept = R / f"outputs/regime_lab/basefirst/small_444_{n}_tl60.pkl"
if kept.is_file():
    cand["kept60"] = lift(pickle.loads(kept.read_bytes())["x"])
for k, (full, J, v, vr) in cand.items():
    print(f"start candidate {k}: J {J/bn:.2f} bn, max row violation {v:.3g} (relative {vr:.2g})", flush=True)
name = min(cand, key=lambda k: cand[k][1])
h = highspy.Highs()
log = SP / f"milp_{n}.log"
log.unlink(missing_ok=True)
for key, val in (("log_to_console", False), ("log_file", str(log)), ("threads", 1), ("time_limit", tl), ("mip_rel_gap", 1e-4)):
    h.setOptionValue(key, val)
h.passModel(L.highs_lp(c, lb, ub, A, rl, ru, off))
idx = np.arange(ncol, ncol + T * G, dtype=np.int32)
h.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx)))
sol = highspy.HighsSolution()
sol.col_value = cand[name][0]
print("setSolution", name, h.setSolution(sol), flush=True)
t0 = time.time()
h.run()
info = h.getInfo()
J, Jb = info.objective_function_value, info.mip_dual_bound
room = J_na - J_or
print(f"MILP {time.time()-t0:.0f} s wall, status {h.modelStatusToString(h.getModelStatus())}, nodes {info.mip_node_count}")
print(f"incumbent {J/bn:.2f} bn (was {J_inc0/bn:.2f}), dual bound {Jb/bn:.2f} bn (was {J_inc0*(1-gap0)/bn:.2f}), gap {100*info.mip_gap:.2f} %")
print(f"episode RSS: best executed {(J_na - d['J']/100)/room:.4f}, incumbent {(J_na-J)/room:.4f}, bound {(J_na-Jb)/room:.4f}; was incumbent {(J_na-J_inc0)/room:.4f}, bound {(J_na-J_inc0*(1-gap0))/room:.4f}")
print(f"best executed - incumbent {(d['J']/100-J)/bn:.1f} bn; incumbent - bound {(J-Jb)/bn:.1f} bn; best executed - bound {(d['J']/100-Jb)/bn:.1f} bn")
x = np.asarray(h.getSolution().col_value)
pickle.dump({"x": x[:ncol], "z": x[ncol:], "J": J, "bound": Jb, "gap": info.mip_gap, "seconds": time.time() - t0, "start": name}, open(SP / f"milp_{n}.pkl", "wb"))
print(f"total wall {time.time()-t_all:.0f} s")
