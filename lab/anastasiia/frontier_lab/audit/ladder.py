import pickle, sys, numpy as np
from pathlib import Path
R = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(R / "lab/anastasiia/frontier_lab"))
import opening as O
refs = O._refs("small", 444, 64)
W = O.WEIGHTS
z = np.load(R / "outputs/plan_stats/20261006_040041/episodes.npz")
ns = list(range(24))
level = np.array([refs[n]["stratum"] for n in range(64)])
naive = np.array([refs[n]["J_naive_cents"] for n in range(64)], float)
oracle = np.array([refs[n]["J_oracle_cents"] for n in range(64)], float)
assert np.allclose(z["J_relaxed"][:40] * 100, oracle[:40], rtol=1e-6)
plan = z["J_plan"] * 100.0; gap = z["mip_gap"]; bound = plan * (1 - gap)
data = {n: pickle.loads((R / f"outputs/frontier_lab/opening/runs/h3_s_small_444_{n}.pkl").read_bytes()) for n in ns}
def c(n, r, base): return float(base if "same_as" in r else r["J"])
start = np.array([data[n]["_meta"]["J_start"] for n in ns], float)
base = np.array([data[n]["base"]["J"] for n in ns], float)
best = np.array([min(c(n, r, base[i]) for k, r in data[n].items() if not k.startswith("_")) for i, n in enumerate(ns)])
nst = [sum(1 for k in data[n] if not k.startswith("_")) for n in ns]
def rss(J, pick):
    pick = np.asarray(pick)
    lv, na, orc = level[pick], naive[pick], oracle[pick]
    return O._rss(lv, na, orc, np.asarray(J))
P = np.arange(24)
print("starts per episode", sorted(set(nst)))
print("levels", [int((level[P] == s).sum()) for s in W], " status", z["status"][:24].tolist())
for name, J in [("start (h3_s played)", start), ("base descent", base), ("best start", best), ("MILP incumbent", plan[P]), ("MILP bound", bound[P]),
                ("max(bound, .) trivially", np.maximum(bound[P], oracle[P]))]:
    print(f"{name:26s} {rss(J, P):.4f}   0-7: {rss(J[:8], P[:8]):.4f}")
bn = 1e11
print(" ep lvl  room   start-best best-inc inc-bound bound-orc  gap%   best-bound  RSSep(best) RSSep(bound)")
for i, n in enumerate(ns):
    print(f"{n:3d} {level[n]}  {(naive[n]-oracle[n])/bn:7.1f} {(start[i]-best[i])/bn:8.1f} {(best[i]-plan[n])/bn:8.1f} {(plan[n]-bound[n])/bn:8.1f} {(bound[n]-oracle[n])/bn:8.1f} {100*gap[n]:6.2f} {(best[i]-bound[n])/bn:8.1f}   {(naive[n]-best[i])/(naive[n]-oracle[n]):.3f}  {(naive[n]-bound[n])/(naive[n]-oracle[n]):.3f}")
d = (best - bound[P]) / bn
print("best-bound: mean %.1f, sorted" % d.mean(), np.round(np.sort(d)[::-1], 1).tolist())
print("share of top 4 / top 8:", np.sort(d)[::-1][:4].sum() / d.sum(), np.sort(d)[::-1][:8].sum() / d.sum())
# weighted contributions by level (in RSS units)
den = sum(w * (naive - oracle)[P][level[P] == s].mean() for s, w in W.items())
for nm, a, b in [("start->best", start, best), ("best->inc", best, plan[P]), ("inc->bound", plan[P], bound[P]), ("bound->oracle", bound[P], oracle[P])]:
    parts = [w * (a - b)[level[P] == s].mean() / den for s, w in W.items()]
    print(f"{nm:14s} total {sum(parts):.4f}  by level " + " ".join(f"{x:.4f}" for x in parts))
print("0.01 RSS in bn/ep (weighted room):", den / bn / 100)
# episodes with tight MILP (gap<=0.2%): what is best-inc there
tight = gap[P] <= 0.002
print("tight eps:", int(tight.sum()), "mean best-inc bn", ((best - plan[P])[tight] / bn).mean(), " loose eps mean best-inc", ((best - plan[P])[~tight] / bn).mean(), "loose inc-bound", ((plan[P] - bound[P])[~tight] / bn).mean())
print("tight: sum room share", (naive - oracle)[P][tight].sum() / (naive - oracle)[P].sum())
print("eps where best executed < incumbent:", [(n, round((plan[n] - best[i]) / bn, 1)) for i, n in enumerate(ns) if best[i] < plan[n]])
