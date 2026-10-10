import pickle, sys, glob, numpy as np
from pathlib import Path
R = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3])); SP = Path(__file__).resolve().parent
sys.path.insert(0, str(R / "lab/anastasiia/frontier_lab"))
import opening as O
W = O.WEIGHTS; bn = 1e11
refs = O._refs("small", 444, 64)
level = np.array([r["stratum"] for r in refs])[:24]; naive = np.array([r["J_naive_cents"] for r in refs], float)[:24]; oracle = np.array([r["J_oracle_cents"] for r in refs], float)[:24]
z = np.load(R / "outputs/plan_stats/20261006_040041/episodes.npz")
plan = z["J_plan"][:24] * 100; gap = z["mip_gap"][:24]; bound = plan * (1 - gap)
P = np.arange(24)
def kept(tag): return pickle.loads((R / f"outputs/hazard_lab/play/{tag}_small_444.pkl").read_bytes())
h3 = np.array([kept("h3_s")[n]["J"] for n in P], float); win = np.array([kept("truthall_s")[n]["J"] for n in P], float); allf = np.array([kept("truthall_h0_s")[n]["J"] for n in P], float); p3 = np.array([kept("h3p3_s")[n]["J"] for n in P], float)
op = {n: pickle.loads((R / f"outputs/frontier_lab/opening/runs/h3_s_small_444_{n}.pkl").read_bytes()) for n in P}
desc = np.array([op[n]["base"]["J"] for n in P], float)
best = np.array([min((op[n]["base"]["J"] if "same_as" in r else r["J"]) for k, r in op[n].items() if not k.startswith("_")) for n in P], float)
rss = lambda J, pick=slice(None): O._rss(level[pick], naive[pick], oracle[pick], np.asarray(J, float)[pick])
# every executed plan with the whole future that is on disk or was played here
exe = np.minimum(best, allf)
print("eps 0-7: saved descents of regime_lab against the best of 28 starts (bn, negative = cheaper)")
for n in range(8):
    row = []
    for f in sorted(glob.glob(str(R / f"outputs/regime_lab/*/small_444_{n}_*.pkl"))):
        d = pickle.loads(open(f, "rb").read())
        if isinstance(d, dict) and "J" in d and "acts" in d:
            row.append((Path(f).parent.name + ":" + Path(f).stem.split("_")[-1], d["J"]))
    if row:
        k, J = min(row, key=lambda kv: kv[1])
        print(f"  ep {n}: best28 {best[n] / bn:8.1f}; best saved {k} {J / bn:8.1f} ({(J - best[n]) / bn:+.1f})")
        exe[n] = min(exe[n], J)
new_bound = bound.copy(); new_inc = plan.copy()
for n in P:
    for f in (SP / f"transplant_{n}_new.pkl", SP / f"transplant_{n}_kept60.pkl"):
        if f.is_file():
            d = pickle.loads(f.read_bytes()); J = min(d["res"].values())
            print(f"  ep {n} transplant {f.stem.split('_')[-1]}: best {J / bn:.1f} against best28 {best[n] / bn:.1f} ({(J - best[n]) / bn:+.1f})")
            exe[n] = min(exe[n], J)
    f = SP / f"milp_{n}.pkl"
    if f.is_file():
        d = pickle.loads(f.read_bytes())
        print(f"  ep {n} MILP re-solve: incumbent {d['J'] / 1e9:.1f} (kept {plan[n] / bn:.1f}), bound {d['bound'] / 1e9:.1f} (kept {bound[n] / bn:.1f})")
        new_bound[n] = max(new_bound[n], d["bound"] * 100); new_inc[n] = min(new_inc[n], d["J"] * 100)
print("\nSmall 444 eps 0-23:")
for nm, J in [("h3_s", h3), ("h3p3_s", p3), ("told the window", win), ("told everything", allf), ("descent", desc), ("best of 28 starts", best), ("best executed, everything on disk + here", exe),
              ("MILP incumbent (kept)", plan), ("MILP incumbent (best known)", new_inc), ("min(incumbent, executed)", np.minimum(new_inc, exe)), ("MILP bound (kept)", bound), ("MILP bound (best known)", new_bound)]:
    print(f"  {nm:42s} {rss(J):.4f}")
# bootstrap of rung sizes (episodes resampled inside harm levels)
rng = np.random.default_rng(0)
groups = [np.flatnonzero(level == s) for s in W]
picks = [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(4000)]
def iv(a, b):
    v = [rss(b, p) - rss(a, p) for p in picks]
    return rss(b) - rss(a), np.percentile(v, 5), np.percentile(v, 95)
print("\nrung sizes on eps 0-23 with 90 % intervals (resampling inside levels):")
for nm, a, b in [("h3_s -> told the window", h3, win), ("told the window -> told everything", win, allf), ("h3_s -> told everything", h3, allf), ("told everything -> descent", allf, desc), ("descent -> best28", desc, best),
                 ("told everything -> best executed", allf, exe), ("h3_s -> best executed", h3, exe), ("best executed -> incumbent(best known)", exe, new_inc), ("best executed -> bound(best known)", exe, new_bound), ("bound -> clairvoyant", new_bound, oracle), ("h3_s -> h3p3_s", h3, p3)]:
    d, lo, hi = iv(a, b)
    print(f"  {nm:40s} {d:+.4f} ({lo:+.4f} to {hi:+.4f})")
# conversion to the formal set: rung sizes in bn per episode by level; level 3 as measured (3 episodes) and pooled with levels 1-2
print("\nconversion: weighted bn per episode; (a) levels as measured, (b) level 3 := mean of levels 1 and 2; RSS on Small 222 (0.01 = 10.1 bn)")
tot_a = tot_b = 0.0
for nm, a, b in [("h3_s -> told the window", h3, win), ("told the window -> told everything", win, allf), ("told everything -> best executed", allf, exe), ("best executed -> incumbent", exe, new_inc), ("incumbent -> bound", new_inc, new_bound), ("bound -> clairvoyant", new_bound, oracle)]:
    m = {s: ((a - b)[level == s]).mean() / bn for s in W}
    va = sum(W[s] * m[s] for s in W); vb = va - W[3] * m[3] + W[3] * (m[1] + m[2]) / 2
    tot_a += va; tot_b += vb
    print(f"  {nm:36s} (a) {va:5.1f} bn = {va / 1010:.4f}   (b) {vb:5.1f} bn = {vb / 1010:.4f}   by level " + " ".join(f"{m[s]:5.1f}" for s in W))
print(f"  sum (a) {tot_a:.1f} bn, (b) {tot_b:.1f} bn; the formal gap of anastasiia_plan_hazard on Small 222 is 102 bn (0.101)")
tight = gap <= 0.002
print("\ntight episodes:", [int(n) for n in P[tight]], "best executed - bound(best known), bn:", np.round(((exe - new_bound) / bn)[tight], 1).tolist(), "mean %.1f" % ((exe - new_bound) / bn)[tight].mean(), "| share of their room %.4f" % ((exe - new_bound)[tight].sum() / (naive - oracle)[tight].sum()))
print("loose episodes:", [int(n) for n in P[~tight]], "best executed - bound, bn:", np.round(((exe - new_bound) / bn)[~tight], 1).tolist(), "mean %.1f" % ((exe - new_bound) / bn)[~tight].mean())
print("room share of tight episodes: %.3f" % ((naive - oracle)[tight].sum() / (naive - oracle).sum()))
