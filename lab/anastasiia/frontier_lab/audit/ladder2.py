import json, pickle, sys, numpy as np
from pathlib import Path
R = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(R / "lab/anastasiia/frontier_lab"))
import opening as O
W = O.WEIGHTS
bn = 1e11  # cents per bn USD
PLAY = R / "outputs/hazard_lab/play"

def kept(tag, task, root):
    p = PLAY / f"{tag}_{task}_{root}.pkl"
    return pickle.loads(p.read_bytes()) if p.is_file() else {}

def rss(level, naive, oracle, J):
    return O._rss(np.asarray(level), np.asarray(naive, float), np.asarray(oracle, float), np.asarray(J, float))

def bylevel(level, naive, oracle, J):
    out = []
    for s in W:
        m = level == s
        out.append(float((naive - J)[m].sum() / (naive - oracle)[m].sum()) if m.any() else float("nan"))
    return out

# ---------------- Small 444 ----------------
refs = O._refs("small", 444, 64)
level = np.array([r["stratum"] for r in refs]); naive = np.array([r["J_naive_cents"] for r in refs], float); oracle = np.array([r["J_oracle_cents"] for r in refs], float)
z = np.load(R / "outputs/plan_stats/20261006_040041/episodes.npz")
plan = np.full(64, np.nan); plan[:40] = z["J_plan"] * 100; gap = np.full(64, np.nan); gap[:40] = z["mip_gap"]; bound = plan * (1 - gap)
tags = ["base_s", "w50as_s", "h3_s", "h3p3_s", "h3max_s", "ends_all_s", "truthall_s", "truthall_h0_s", "w50as_h0_s"]
D = {t: kept(t, "small", 444) for t in tags}
print("episodes kept:", {t: len(D[t]) for t in tags})
P24 = np.arange(24)
op = {n: pickle.loads((R / f"outputs/frontier_lab/opening/runs/h3_s_small_444_{n}.pkl").read_bytes()) for n in P24}
desc = np.array([op[n]["base"]["J"] for n in P24], float)
best = np.array([min((op[n]["base"]["J"] if "same_as" in r else r["J"]) for k, r in op[n].items() if not k.startswith("_")) for n in P24], float)
def col(tag, ns): return np.array([D[tag][n]["J"] for n in ns], float)
print("\n== Small 444 eps 0-23, levels", [int((level[P24] == s).sum()) for s in W])
L = {}
for t in tags:
    if all(n in D[t] for n in P24):
        L[t] = col(t, P24)
L["descent(base)"] = desc; L["best of 28 starts"] = best
L["min(best start, truthall_h0)"] = np.minimum(best, L["truthall_h0_s"])
L["MILP incumbent"] = plan[:24]; L["min(incumbent, best exec)"] = np.minimum(plan[:24], L["min(best start, truthall_h0)"]); L["MILP bound"] = bound[:24]
for k, J in L.items():
    print(f"{k:30s} {rss(level[P24], naive[P24], oracle[P24], J):.4f}  levels " + " / ".join(f"{x:.3f}" for x in bylevel(level[P24], naive[P24], oracle[P24], J)))
th = L["truthall_h0_s"]
print("truthall_h0 cheaper than best descent in eps:", [(int(n), round((best[n] - th[n]) / bn, 1)) for n in P24 if th[n] < best[n]])
print("truthall_h0 minus best descent, bn: mean %.1f median %.1f max %.1f" % (((th - best) / bn).mean(), np.median((th - best) / bn), ((th - best) / bn).max()))
den = sum(w * (naive - oracle)[P24][level[P24] == s].mean() for s, w in W.items())
print("0.01 RSS = %.2f bn/ep" % (den / bn / 100))
tight = gap[:24] <= 0.002
def contrib(a, b, mask):
    return sum(w * (((a - b) * mask)[level[P24] == s]).sum() / (level[P24] == s).sum() for s, w in W.items()) / den
print("tight episodes (MILP gap <= 0.2 %):", int(tight.sum()), "of 24; by level", [int((tight & (level[P24] == s)).sum()) for s in W])
print("rung                         all     tight   loose")
seq = [("h3_s -> truthall(window)", L["h3_s"], L["truthall_s"]), ("window -> truthall_h0", L["truthall_s"], th), ("truthall_h0 -> descent", th, desc),
       ("descent -> best start", desc, best), ("best start -> incumbent", best, plan[:24]), ("incumbent -> bound", plan[:24], bound[:24]), ("bound -> clairvoyant", bound[:24], oracle[P24]),
       ("h3_s -> h3p3_s", L["h3_s"], L["h3p3_s"]), ("h3_s -> clairvoyant", L["h3_s"], oracle[P24]), ("best start -> bound", best, bound[:24])]
for nm, a, b in seq:
    print(f"{nm:28s} {contrib(a, b, 1.0):+.4f} {contrib(a, b, tight):+.4f} {contrib(a, b, ~tight):+.4f}")
# per-level mean bn per episode of the main pieces (for conversion)
print("per level, mean bn/episode:   L1      L2      L3      L4")
for nm, a, b in seq:
    print(f"{nm:28s} " + " ".join(f"{((a - b)[level[P24] == s]).mean() / bn:7.1f}" for s in W))
print("room                         " + " ".join(f"{((naive - oracle)[P24][level[P24] == s]).mean() / bn:7.1f}" for s in W))

# ---------------- Small 444 x64: model, passes, window truth ----------------
P64 = np.arange(64)
print("\n== Small 444 eps 0-63, levels", [int((level == s).sum()) for s in W])
for t in ["w50as_s", "h3_s", "h3p3_s", "truthall_s"]:
    if all(n in D[t] for n in P64):
        J = col(t, P64)
        print(f"{t:14s} {rss(level, naive, oracle, J):.4f}  levels " + " / ".join(f"{x:.3f}" for x in bylevel(level, naive, oracle, J)) + "  gap bn/ep by level " + " ".join(f"{((J - oracle)[level == s]).mean() / bn:6.1f}" for s in W))
J3, Jt, Jp = col("h3_s", P64), col("truthall_s", P64), col("h3p3_s", P64)
print("mean bn/ep by level: h3->window", " ".join(f"{((J3 - Jt)[level == s]).mean() / bn:6.1f}" for s in W), "| h3->p3", " ".join(f"{((J3 - Jp)[level == s]).mean() / bn:6.1f}" for s in W))
print("corr per episode (h3->p3 gain, h3->window gain): %.2f" % np.corrcoef(J3 - Jp, J3 - Jt)[0, 1])
# eps 0-39: model gap vs tax
P40 = np.arange(40)
tax = (bound - oracle)[:40]; g = (J3 - oracle)[:40]
print("eps 0-39: corr(model gap to clairvoyant, bound - clairvoyant) = %.2f; mean gap %.1f bn = tax(bound) %.1f + rest %.1f; rest by level" % (np.corrcoef(g, tax)[0, 1], g.mean() / bn, tax.mean() / bn, (g - tax).mean() / bn),
      " ".join(f"{((g - tax)[level[:40] == s]).mean() / bn:6.1f}" for s in W), "| tax by level", " ".join(f"{(tax[level[:40] == s]).mean() / bn:6.1f}" for s in W),
      "| model->incumbent by level", " ".join(f"{((J3[:40] - plan[:40])[level[:40] == s]).mean() / bn:6.1f}" for s in W))

# ---------------- formal records ----------------
print("\n== formal records")
for model in ["anastasiia_plan_hazard", "anastasiia_plan_hull3", "anastasiia_rules_v2"]:
    rec = json.loads((R / f"hub/eval/records/{model}.json").read_text())
    for name, st in rec["sets"].items():
        rows = [r for r in st["rows"] if not r.get("excluded")]
        lv = np.array([r["stratum"] for r in rows]); na = np.array([r["J_naive_cents"] for r in rows], float); orc = np.array([r["J_clairvoyant_cents"] for r in rows], float); J = np.array([r["J_policy_cents"] for r in rows], float)
        d = sum(w * (na - orc)[lv == s].mean() for s, w in W.items())
        print(f"{model:24s} {name:10s} n={len(rows)} levels {[int((lv == s).sum()) for s in W]} RSS {rss(lv, na, orc, J):.4f} by level " + " / ".join(f"{x:.3f}" for x in bylevel(lv, na, orc, J))
              + f" | 0.01={d / bn / 100:.1f} bn | gap bn/ep " + " ".join(f"{((J - orc)[lv == s]).mean() / bn:6.1f}" for s in W) + " | room " + " ".join(f"{((na - orc)[lv == s]).mean() / bn:7.1f}" for s in W)
              + " | share of weighted gap " + " ".join(f"{100 * w * ((J - orc)[lv == s]).mean() / sum(ww * ((J - orc)[lv == ss]).mean() for ss, ww in W.items()):.0f}%" for s, w in W.items()))
        if model == "anastasiia_plan_hazard":
            # first 64 / 128 and bootstrap spread of 24-episode subsets (first-n composition) of the formal set
            rng = np.random.default_rng(0)
            for k in (24, 64):
                vals = []
                for _ in range(2000):
                    p = rng.choice(len(rows), k, replace=False)
                    if set(lv[p].tolist()) != set(W): continue
                    vals.append(rss(lv[p], na[p], orc[p], J[p]))
                print(f"    random {k}-episode subsets of this set: RSS 5 / 50 / 95 %: " + " / ".join(f"{x:.3f}" for x in np.percentile(vals, [5, 50, 95])))

# ---------------- Full 444 x16 ----------------
print("\n== Full 444 eps 0-15")
try:
    rf = O._refs("full", 444, 16)
    lvf = np.array([r["stratum"] for r in rf]); naf = np.array([r["J_naive_cents"] for r in rf], float); orf = np.array([r["J_oracle_cents"] for r in rf], float)
    print("levels", [int((lvf == s).sum()) for s in W])
    for t in ["base_f", "w50as_f", "h3_f", "ends_all_f", "truthall_f", "look1_f"]:
        d = kept(t, "full", 444)
        ns = [n for n in range(16) if n in d]
        if len(ns) < 16: print(t, "kept", len(ns)); continue
        J = np.array([d[n]["J"] for n in ns], float)
        print(f"{t:12s} all saved over all room {((naf - J).sum() / (naf - orf).sum()):.4f}; gap bn/ep {((J - orf).mean() / bn):.1f}; room bn/ep {((naf - orf).mean() / bn):.1f}")
except Exception as e:
    print("Full refs failed:", repr(e)[:200])
