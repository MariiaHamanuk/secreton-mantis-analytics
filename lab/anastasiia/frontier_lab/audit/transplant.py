"""Is the integer plan's set of whole weeks executable? Its whole-week pattern over all 52 weeks is put into the team's
own cell and descent, from the best executed plan. Usage: transplant.py <episode> <kept60|new>. Scratchpad only."""
import pickle, sys, time
from pathlib import Path
import numpy as np

R = Path(str(__import__("pathlib").Path(__file__).resolve().parents[3]))
SP = Path(__file__).resolve().parent
sys.path[:0] = [str(R / "lab/anastasiia/frontier_lab")]
import opening as O  # noqa: E402
import core  # noqa: E402

n, src = int(sys.argv[1]), sys.argv[2]
bn = 1e11
ep = core.Episode.of("small", 444, n)
inst, marks, m = ep.inst, ep.marks, ep.m
T, nc, tmpl, G = m.T, m.meta["nc"], m.meta["template"], len(inst.grids)
tb = O._tables(ep)
ybar = np.asarray(marks.y_bar, dtype=float)
x = pickle.loads((SP / f"milp_{n}.pkl").read_bytes())["x"] if src == "new" else pickle.loads((R / f"outputs/regime_lab/basefirst/small_444_{n}_tl60.pkl").read_bytes())["x"]
x = np.asarray(x, float)
Jx = float(m.objective() @ x) + ep.offset
pshed = np.array([[x[t * nc + tmpl[("ysh", go)]] for go in range(G)] for t in range(T)])
print(f"ep {n}: integer plan ({src}) own cost {Jx / 1e9:.1f} bn", flush=True)

# the best executed plan (replayed by milp_start.py or rebuilt here)
bp = SP / f"best_exec_{n}.pkl"
if bp.is_file():
    best_acts = pickle.loads(bp.read_bytes())["acts"]
else:
    start = pickle.loads(O._start_path("h3_s", "small", 444, n).read_bytes())
    base = ep.validated(start["actions"])
    data = pickle.loads(O._run_path("h3_s", "small", 444, n).read_bytes())
    cands = {k: (data["base"]["J"] if "same_as" in r else r["J"]) for k, r in data.items() if not k.startswith("_")}
    label = min(cands, key=cands.get); r = data[label]
    acts = base if (label == "base" or "same_as" in r) else O.pattern(base, tb, {(t, tb["names"].index(g)) for t, g in r["cells"]}) if label.startswith("plan_") else O.opening(base, tb, [tb["names"].index(g) for g in r["grids"]], r["K"], "v" in r["parts"], "w" in r["parts"], r.get("offset", 0))
    best_acts = O._descent(core, ep, acts, 60, 3, 8)["acts"]
recs0, J0 = ep.simulate(best_acts)
print(f"best executed plan: {J0 / bn:.1f} bn; above the integer plan by {(J0 / 100 - Jx) / 1e9:.1f} bn", flush=True)
shed0 = np.array([r.shed for r in recs0])
F = tb["with_fabs"]
short_p = pshed > 5e-3 * ybar   # the plan leaves the week short
whole_p = pshed < 5e-4 * ybar
short_0 = shed0 > 5e-3 * ybar
whole_0 = shed0 < 5e-4 * ybar
print("grid-weeks of grids with fabs: plan whole %d, executed whole %d; plan whole & executed short %d; plan short & executed whole %d" % (
    int(whole_p[:, F].sum()), int(whole_0[:, F].sum()), int((whole_p & short_0)[:, F].sum()), int((short_p & whole_0)[:, F].sum())), flush=True)

def descent(acts, tag):
    t0 = time.process_time()
    d = O._descent(core, ep, acts, 60, 3, 8)
    print(f"  {tag:46s} start {d['J0'] / bn:8.1f} -> {d['J'] / bn:8.1f} bn ({len(d['hist'])} passes, {time.process_time() - t0:.0f} s); to best executed {(d['J'] - J0) / bn:+.1f}", flush=True)
    return d

def soft_cell(acts, until):
    """One cell from the trajectory of ``acts`` with every short week the plan closes asked to be whole ("SOFT")."""
    recs, J = ep.simulate(acts)
    mode, ref = ep.regimes(recs)
    asked = 0
    for (t, gi), gm in list(mode["grid"].items()):
        if gi in F and gm == "OFF" and t <= until and whole_p[t - 1, gi]:
            mode["grid"][(t, gi)] = "SOFT"; asked += 1
    C = ep.cell(mode, ref)
    sol = ep.solve(C, method="simplex")
    if sol["status"] != "Optimal":
        return None, asked, sol["status"], J
    a2 = ep.actions(sol["x"])
    r2, J2 = ep.simulate(a2)
    closed = sum(1 for (t, gi), gm in mode["grid"].items() if gm == "SOFT" and r2[t - 1].shed[gi] < 5e-4 * ybar[t - 1, gi])
    return a2, asked, f"claim {sol['J'] / 1e9:.1f} bn, played {J2 / bn:.1f} bn, {closed} of {asked} asked weeks closed", J2

out = {}
# (a) the plan's own actions, played blind, then the descent with hull rounds and the search
out["plan actions"] = descent(ep.actions(x), "plan's own actions (blind replay)")
# (b) from the best executed plan: ask for the plan's whole weeks (SOFT), then descend
for until in (T - 12, T):
    a2, asked, note, J2 = soft_cell(best_acts, until)
    print(f"  ask the plan's whole weeks up to week {until}: {asked} asked; {note}", flush=True)
    if a2 is not None:
        out[f"soft{until}"] = descent(a2, f"best executed + plan's whole weeks (<= {until})")
# (c) open the weeks the plan leaves short (valves, wafers), all 52 weeks, then ask for its whole weeks, then descend
cells = {(t, gi) for t in range(2, T + 1) for gi in F if short_p[t - 1, gi] and whole_0[t - 1, gi]}
a1 = O.pattern(best_acts, tb, cells)
r1, J1 = ep.simulate(a1)
print(f"  cut {len(cells)} grid-weeks the plan leaves short: played {J1 / bn:.1f} bn", flush=True)
out["cut"] = descent(a1, "best executed with the plan's short weeks cut")
a2, asked, note, J2 = soft_cell(a1, T - 12)
print(f"  cut, then ask the plan's whole weeks: {asked} asked; {note}", flush=True)
if a2 is not None:
    out["cut+soft"] = descent(a2, "cut + plan's whole weeks")
res = {k: int(d["J"]) for k, d in out.items()}
bestk = min(res, key=res.get)
print(f"ep {n}: best executed {J0 / bn:.1f}; best transplant '{bestk}' {res[bestk] / bn:.1f} ({(res[bestk] - J0) / bn:+.1f}); integer plan {Jx / 1e9:.1f}", flush=True)
pickle.dump({"J0": J0, "Jx": Jx, "res": res}, open(SP / f"transplant_{n}_{src}.pkl", "wb"))
if res[bestk] < J0:
    pickle.dump({"acts": out[bestk]["acts"], "J": res[bestk]}, open(SP / f"transplant_best_{n}.pkl", "wb"))
