"""G0 of R2 (``notes/r_rl.md``): what a unit of stock is worth to the teacher against the model, and how much of the
difference the model could tell from what it sees.

    uv run python lab/anastasiia/frontier_lab/prices/audit.py classes --tag=zero --episodes=0-23
    uv run python lab/anastasiia/frontier_lab/prices/audit.py slots zero zero0 --episodes=0-11

Reads the weeks kept by ``play.py run zero`` (the plain model's own play with the teacher beside it). A row is a
(episode, week, cut, stock slot): the worth of a unit held at the end of the cut's week in the model's program and in
the teacher's (``play.slope``; the cell of each planner's plan with its short weeks as "HULL"), as a share of the
unit's own worth (``play.worth``: a fuel's price of shed base load, a chip's penalty of a lost sale). A cell's duals
also hold the price of staying in its regimes, thousands of times a unit's worth in 0.1 to 0.8 % of the rows: both
worths are cut to -1..10 of the unit's worth before anything is averaged.

By class of stock (commodity, kind of node) and cut (after weeks 1, 4, 13 of the window; "end": the window's last
week where the episode goes on, the model's worth there being its end credit):

- the class's weight: the mean of ``|difference| x the slot's usual stock x the unit's worth`` (bn USD a state), the
  usual stock being the slot's mean stock in the model's kept play; its share of all classes at that cut;
- the model's and the teacher's mean worth (shares of the unit's worth, weighted by the usual stock), and their
  ratio with a 90 % interval (episodes drawn again); the share of rows where the two differ by over a tenth of the
  unit's worth;
- the share of the variance of ``difference x usual stock`` explained out of episode (folds of whole episodes) by
  what the model sees: its own worth of the slot at every cut, the slot's stock and planned stock, the forecast's
  share of nominal capacity into and out of the slot's node in weeks 1, 4 and 13, announced prohibitions on those
  edges, the grids' output, the warning scores, the week. A ridge and boosted trees.
"""

import importlib
import pickle
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import play as P  # noqa: E402


LO, HI = -1.0, 10.0  # a worth is read between these shares of the unit's own worth


def _kind(inst, node: int) -> str:
    nd = inst.nodes[node]
    for attr in ("grid", "fab", "osat"):
        if getattr(nd, attr, None) is not None:
            return attr
    for key in ("sink", "term", "src", "mat"):
        if nd.id.startswith(key):
            return key
    return nd.id.split("_")[0]


def rows_of(tag: str, task: str, entropy: int, episodes) -> dict:
    inst, _params = importlib.import_module("shockbench_flow.hosting.tasks").task_generator(task)
    com = [c.id for c in inst.commodities]
    W = P.worth(inst)
    kept = P.kept(tag, task, entropy)
    base = pickle.loads((ROOT / "outputs" / "hazard_lab" / "play" / f"h3_s_{task}_{entropy}.pkl").read_bytes())
    usual = np.mean([r["stock"].mean(axis=0) for r in base.values()], axis=0)  # by stock slot
    out = {k: [] for k in ("ep", "week", "cut", "slot", "cls", "vS", "vT", "X", "raw")}
    for n in P._numbers(episodes):
        if n not in kept:
            continue
        for w, a in enumerate(kept[n]["audit"], start=1):
            if a is None:
                continue
            slots, cuts = a["slots"], a["cuts"]
            w_slot = np.array([W[inst.stock_slots[s].k] for s in slots])
            label = [("end" if h == a["TS"] else str(h)) for h in cuts]
            rel = {lab: np.clip(a["vS"][i] / w_slot, LO, HI) for i, lab in enumerate(label)}
            for i, h in enumerate(cuts):
                if np.isnan(a["vS"][i]).all():  # the window's end is the episode's: nothing past it
                    continue
                vS, vT = a["vS"][i] / w_slot, a["vT"][i] / w_slot
                for j, s in enumerate(slots):
                    sl = inst.stock_slots[s]
                    u = max(usual[s], 1e-9)
                    own = [float(np.nan_to_num(rel[lab][j])) if lab in rel else 0.0 for lab in ("1", "4", "13", "end")]
                    x = [w / inst.T, a["TS"] / 26.0, float(a["TT"] > a["TS"]), float(np.clip(vS[j], LO, HI)), *own,
                         a["i0"][j] / u, np.nan_to_num(a["iS"][i, j]) / u, *a["routes"][j], *a["pending"][j],
                         float(a["grids"].min()), float(a["grids"].mean()), float(a["warning"].max()), float(a["warning"].mean())]  # fmt: skip
                    out["ep"].append(n), out["week"].append(w), out["cut"].append(label[i]), out["slot"].append(s)
                    out["cls"].append(f"{com[sl.k]} at {_kind(inst, sl.node)}")
                    out["raw"].append(max(abs(vS[j]), abs(vT[j])))
                    out["vS"].append(float(np.clip(vS[j], LO, HI))), out["vT"].append(float(np.clip(vT[j], LO, HI)))
                    out["X"].append(x)
    arr = {k: np.array(v) for k, v in out.items()}
    arr["usual"] = usual
    arr["dollars"] = np.array([usual[s] * W[inst.stock_slots[s].k] for s in range(len(inst.stock_slots))])
    return arr


# ----- two small regressors ------------------------------------------------------------------------------------------
def ridge(Xa, ya, Xb, alpha: float = 1.0) -> np.ndarray:
    mu, sd = Xa.mean(axis=0), Xa.std(axis=0)
    sd[sd < 1e-12] = 1.0
    A, B = (Xa - mu) / sd, (Xb - mu) / sd
    A, B = np.hstack([A, A**2]), np.hstack([B, B**2])
    w = np.linalg.solve(A.T @ A + alpha * len(A) / 100.0 * np.eye(A.shape[1]), A.T @ (ya - ya.mean()))
    return ya.mean() + B @ w


def trees(Xa, ya, Xb, rounds: int = 120, rate: float = 0.1, bins: int = 24, leaf: int = 30) -> np.ndarray:
    """Boosted trees of depth two on quantile bins."""
    F = Xa.shape[1]
    edges = [np.unique(np.quantile(Xa[:, f], np.linspace(0, 1, bins + 1)[1:-1])) for f in range(F)]
    Ba = np.stack([np.searchsorted(edges[f], Xa[:, f]) for f in range(F)], axis=1)
    Bb = np.stack([np.searchsorted(edges[f], Xb[:, f]) for f in range(F)], axis=1)

    def split(idx, g):
        best = (0.0, None, None)
        tot, n = g[idx].sum(), len(idx)
        if n < 2 * leaf:
            return None
        for f in range(F):
            nb = len(edges[f]) + 1
            s = np.cumsum(np.bincount(Ba[idx, f], weights=g[idx], minlength=nb))[:-1]
            c = np.cumsum(np.bincount(Ba[idx, f], minlength=nb))[:-1]
            ok = (c >= leaf) & (n - c >= leaf)
            if not ok.any():
                continue
            gain = np.where(ok, s**2 / np.maximum(c, 1) + (tot - s) ** 2 / np.maximum(n - c, 1) - tot**2 / n, -1.0)
            b = int(np.argmax(gain))
            if gain[b] > best[0]:
                best = (float(gain[b]), f, b)
        return None if best[1] is None else best[1:]

    pa, pb = np.full(len(ya), ya.mean()), np.full(len(Xb), ya.mean())
    for _ in range(rounds):
        g = ya - pa
        nodes = [(np.arange(len(ya)), np.arange(len(Xb)))]
        for _depth in range(2):
            nxt = []
            for ia, ib in nodes:
                sp = split(ia, g)
                if sp is None:
                    nxt.append((ia, ib))
                    continue
                f, b = sp
                nxt += [(ia[Ba[ia, f] <= b], ib[Bb[ib, f] <= b]), (ia[Ba[ia, f] > b], ib[Bb[ib, f] > b])]
            nodes = nxt
        for ia, ib in nodes:
            v = rate * g[ia].mean() if len(ia) else 0.0
            pa[ia] += v
            pb[ib] += v
    return pb


def explained(X, y, groups, folds: int = 6) -> tuple[float, float]:
    """Out-of-episode share of the variance of ``y`` a ridge and the trees explain."""
    ids = np.unique(groups)
    if len(ids) < folds or y.std() < 1e-12:
        return float("nan"), float("nan")
    order = np.random.default_rng(0).permutation(ids)
    sse = {"ridge": 0.0, "trees": 0.0}
    sst = 0.0
    for k in range(folds):
        test = np.isin(groups, order[k::folds])
        if test.sum() == 0 or (~test).sum() < 50:
            continue
        ya, yb = y[~test], y[test]
        sst += float(((yb - ya.mean()) ** 2).sum())
        sse["ridge"] += float(((yb - ridge(X[~test], ya, X[test])) ** 2).sum())
        sse["trees"] += float(((yb - trees(X[~test], ya, X[test])) ** 2).sum())
    return 1.0 - sse["ridge"] / sst, 1.0 - sse["trees"] / sst


def main(
    tag: str = "zero", task: str = "small", entropy: int = 444, episodes="0-23", least: float = 0.02, draws: int = 2000
) -> None:
    d = rows_of(tag, task, entropy, episodes)
    eps = np.unique(d["ep"])
    usd = d["dollars"][d["slot"]] / 1e9  # bn USD: the slot's usual stock at the unit's worth
    diff = (d["vT"] - d["vS"]) * usd
    states = len(np.unique(d["ep"] * 1000 + d["week"]))
    print(
        f"{tag} {task} {entropy}: {len(eps)} episodes ({eps.min()}..{eps.max()}), {states} states, {len(diff)} rows; "
        f"rows with a worth outside {LO:g}..{HI:g} of the unit's: {np.mean((d['raw'] > HI)):.4f}"
    )
    rng = np.random.default_rng(0)
    picks = [rng.choice(eps, len(eps)) for _ in range(draws)]
    for cut in ("1", "4", "13", "end"):
        at = d["cut"] == cut
        total = np.abs(diff[at]).sum()
        n_states = len(np.unique(d["ep"][at] * 1000 + d["week"][at]))
        print(
            f"\ncut after week {cut}: {n_states} states; all classes: mean |difference| x usual stock {total / n_states:.2f} bn a state"
        )
        print(
            f"  {'class':22s} {'share':>6s} {'bn/state':>9s} {'model':>7s} {'teacher':>8s} {'ratio (90 %)':>22s} {'differ':>7s} {'explained: ridge / trees':>25s}"
        )
        table = [(np.abs(diff[at & (d["cls"] == cls)]).sum(), cls) for cls in sorted(set(d["cls"][at]))]
        for mass, cls in sorted(table, reverse=True):
            if mass / total < least:
                continue
            m = at & (d["cls"] == cls)
            wS, wT = d["vS"][m] * usd[m], d["vT"][m] * usd[m]
            ratio = wT.sum() / wS.sum() if abs(wS.sum()) > 0 else float("nan")
            by = {e: (wS[d["ep"][m] == e].sum(), wT[d["ep"][m] == e].sum()) for e in eps}
            rs = [sum(by[e][1] for e in p) / max(sum(by[e][0] for e in p), 1e-12) for p in picks]
            lo, hi = np.percentile(rs, [5, 95])
            r2 = explained(d["X"][m], diff[m], d["ep"][m])
            print(
                f"  {cls:22s} {mass / total:6.2f} {mass / n_states:9.3f} {np.average(d['vS'][m], weights=usd[m]):7.3f} "
                f"{np.average(d['vT'][m], weights=usd[m]):8.3f} {ratio:8.3f} ({lo:.3f} to {hi:.3f}) "
                f"{np.mean(np.abs(d['vT'][m] - d['vS'][m]) > 0.1):7.2f} {r2[0]:14.2f} / {r2[1]:.2f}"
            )


def slots(*tags: str, task: str = "small", entropy: int = 444, episodes="0-23", cuts=("1", "end")) -> None:
    """By stock slot and cut: the model's mean worth of a unit (share of the unit's own worth) and each tag's
    "teacher's" (``zero``: the planner told the future; ``zero0``: the plain model with its window to the episode's
    end), over the states every tag has."""
    inst, _params = importlib.import_module("shockbench_flow.hosting.tasks").task_generator(task)
    com = [c.id for c in inst.commodities]
    data = {t: rows_of(t, task, entropy, episodes) for t in tags}
    keys = {t: d["ep"] * 10**6 + d["week"] * 1000 + d["slot"] for t, d in data.items()}
    for cut in cuts:
        common = None
        for t in tags:
            k = set(keys[t][data[t]["cut"] == cut].tolist())
            common = k if common is None else common & k
        print(
            f"\ncut after week {cut}: {len(common) // max(1, len(set(data[tags[0]]['slot'])))} states every tag has; mean worth as a share of the unit's own"
        )
        print(
            f"  {'slot':34s} {'model':>7s} "
            + " ".join(f"{t:>9s}" for t in tags)
            + "   share of states where the model's is over half and the tag's under a fifth"
        )
        d0 = data[tags[0]]
        for s in sorted(set(d0["slot"].tolist())):
            sl = inst.stock_slots[s]
            row, low = [], []
            for t in tags:
                d = data[t]
                m = (d["cut"] == cut) & (d["slot"] == s) & np.isin(keys[t], list(common))
                if t == tags[0]:
                    model = d["vS"][m].mean() if m.any() else float("nan")
                row.append(d["vT"][m].mean() if m.any() else float("nan"))
                low.append(np.mean((d["vS"][m] > 0.5) & (d["vT"][m] < 0.2)) if m.any() else float("nan"))
            if max(abs(model), *[abs(r) for r in row]) < 0.02:
                continue
            print(
                f"  {inst.nodes[sl.node].id + ' ' + com[sl.k]:34s} {model:7.3f} "
                + " ".join(f"{r:9.3f}" for r in row)
                + "   "
                + " ".join(f"{v:5.2f}" for v in low)
            )


def split(
    told: str = "zero",
    blind: str = "zero0",
    task: str = "small",
    entropy: int = 444,
    episodes="0-11",
    least: float = 0.02,
) -> None:
    """The teacher's worth less the model's, split in two on the states both tags have: what a planner with no
    foresight and its window to the episode's end already says (``blind`` less the model: a longer window, another
    plan) and what the future adds (``told`` less ``blind``). By class and cut: the mean absolute size of each part
    times the usual stock (bn USD a state), the share of the whole difference's variance that is the future's part,
    and the share of the future's part explained out of episode by what the model sees, the blind planner's worth of
    the slot included (the model could compute it)."""
    a, b = rows_of(told, task, entropy, episodes), rows_of(blind, task, entropy, episodes)
    cut_id = {"1": 1, "4": 2, "13": 3, "end": 4}
    ka = a["ep"] * 10**7 + a["week"] * 10**5 + np.array([cut_id[c] for c in a["cut"]]) * 10**3 + a["slot"]
    kb = b["ep"] * 10**7 + b["week"] * 10**5 + np.array([cut_id[c] for c in b["cut"]]) * 10**3 + b["slot"]
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    usd = a["dollars"][a["slot"][ia]] / 1e9
    vS, vT, vB = a["vS"][ia], a["vT"][ia], b["vT"][ib]
    whole, longer, future = (vT - vS) * usd, (vB - vS) * usd, (vT - vB) * usd
    X = np.hstack([a["X"][ia], vB[:, None]])
    ep, cut, cls, week = a["ep"][ia], a["cut"][ia], a["cls"][ia], a["week"][ia]
    print(f"{told} against {blind}, {task} {entropy}: {len(np.unique(ep))} episodes, {len(common)} rows")
    for c in ("1", "4", "13", "end"):
        at = cut == c
        n_states = len(np.unique(ep[at] * 1000 + week[at]))
        total = np.abs(whole[at]).sum()
        print(
            f"\ncut after week {c}: {n_states} states; mean |.| x usual stock, bn a state: the whole difference {total / n_states:.2f}, "
            f"the longer window's part {np.abs(longer[at]).sum() / n_states:.2f}, the future's part {np.abs(future[at]).sum() / n_states:.2f}; "
            f"correlation of the whole with the longer window's part {np.corrcoef(whole[at], longer[at])[0, 1]:+.2f}"
        )
        print(
            f"  {'class':22s} {'whole':>8s} {'longer':>8s} {'future':>8s} {'mean: model':>12s} {'blind':>7s} {'told':>7s} {'future explained: ridge / trees':>32s}"
        )
        table = [(np.abs(whole[at & (cls == k)]).sum(), k) for k in sorted(set(cls[at]))]
        for mass, k in sorted(table, reverse=True):
            if mass / total < least:
                continue
            m = at & (cls == k)
            r2 = explained(X[m], future[m], ep[m])
            print(
                f"  {k:22s} {mass / n_states:8.3f} {np.abs(longer[m]).sum() / n_states:8.3f} {np.abs(future[m]).sum() / n_states:8.3f} "
                f"{np.average(vS[m], weights=usd[m]):12.3f} {np.average(vB[m], weights=usd[m]):7.3f} {np.average(vT[m], weights=usd[m]):7.3f} "
                f"{r2[0]:21.2f} / {r2[1]:.2f}"
            )


if __name__ == "__main__":
    fire.Fire({"classes": main, "slots": slots, "split": split})
