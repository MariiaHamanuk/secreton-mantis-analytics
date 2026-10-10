"""Where two executed plans of one Full episode differ (notes/u_fullceil.md): the told planner's play against the
best executed plan with the whole future (or any two plans kept by ``record.py``, ``descend.py`` and ``milp.py``).

    uv run python lab/anastasiia/frontier_lab/fullceil/diff.py --only=2,3,9
    uv run python lab/anastasiia/frontier_lab/fullceil/diff.py --only=2 --a=start:tah0_f --b=run:tah0_f --detail

Plans: ``start:<tag>`` (the actions ``record.py`` kept), ``run:<label>`` (where a descent ended), ``milp[:name]`` (the
integer plan's actions played blind), ``best`` (the cheapest run of the episode). Both plans are played again on the
simulator here, so every number is of an executed trajectory. Differences are A minus B: positive = B is cheaper.
"""

import pickle

import common as K
import fire
import numpy as np
from descend import milp_path, run_path
from milp import best_run
from record import start_path


ITEMS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def plan_acts(ep, plan: str, task: str, entropy: int, n: int) -> tuple[list, str]:
    if plan == "best":
        label, run = best_run(task, entropy, n)
        return run["acts"], f"run:{label}"
    if plan.startswith("start:"):
        return ep.validated(pickle.loads(start_path(plan[6:], task, entropy, n).read_bytes())["actions"]), plan
    if plan.startswith("run:"):
        return pickle.loads(run_path(plan[4:], task, entropy, n).read_bytes())["acts"], plan
    if plan.startswith("milp"):
        return pickle.loads(milp_path(task, entropy, n, plan.replace(":", "_")).read_bytes())["acts"], plan
    raise ValueError(plan)


def played(ep, acts: list) -> dict:
    """The simulator's trajectory of ``acts`` as arrays by week."""
    inst = ep.inst
    recs, J = ep.simulate(acts)
    T, G, Kn = len(recs), len(inst.grids), len(inst.commodities)
    seg = np.zeros((T, G, Kn + 1))  # last column: the segment without fuel
    sent = np.zeros((T, len(inst.action_slots)))
    for t, r in enumerate(recs):
        for (gi, k), q in r.segment.items():
            seg[t, gi, -1 if k is None else k] = q
        for s, q in r.executed.items():
            sent[t, s] = q
    return {
        "J": J, "costs": np.array([[getattr(r.costs, c) for c in ITEMS] for r in recs]),
        "shed": np.array([r.shed for r in recs]), "lots": np.array([r.lots_started for r in recs]),
        "energy": np.array([r.energy for r in recs]), "lost": np.array([r.lost for r in recs]),
        "served": np.array([r.served for r in recs]), "stock": np.array([r.stock for r in recs]),
        "disposal": np.array([r.disposal for r in recs]), "lift": np.array([r.lift for r in recs]),
        "scrapped": np.array([r.scrapped for r in recs]), "demand": np.array([r.demand for r in recs]),
        "packaged": np.array([[r.packaged.get((oi, k), 0.0) for k in range(Kn)] for oi in range(len(inst.osats)) for r in recs])
        .reshape(len(inst.osats), T, Kn).sum(axis=0),  # (T, K): packaged of each chip, all plants
        "segment": seg, "sent": sent,
    }  # fmt: skip


def tables(ep) -> dict:
    inst = ep.inst
    N = inst.nodes
    Kid = [c.id for c in inst.commodities]
    packed = {}  # raw chip -> its packaged chip
    for o in inst.osats:
        packed.update(N[o].osat.packages)
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
    fab_grid = {fi: gi for gi, fabs in enumerate(inst.grid_fabs) for fi in fabs}
    return {
        "K": Kid, "grids": [N[g].id for g in inst.grids], "fabs": [N[f].id for f in inst.fabs],
        "fab_grid": np.array([fab_grid.get(fi, -1) for fi in range(len(inst.fabs))]),
        "fab_pi": np.array([pi.get(packed.get(N[f].fab.product, -1), 0.0) for f in inst.fabs]),  # USD a lot's chip is worth unmet
        "fab_cap": np.array([N[f].fab.cap0 for f in inst.fabs]), "fab_product": [Kid[N[f].fab.product] for f in inst.fabs],
        "demands": [(N[d.node].id, Kid[d.k]) for d in inst.demands], "pi": np.array([d.pi for d in inst.demands]),
        "voll": np.array([N[g].grid.voll for g in inst.grids]),
        "with_fabs": [gi for gi, fabs in enumerate(inst.grid_fabs) if fabs],
        "burn": {(gi, k): N[g].grid.shares[k] * N[g].grid.deliverable for gi, g in enumerate(inst.grids) for k in N[g].grid.fuels
                 if N[g].grid.shares.get(k, 0.0) > 0},
        "slot_k": np.array([k for _e, k, _lane in inst.action_slots]),
        "stock_node": [N[s.node].id for s in inst.stock_slots], "stock_k": np.array([s.k for s in inst.stock_slots]),
        "stock_type": [N[s.node].type for s in inst.stock_slots],
        "hold": np.array([0.0 if N[s.node].type == "chokepoint" else s.holding for s in inst.stock_slots]),  # USD a unit a week
        "disp": np.array([inst.commodities[s.k].disposal_cost for s in inst.stock_slots]),  # USD a unit thrown away
    }  # fmt: skip


def whole(ep, shed: np.ndarray) -> np.ndarray:
    """(T, G) True where the grid serves its whole base load that week."""
    ybar = np.asarray(ep.marks.y_bar, dtype=float)[: len(shed)]
    return shed <= 1e-6 * np.maximum(1.0, ybar)


def compare(ep, tb: dict, A: dict, B: dict) -> dict:
    """The numbers of one pair of plans; money in bn USD, A minus B."""
    bn = 1e9
    T = len(A["shed"])
    blocks = [slice(i, min(i + 13, T)) for i in range(0, T, 13)]
    out = {"J": (A["J"] / K.BN, B["J"] / K.BN), "items": (A["costs"].sum(0) - B["costs"].sum(0)) / bn}
    lostA, lostB = A["lost"] * tb["pi"], B["lost"] * tb["pi"]
    out["lost_by_demand"] = (lostA.sum(0) - lostB.sum(0)) / bn
    out["lost_by_block"] = np.array([(lostA[b].sum() - lostB[b].sum()) / bn for b in blocks])
    shedA, shedB = A["shed"] * tb["voll"], B["shed"] * tb["voll"]
    out["shed_by_grid"] = (shedA.sum(0) - shedB.sum(0)) / bn
    out["shed_by_block"] = np.array([(shedA[b].sum() - shedB[b].sum()) / bn for b in blocks])
    wA, wB = whole(ep, A["shed"]), whole(ep, B["shed"])
    out["whole_weeks"] = (wA.sum(0), wB.sum(0))
    out["whole_only_A"], out["whole_only_B"] = (wA & ~wB).sum(0), (wB & ~wA).sum(0)
    valA, valB = A["lots"] * tb["fab_pi"], B["lots"] * tb["fab_pi"]
    out["lots_value_by_fab"] = (valA.sum(0) / bn, valB.sum(0) / bn)
    out["lots_value_by_block"] = np.array([(valA[b].sum() / bn, valB[b].sum() / bn) for b in blocks])
    out["lots_share_of_cap"] = (A["lots"].sum(0) / (T * tb["fab_cap"]), B["lots"].sum(0) / (T * tb["fab_cap"]))
    out["energy_by_grid"] = tuple(np.array([P["energy"][:, tb["fab_grid"] == gi].sum() for gi in range(len(tb["grids"]))]) for P in (A, B))
    # fuel burned by grid and fuel, in weeks of the fuel's full burn
    out["burn_weeks"] = {key: (A["segment"][:, key[0], key[1]].sum() / full, B["segment"][:, key[0], key[1]].sum() / full)
                         for key, full in tb["burn"].items()}
    # fuel thrown away and lifted, by fuel
    fuels = sorted({k for _gi, k in tb["burn"]})
    out["thrown"] = {tb["K"][k]: (A["disposal"][:, tb["stock_k"] == k].sum(), B["disposal"][:, tb["stock_k"] == k].sum()) for k in fuels}
    out["lifted"] = {tb["K"][k]: (A["lift"][:, tb["stock_k"] == k].sum(), B["lift"][:, tb["stock_k"] == k].sum()) for k in fuels}
    out["served"] = (A["served"].sum(0), B["served"].sum(0))
    # chips left at the end, by kind of place
    chips = {}
    for name in ("chip_le_raw", "chip_mat_raw", "chip_le", "chip_mat"):
        k = tb["K"].index(name)
        for kind in sorted(set(tb["stock_type"])):
            m = (tb["stock_k"] == k) & (np.array(tb["stock_type"]) == kind)
            if m.any() and (A["stock"][-1, m].sum() > 0 or B["stock"][-1, m].sum() > 0):
                chips[(name, kind)] = (A["stock"][-1, m].sum(), B["stock"][-1, m].sum())
    out["chips_left"] = chips
    # holding and disposal by commodity, bn, A - B
    out["holding_by_k"] = {name: float(((A["stock"] - B["stock"])[:, tb["stock_k"] == k] * tb["hold"][tb["stock_k"] == k]).sum()) / bn
                           for k, name in enumerate(tb["K"])}
    out["disposal_by_k"] = {name: float(((A["disposal"] - B["disposal"])[:, tb["stock_k"] == k] * tb["disp"][tb["stock_k"] == k]).sum()) / bn
                            for k, name in enumerate(tb["K"])}
    # the chips' balance, thousand units, by family: lots started -> raw chips packaged -> sold; what never sold
    bal = {}
    for raw, fin in (("chip_le_raw", "chip_le"), ("chip_mat_raw", "chip_mat")):
        kr, kf = tb["K"].index(raw), tb["K"].index(fin)
        fabs = np.array([p == raw for p in tb["fab_product"]])
        dem = np.array([d[1] == fin for d in tb["demands"]])
        kinds = np.array(tb["stock_type"])
        row = {}
        for name, P in (("A", A), ("B", B)):
            row[name] = {
                "lots": P["lots"][:, fabs].sum(), "lots_last12": P["lots"][-12:, fabs].sum(), "scrapped": P["scrapped"][:, fabs].sum(),
                "packaged": P["packaged"][:, kf].sum(), "sold": P["served"][:, dem].sum(), "demand": P["demand"][:, dem].sum(),
                "thrown_raw": P["disposal"][:, tb["stock_k"] == kr].sum(), "thrown_fin": P["disposal"][:, tb["stock_k"] == kf].sum(),
                "end_raw_fab": P["stock"][-1, (tb["stock_k"] == kr) & (kinds == "fab")].sum(),
                "end_raw_strait": P["stock"][-1, (tb["stock_k"] == kr) & (kinds == "chokepoint")].sum(),
                "end_raw_plant": P["stock"][-1, (tb["stock_k"] == kr) & (kinds == "osat")].sum(),
                "end_fin_plant": P["stock"][-1, (tb["stock_k"] == kf) & (kinds == "osat")].sum(),
                "end_fin_strait": P["stock"][-1, (tb["stock_k"] == kf) & (kinds == "chokepoint")].sum(),
            }  # fmt: skip
        bal[fin] = row
    out["balance"] = bal
    # how far the dispatches are apart, by commodity: sum of |A - B| over all of both
    out["sent_apart"] = {tb["K"][k]: float(np.abs(A["sent"][:, tb["slot_k"] == k] - B["sent"][:, tb["slot_k"] == k]).sum()
                                           / max(1.0, (A["sent"][:, tb["slot_k"] == k] + B["sent"][:, tb["slot_k"] == k]).sum()))
                         for k in range(len(tb["K"]))}  # fmt: skip
    return out


def show(n: int, tb: dict, c: dict, names: tuple[str, str], detail: bool) -> None:
    G = tb["grids"]
    print(f"\n=== ep {n}: A = {names[0]} {c['J'][0]:.1f} bn, B = {names[1]} {c['J'][1]:.1f} bn, A - B = {c['J'][0] - c['J'][1]:+.1f} bn")
    print("  by item, bn: " + ", ".join(f"{k} {v:+.1f}" for k, v in zip(ITEMS, c["items"]) if abs(v) >= 0.05))
    order = np.argsort(-np.abs(c["lost_by_demand"]))[:8]
    print("  unmet demand by market and chip, bn: " + ", ".join(f"{tb['demands'][i][0]}/{tb['demands'][i][1]} {c['lost_by_demand'][i]:+.1f}" for i in order if abs(c["lost_by_demand"][i]) >= 0.05))
    print("  shed load by grid, bn: " + ", ".join(f"{G[gi]} {v:+.1f}" for gi, v in enumerate(c["shed_by_grid"]) if abs(v) >= 0.05))
    print("  by 13 weeks, unmet: " + " ".join(f"{v:+6.1f}" for v in c["lost_by_block"]) + " | shed: " + " ".join(f"{v:+6.1f}" for v in c["shed_by_block"]))
    print("  whole weeks A / B (only A, only B): " + ", ".join(
        f"{G[gi]} {c['whole_weeks'][0][gi]}/{c['whole_weeks'][1][gi]} ({c['whole_only_A'][gi]}, {c['whole_only_B'][gi]})" for gi in tb["with_fabs"]))
    vA, vB = c["lots_value_by_fab"]
    order = np.argsort(-np.abs(vA - vB))[:8]
    print("  lots at the chip's penalty, bn, A / B: total "
          f"{vA.sum():.0f} / {vB.sum():.0f}; " + ", ".join(f"{tb['fabs'][i]} {vA[i]:.0f}/{vB[i]:.0f}" for i in order if abs(vA[i] - vB[i]) >= 0.5))
    if not detail:
        return
    print("  lots at the chip's penalty by 13 weeks, A / B: " + " ".join(f"{a:.0f}/{b:.0f}" for a, b in c["lots_value_by_block"]))
    print("  fuel burned, weeks of full burn, A / B: " + ", ".join(
        f"{G[gi]}/{tb['K'][k]} {a:.1f}/{b:.1f}" for (gi, k), (a, b) in c["burn_weeks"].items() if abs(a - b) >= 0.3))
    print("  fuel thrown away, thousand units, A / B: " + ", ".join(f"{k} {a / 1e3:.0f}/{b / 1e3:.0f}" for k, (a, b) in c["thrown"].items())
          + " | lifted: " + ", ".join(f"{k} {a / 1e3:.0f}/{b / 1e3:.0f}" for k, (a, b) in c["lifted"].items()))
    print("  chips left at the end, thousand, A / B: " + ", ".join(f"{k[0]}@{k[1]} {a / 1e3:.0f}/{b / 1e3:.0f}" for k, (a, b) in c["chips_left"].items()))
    print("  dispatches apart (sum |A-B| / sum (A+B)): " + ", ".join(f"{k} {v:.2f}" for k, v in c["sent_apart"].items()))
    print("  holding by commodity, bn: " + ", ".join(f"{k} {v:+.1f}" for k, v in c["holding_by_k"].items() if abs(v) >= 0.05)
          + " | disposal: " + ", ".join(f"{k} {v:+.1f}" for k, v in c["disposal_by_k"].items() if abs(v) >= 0.05))
    for fin, row in c["balance"].items():
        print(f"  {fin} balance, thousand, A / B: " + ", ".join(f"{k} {row['A'][k] / 1e3:.0f}/{row['B'][k] / 1e3:.0f}" for k in row["A"]))


def main(only: str | tuple | int = "", a: str = "start:tah0_f", b: str = "best", task: str = "full", entropy: int = 444,
         first: int = 0, episodes: int = 16, detail: bool = False, save: str = "") -> None:
    import core  # regime_lab's

    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)] if only != "" else list(range(first, first + episodes))
    kept, tb = {}, None
    for n in ns:
        ep = core.Episode.of(task, entropy, n)
        tb = tb or tables(ep)
        try:
            (actsA, nameA), (actsB, nameB) = plan_acts(ep, a, task, entropy, n), plan_acts(ep, b, task, entropy, n)
        except (FileNotFoundError, TypeError):
            continue
        c = compare(ep, tb, played(ep, actsA), played(ep, actsB))
        kept[n] = c
        show(n, tb, c, (nameA, nameB), detail)
    if len(kept) > 1:
        G = tb["grids"]
        m = len(kept)
        print(f"\n=== mean of {m} episodes {sorted(kept)}, A - B, bn an episode: {np.mean([c['J'][0] - c['J'][1] for c in kept.values()]):+.1f}")
        items = np.mean([c["items"] for c in kept.values()], axis=0)
        print("  by item: " + ", ".join(f"{k} {v:+.1f}" for k, v in zip(ITEMS, items)))
        lost = np.mean([c["lost_by_demand"] for c in kept.values()], axis=0)
        print("  unmet demand: " + ", ".join(f"{tb['demands'][i][0]}/{tb['demands'][i][1]} {lost[i]:+.1f}" for i in np.argsort(-np.abs(lost))[:10]))
        kk = np.array([d[1] for d in tb["demands"]])
        print("  unmet demand by chip: " + ", ".join(f"{k} {lost[kk == k].sum():+.1f}" for k in ("chip_le", "chip_mat")))
        shed = np.mean([c["shed_by_grid"] for c in kept.values()], axis=0)
        print("  shed load by grid: " + ", ".join(f"{G[gi]} {v:+.1f}" for gi, v in enumerate(shed)))
        print("  by 13 weeks, unmet: " + " ".join(f"{v:+6.1f}" for v in np.mean([c["lost_by_block"] for c in kept.values()], axis=0))
              + " | shed: " + " ".join(f"{v:+6.1f}" for v in np.mean([c["shed_by_block"] for c in kept.values()], axis=0)))
        wA = np.mean([c["whole_weeks"][0] for c in kept.values()], axis=0)
        wB = np.mean([c["whole_weeks"][1] for c in kept.values()], axis=0)
        oA = np.mean([c["whole_only_A"] for c in kept.values()], axis=0)
        oB = np.mean([c["whole_only_B"] for c in kept.values()], axis=0)
        print("  whole weeks A / B (only A, only B): " + ", ".join(f"{G[gi]} {wA[gi]:.1f}/{wB[gi]:.1f} ({oA[gi]:.1f}, {oB[gi]:.1f})" for gi in tb["with_fabs"]))
        vA = np.mean([c["lots_value_by_fab"][0] for c in kept.values()], axis=0)
        vB = np.mean([c["lots_value_by_fab"][1] for c in kept.values()], axis=0)
        print(f"  lots at the chip's penalty, A / B: total {vA.sum():.0f} / {vB.sum():.0f}; " + ", ".join(
            f"{tb['fabs'][i]} {vA[i]:.0f}/{vB[i]:.0f}" for i in np.argsort(-np.abs(vA - vB))[:8]))
        for key, word in (("holding_by_k", "holding"), ("disposal_by_k", "disposal")):
            print(f"  {word} by commodity: " + ", ".join(f"{k} {np.mean([c[key][k] for c in kept.values()]):+.1f}" for k in tb["K"]))
        for fin in ("chip_le", "chip_mat"):
            keys = list(next(iter(kept.values()))["balance"][fin]["A"])
            print(f"  {fin} balance, thousand, A / B: " + ", ".join(
                f"{k} {np.mean([c['balance'][fin]['A'][k] for c in kept.values()]) / 1e3:.0f}/{np.mean([c['balance'][fin]['B'][k] for c in kept.values()]) / 1e3:.0f}"
                for k in keys))
        thr = {k: (np.mean([c["thrown"][k][0] for c in kept.values()]), np.mean([c["thrown"][k][1] for c in kept.values()])) for k in next(iter(kept.values()))["thrown"]}
        lif = {k: (np.mean([c["lifted"][k][0] for c in kept.values()]), np.mean([c["lifted"][k][1] for c in kept.values()])) for k in next(iter(kept.values()))["lifted"]}
        print("  fuel thrown away, thousand units, A / B: " + ", ".join(f"{k} {x / 1e3:.0f}/{y / 1e3:.0f}" for k, (x, y) in thr.items())
              + " | lifted: " + ", ".join(f"{k} {x / 1e3:.0f}/{y / 1e3:.0f}" for k, (x, y) in lif.items()))
        lb = np.mean([c["lots_value_by_block"] for c in kept.values()], axis=0)
        print("  lots at the chip's penalty by 13 weeks, A / B: " + " ".join(f"{x:.0f}/{y:.0f}" for x, y in lb))
    if save:
        path = K.OUT / "diff" / f"{save}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"a": a, "b": b, "episodes": kept}))


if __name__ == "__main__":
    fire.Fire(main)
