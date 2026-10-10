"""Remaining duration of running events: the law given what is observed (age, phase) against what happened."""

import glob
import math
import pickle
import sys

import numpy as np
from metrics import summary
from model import CLOSURE, CONFLICT, ENERGY, SANCTION, STRIKE, WEATHER, Gen
from predict import Acc


HD = (1, 4, 13, 26)


def main(task, pattern):
    g = Gen(task)
    p, T = g.p, g.T
    eps = []
    for path in sorted(glob.glob(pattern)):
        eps += pickle.load(open(path, "rb"))
    laws = dict(p.laws.duration)
    W = float(p.marks.war_profile_window)
    grid = np.concatenate([np.arange(0, 60, 0.05), np.arange(60, 4200, 1.0)])

    def cdf_fn(law, first_phase=False):
        vals = np.array([law.cdf(x) if x > 0 else 0.0 for x in grid])
        if first_phase:  # the first phase of a war profile lasts max(d, W)
            vals = np.where(grid >= W, vals, 0.0)
        return lambda x: np.interp(np.asarray(x, dtype=float), grid, vals)

    kinds = {
        "sanction": (SANCTION, cdf_fn(laws["sanction"])),
        "militarised_closure": (CLOSURE, cdf_fn(laws["militarised_closure"])),
        "conflict_W0": (CONFLICT, cdf_fn(laws["regional_conflict"], True)),
        "energy_shock": (ENERGY, cdf_fn(laws["energy_shock"])),
        "weather_closure": (WEATHER, cdf_fn(laws["weather_closure"])),
        "stoppage": (STRIKE, cdf_fn(p.poisson.stoppage_duration)),
        "slowdown": (STRIKE, cdf_fn(p.poisson.slowdown_duration)),
    }
    a0 = np.concatenate([np.arange(0.25, 60, 0.5), np.arange(60.5, 3725, 2.0)])
    w0 = np.concatenate([np.full(120, 0.5), np.full(len(a0) - 120, 2.0)])

    def carried(F, s, h):
        """P(end within h | runs at s, began before 0 at an unknown time): the stationary mixture over the age."""
        num = (w0 * (F(a0 + s + h) - F(a0 + s))).sum()
        den = (w0 * (1 - F(a0 + s))).sum()
        return num / max(den, 1e-12)

    table = {k: np.array([[carried(F, s, h) for h in HD] for s in range(T)]) for k, (_c, F) in kinds.items()}
    acc = Acc()
    counts = {}
    for e in eps:
        st = e["stored"]
        for row in st:
            ty, cp, on, dur, sev, der = int(row[0]), int(row[3]), row[7], row[8], row[9], int(row[11])
            if ty == SANCTION:
                kind, end = "sanction", on + dur
            elif ty == CLOSURE and not der:
                kind, end = "militarised_closure", on + dur
            elif ty == CONFLICT and cp >= 0:
                kind, end = "conflict_W0", on + max(dur, W)
            elif ty == ENERGY:
                kind, end = "energy_shock", on + dur
            elif ty == WEATHER:
                kind, end = "weather_closure", on + dur
            elif ty == STRIKE:
                kind, end = ("stoppage" if sev >= 0.9 else "slowdown"), on + dur
            else:
                continue
            F = kinds[kind][1]
            new = on > 0
            first = max(0, int(math.ceil(on)))
            for s in range(first, T):
                if end <= s:
                    break
                for hi, h in enumerate(HD):
                    if s + h > T:
                        continue
                    y = float(end <= s + h)
                    a_true = s - on
                    p_true = float((F(a_true + h) - F(a_true)) / max(1 - F(a_true), 1e-12))
                    if new:
                        a_obs = s - math.ceil(on) + 0.5
                        p_obs = float((F(a_obs + h) - F(a_obs)) / max(1 - F(a_obs), 1e-12))
                    else:
                        p_obs = float(table[kind][s, hi])
                    grp = "new" if new else "carried"
                    for gname in (grp, "all"):
                        acc.add((kind, gname, h, "observed"), [p_obs], [y])
                        acc.add((kind, gname, h, "true_age"), [p_true], [y])
                        counts[(kind, gname, h)] = counts.get((kind, gname, h), 0) + 1
    print(f"# {task}, {len(eps)} episodes: P(ends within h | runs now)")
    print(
        "kind                 group    h   event-weeks  base    | observed age: AUROC top10% KS  E[p|end] point pred/obs | true age: AUROC KS E[p|end] point"  # noqa: E501
    )
    for kind in kinds:
        for grp in ("new", "carried", "all"):
            for h in HD:
                if (kind, grp, h, "observed") not in acc.h:
                    continue
                a, b = summary(acc.h[(kind, grp, h, "observed")]), summary(acc.h[(kind, grp, h, "true_age")])
                if a["npos"] < 5:
                    continue
                print(
                    f"{kind:20s} {grp:8s} {h:3d} {int(a['n']):9d}  {a['base']:.4f}  | {a['auc']:.3f} {a['top10']:.3f} {a['ks']:.3f} {a['share_two']:.3f} {a['share_point']:.3f} {a['cal_all']:.2f} | {b['auc']:.3f} {b['ks']:.3f} {b['share_two']:.3f} {b['share_point']:.3f}"  # noqa: E501
                )
    pickle.dump(dict(hist=acc.h, counts=counts), open(sys.argv[3], "wb"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
