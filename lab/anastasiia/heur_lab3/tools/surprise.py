"""Does a model lose more in episodes whose network changes after week 1? Its score against how much the future differs.

    uv run python lab/anastasiia/heur_lab3/tools/surprise.py anastasiia_rules_v2 --set=small-111
    uv run python lab/anastasiia/heur_lab3/tools/surprise.py agents/anastasiia_rules_v2 --set=full-111 --costs=x.json

"It stays as it is" is the cheap forecast every rule agent uses. In an episode whose network never changes after
week 1 that forecast is exact, so whatever the agent loses there is control, not foresight. This script measures, per
episode of a set kept in ``hub/eval/records/<model>.json`` (or of a file written by ``costs.py --out``, whose key
is the agent's path as it was given there), how much the true network (the generator's marks: no agent
is played) departs from its week-1 state, and prints the model's score by thirds of that measure and the rank
correlations.

Per episode, for the fuel slots and for the chip-chain slots (a slot's capacity is its first edge's, zero while any
edge of its route is prohibited for its commodity), as shares of the nominal capacity over weeks 2..T:

- ``cut1``: capacity missing in week 1 (the damage the agent sees at once);
- ``loss``: capacity present in week 1 and missing later (a bad surprise);
- ``gain``: capacity missing in week 1 and present later (a good surprise);

and ``power``: the grids' deliverable output lost after week 1, as a share of week 1's.
The measures are kept in ``outputs/heur3/data/surprise_<set>.json`` and reused.
"""

import json
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[4]
LEVEL_WEIGHTS = (0.50, 0.30, 0.15, 0.05)
FUELS = ("lng", "crude", "nucfuel")


def measures(task: str, entropy: int, n: int) -> dict:
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks

    inst, params = task_generator(task)
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "dev" if entropy == 0 else "train"))
    u, banned = np.asarray(marks.u), np.asarray(marks.prohibited)
    names = [c.id for c in inst.commodities]
    out = {}
    for group in ("fuel", "chips"):
        cap, nominal = [], []
        for e, k, lane in inst.action_slots:
            if (names[k] in FUELS) != (group == "fuel"):
                continue
            route = [e] if lane is None else list(inst.lanes[lane].edges)
            cap.append(u[:, e] * (1.0 - banned[:, route, k].max(axis=1)))
            nominal.append(float(inst.edges[e].u0))
        cap, nominal = np.stack(cap, axis=1), np.array(nominal)  # (T, slots), (slots,)
        later = (cap.shape[0] - 1) * nominal.sum()
        out[f"{group}_cut1"] = float(1.0 - cap[0].sum() / nominal.sum())
        out[f"{group}_loss"] = float(np.maximum(cap[0] - cap[1:], 0.0).sum() / later)
        out[f"{group}_gain"] = float(np.maximum(cap[1:] - cap[0], 0.0).sum() / later)
    g = np.asarray(marks.G_bar)[: inst.T]
    out["power"] = float(np.maximum(g[0] - g[1:], 0.0).sum() / ((len(g) - 1) * g[0].sum()))
    return out


def pooled(level: np.ndarray, saved: np.ndarray, room: np.ndarray) -> float:
    present = [s for s in (1, 2, 3, 4) if (level == s).any()]
    w = {s: LEVEL_WEIGHTS[s - 1] for s in present}
    return float(
        sum(w[s] * saved[level == s].mean() for s in present) / sum(w[s] * room[level == s].mean() for s in present)
    )


def ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def main(model: str, set: str = "small-111", costs: str = "", n_jobs: int = 2) -> None:  # noqa: A002
    if costs:
        rows = json.loads(Path(costs).read_text())[model]
    else:
        rows = json.loads((ROOT / "hub/eval/records" / f"{model}.json").read_text())["sets"][set]["rows"]
    task, entropy = set.split("-")[0], int(set.split("-")[1])
    kept = ROOT / "outputs/heur3/data" / f"surprise_{set}.json"
    if kept.is_file() and len(json.loads(kept.read_text())) == len(rows):
        m = json.loads(kept.read_text())
    else:
        m = Parallel(n_jobs=n_jobs)(delayed(measures)(task, entropy, r["episode"]) for r in rows)
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(json.dumps(m))
    ok = np.array([r["excluded"] is None for r in rows])
    level = np.array([r["stratum"] for r in rows])[ok]
    naive = np.array([r["J_naive_cents"] for r in rows], dtype=float)[ok] / 1e11
    room = naive - np.array([r["J_clairvoyant_cents"] for r in rows], dtype=float)[ok] / 1e11
    saved = naive - np.array([r["J_policy_cents"] for r in rows], dtype=float)[ok] / 1e11
    M = {k: np.array([x[k] for x in m])[ok] for k in m[0]}
    M["fuel_change"] = M["fuel_loss"] + M["fuel_gain"]
    M["chips_change"] = M["chips_loss"] + M["chips_gain"]
    M["any_change"] = M["fuel_change"] + M["chips_change"] + M["power"]
    score, gap = saved / room, room - saved
    print(f"{model} on {set}: {len(level)} episodes, score {pooled(level, saved, room):.4f}")
    print(f"mean gap to the clairvoyant plan {gap.mean():.0f} bn USD per episode of {room.mean():.0f} attainable\n")
    head = f"{'measure':>13} {'median':>7} {'p90':>6}"
    print(f"{head} | rank corr. with score, gap bn | score by thirds (low, mid, high)")
    for k, v in M.items():
        r_score = np.corrcoef(ranks(v), ranks(score))[0, 1]
        r_gap = np.corrcoef(ranks(v), ranks(gap))[0, 1]
        third = np.argsort(np.argsort(v, kind="stable"), kind="stable") * 3 // len(v)
        by = [saved[third == i].sum() / room[third == i].sum() for i in range(3)]
        gaps = [gap[third == i].mean() for i in range(3)]
        print(
            f"{k:>13} {np.median(v):7.3f} {np.quantile(v, 0.9):6.3f} | {r_score:+.2f} {r_gap:+.2f} | "
            f"{by[0]:.3f} {by[1]:.3f} {by[2]:.3f} | gap bn {gaps[0]:.0f} {gaps[1]:.0f} {gaps[2]:.0f}"
        )
    quiet = M["any_change"] <= np.quantile(M["any_change"], 0.2)
    print(
        f"\nthe quietest fifth of the episodes (any_change <= {M['any_change'][quiet].max():.3f}): "
        f"score {saved[quiet].sum() / room[quiet].sum():.3f}, gap {gap[quiet].mean():.0f} bn; "
        f"the rest: {saved[~quiet].sum() / room[~quiet].sum():.3f}, {gap[~quiet].mean():.0f} bn"
    )


if __name__ == "__main__":
    fire.Fire(main)
