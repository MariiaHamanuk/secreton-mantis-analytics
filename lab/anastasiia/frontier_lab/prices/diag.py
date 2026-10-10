"""What the prices did to a priced play (G1 of R2): how far the week's requests moved, whether towards the teacher's,
and where the cost went against the plain model's kept play.

    uv run python lab/anastasiia/frontier_lab/prices/diag.py cut1 req --episodes=0-23

For every week of a priced play three planners saw the same observation: the teacher (told the future), the plain
model and the player (the model with the prices). By kind of slot, summed over the weeks: the distance between the
plain model's requests and the teacher's (sum of absolute differences, each slot in units of its mean executed flow
in the model's kept play, requests cut at three times that flow: the model asks above capacity on a cut edge, the
teacher does not), the distance the prices moved the player from the plain model, and the share of the distance to the
teacher that the player closed (1: the player asks what the teacher asks; negative: the prices moved it away).
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


def main(*tags: str, task: str = "small", entropy: int = 444, episodes="0-23", base: str = "h3_s") -> None:
    inst, _params = importlib.import_module("shockbench_flow.hosting.tasks").task_generator(task)
    com = [c.id for c in inst.commodities]
    slots = list(inst.action_slots)

    def kind(node: int) -> str:
        nd = inst.nodes[node]
        for attr in ("grid", "fab", "osat"):
            if getattr(nd, attr, None) is not None:
                return attr
        return next((key for key in ("sink", "term", "src", "mat") if nd.id.startswith(key)), nd.id.split("_")[0])

    tail = [inst.edges[e].tail for e, _k, _l in slots]
    head = [inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head for e, _k, lane in slots]
    group = np.array([f"{com[k]}: {kind(tail[s])}->{kind(head[s])}" for s, (_e, k, _l) in enumerate(slots)])
    kept0 = pickle.loads((ROOT / "outputs" / "hazard_lab" / "play" / f"{base}_{task}_{entropy}.pkl").read_bytes())
    usual = np.mean([r["sent"].mean(axis=0) for r in kept0.values()], axis=0)
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    for tag in tags:
        kept = P.kept(tag, task, entropy)
        ns = [n for n in P._numbers(episodes) if n in kept]
        if not ns:
            continue
        gap, moved, left = {}, {}, {}
        sizes, missed, errors, weeks = [], 0, 0, 0
        for n in ns:
            r = kept[n]
            a = np.minimum(r["asked"], 3.0 * usual[None, None, :]) / np.maximum(usual, 1e-9)[None, None, :]
            a[:, :, usual <= 0] = 0.0
            sizes += r["notes"]["prices"]
            missed += r["notes"]["missed"]["teacher"] + r["notes"]["missed"]["model"]
            errors += r["notes"].get("errors", 0)
            weeks += len(r["asked"])
            for g in np.unique(group):
                m = group == g
                gap[g] = gap.get(g, 0.0) + np.abs(a[:, 1, m] - a[:, 0, m]).sum()
                moved[g] = moved.get(g, 0.0) + np.abs(a[:, 2, m] - a[:, 1, m]).sum()
                left[g] = left.get(g, 0.0) + np.abs(a[:, 2, m] - a[:, 0, m]).sum()
        d = np.array([(kept0[n]["J"] - kept[n]["J"]) / 1e11 for n in ns])
        print(f"\n{tag}: {len(ns)} episodes, saved against {base} {d.mean():+.2f} bn an episode (sd {d.std(ddof=1) if len(d) > 1 else 0:.1f}), cheaper in "
              f"{int((d > 0).sum())}; weeks {weeks}, without prices {int(np.sum(np.array(sizes) == 0))}, a planner without a cell "
              f"{missed}, weeks with an error {errors}; sum of |price| a week: median {np.median(sizes):.3g}, 90th percentile {np.percentile(sizes, 90):.3g}")  # fmt: skip
        print(
            f"  {'kind of slot':28s} {'model to teacher':>17s} {'player to model':>16s} {'player to teacher':>18s} {'closed':>7s}"
        )
        for g in sorted(gap, key=lambda g: -gap[g]):
            if gap[g] > 0:
                print(
                    f"  {g:28s} {gap[g] / weeks:17.3f} {moved[g] / weeks:16.3f} {left[g] / weeks:18.3f} {1 - left[g] / gap[g]:7.2f}"
                )
        T = kept[ns[0]]["costs"].shape[0]
        Q = [(0, T // 4), (T // 4, T // 2), (T // 2, 3 * T // 4), (3 * T // 4, T)]
        for ci in (5, 7):
            row = "  ".join(
                f"{np.mean([(kept0[n]['costs'][a:b, ci].sum() - kept[n]['costs'][a:b, ci].sum()) / 1e9 for n in ns]):+6.2f}"
                for a, b in Q
            )
            print(f"  saved in {comp[ci]:9s} by quarter, bn an episode: {row}")


if __name__ == "__main__":
    fire.Fire(main)
