"""What a traced scenario play asks above the point plan, and what of it the environment executed.

    uv run python lab/anastasiia/frontier_lab/mech/hedge.py two2bep

Reads the traces of ``mech/trace.py``. A week counts when the joint first week was played. Per kind of slot and by
whether the slot's first edge is cut at the instant the week starts (its observed capacity under the nominal one):
the requests of the point plan, what the played action asks more and less, and what the environment executed above
the point plan's request (cargo that moved only because more was asked). The same for the tanker releases by whether
their strait is shut or their out-edge cut. Then the weeks split by what the point block gives up in the program
(a hedge: the model's own forecast pays for the other futures).
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scen"), str(HERE.parents[1] / "hazard_lab")]


def main(tag: str, task: str = "small", entropy: int = 444, base: str = "") -> None:
    import trace as T

    data = T.kept(tag, task, entropy)
    if not data:
        print("nothing kept")
        return
    lab = next(iter(data.values()))["labels"]
    kinds = np.array([s["kind"] for s in lab["slots"]])
    edge = np.array([s["e"] for s in lab["slots"]])
    u0 = np.array([np.inf if s["u0"] is None else s["u0"] for s in lab["slots"]])
    names = ("order", "valve", "wafer", "raw", "pack")
    tally = {(k, c): np.zeros(6) for k in names for c in (False, True)}  # point, more, less, executed above, slots, weeks
    weeks = played = hedged = 0
    gain_point, gain_rest = [], []
    for r in data.values():
        for t, (w, row) in enumerate(zip(r["trace"], r["notes"]["rows"])):
            weeks += 1
            if row["pick"] != "joint":
                continue
            played += 1
            g0 = row["start"][0] - row["played"][0]
            gain_point.append(g0)
            gain_rest.append(float(np.mean(np.array(row["start"][1:]) - np.array(row["played"][1:]))))
            hedged += g0 < -1e-6
            p, f, x = w["point"]["flows"], w["final"]["flows"], r["sent"][t]
            cut = w["u"][edge] < 0.999 * u0
            for k in names:
                for c in (False, True):
                    m = (kinds == k) & (cut == c)
                    d = f[m] - p[m]
                    tally[(k, c)] += (p[m].sum(), np.maximum(d, 0).sum(), np.maximum(-d, 0).sum(),
                                      np.maximum(x[m] - p[m], 0).sum(), m.sum(), 1)
    print(f"{tag} {task} {entropy}: {len(data)} episodes, {weeks} weeks, the joint first week played in {played} "
          f"({played / weeks:.2f}); the point block pays in {hedged} of them")
    print(f"  the program's claim in those weeks, bn: the point block {np.mean(gain_point):+.2f} (median "
          f"{np.median(gain_point):+.2f}), the scenarios' blocks {np.mean(gain_rest):+.2f}")
    print("  kind, first edge cut now: slots a week; the point plan's requests a week; asked more; asked less; "
          "executed above the point plan's request (all as shares of the point plan's requests of the kind)")
    for k in names:
        whole = tally[(k, False)][0] + tally[(k, True)][0]
        for c in (False, True):
            a = tally[(k, c)]
            print(f"    {k:6s} {'cut ' if c else 'open'}: {a[4] / max(a[5], 1):5.1f}  {a[0] / max(a[5], 1):10.0f}  "
                  f"{a[1] / max(whole, 1e-9):+.4f}  {-a[2] / max(whole, 1e-9):+.4f}  {a[3] / max(whole, 1e-9):+.4f}")
    # the tanker releases and the holds
    rel = np.zeros(5)
    for r in data.values():
        for w, row in zip(r["trace"], r["notes"]["rows"]):
            if row["pick"] != "joint":
                continue
            pair = np.array([x["pair"] for x in lab["releases"]])
            pa = np.where(w["point"]["release_mode"][pair] == 1, w["point"]["override_qty"], 0.0)
            fa = np.where(w["final"]["release_mode"][pair] == 1, w["final"]["override_qty"], 0.0)
            rel += (pa.sum(), np.maximum(fa - pa, 0).sum(), np.maximum(pa - fa, 0).sum(),
                    np.sum(w["point"]["release_mode"] != w["final"]["release_mode"]), 1)
    print(f"  tanker releases a week: the point plan's {rel[0] / max(rel[4], 1):.0f}, asked more {rel[1] / max(rel[0], 1e-9):+.4f}, "
          f"less {-rel[2] / max(rel[0], 1e-9):+.4f}; pairs whose release mode differs {rel[3] / max(rel[4], 1):.2f} a week")
    if base:
        import play as P

        b = P.kept(base, task, entropy)
        ns = [n for n in sorted(data) if n in b]
        d = np.array([(b[n]["J"] - data[n]["J"]) / 1e11 for n in ns])
        print(f"  against {base} on the same {len(ns)} episodes: {d.mean():+.2f} bn an episode, cheaper in {int((d > 0).sum())}")


if __name__ == "__main__":
    fire.Fire(main)
