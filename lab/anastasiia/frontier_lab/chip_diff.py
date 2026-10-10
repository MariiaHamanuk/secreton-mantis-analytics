"""Where a play that knew the future sold more chips than a play that did not: by market, plant, fab and week.

    uv run python lab/anastasiia/frontier_lab/chip_diff.py w50as_s truthall_h0_s --episodes=24

Reads two kept plays of the same episodes (``outputs/hazard_lab/play/<tag>_<task>_<entropy>.pkl``: executed flows by
slot and week, lots, sales, stocks) and the scenario's own marks. Printed, per episode on average, the second tag
less the first: chips sold by market and quarter; lots by fab and quarter; raw chips sent from each fab to each
plant; packaged chips sent from each plant to each market; packaged chips left on the plants at the end; and, for
the packaged chip of the leading kind, how the plants' stock at the end splits by whether the plant's exits to the
markets with unmet demand were prohibited in the last weeks.
"""

import importlib
import pickle
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sample_omega = importlib.import_module("shockbench_flow.disruption.sampler").sample_omega
task_generator = importlib.import_module("shockbench_flow.hosting.tasks").task_generator
compute_marks = importlib.import_module("shockbench_flow.marks").compute_marks


def main(a: str, b: str, task: str = "small", entropy: int = 444, episodes: int = 24, first: int = 0) -> None:
    A = pickle.loads((ROOT / f"outputs/hazard_lab/play/{a}_{task}_{entropy}.pkl").read_bytes())
    B = pickle.loads((ROOT / f"outputs/hazard_lab/play/{b}_{task}_{entropy}.pkl").read_bytes())
    inst0, params = task_generator(task)
    com = [c.id for c in inst0.commodities]
    name = [n.id for n in inst0.nodes]
    ns = [n for n in range(first, first + episodes) if n in A and n in B]
    T = inst0.T
    Q = [(0, T // 4), (T // 4, T // 2), (T // 2, 3 * T // 4), (3 * T // 4, T)]
    slots = list(inst0.action_slots)

    def head(s):  # where a slot's cargo ends up: the head of its edge, or of its lane's last edge
        e, _k, lane = slots[s]
        return inst0.edges[e if lane is None else inst0.lanes[lane].edges[-1]].head

    tail = [inst0.edges[e].tail for e, _k, _l in slots]
    sold = {}  # (market, chip) -> by quarter, thousand
    lots = {}  # fab -> by quarter, thousand
    moved = {}  # (from, to, commodity) -> by quarter, thousand
    left, banned_left = {}, {}
    for n in ns:
        marks = compute_marks(inst0, sample_omega(inst0, params, entropy, n, "train"))
        inst = inst0.at_digest(marks.instance_digest)
        for sign, D in ((-1, A), (1, B)):
            r = D[n]
            for di, d in enumerate(inst.demands):
                key = (name[d.node], com[d.k])
                sold.setdefault(key, np.zeros(4))
                sold[key] += sign * np.array([r["served"][q0:q1, di].sum() for q0, q1 in Q]) / 1e3
            for fi, f in enumerate(inst.fabs):
                lots.setdefault(name[f], np.zeros(4))
                lots[name[f]] += sign * np.array([r["lots"][q0:q1, fi].sum() for q0, q1 in Q]) / 1e3
            for s, (e, k, _lane) in enumerate(slots):
                if not com[k].startswith("chip"):
                    continue
                key = (name[tail[s]], name[head(s)], com[k])
                moved.setdefault(key, np.zeros(4))
                moved[key] += sign * np.array([r["sent"][q0:q1, s].sum() for q0, q1 in Q]) / 1e3
            for o in inst.osats:
                for pk in inst.nodes[o].osat.packages.values():
                    si = inst.slot_index[(o, pk)]
                    key = (name[o], com[pk])
                    left.setdefault(key, np.zeros(2))
                    left[key][0 if sign < 0 else 1] += r["stock"][-1, si] / 1e3
                    # was every direct exit of this plant for this chip prohibited in the last eight weeks?
                    exits = [s for s, (e, k, lane) in enumerate(slots) if tail[s] == o and k == pk]
                    shut = all(marks.prohibited[T - 8 :, slots[s][0], pk].all() for s in exits) if exits else True
                    banned_left.setdefault(key, np.zeros(2))
                    banned_left[key][0 if sign < 0 else 1] += (r["stock"][-1, si] / 1e3) * shut
    m = len(ns)
    print(f"{task} {entropy}, {m} episodes from {first}: {b} less {a}, thousand units an episode, by quarter of the episode")
    print("chips sold, by market:")
    for key, v in sorted(sold.items(), key=lambda kv: -abs(kv[1].sum())):
        if abs(v.sum()) / m > 0.5:
            print(f"  {key[0]:10s} {key[1]:9s} {np.round(v / m, 1)}  total {v.sum() / m:+.1f}")
    print("lots started, by fab:")
    for key, v in sorted(lots.items(), key=lambda kv: -abs(kv[1].sum())):
        print(f"  {key:20s} {np.round(v / m, 1)}  total {v.sum() / m:+.1f}")
    print("chips moved (from, to, what), the largest differences:")
    for key, v in sorted(moved.items(), key=lambda kv: -abs(kv[1].sum()))[:22]:
        if abs(v.sum()) / m > 1.0:
            print(f"  {key[0]:18s} -> {key[1]:10s} {key[2]:12s} {np.round(v / m, 1)}  total {v.sum() / m:+.1f}")
    print(f"packaged chips on the plants at the end, thousand an episode ({a} / {b}); in brackets the part at a plant whose exits were all prohibited in the last 8 weeks:")
    for key in sorted(left):
        v, w = left[key] / m, banned_left[key] / m
        print(f"  {key[0]:10s} {key[1]:9s} {v[0]:7.1f} / {v[1]:7.1f}   [{w[0]:6.1f} / {w[1]:6.1f}]")


if __name__ == "__main__":
    fire.Fire(main)
