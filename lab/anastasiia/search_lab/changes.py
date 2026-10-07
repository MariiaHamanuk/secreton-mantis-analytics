"""What a search changed in an agent's actions: by fuel and destination, by week, and against the edges' capacity.

    uv run python lab/anastasiia/search_lab/changes.py outputs/search_lab/small_444_ep9_fuel.pkl --moves=fuel

Reads a result file of ``search.py run --out``. Per commodity and destination of the slot's route: the quantity
requested as played and after the search, by third of the episode. Then, for the dispatches the search raised, whether
their first edge had room as played (the request was below the edge's true capacity that week) or was full (a larger
request only takes a larger share of the same edge from the other slots on it).
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import world  # noqa: E402


def main(path: str, moves: str = "fuel") -> None:
    d = pickle.loads(Path(path).read_bytes())
    task, entropy = d["task"], d["entropy"]
    tot, room_units, full_units, lower_units = {}, 0.0, 0.0, 0.0
    weeks_changed = np.zeros(3)
    for r in d["results"]:
        inst, marks = world(task, entropy, r["episode"])
        N, E, K, T = inst.nodes, inst.edges, [c.id for c in inst.commodities], inst.T
        u = np.asarray(marks.u)
        before, after = r["flows"], r["sets"][moves]["flows"]
        first_edge = [e for e, _k, _l in inst.action_slots]
        for t in range(T):
            load = {}
            for s, q in before[t].items():
                load[first_edge[s]] = load.get(first_edge[s], 0.0) + q
            for s in set(before[t]) | set(after[t]):
                e, k, lane = inst.action_slots[s]
                q0, q1 = before[t].get(s, 0.0), after[t].get(s, 0.0)
                dest = N[E[e].head if lane is None else E[inst.lanes[lane].edges[-1]].head].id
                a = tot.setdefault((K[k], dest), np.zeros((2, 3)))
                third = min(2, 3 * t // T)
                a[0, third] += q0
                a[1, third] += q1
                if abs(q1 - q0) > 1e-9:
                    weeks_changed[third] += 1
                if q1 > q0 + 1e-9:
                    if load.get(e, 0.0) < u[t, e] - 1e-6:
                        room_units += q1 - q0
                    else:
                        full_units += q1 - q0
                elif q1 < q0 - 1e-9:
                    lower_units += q0 - q1
    n = len(d["results"])
    print(f"{path}: {n} episode(s), move set {moves}; quantities per episode, by third of the episode")
    print(f"{'commodity':>9} {'to':>10} | {'as played':>26} | {'after the search':>26} | change")
    for (k, dest), a in sorted(tot.items()):
        if np.abs(a[1] - a[0]).sum() < 1e-6:
            continue
        a = a / n
        cells0 = " ".join(f"{x:8.0f}" for x in a[0])
        cells1 = " ".join(f"{x:8.0f}" for x in a[1])
        print(f"{k:>9} {dest:>10} | {cells0} | {cells1} | {a[1].sum() - a[0].sum():+9.0f}")
    print(f"\ndispatches changed per episode, by third: {(weeks_changed / n).round(0).tolist()}")
    print(
        f"units added per episode: {room_units / n:,.0f} on edges with room as played, {full_units / n:,.0f} on full"
        f" edges; units removed: {lower_units / n:,.0f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
