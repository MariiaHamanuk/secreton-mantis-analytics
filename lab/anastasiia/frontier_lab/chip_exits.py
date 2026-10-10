"""Lost sales of the leading chip that an exit with spare room could have served, and how many of them from chips
already at that plant.

    uv run python lab/anastasiia/frontier_lab/chip_exits.py w50as_s truthall_h0_s --episodes=24

For each tag (kept plays of ``hazard_lab/play.py``), per episode on average, thousand leading chips: sold and lost;
packaged chips left at the plants at the end; the capacity of the plants' exits into the markets that want the chip
(a prohibited exit carries nothing) and what was sent through them; then, week by week and exit by exit, the room
left on the exit (the mature chip shares the edge) met with the sales its market lost in the week of arrival: the
total, and the part of it the plant could have filled from the packaged chips it held at the week's start.
The first number says whether the exits or the chips are short; the second is the size of "kept for a later exit".
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


def main(*tags: str, task: str = "small", entropy: int = 444, episodes: int = 24) -> None:
    N = episodes
    D = {t: pickle.loads((ROOT / f"outputs/hazard_lab/play/{t}_{task}_{entropy}.pkl").read_bytes()) for t in tags}
    inst0, params = task_generator(task)
    com = [c.id for c in inst0.commodities]; kle = com.index("chip_le"); kraw = com.index("chip_le_raw")
    name = [n.id for n in inst0.nodes]; slots = list(inst0.action_slots); T = inst0.T
    tail = [inst0.edges[e].tail for e, k, l in slots]; head = [inst0.edges[e].head for e, k, l in slots]
    osats = [o for o in inst0.osats if kraw in inst0.nodes[o].osat.packages]
    dem = {d.node: di for di, d in enumerate(inst0.demands) if d.k == kle and d.dbar > 0}
    ex = [s for s, (e, k, l) in enumerate(slots) if tail[s] in osats and k == kle and l is None and head[s] in dem]
    print("exits of packaged leading chips to markets that want them:", len(ex), "| markets", [name[m] for m in dem])
    ns = [n for n in range(N) if all(n in D[t] for t in tags)]
    out = {t: dict(lost=0.0, sold=0.0, end=0.0, end_shut=0.0, feas=0.0, feas_own=0.0, cap=0.0, sent=0.0, wasted_cap=0.0) for t in tags}
    bym = {t: {m: np.zeros(3) for m in dem} for t in tags}
    for n in ns:
        marks = compute_marks(inst0, sample_omega(inst0, params, entropy, n, "train"))
        inst = inst0.at_digest(marks.instance_digest)
        cap = np.array([[0.0 if marks.prohibited[w][slots[s][0], kle] else float(marks.u[w][slots[s][0]]) for s in ex] for w in range(T)])
        tau = np.array([inst.edges[slots[s][0]].tau for s in ex])
        for t in tags:
            r, o = D[t][n], out[t]
            sent = r["sent"][:, ex]
            free = np.maximum(cap - sent, 0.0)
            # edges share capacity with the mature chip: take what both sent
            for j, s in enumerate(ex):
                e = slots[s][0]
                both = [s2 for s2, (e2, k2, l2) in enumerate(slots) if e2 == e]
                free[:, j] = np.maximum(cap[:, j] - r["sent"][:, both].sum(axis=1), 0.0)
            lost = {m: r["lost"][:, di].copy() for m, di in dem.items()}
            o["lost"] += sum(x.sum() for x in lost.values()); o["sold"] += sum(r["served"][:, di].sum() for di in dem.values())
            stock = {p: np.concatenate([[np.nan], r["stock"][:-1, inst.slot_index[(p, kle)]]]) for p in osats}  # at the week's start
            o["end"] += sum(r["stock"][-1, inst.slot_index[(p, kle)]] for p in osats)
            o["cap"] += cap.sum(); o["sent"] += sent.sum()
            for w in range(1, T):
                left = {p: stock[p][w] - sum(sent[w, j] for j, s in enumerate(ex) if tail[s] == p) for p in osats}
                for j, s in enumerate(ex):
                    m, a = head[s], w + int(tau[j])
                    if a >= T or free[w, j] <= 0: continue
                    room = min(free[w, j], lost[m][a])
                    if room <= 0: continue
                    own = min(room, max(left[tail[s]], 0.0))
                    o["feas"] += room; o["feas_own"] += own
                    bym[t][m] += (room, own, 0)
                    lost[m][a] -= room; left[tail[s]] -= own
    print(f"{len(ns)} episodes, per episode, thousand leading chips")
    for t in tags:
        o = {k: v / len(ns) / 1e3 for k, v in out[t].items()}
        print(f"{t:16s} sold {o['sold']:7.0f} lost {o['lost']:7.0f} | at plants at the end {o['end']:5.0f} | exits: capacity {o['cap']:7.0f} sent {o['sent']:7.0f} | lost sales an exit with room could have served {o['feas']:6.0f}, of them from chips on hand at that plant {o['feas_own']:6.0f}")
        print("     by market (room, on hand):", {name[m]: tuple(round(x / len(ns) / 1e3) for x in v[:2]) for m, v in bym[t].items()})


if __name__ == "__main__":
    fire.Fire(main)
