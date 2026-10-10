"""Where the leading chips of a kept play end up, and what could be known of a plant's exits when its chips were sent.

    uv run python lab/anastasiia/frontier_lab/chip_trace.py w50as_s truthall_h0_s --episodes=24

For each tag, per episode on average (thousand units): lots of the leading chip started; raw chips sent from fabs to
plants; packaged chips sent from plants to markets; packaged and raw chips left at the end, by where they sit; and
the balance, which is what was thrown away (stores that overflowed) or is still on the way.

Then, for the first tag, every week a fab sent raw leading chips to a plant: the state of that plant's exits to the
markets with unmet demand in the week of dispatch and in the weeks the chips could leave it (4 to 12 weeks later),
as the scenario has them. "Open" is the share of the exits' nominal capacity not prohibited and not cut. Printed: raw
chips sent, split by how open the plant's exits were at dispatch and how open they turned out to be, so that chips
sent to a plant whose exits were already shut are told from chips whose plant was shut afterwards.
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
sim = importlib.import_module("shockbench_flow.dynamics.sim")


def main(*tags: str, task: str = "small", entropy: int = 444, episodes: int = 24, first: int = 0) -> None:
    kept = {t: pickle.loads((ROOT / f"outputs/hazard_lab/play/{t}_{task}_{entropy}.pkl").read_bytes()) for t in tags}
    inst0, params = task_generator(task)
    com = [c.id for c in inst0.commodities]
    kle, kraw = com.index("chip_le"), com.index("chip_le_raw")
    name = [n.id for n in inst0.nodes]
    slots = list(inst0.action_slots)
    T = inst0.T
    ns = [n for n in range(first, first + episodes) if all(n in kept[t] for t in tags)]
    tail = [inst0.edges[e].tail for e, _k, _l in slots]
    head = [inst0.edges[e if lane is None else inst0.lanes[lane].edges[-1]].head for e, _k, lane in slots]
    fabs_le = [fi for fi, f in enumerate(inst0.fabs) if inst0.nodes[f].fab.product == kraw]
    osats = [o for o in inst0.osats if kraw in inst0.nodes[o].osat.packages]
    raw_slots = [s for s, (e, k, l) in enumerate(slots) if k == kraw and tail[s] in inst0.fabs and head[s] in osats]
    out_slots = [s for s, (e, k, l) in enumerate(slots) if k == kle and tail[s] in osats]
    book = {t: dict(lots=0.0, raw_sent=0.0, out_sent=0.0, sold=0.0, at_plants=0.0, at_fabs=0.0, at_markets=0.0) for t in tags}
    grid = np.zeros((3, 3))  # exits at dispatch (shut / part / open) x exits when the chips could leave
    grid_n = 0.0
    for n in ns:
        marks = compute_marks(inst0, sample_omega(inst0, params, entropy, n, "train"))
        inst = inst0.at_digest(marks.instance_digest)
        dem = [di for di, d in enumerate(inst.demands) if d.k == kle]
        for t in tags:
            r, b = kept[t][n], book[t]
            b["lots"] += r["lots"][:, fabs_le].sum() / 1e3
            b["raw_sent"] += r["sent"][:, raw_slots].sum() / 1e3
            b["out_sent"] += r["sent"][:, out_slots].sum() / 1e3
            b["sold"] += r["served"][:, dem].sum() / 1e3
            b["at_plants"] += sum(r["stock"][-1, inst.slot_index[(o, kle)]] + r["stock"][-1, inst.slot_index[(o, kraw)]] for o in osats) / 1e3
            b["at_fabs"] += sum(r["stock"][-1, inst.slot_index[(inst.fabs[fi], kraw)]] for fi in fabs_le) / 1e3
            b["at_markets"] += sum(r["stock"][-1, inst.slot_index[(inst.demands[di].node, kle)]] for di in dem
                                   if (inst.demands[di].node, kle) in inst.slot_index) / 1e3
        # the first tag: exits of a plant at dispatch and later
        r = kept[tags[0]][n]
        u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
        open_share = {}
        for o in osats:
            ex = [s for s in out_slots if tail[s] == o and slots[s][2] is None]
            if not ex:
                continue
            edges = [slots[s][0] for s in ex]
            nominal = sum(u0[e] for e in edges)
            free = np.array([sum(0.0 if marks.prohibited[w][e, kle] else float(marks.u[w][e]) for e in edges) for w in range(T)])
            open_share[o] = free / nominal
        for s in raw_slots:
            o = head[s]
            if o not in open_share:
                continue
            lag = inst.edges[slots[s][0]].tau + inst.nodes[o].osat.tau + 1  # weeks until the packaged chip is on hand
            for w in np.flatnonzero(r["sent"][:, s] > 0):
                later = open_share[o][min(w + lag, T - 1) : min(w + lag + 8, T)]
                if not len(later):
                    continue
                now, then = open_share[o][w], float(later.mean())
                i = 0 if now < 0.2 else 2 if now > 0.8 else 1
                j = 0 if then < 0.2 else 2 if then > 0.8 else 1
                grid[i, j] += r["sent"][w, s] / 1e3
                grid_n += r["sent"][w, s] / 1e3
    m = len(ns)
    print(f"{task} {entropy}, {m} episodes from {first}; the leading chip, thousand units an episode")
    print(f"{'':16s} {'lots':>8} {'raw sent':>9} {'out of plants':>14} {'sold':>8} {'left at plants':>15} {'at fabs':>8} {'thrown away or on the way':>26}")
    for t in tags:
        b = {k: v / m for k, v in book[t].items()}
        lost = b["lots"] - b["sold"] - b["at_plants"] - b["at_fabs"] - b["at_markets"]
        print(f"{t:16s} {b['lots']:8.0f} {b['raw_sent']:9.0f} {b['out_sent']:14.0f} {b['sold']:8.0f} {b['at_plants']:15.0f} {b['at_fabs']:8.0f} {lost:26.0f}")
    print(f"\n{tags[0]}: raw leading chips sent to plants, by how open the plant's exits were in the week of dispatch (rows) and in the eight weeks its chips could leave (columns); thousand an episode")
    print(f"{'':28s} {'later shut (<20%)':>18} {'later part':>12} {'later open (>80%)':>18}")
    for i, label in enumerate(("shut at dispatch (<20%)", "part open at dispatch", "open at dispatch (>80%)")):
        print(f"{label:28s} {grid[i, 0] / m:18.0f} {grid[i, 1] / m:12.0f} {grid[i, 2] / m:18.0f}")
    print(f"sent in all {grid_n / m:.0f}")


if __name__ == "__main__":
    fire.Fire(main)
