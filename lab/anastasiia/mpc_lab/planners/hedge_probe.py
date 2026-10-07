"""Open-loop probe: hedge fuel ORDERS across alternate lanes to the same terminal, on the hybrid's played flows.

    uv run python lab/anastasiia/mpc_lab/planners/hedge_probe.py --episodes=12

The fuel search's gain sits in episode-specific lane choice (reports/lsf_signal.md): x1.5 on exactly the lanes that
survive. An honest rule cannot pick the survivor, but it can pay a small premium to spread each week's order of one
(terminal, fuel) across the open alternate lanes, so a closure hurts less. This probe measures that first order of
that idea on the played trajectories (replay, no closed-loop reaction): redistribute each week's group total across
the group's lanes open that week (first edge not prohibited), by a mix of the played split and the uniform split.
A lane's capacity still clips in the replay, so the premium (slower or thinner lanes) is priced in.
"""

import os
import sys

import fire
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner_LSF as F  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "outputs", "mpc_research")


def order_groups(inst, fs):
    """(terminal node, fuel) -> [order slots of its alternate lanes], only groups with >= 2 lanes."""
    groups = {}
    for s, (fuel, kind) in fs.items():
        if kind != "order":
            continue
        e, k, lane = inst.action_slots[s]
        dest = inst.edges[inst.lanes[lane].edges[-1]].head if lane is not None else inst.edges[e].head
        groups.setdefault((dest, fuel), []).append(s)
    return {g: slots for g, slots in groups.items() if len(slots) >= 2}


def hedge(inst, marks, fs, groups, flows, T, mix, lo=1, hi=10**6, fuels=None):
    """Redistribute each week's group total: mix * uniform over open lanes + (1 - mix) * the played split."""
    out = []
    for t, fl in enumerate(flows, start=1):
        if not lo <= t <= hi:
            out.append(fl)
            continue
        fl = dict(fl)
        for (dest, fuel), slots in groups.items():
            if fuels is not None and fuel not in fuels:
                continue
            total = sum(fl.get(s, 0.0) for s in slots)
            if total <= 0.0:
                continue
            open_slots = [
                s for s in slots if not marks.prohibited[t - 1, inst.action_slots[s][0], inst.action_slots[s][1]]
            ]
            if len(open_slots) < 2:
                continue
            uni = total / len(open_slots)
            for s in slots:
                played = fl.get(s, 0.0)
                want = (1.0 - mix) * played + (mix * uni if s in open_slots else 0.0)
                if want > 0.0:
                    fl[s] = want
                elif s in fl:
                    del fl[s]
        out.append(fl)
    return out


def main(episodes=12):
    worlds = []
    for n in range(episodes):
        d = F.play(n)
        inst, _omega, marks = F.world(n)
        acts, _ = F.wire_actions(d, inst, marks)
        worlds.append((F.FReplay(inst, marks, acts), inst, marks))
    inst0 = worlds[0][1]
    fs = F.fuel_slots(inst0)
    groups = order_groups(inst0, fs)
    print(f"order groups with alternate lanes: {len(groups)}:")
    for (dest, fuel), slots in sorted(groups.items()):
        print(f"  {inst0.nodes[dest].id:12s} {fuel:6s} {len(slots)} lanes")
    variants = {
        "uni mix=1.0, all weeks": dict(mix=1.0),
        "mix=0.5, all weeks": dict(mix=0.5),
        "mix=0.25, all weeks": dict(mix=0.25),
        "mix=0.5, weeks 1-26": dict(mix=0.5, hi=26),
        "mix=0.25, weeks 1-26": dict(mix=0.25, hi=26),
        "mix=0.5, lng only": dict(mix=0.5, fuels={"lng"}),
        "mix=0.25, lng only": dict(mix=0.25, fuels={"lng"}),
    }
    lines = ["| варіант | виграш, млрд USD/еп (середнє) | медіана | + | − |", "|---|---|---|---|---|"]
    for name, kw in variants.items():
        dj = []
        for R, inst, marks in worlds:
            flows = hedge(inst, marks, fs, groups, R.flows, R.T, **kw)
            dj.append((R._run(1, flows, keep=False) - R.J) / 1e11)  # + is worse
        dj = np.array(dj)
        lines.append(
            f"| {name} | {-dj.mean():+.2f} | {-np.median(dj):+.2f} | {int((dj < -1e-6).sum())} | {int((dj > 1e-6).sum())} |"
        )
        print(lines[-1], flush=True)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "hedge_probe.md"), "w") as f:
        f.write(text)


if __name__ == "__main__":
    fire.Fire(main)
