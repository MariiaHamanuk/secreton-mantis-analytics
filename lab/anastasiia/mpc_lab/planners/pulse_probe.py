"""Open-loop probe: hold the rationed fuel at the terminal in weeks 1..k, release the held amount at k+1..k+m.

    uv run python lab/anastasiia/mpc_lab/planners/pulse_probe.py --episodes=12

The fuel search's shed relief lands in weeks 7-13 while weeks 1-6 shed a little MORE, and the grid's closing stock is
higher early (reports/lsf_signal.md): the shape of a hold -> prime -> run cycle anchored at the episode start, before
the supply dips of weeks 5-8. The earlier distillation pushed the valves UP early (burn more) and lost; this probe
tests the opposite reading on the played trajectories of the hybrid: withhold a share of the terminal -> grid valve
flow of the rationed fuel for the first k weeks, then add everything withheld onto the next m weeks' valves.
Open loop (no agent reaction), per grid; capacity and storage still clip in the replay.
"""

import os
import sys

import fire
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner_LSF as F  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "outputs", "mpc_research")


def valve_slots(inst, fs, fuel="lng"):
    return [s for s, (f, kind) in fs.items() if kind == "valve" and f == fuel]


def pulse(slots, flows, T, k, m, share):
    """Withhold ``share`` of each slot's flow in weeks 1..k; add the held total onto weeks k+1..k+m, split evenly."""
    out = [dict(fl) for fl in flows]
    for s in slots:
        held = 0.0
        for t in range(1, min(k, T) + 1):
            q = out[t - 1].get(s, 0.0)
            if q > 0.0:
                out[t - 1][s] = q * (1.0 - share)
                held += q * share
        if held <= 0.0:
            continue
        upto = min(k + m, T)
        for t in range(k + 1, upto + 1):
            out[t - 1][s] = out[t - 1].get(s, 0.0) + held / (upto - k)
    return out


def main(episodes=12, fuel="lng"):
    worlds = []
    for n in range(episodes):
        d = F.play(n)
        inst, _omega, marks = F.world(n)
        acts, _ = F.wire_actions(d, inst, marks)
        worlds.append(F.FReplay(inst, marks, acts))
    fs = F.fuel_slots(worlds[0].inst)
    slots = valve_slots(worlds[0].inst, fs, fuel)
    print(f"{fuel} valve slots: {len(slots)}")
    lines = ["| hold k / release m / частка | виграш, млрд USD/еп (середнє) | медіана | + | − |", "|---|---|---|---|---|"]
    for k in (2, 3, 4, 6):
        for m in (1, 2):
            for share in (1.0, 0.5):
                dj = np.array(
                    [(R._run(1, pulse(slots, R.flows, R.T, k, m, share), keep=False) - R.J) / 1e11 for R in worlds]
                )
                lines.append(
                    f"| k={k} m={m} share={share} | {-dj.mean():+.2f} | {-np.median(dj):+.2f} "
                    f"| {int((dj < -1e-6).sum())} | {int((dj > 1e-6).sum())} |"
                )
                print(lines[-1], flush=True)
    with open(os.path.join(OUT, "pulse_probe.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
