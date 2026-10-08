"""How the network's disruptions run in the scenarios themselves: onsets, ends and lengths, field by field.

    uv run python lab/anastasiia/plan_lab/spells.py --task=small --entropy=444 --episodes=16

For edge capacities, strait openness and throughput, grid output, supply and prohibitions: the share of
element-weeks off the nominal value, how many spells start after week 1 and how many end before the episode does
(per episode), the median length of a spell, and the chance that a spell that is on in week t is over by week
t + 4, t + 13 and t + 26 (what a forecast "it stays as it is" gets wrong inside a window of 26 weeks).
"""

import fire
import numpy as np


def main(task: str = "small", entropy: int = 444, episodes: int = 16, first: int = 0) -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import regime

    fields = {"u": "edge capacity", "o": "strait openness", "kappa": "strait throughput", "G_bar": "grid output",
              "supply": "supply", "prohibited": "prohibition (edge, good)", "tariff": "tariff (edge, good)"}
    stat = {name: {"weeks": 0, "off": 0, "onsets": 0, "ends": 0, "lengths": [], "over": {4: [0, 0], 13: [0, 0], 26: [0, 0]},
                   "from1": 0, "spells": 0} for name in fields}
    for n in range(first, first + episodes):
        inst, _omega, marks = regime.world(task, entropy, n)
        T = inst.T
        for name in fields:
            a = np.asarray(getattr(marks, name), dtype=float).reshape(T, -1)
            if name in ("prohibited", "tariff"):
                calm = np.zeros(a.shape[1])
            else:
                calm = np.max(a, axis=0)  # the nominal value: the largest the element ever has
            off = ~np.isclose(a, calm[None], rtol=1e-3, atol=1e-9)
            s = stat[name]
            s["weeks"] += off.size
            s["off"] += int(off.sum())
            for j in range(off.shape[1]):
                col = off[:, j]
                if not col.any():
                    continue
                t = 0
                while t < T:
                    if col[t]:
                        e = t
                        while e < T and col[e]:
                            e += 1
                        s["spells"] += 1
                        s["from1"] += t == 0
                        s["onsets"] += t > 0
                        s["ends"] += e < T
                        s["lengths"].append(e - t)
                        for w in range(t, e):  # a week the spell is on: is it over h weeks later (if the episode lasts)
                            for h, c in s["over"].items():
                                if w + h < T:
                                    c[1] += 1
                                    c[0] += w + h >= e
                        t = e
                    else:
                        t += 1
    print(f"{task} {entropy}, episodes {first}..{first + episodes - 1}; per episode")
    print(f"{'field':26} {'off, % of weeks':>15} {'spells':>7} {'on from week 1':>15} {'start later':>12} {'end early':>10} "
          f"{'median weeks':>13} {'over in 4 / 13 / 26 weeks':>27}")
    for name, label in fields.items():
        s = stat[name]
        over = " / ".join(f"{c[0] / c[1]:.0%}" if c[1] else "-" for c in s["over"].values())
        med = np.median(s["lengths"]) if s["lengths"] else float("nan")
        print(f"{label:26} {100 * s['off'] / max(s['weeks'], 1):15.1f} {s['spells'] / episodes:7.1f} {s['from1'] / episodes:15.1f} "
              f"{s['onsets'] / episodes:12.1f} {s['ends'] / episodes:10.1f} {med:13.0f} {over:>27}")


if __name__ == "__main__":
    fire.Fire(main)
