"""Where a played plan loses chips it has made: raw chips thrown away at the fabs, chips left at the end, by node.

    uv run python lab/anastasiia/frontier_lab/fullceil/waste.py --only=2 --a=start:tah0_f --b=run:tah0_f
"""

import common as K  # noqa: F401 - sets the paths
import fire
import numpy as np
from diff import plan_acts, played, tables


def main(only: str | tuple | int = "2", a: str = "start:tah0_f", b: str = "best", task: str = "full", entropy: int = 444,
         top: int = 8) -> None:
    import core  # regime_lab's

    ns = [int(n) for n in (str(only).split(",") if not isinstance(only, tuple) else only)]
    for n in ns:
        ep = core.Episode.of(task, entropy, n)
        tb = tables(ep)
        inst = ep.inst
        chips = [k for k, name in enumerate(tb["K"]) if name.startswith("chip")]
        pi = {}
        for d in inst.demands:
            pi[d.k] = max(pi.get(d.k, 0.0), d.pi)
        for o in inst.osats:
            for raw, fin in inst.nodes[o].osat.packages.items():
                pi[raw] = pi.get(fin, 0.0)
        print(f"=== ep {n}")
        for name, plan in (("A", a), ("B", b)):
            acts, label = plan_acts(ep, plan, task, entropy, n)
            P = played(ep, acts)
            thrown = P["disposal"].sum(axis=0)
            rows = [(thrown[s] * pi.get(int(tb["stock_k"][s]), 0.0) / 1e9, s) for s in range(len(thrown)) if tb["stock_k"][s] in chips and thrown[s] > 0]
            rows.sort(reverse=True)
            print(f"  {name} = {label} ({P['J'] / K.BN:.1f} bn): chips thrown away, bn at the chip's penalty: "
                  + (", ".join(f"{tb['stock_node'][s]}/{tb['K'][tb['stock_k'][s]]} {v:.1f} (weeks {int(np.flatnonzero(P['disposal'][:, s] > 0)[0]) + 1}-"
                               f"{int(np.flatnonzero(P['disposal'][:, s] > 0)[-1]) + 1}, {int((P['disposal'][:, s] > 0).sum())} of them)" for v, s in rows[:top]) or "none"))
            left = [(P["stock"][-1, s] * pi.get(int(tb["stock_k"][s]), 0.0) / 1e9, s) for s in range(len(thrown)) if tb["stock_k"][s] in chips and P["stock"][-1, s] > 0]
            left.sort(reverse=True)
            print("      chips left at the end, bn: " + (", ".join(f"{tb['stock_node'][s]}/{tb['K'][tb['stock_k'][s]]} {v:.1f}" for v, s in left[:top]) or "none"))


if __name__ == "__main__":
    fire.Fire(main)
