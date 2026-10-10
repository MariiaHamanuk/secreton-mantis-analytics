"""A price on packaged chips that end a week at a plant: a sale put off is a sale at risk.

The week's program trusts its window. A packaged chip kept at a plant for a later exit without a tariff costs it the
holding rate alone (16 USD a week for a leading chip), against 2 000 to 5 000 USD of tariff on the exit that is open
now, so it keeps the chip for weeks while the market behind the open exit loses sales. In play the later exit is
prohibited or cut before the chip leaves, or the episode ends with the chip at the plant. ``RATE`` of the chip's
shortage penalty is charged for every week a packaged chip ends at a plant.
"""

import numpy as np


RATE = 0.01  # of the commodity's shortage penalty, a unit a week


def _setup(ep) -> list:
    inst = ep.inst
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), float(d.pi))
    out = []
    for o in inst.osats:
        for _raw, k in sorted(inst.nodes[o].osat.packages.items()):
            if pi.get(k, 0.0) > 0.0 and (o, k) in inst.slot_index and ep.has("I", ep.slot(o, k)):
                out.append((ep.slot(o, k), pi[k]))
    return out


def add(ep, mode, ref):
    items = getattr(ep, "_hold", None)
    if items is None:
        items = ep._hold = _setup(ep)
    if not items or RATE <= 0.0:
        return None
    v = np.zeros(ep.N)
    for t in range(1, ep.T + 1):
        for s, pi in items:
            v[ep.col("I", t, s)] += RATE * pi
    return v
