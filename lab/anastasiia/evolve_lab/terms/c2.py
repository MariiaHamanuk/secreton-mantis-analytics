"""Price on raw chips queuing in front of an OSAT that already runs at its throughput cap.

Idea: where the OSAT's regime is "at capacity" (mode["osat"] == 0) extra raw chips cannot be processed this week, so
they only wait or are thrown away. Mechanism: a price on the raw-chip stock column ``I`` of that OSAT of 2 % of the
packaged chip's shortage penalty per unit-week, tripled when the OSAT's throughput is cut (restoration below 1).
Expected effect: the program stops pushing more raw chips to a saturated plant than it can pack, and sends them to
another plant or holds the lots upstream. Magnitude: 2 %/week is small against pi, so only real queues feel it.
"""

import numpy as np


def _setup(ep):
    inst = ep.inst
    pi = {}
    for d in inst.demands:
        pi[d.k] = max(pi.get(d.k, 0.0), float(d.pi))
    out = []
    for oi, o in enumerate(inst.osats):
        osat = inst.nodes[o].osat
        for kr, kp in sorted(osat.packages.items()):
            base = 0.02 * pi.get(kp, 0.0)
            s = ep.slot(o, kr)
            if base > 0.0 and ep.has("I", s):
                out.append((oi, s, base))
    nom = {oi: float(inst.nodes[o].osat.thr) for oi, o in enumerate(inst.osats)}
    return out, nom


def add(ep, mode, ref):
    st = getattr(ep, "_c2", None)
    if st is None:
        st = ep._c2 = _setup(ep)
    items, nom = st
    if not items:
        return None
    v = np.zeros(ep.N)
    for t in range(1, ep.T + 1):
        for oi, s, base in items:
            if mode["osat"][(t, oi)] != 0:
                continue
            thr = ep.osat_thr(t, oi)
            if thr <= 0.0 or nom[oi] <= 0.0:
                continue
            cut = max(0.0, 1.0 - thr / nom[oi])
            v[ep.col("I", t, s)] += base * (1.0 + 2.0 * min(1.0, cut))
    return v
