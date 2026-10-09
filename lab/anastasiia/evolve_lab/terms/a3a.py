"""Refinement of a3: reward fuel only where its burn is a choice and the closing week lacks that fuel.

Parent a3's per-segment reward on G(t) in HULL/SOFT weeks was checked with `ep.has("G", t, gi, k)`, a key that never
exists (the key has no week), so only the stock reward acted. Here the G reward (4 % of voll per GWh) is applied
with the correct key and only where the fuel regime of that week is R or S (under F burn is forced); the stock
reward (2 % of voll on scarce fuel the week before) needs the closing week's regime of that fuel to be R or S.
Expected: premium aimed at the fuel that gates the closure. Off in the last two weeks of the episode.
"""

import numpy as np


GB, IB = 0.04, 0.02


def _scarce(ep):
    cached = getattr(ep, "_a3a", None)
    if cached is not None:
        return cached
    inst = ep.inst
    scarce = set()
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        for k in ga.fuels:
            burn = float(ga.shares[k]) * float(ga.deliverable)
            if burn > 0 and (g, k) in inst.slot_index and float(ep.i0[ep.slot(g, k)]) < 2.0 * burn:
                scarce.add((gi, k))
    ep._a3a = scarce
    return scarce


def add(ep, mode, ref):
    out = np.zeros(ep.N)
    inst, T = ep.inst, ep.T
    left = ep.info.get("left", 10**6) if getattr(ep, "info", None) else 10**6
    last = min(T, int(left) - 2)
    scarce = _scarce(ep)
    fuel = mode["fuel"]
    closing = {key for key, gm in mode["grid"].items() if gm in ("HULL", "SOFT")}
    for t in range(1, last + 1):
        for gi, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            if ga.voll <= 0:
                continue
            for k in ga.fuels:
                if (t, gi) in closing and ep.has("G", gi, k) and fuel.get((t, gi, k)) in ("R", "S"):
                    out[ep.col("G", t, gi, k)] -= GB * float(ga.voll)
                if (t < T and (t + 1, gi) in closing and (gi, k) in scarce and (g, k) in inst.slot_index
                        and fuel.get((t + 1, gi, k)) in ("R", "S")):
                    s = ep.slot(g, k)
                    if ep.has("I", s):
                        out[ep.col("I", t, s)] -= IB * float(ga.voll)
    return out
