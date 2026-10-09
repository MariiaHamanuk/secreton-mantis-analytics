"""Closing-week premium: pay for fuel burned in the weeks the cell asks to close ("HULL"/"SOFT"), and for the
stock that feeds them.

Idea: (1) a reward of 4 % of the grid's voll per GWh on every fuel segment G(t) in a HULL or SOFT week, which is
bounded by the segment's cap so it cannot be hoarded; (2) a reward of 2 % of voll per GWh on the end stock I(t) of
a scarce fuel (opening stock under two weeks of its burn) in the week before such a week. Mechanism: a unit of fuel
that closes a whole week is worth 25-40 m USD against 4 m as base load, which the cell's linear pricing only
partly sees; these terms (about 1 % of the closure value) tilt fuel toward weeks that can close. Off in the last two
weeks of the episode.
"""

import numpy as np


GB, IB = 0.04, 0.02


def _scarce(ep):
    cached = getattr(ep, "_a3", None)
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
    ep._a3 = scarce
    return scarce


def add(ep, mode, ref):
    out = np.zeros(ep.N)
    inst, T = ep.inst, ep.T
    left = ep.info.get("left", 10**6) if getattr(ep, "info", None) else 10**6
    last = min(T, int(left) - 2)
    scarce = _scarce(ep)
    closing = {key for key, gm in mode["grid"].items() if gm in ("HULL", "SOFT")}
    for t in range(1, last + 1):
        for gi, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            if ga.voll <= 0:
                continue
            for k in ga.fuels:
                if (t, gi) in closing and ep.has("G", t, gi, k):
                    out[ep.col("G", t, gi, k)] -= GB * float(ga.voll)
                if t < T and (t + 1, gi) in closing and (gi, k) in scarce and (g, k) in inst.slot_index:
                    s = ep.slot(g, k)
                    if ep.has("I", s):
                        out[ep.col("I", t, s)] -= IB * float(ga.voll)
    return out
