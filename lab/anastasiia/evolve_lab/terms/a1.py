"""Threshold climb: a decaying reward for stock I of a grid's rationed fuel while it is still below psi * ibar.

Mechanism: in calm episodes the rationed-fuel stock never reaches the threshold, so the grid cannot run at full
output and its fabs get no energy. A reward of 5 % of the grid's voll per GWh held (about 0.2 % of the 25-40 m USD a
closed week is worth) is paid on I(t) of the rationed fuel in the first weeks of a window whose opening stock i0 is
under the threshold; it falls linearly to 0 over the weeks the climb needs at half the grid's burn, is off when the
next week is already forced to the threshold (regime "F": nothing left to climb), and off in the last two weeks of
the episode (fuel that arrives then cannot be burned). Expected: earlier arrival of fuel, less late hoarding.
"""

import math

import numpy as np


EPS = 0.05  # share of the grid's voll per GWh of stock, at its largest


def _plan(ep):
    """(grid ordinal, column slot, weekly weights as a list) per grid with a rationed fuel below its threshold."""
    cached = getattr(ep, "_a1", None)
    if cached is not None:
        return cached
    inst, T = ep.inst, ep.T
    left = ep.info.get("left", 10**6) if getattr(ep, "info", None) else 10**6
    plan = []
    for gi, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        r = ga.rationed
        if r is None or (g, r) not in inst.slot_index or ga.voll <= 0:
            continue
        thr = ep.psi * float(ga.ibar.get(r, 0.0))
        burn = float(ga.shares.get(r, 0.0)) * float(ga.deliverable)
        s = ep.slot(g, r)
        deficit = thr - float(ep.i0[s])
        if thr <= 0 or burn <= 0 or deficit <= 0:
            continue
        tw = min(T - 1, 2 + int(math.ceil(deficit / (0.5 * burn))))
        tw = max(0, min(tw, int(left) - 2))
        if tw <= 0:
            continue
        w = [EPS * float(ga.voll) * (1.0 - t / tw) for t in range(tw)]  # week t+1 gets w[t]
        plan.append((gi, r, s, w))
    ep._a1 = plan
    return plan


def add(ep, mode, ref):
    out = np.zeros(ep.N)
    for gi, r, s, w in _plan(ep):
        for t in range(1, len(w) + 1):
            if t >= ep.T or not ep.has("I", s):
                break
            if mode["fuel"].get((t + 1, gi, r)) == "F":
                continue
            out[ep.col("I", t, s)] -= w[t - 1]
    return out
