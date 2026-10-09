"""Parking price and lift tilt: fuel should flow, not sit.

Idea: (1) a small price on fuel parked in strait queues Q (cannot be burned this week) in near weeks, falling to 0
over the first 6 weeks; (2) a tilt on lifting fuel from its source that grows linearly from 0 to 1 % of the fuel's
value over the window, so that a lift available now is taken now. Both are priced from the grid voll (the burn
value of the fuel, the cheapest consuming grid): 3 % and 1 % of it per GWh. Targets late arrival that leaves a
rationed grid below the threshold; expected effect: dispatch through open routes earlier, fewer queued GWh.
No term on the last week (its columns are credited by the end valuation) or on non-fuel commodities.
"""

import numpy as np


QP, LT, NEAR = 0.03, 0.01, 6


def _plan(ep):
    cached = getattr(ep, "_a2", None)
    if cached is not None:
        return cached
    inst, T = ep.inst, ep.T
    value = {}
    for g in inst.grids:
        ga = inst.nodes[g].grid
        if ga.voll <= 0:
            continue
        for k in ga.fuels:
            if ga.shares[k] > 0:
                value[k] = min(value.get(k, float("inf")), float(ga.voll))
    left = ep.info.get("left", 10**6) if getattr(ep, "info", None) else 10**6
    last = max(0, min(T - 1, int(left) - 2))  # weeks 1..last carry a term
    qcols = []  # (key, price)
    lcols = []
    for key in sorted(ep.tm, key=repr):
        if key[0] == "Q" and key[2] in value:
            qcols.append((key[1:], value[key[2]] * QP))
        elif key[0] == "lift" and inst.stock_slots[key[1]].k in value:
            lcols.append((key[1], value[inst.stock_slots[key[1]].k] * LT))
    plan = (last, qcols, lcols)
    ep._a2 = plan
    return plan


def add(ep, mode, ref):
    out = np.zeros(ep.N)
    last, qcols, lcols = _plan(ep)
    for t in range(1, last + 1):
        wq = max(0.0, 1.0 - (t - 1) / float(NEAR))
        wl = (t - 1) / float(max(1, ep.T - 1))
        if wq > 0:
            for key, p in qcols:
                out[ep.col("Q", t, *key)] += p * wq
        if wl > 0:
            for s, p in lcols:
                out[ep.col("lift", t, s)] += p * wl
    return out
