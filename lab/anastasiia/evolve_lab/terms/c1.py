"""Anchor price that rises with the week of the window (chip chain: wafer orders).

Idea: the rules' plan is firm in the near weeks (this week's wafer order is what gets executed) and the program's own
far-week wafer flows rest on forecasts that are worth nothing, so leaving the rules' plan should be cheap in week 1
and dear far out. Mechanism: the anchor columns of wafer slots (``jdev``, ``jdev + 1``) get an extra price that rises
linearly from -15 % of the existing 5000 USD per wafer unit in week 1 to +50 % in the last week of the window.
Expected effect: fewer speculative far-week wafer reroutings that distort the first week, same freedom now.
"""

import numpy as np


_ANCHOR = 5000.0  # the model's anchor price per wafer unit, USD
_LO, _HI = -0.15, 0.50  # share of it added in the first and in the last week of the window


def _wafer_slots(ep):
    inst = ep.inst
    wafers = {inst.nodes[f].fab.input for f in inst.fabs}
    return [s for s, (_e, k, _lane) in enumerate(inst.action_slots) if k in wafers]


def add(ep, mode, ref):
    if not ep.anchored:
        return None
    v = getattr(ep, "_c1", None)
    if v is None:
        v = np.zeros(ep.N)
        T = ep.T
        for s in _wafer_slots(ep):
            for t in range(1, T + 1):
                frac = (t - 1) / (T - 1) if T > 1 else 0.0
                j = ep.jdev(t, s)
                v[j] = v[j + 1] = _ANCHOR * (_LO + (_HI - _LO) * frac)
        ep._c1 = v
    return v
