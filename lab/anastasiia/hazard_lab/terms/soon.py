"""A sale lost in a near week of the window is dearer than one lost in a far week.

The week's program prices a sale planned twenty weeks ahead as it prices this week's, though by then the routes it
counts on have changed. A unit of demand lost in week t of the window costs ``RATE`` of its shortage penalty more
for every week t is before the middle of a window of ``SPAN`` weeks and as much less for every week after it, so a
chip sold n weeks sooner is worth ``RATE * n`` of the penalty more and the price of a lost sale is unchanged on
average. Only the timing of sales moves: a chip that is not sold in the window is charged nothing.
"""

import numpy as np


RATE = 0.01  # of the shortage penalty, a unit a week
SPAN = 26  # weeks of the model's window


def add(ep, mode, ref):
    if RATE <= 0.0:
        return None
    v = np.zeros(ep.N)
    hit = False
    for do, d in enumerate(ep.inst.demands):
        if not ep.has("U", do):
            continue
        for t in range(1, ep.T + 1):
            v[ep.col("U", t, do)] += RATE * float(d.pi) * (0.5 * (SPAN - 1) - (t - 1))
            hit = True
    return v if hit else None
