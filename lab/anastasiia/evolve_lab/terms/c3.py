"""Episode-end-aware asymmetric anchor on wafer orders (state-dependent, not week-of-window).

Idea: a wafer ordered at episode week w becomes a sold chip only about fab.tau + osat.tau + 4 weeks later (14 for
advanced fabs, 12 for mature); in a window that stops before the episode does, the end credit may still price such
late wafers as if they sold. Mechanism: for wafer slots, in window weeks within 6 weeks of that horizon (or beyond it)
sending MORE than the rules' plan costs up to +5000 USD (100 % of the anchor price) extra and sending LESS up to
1250 USD (25 %) less, ramping linearly. Expected effect: late-episode over-ordering of dead wafers is not rewarded.
"""

import numpy as np


_ANCHOR = 5000.0
_RAMP = 6.0


def _setup(ep):
    inst = ep.inst
    osat_tau = min((inst.nodes[o].osat.tau for o in inst.osats), default=0)
    lag = {}
    for f in inst.fabs:
        fa = inst.nodes[f].fab
        lag[fa.input] = min(lag.get(fa.input, 10**9), int(fa.tau) + int(osat_tau) + 4)
    slots = [(s, lag[k]) for s, (_e, k, _lane) in enumerate(inst.action_slots) if k in lag]
    return slots


def add(ep, mode, ref):
    if not ep.anchored:
        return None
    left = int(ep.info.get("left", 0)) if isinstance(ep.info, dict) else 0
    if left <= ep.T or left <= 0:  # the window reaches the episode's end: the program sees the end itself
        return None
    v = getattr(ep, "_c3", None)
    if v is None:
        v = np.zeros(ep.N)
        for s, lag in _setup(ep):
            for t in range(1, ep.T + 1):
                r = min(1.0, max(0.0, (t + lag - left + _RAMP) / _RAMP))
                if r > 0.0:
                    j = ep.jdev(t, s)
                    v[j] = _ANCHOR * r
                    v[j + 1] = -0.25 * _ANCHOR * r
        ep._c3 = v
    return v
