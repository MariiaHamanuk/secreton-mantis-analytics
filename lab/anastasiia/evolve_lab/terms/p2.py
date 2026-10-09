"""A placebo, not a candidate: a price a thousand times below a cent on every column, in a fixed pattern. It can
only change which of several equally good vertices the solver returns, so its difference with the base shows what
"no effect" looks like on a set of episodes."""

import numpy as np

SEED = 2


def add(ep, mode, ref):
    return 1e-5 * np.sin(SEED + (0.37 + 0.11 * SEED) * np.arange(ep.N))
