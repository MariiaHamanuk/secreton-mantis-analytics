"""The team's approved episode sets.

An episode is (task, root, index): the generator rebuilds it bit for bit from those three with the package version
pinned in ``uv.lock``, so a set is fixed by its root and its count, and nothing but the reference costs of its
episodes is kept in git (``hub/refcache``).

- ``FORMAL``: the sets of ``hub/FORMAL_RESULTS.md``. Nothing is ever tuned on them.
- ``TUNING``: where variants are compared and parameters chosen. A model tuned here looks a little better here.
"""

FORMAL = {
    "small": {"entropy": 222, "episodes": 256},
    "full": {"entropy": 222, "episodes": 128},
}
FORMAL_FULL_LONG = 256  # the optional longer Full set (the same root): its first 128 episodes are the formal ones
TUNING = {
    "small": {"entropy": 111, "episodes": 256},
    "full": {"entropy": 111, "episodes": 256},
}
PREFIXES = (64, 128, 256)  # a run on n episodes also gives the score on its first 64 and 128
LEVEL_WEIGHTS = (0.50, 0.30, 0.15, 0.05)  # the board's weight of each harm level
