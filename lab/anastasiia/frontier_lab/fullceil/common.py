"""Shared pieces of the Full ceiling study (notes/u_fullceil.md): the episode with its whole future, the kept plays as
starts, the references and the score.

Everything is offline with the whole future known (Full, root 444). A cost counts only when the simulator played it
(``Episode.simulate``); a program's own value and a bound are named as such wherever they are printed.
"""

import pickle
import resource
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LAB = HERE.parents[1]
sys.path[:0] = [str(LAB / "regime_lab"), str(LAB / "mpc_lab"), str(LAB / "stats_lab"), str(HERE.parent)]

OUT = ROOT / "outputs" / "frontier_lab" / "fullceil"
PLAY = ROOT / "outputs" / "hazard_lab" / "play"
WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}  # the board's weight of each harm level
BN = 1e11  # cents in a bn USD


def rss_mb() -> float:
    """Peak resident memory of this process, MB (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)


def kept(tag: str, task: str = "full", entropy: int = 444) -> dict:
    """The kept play of ``tag`` (``hazard_lab/play.py run``): episode -> its record."""
    return pickle.loads((PLAY / f"{tag}_{task}_{entropy}.pkl").read_bytes())


def sent_actions(ep, sent: np.ndarray) -> list[tuple]:
    """A kept play's executed dispatches as weekly (flows, overrides, holds). The play's tanker releases are not
    kept: the straits release by their default rule, so the replay need not cost what the play did."""
    return [({int(s): float(q) for s, q in enumerate(row) if q > 0.0}, {}, frozenset()) for row in sent[: ep.T]]


def references(task: str = "full", entropy: int = 444, upto: int = 64) -> list[dict]:
    import sbf_starter  # noqa: F401 - points the package at the team's reference cache
    from sbf_starter import scoring

    return list(scoring.episode_set(task, upto, entropy=entropy, n_jobs=1, verbose=False).references)


def rss(level: np.ndarray, naive: np.ndarray, oracle: np.ndarray, J: np.ndarray) -> float:
    """The board's score; a set without one of the four levels: all that was saved over all that could be."""
    if set(level.tolist()) == set(WEIGHTS):
        num = sum(w * (naive - J)[level == s].mean() for s, w in WEIGHTS.items())
        return float(num / sum(w * (naive - oracle)[level == s].mean() for s, w in WEIGHTS.items()))
    return float((naive - J).sum() / (naive - oracle).sum())


def weeks_of(recs: list) -> dict:
    """What a played plan did by week, for the comparison of plans (the fields ``hazard_lab/play.py`` keeps)."""
    comp = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
    return {
        "costs": np.array([[getattr(r.costs, c) for c in comp] for r in recs]),
        "lots": np.array([r.lots_started for r in recs]), "shed": np.array([r.shed for r in recs]),
        "lost": np.array([r.lost for r in recs]), "served": np.array([r.served for r in recs]),
        "energy": np.array([r.energy for r in recs]), "stock": np.array([r.stock for r in recs]),
    }  # fmt: skip


def sent_of(ep, recs: list) -> np.ndarray:
    out = np.zeros((len(recs), len(ep.inst.action_slots)))
    for t, r in enumerate(recs):
        for slot, q in r.executed.items():
            out[t, slot] = q
    return out
