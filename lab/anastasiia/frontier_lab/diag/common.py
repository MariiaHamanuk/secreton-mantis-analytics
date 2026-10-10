"""What every script of this folder reads: the kept plays of ``hazard_lab/play.py``, the references and the names."""

import pickle
import sys
from functools import cache
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PLAY = ROOT / "outputs" / "hazard_lab" / "play"
OUT = ROOT / "outputs" / "frontier_lab" / "diag"
ITEMS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")
WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}  # the board's weight of each harm level
BN = 1e9  # the plays keep weekly costs in USD
sys.path.insert(0, str(ROOT / "lab" / "anastasiia" / "hazard_lab"))


@cache
def kept(tag: str, task: str = "full", entropy: int = 444) -> dict:
    return pickle.loads((PLAY / f"{tag}_{task}_{entropy}.pkl").read_bytes())


@cache
def clair(task: str = "full", entropy: int = 444) -> dict:
    return pickle.loads((OUT / f"clair_{task}_{entropy}.pkl").read_bytes())


@cache
def refs(task: str = "full", entropy: int = 444, episodes: int = 112) -> list[dict]:
    import play

    return play.references(task, entropy, episodes)


@cache
def instance(task: str = "full"):
    from shockbench_flow.hosting.tasks import task_generator

    return task_generator(task)[0]


class Names:
    """Ordinals to readable names, and the groupings the tables use."""

    def __init__(self, task: str = "full") -> None:
        inst = self.inst = instance(task)
        N = inst.nodes
        self.K = [c.id for c in inst.commodities]
        self.grids = [N[g].id for g in inst.grids]
        self.fabs = [N[f].id for f in inst.fabs]
        self.fab_grid = np.array([-1 if N[f].fab.grid is None else inst.grid_ordinal[N[f].fab.grid] for f in inst.fabs])
        self.demands = [(N[d.node].id, self.K[d.k]) for d in inst.demands]
        self.demand_k = np.array([d.k for d in inst.demands])
        self.penalty = np.array([d.pi for d in inst.demands])  # USD a unit of unmet demand
        self.slot_k = np.array([k for _e, k, _lane in inst.action_slots])
        self.slot_edge = np.array([e for e, _k, _lane in inst.action_slots])
        self.slot_tail = [N[inst.edges[e].tail].id for e in self.slot_edge]
        self.slot_head = [N[inst.edges[e].head].id for e in self.slot_edge]
        self.stock = [(N[s.node].id, self.K[s.k]) for s in inst.stock_slots]


def level_weights(levels: np.ndarray) -> np.ndarray:
    """A weight per episode so that a weighted mean over the episodes is the board's mix of harm levels (the levels
    present share the whole weight in the board's proportions)."""
    present = {s: WEIGHTS[s] for s in set(levels.tolist())}
    total = sum(present.values())
    return np.array([present[s] / total / (levels == s).sum() for s in levels])


def paired(
    delta: np.ndarray, levels: np.ndarray, room: np.ndarray, draws: int = 4000, seed: int = 0
) -> tuple[float, float, float]:
    """A saving per episode (the same unit as ``room``) as a score difference: the board's weighting, and the 5th and
    95th percentiles when the episodes are drawn again inside each harm level."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(levels == s) for s in sorted(set(levels.tolist()))]
    w = {s: WEIGHTS[s] for s in set(levels.tolist())}

    def score(idx_by_level) -> float:
        num = sum(w[int(levels[i[0]])] * delta[i].mean() for i in idx_by_level)
        den = sum(w[int(levels[i[0]])] * room[i].mean() for i in idx_by_level)
        return num / den

    point = score(groups)
    boot = [score([rng.choice(g, len(g)) for g in groups]) for _ in range(draws)]
    lo, hi = np.percentile(boot, [5, 95])
    return float(point), float(lo), float(hi)
