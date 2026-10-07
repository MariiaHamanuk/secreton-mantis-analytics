"""Diagnostic, not a submission: ``anastasiia_hybrid_hub`` whose wafer orders are held to the plan's lot schedule.

The plan that knows the future and serves the base load first (``lab/anastasiia/stats_lab/plan_stats.py``, root 444)
starts half of the hybrid's lots in the first quarter and sells as much. The agent cannot start lots, only send
wafers, so the brake is on the wafers: a fab is sent no more than brings its wafers on hand and on the way up to the
lots the plan has started by week t + ``slack``, less the lots the agent has started so far. Only a brake: the rules'
order is scaled down, never up. The fuel rules run after it and read the smaller orders.

``PLAN_LAB_PARAMS`` (set by ``harness.py --params``): ``until`` (the last week braked), ``slack`` (weeks of the plan's
lots a fab may hold ahead), ``fabs`` ("all", or a list of fab ordinals of the instance to brake).
The truth comes through ``tell_truth``; without it the agent is the hybrid.
"""

import builtins
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
BASE = ROOT / "agents" / "anastasiia_hybrid_hub"
PLANS = {  # (task, root) -> the plan's weekly arrays and the episodes its file leaves out
    ("small", 444): (ROOT / "outputs/plan_stats/20261006_040041/episodes.npz", ()),
    ("full", 444): (ROOT / "outputs/heur4/lead/plan_full_444_x20/episodes.npz", (3,)),
}
PARAMS = {"until": 13, "slack": 1, "fabs": "all"} | dict(getattr(builtins, "PLAN_LAB_PARAMS", {}))

_spec = importlib.util.spec_from_file_location("plan_lab_hub_base", BASE / "agent.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Agent(_base.Agent):
    def __init__(self, config=None):
        super().__init__(config)
        self.p = dict(PARAMS)
        self.plan_cum = None  # (T, fabs in the chip rules' order): the plan's lots started by the end of each week
        self.started = np.zeros(len(self.chips.fabs))  # lots this agent has started so far, per fab
        self.notes = {"cut": 0.0}

    def tell_truth(self, truth) -> None:
        path, left_out = PLANS[(truth["task"], truth["entropy"])]
        n = truth["episode"]
        if n in left_out:
            return
        row = n - sum(1 for x in left_out if x < n)
        lots = np.load(path)["plan_lots"]
        if row >= len(lots):
            return
        order = [truth["inst"].fab_ordinal[f] for f in self.chips.fabs]
        self.plan_cum = np.cumsum(lots[row][:, order], axis=0)

    def _brake(self, observation, flows) -> None:
        c = self.chips
        week = int(observation["week"][0])
        live = observation["wip.qty.observed"] == 1
        node, out_week, qty = observation["wip.node"][live], observation["wip.out_week"][live], observation["wip.qty"][live]
        for i, f in enumerate(c.fabs):  # lots started last week: in process, out in week (t - 1) + tau
            self.started[i] += qty[(node == f) & (out_week == week - 1 + c.fab[f]["tau"])].sum()
        if self.plan_cum is None or week > self.p["until"]:
            return
        moving = observation["pipeline.qty.observed"] == 1
        edge, k, q = observation["pipeline.edge"][moving], observation["pipeline.k"][moving], observation["pipeline.qty"][moving]
        lane = np.where(observation["pipeline.lane.observed"][moving] == 1, observation["pipeline.lane"][moving], -1)
        head = np.array([c.edges["head"][c.lanes["edges"][ln][-1]] if ln >= 0 else c.edges["head"][e] for e, ln in zip(edge, lane)])
        T = len(self.plan_cum)
        for i, f in enumerate(c.fabs):
            if self.p["fabs"] != "all" and i not in self.p["fabs"]:
                continue
            inp = c.fab[f]["inp"]
            on_hand = float(observation["stock.qty"][c.stock_ix[(f, inp)]]) + float(q[(head == f) & (k == inp)].sum())
            allowed = self.plan_cum[min(week + self.p["slack"], T) - 1, i] - self.started[i]
            room = max(0.0, allowed - on_hand)
            slots = c.wafer_slots[f]
            asked = float(flows[slots].sum())
            if asked > room:
                flows[slots] *= room / asked
                self.notes["cut"] += asked - room

    def act(self, observation):
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)
        planned = self._planned(observation)
        if planned is not None:
            flows[self.take] = planned[self.take]
            self.planned_weeks += 1
        self._brake(observation, flows)
        self.fuel.fill(observation, flows)
        return {"flows": flows, **self.strait.fill(observation)}
