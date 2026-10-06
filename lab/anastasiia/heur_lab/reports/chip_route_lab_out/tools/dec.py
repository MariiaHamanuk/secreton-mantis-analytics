"""Scratch: decode an episode record into named per-week arrays for the chip chain."""
import numpy as np

from shockbench_flow_agent.convert import layout_tables
from shockbench_flow.information.flat import FlatLayout


class Rec:
    def __init__(self, rec):
        self.rec = rec
        self.static = rec["static"]
        self.obs = rec["obs"]
        self.T = int(rec["meta"]["T"])
        self.layout = layout_tables(FlatLayout.from_static(self.static))
        st = self.static
        self.node = st["nodes"]["id"]
        self.com = st["commodities"]["id"]
        self.node_ix = {n: i for i, n in enumerate(self.node)}
        self.com_ix = {c: i for i, c in enumerate(self.com)}
        self.stock_slots = [tuple(x) for x in self.layout["stock_slots"]]
        self.slot_of = {nk: i for i, nk in enumerate(self.stock_slots)}
        self.demands = [tuple(x) for x in self.layout["demands"]]
        self.fabs = list(self.layout["fabs"])
        self.osats = list(self.layout["osats"])
        self.grids = list(self.layout["grids"])
        inst = st["instance"]
        self.inst = inst
        self.nodes_attr = inst["nodes"]
        self.costs = np.asarray(rec["costs"])
        self.flows = np.asarray(rec["action"]["flows"])  # (T, slots) requested

    def stock(self, node: str, com: str) -> np.ndarray:
        """stock at the start of weeks 1..T+1 (the state at the end of week t-1)"""
        s = self.slot_of[(self.node_ix[node], self.com_ix[com])]
        return np.asarray(self.obs["stock.qty"])[:, s]

    def wip_starts(self, node: str) -> np.ndarray:
        """lots started in weeks 1..T by a fab (read off the next observation)"""
        n = self.node_ix[node]
        tau = self.fab_attr(node)["tau"]
        out = np.zeros(self.T)
        for t in range(1, self.T + 1):  # observation of week t+1 holds the start of week t
            ob = self.obs
            live = np.asarray(ob["wip.qty.observed"][t]) == 1
            for nd, q, ow in zip(np.asarray(ob["wip.node"][t])[live], np.asarray(ob["wip.qty"][t])[live], np.asarray(ob["wip.out_week"][t])[live]):
                if int(nd) == n and int(ow) == t + tau:
                    out[t - 1] += q
        return out

    def fab_attr(self, node: str) -> dict:
        return self.nodes_attr[self.node_ix[node]]["fab"]
