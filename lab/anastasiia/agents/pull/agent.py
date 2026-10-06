"""Send the maximum, but let the valuable commodity ask first and ask a market only for what it can sell.

The simulator shares an edge's capacity among its commodities, and a stock among the routes that draw on it, in
proportion to their requests. Send-the-maximum therefore gives a cheap commodity half of a route the valuable one
needs, and ships chips to markets that do not want them. This agent keeps send-the-maximum everywhere except:

- ``value_first``: on a route two sold commodities share, the one with the higher shortage penalty asks first and the
  other gets the capacity that is left;
- ``sink_pull``: a route into a market asks for that market's forecast demand (times this margin), split over the
  routes into it by their capacity this week; a market with no demand gets nothing.

A ``params.json`` beside this file replaces ``PARAMS``. Rules tried and dropped (team/PLAN.md): asking a fab, a
packaging plant or a grid only for what it can use.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = {
    "sink_pull": 1.3,  # margin on a market's forecast demand; None: send the maximum into markets
    "value_first": True,  # the commodity with the higher penalty asks first on a shared route
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        self.edge = np.array(slots["edge"])
        self.k = np.array(slots["k"])
        route = [lanes["edges"][lane] if lane is not None else [e] for e, lane in zip(slots["edge"], slots["lane"])]
        self.dest = np.array([edges["head"][r[-1]] for r in route])  # the node a slot's cargo ends at
        self.lead = np.array([sum(edges["tau0"][e] for e in r) for r in route])
        # the shortage penalty per commodity: the largest over the markets that want it; 0 for an input
        pi = np.zeros(len(static["commodities"]["id"]))
        for k, p in zip(static["sinks"]["k"], static["sinks"]["pi"]):
            pi[k] = max(pi[k], p)
        self.order = sorted(set(self.k.tolist()), key=lambda k: -pi[k])
        self.sold = pi > 0  # a commodity some market pays for; inputs share a route as send-the-maximum does
        self.demand_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["demands"])}
        self.horizon = config["spaces"]["observation"]["demand_forecast.qty"]["shape"][1]

    def act(self, observation):
        mask = observation["action_mask"] == 1
        forecast = observation["demand_forecast.qty"]
        left = np.asarray(observation["graph_now.u"], dtype=float).copy()  # capacity not yet asked for, per edge
        left[~np.isfinite(left)] = 0.0
        flows = np.zeros(len(self.edge))
        for k in self.order:
            mine = np.flatnonzero((self.k == k) & mask)
            room = left[self.edge[mine]]
            ask = room.copy()
            if PARAMS["sink_pull"] is not None:
                for node in set(self.dest[mine].tolist()):
                    row = self.demand_row.get((node, k))
                    if row is None:
                        continue
                    into = self.dest[mine] == node
                    wanted = PARAMS["sink_pull"] * forecast[row, np.minimum(self.lead[mine][into], self.horizon - 1)]
                    total = room[into].sum()
                    ask[into] = np.minimum(room[into], wanted * room[into] / total) if total > 0 else 0.0
            flows[mine] = ask
            if PARAMS["value_first"] and self.sold[k]:
                np.subtract.at(left, self.edge[mine], ask)
                np.maximum(left, 0.0, out=left)
        return {"flows": flows}
