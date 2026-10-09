"""The simulator's own weekly step inside an agent: what a request will do, before it is sent.

The package's simulator (a copy ships in the hybrid agent's ``sbfv/``) takes a state, the weeks' network and an
action, and returns everything the week does: what is dispatched, burned, started, packaged, sold and shed, and the
week's cost. ``Model`` gives it the three things an agent has in place of the true ones:

- the state, rebuilt from the observation (``window``). Stocks, shipments on the way, work in process and backlog are
  shown exactly. On Small and Full the cargo queued at the straits is shown per (strait, commodity, lane, next edge,
  arrival week), which is all the default release reads; a lot's entry edge and dispatch week, which only order the
  lots an override takes, follow from its lane.
- the network of the coming weeks: as observed now, unchanged ("it stays as it is"), demand from the forecast. One
  field can do better: a grid in a spell of low output comes back in one step after some weeks, and ``grids``
  (``grid_recovery.json``, written by ``grid_spells.py``) says what to expect by the spell's age: ``blend`` plans the
  expected output, ``step`` plans the spell over from the week it more likely than not is.
- the action in the simulator's form (``wire``): the environment's validity rules applied to the flat arrays.

``flat`` is the way back: the observation a policy would read after some weeks of the model, so that the agent's own
rules can play those weeks (a rollout). The fields that follow from the state are rebuilt, and the grids' output is
the model's for that week; the rest of the network, the forecast and the announcements stay the real observation's.

The observation is read through the hybrid's ``lp_part.Planner`` (its instance, its memory of the network, its state
dict): ``planner._remember(observation)`` must have run for the week (``Planner.flows`` does it). Import this module
after ``lp_part``, which puts ``sbfv`` on the path, or keep it beside ``sbfv/``. Small and Full only: Tiny shows the
queue as a list of lots, which ``flat`` does not write.
"""

from dataclasses import dataclass

import numpy as np
from sbfv.dynamics import sim
from sbfv.policies import lp_common as L


OVERRIDE, HOLD = 1, 2  # release_mode codes
PIPELINE = ("edge", "k", "lane", "qty", "arrival_week")
WIP = ("node", "k", "qty", "out_week")


@dataclass
class Window:
    """Some weeks of the model, from the real week ``week`` on: the simulator's inputs and its state so far."""

    inst: object  # the instance with the observed state as its start and the window as its horizon
    marks: object  # the network of the window's weeks
    state: object  # the simulator's state; ``state.week`` weeks of the window have been played
    week: int  # the real week of the window's first week
    backlog: np.ndarray  # backlog at the start (the model adds it to the first week's demand)


class Model:
    def __init__(self, config, planner, grids: dict | None = None, grid_rule: str = "blend"):
        self.planner = planner
        self.grids, self.grid_rule = grids, grid_rule  # the spells' table (None: a spell stays) and how to read it
        inst = self.inst = planner.inst
        layout, static = config["layout"], config["static"]
        slots, ov = static["action_slots"], static["override_slots"]
        self.slot_edge, self.slot_k = np.array(slots["edge"], dtype=int), np.array(slots["k"], dtype=int)
        self.ov_edge, self.ov_k = np.array(ov["out_edge"], dtype=int), np.array(ov["k"], dtype=int)
        self.pairs = [(int(c), int(k)) for c, k in layout["release_pairs"]]
        index = {pair: i for i, pair in enumerate(self.pairs)}
        self.ov_pair = np.array([index[(int(c), int(k))] for c, k in zip(ov["chokepoint"], ov["k"])], dtype=int)
        self.stock_slot = np.array([inst.slot_index[(int(n), int(k))] for n, k in layout["stock_slots"]], dtype=int)
        keys = layout.get("lot_keys")
        self.lot_row = {tuple(int(x) for x in key): i for i, key in enumerate(keys)} if keys else None

    # ----- from the observation to the simulator ------------------------------------------------------------------
    def window(self, observation, weeks: int, arrays=None, fab_hits=(), pending: bool = False) -> Window:
        """The model of the next ``weeks`` weeks (fewer at the end of the episode), nothing played yet.

        ``arrays`` replaces the forecast of the network (the fields of ``lp_common.WINDOW_FIELDS``, one row per
        week), ``fab_hits`` are the scrap events in it; None and (): the network as observed now. ``pending`` also
        switches the announced prohibitions on from their effective week.
        """
        inst = self.inst
        obs = self.planner._state(observation)
        if self.lot_row is not None:  # the dense table drops these two fields; the lane gives them back
            lots = obs["queue_lots"]
            entry = [inst.lane_through[(lane, c)][0] for c, lane in zip(lots["chokepoint"], lots["lane"])]
            lots["entry_edge"] = entry
            lots["dispatch_week"] = [a - inst.edges[e].tau for a, e in zip(lots["arrival_week"], entry)]
        week = obs["week"]
        weeks = L.window_length(int(weeks), week, inst.T)
        inst_r, backlog = L.rolled_window(inst, obs, weeks)
        if arrays is None:
            arrays = self._forecast(obs, weeks, pending)
        marks = L.window_marks(inst_r, arrays, backlog, fab_hits=fab_hits)
        return Window(inst_r, marks, sim.initial_state(inst_r), week, backlog)

    def forecast(self, observation, weeks: int, pending: bool = False) -> dict:
        """The network of the next ``weeks`` weeks as the model assumes it: ``window``'s default ``arrays``."""
        obs = self.planner._state(observation)
        return self._forecast(obs, L.window_length(int(weeks), obs["week"], self.inst.T), pending)

    def _forecast(self, obs: dict, weeks: int, pending: bool) -> dict:
        arrays = L.persistence_arrays(self.inst, obs, self.planner.memory, weeks)
        if not pending:
            seen = self.planner.memory.values["prohibited"]
            arrays = dict(arrays) | {"prohibited": np.broadcast_to(seen, (weeks, *seen.shape)).copy()}
        if self.grids is not None:
            arrays = dict(arrays) | self._grids_back(weeks)
        return arrays

    def _grids_back(self, weeks: int) -> dict:
        """The grids' output of the next ``weeks`` weeks: a spell of low output ends as spells of its age do."""
        planner = self.planner
        now, calm = planner.memory.values["grid_G"], planner.calm["grid_G"]
        table = np.asarray(self.grids["back" if self.grid_rule == "blend" else "gone"])
        out = np.broadcast_to(now, (weeks, len(now))).copy()
        for g in np.flatnonzero(planner.grid_age > 0):
            back = table[min(int(planner.grid_age[g]), len(table)) - 1, :weeks]
            if len(back) < weeks:  # beyond the table: as its last week
                back = np.r_[back, np.full(weeks - len(back), back[-1])]
            if self.grid_rule != "blend":
                back = (back >= 0.5).astype(float)
            out[:, g] = now[g] + back * (calm[g] - now[g])
        shown = np.vstack([now, out[:-1]])  # what a week's observation shows: the output as the week before left it
        return {"G_bar": out, "G_bar_now": shown}

    @staticmethod
    def restart(w: Window) -> Window:
        """The same window with nothing played: another candidate starts here without rebuilding the network."""
        return Window(w.inst, w.marks, sim.initial_state(w.inst), w.week, w.backlog)

    def wire(self, action, prohibited: np.ndarray) -> tuple[dict, dict, frozenset]:
        """The action as the simulator takes it: requests, strait overrides and holds that pass the validity rules.

        An entry that is zero, negative or not finite, or whose own edge is prohibited for its commodity
        (``prohibited``, edges by commodities), is dropped, as the environment drops it. An override entry of
        quantity 0 stays: it turns the default release of its strait and commodity off.
        """
        flows = np.asarray(action["flows"], dtype=float)
        ok = np.isfinite(flows) & (flows > 0) & ~prohibited[self.slot_edge, self.slot_k]
        sent = {int(s): float(flows[s]) for s in np.flatnonzero(ok)}
        modes = action.get("release_mode")
        if modes is None:
            return sent, {}, frozenset()
        modes = np.asarray(modes)
        qty = action.get("override_qty")
        qty = np.zeros(len(self.ov_pair)) if qty is None else np.asarray(qty, dtype=float)
        live = (modes[self.ov_pair] == OVERRIDE) & np.isfinite(qty) & (qty >= 0)
        live &= ~prohibited[self.ov_edge, self.ov_k]
        overrides = {int(o): float(qty[o]) for o in np.flatnonzero(live)}
        holds = frozenset(self.pairs[p] for p in np.flatnonzero(modes == HOLD))
        return sent, overrides, holds

    def step(self, w: Window, action):
        """Play the window's next week with ``action`` (the agent's flat arrays); the simulator's record of it."""
        flows, overrides, holds = self.wire(action, np.asarray(w.marks.prohibited[w.state.week]))
        return sim.step(w.inst, w.marks, w.state, flows, overrides, holds)

    # ----- from the simulator back to an observation ----------------------------------------------------------------
    def flat(self, observation, w: Window) -> dict:
        """The observation after the weeks ``w`` has played: ``observation`` with the state's fields replaced.

        Rebuilt: the week, stocks, backlog, shipments on the way, queues at the straits, work in process and last
        week's record. Kept from ``observation``: the network, the masks, the demand forecast, the announcements.
        """
        inst, st, shift = self.inst, w.state, w.week - 1
        out = dict(observation)
        out["week"] = np.array([w.week + st.week], dtype=observation["week"].dtype)
        if 0 < st.week < w.inst.T:  # the grids' output as the model has it for the coming week
            shown = np.asarray(w.marks.G_bar_now[st.week])[self.planner.grids]
            out["graph_now.grid.G_bar"] = shown.astype(observation["graph_now.grid.G_bar"].dtype)
        out["stock.qty"] = np.asarray(st.stock, dtype=float)[self.stock_slot]
        out["backlog.qty"] = np.array(w.backlog if st.week == 0 else st.backlog, dtype=float)

        moving: dict[tuple, float] = {}  # one entry per (edge, commodity, lane, arrival week), as the flat view groups
        for s in st.pipeline:
            key = (s.edge, s.k, s.lane, s.arrival_week + shift)
            moving[key] = moving.get(key, 0.0) + s.qty
        rows = [(e, k, lane, q, a) for (e, k, lane, a), q in moving.items()]
        self._list(out, observation, "pipeline", PIPELINE, rows)

        queue = np.zeros_like(observation["queue_lots.qty"])
        for lot in st.lots:
            queue[self.lot_row[(lot.chokepoint, lot.k, lot.lane, lot.next_edge)], lot.arrival_week + shift - 1] += (
                lot.qty
            )
        out["queue_lots.qty"] = queue
        out["queue_lots.qty.observed"] = (queue != 0).astype(observation["queue_lots.qty.observed"].dtype)

        rows = []  # fabs by start week, then plants by week out and commodity: the environment's order
        for fi, f in enumerate(inst.fabs):
            fab = inst.nodes[f].fab
            rows += [
                (f, fab.product, q, start + fab.tau + shift) for start, q in sorted(st.fab_wip.get(fi, {}).items())
            ]
        for oi, o in enumerate(inst.osats):
            for week_out, book in sorted(st.osat_wip.get(oi, {}).items()):
                rows += [(o, k, q, week_out + shift) for k, q in sorted(book.items())]
        self._list(out, observation, "wip", WIP, rows)

        rec = st.last
        if rec is not None:
            asked, done = np.zeros(len(self.slot_edge)), np.zeros(len(self.slot_edge))
            seen = np.zeros(len(self.slot_edge), dtype=observation["last_week.clip.requested.observed"].dtype)
            for s, q in rec.requested.items():
                asked[s], seen[s] = q, 1
            for s, q in rec.executed.items():
                done[s] = q
            last = {
                "clip.requested": asked,
                "clip.executed": done,
                "cost_components": np.array(list(rec.costs.as_dict().values())),
                "sinks.demand": np.array(rec.demand, dtype=float),
                "sinks.served": np.array(rec.served, dtype=float),
                "sinks.lost": np.array(rec.lost, dtype=float),
                "shed.qty": np.array(rec.shed, dtype=float),
            }
            for name, values in last.items():
                out[f"last_week.{name}"] = values
                flags = observation[f"last_week.{name}.observed"]
                out[f"last_week.{name}.observed"] = seen if name.startswith("clip") else np.ones_like(flags)
        return out

    @staticmethod
    def _list(out: dict, observation, block: str, names: tuple, rows: list) -> None:
        """Write ``rows`` as the padded columns ``block.<name>`` and their ``.observed`` flags (a None: 0, unseen)."""
        size = len(observation[f"{block}.qty"])
        rows = rows[:size]
        for j, name in enumerate(names):
            like, flags = observation[f"{block}.{name}"], observation[f"{block}.{name}.observed"]
            column, seen = np.zeros_like(like), np.zeros_like(flags)
            for i, row in enumerate(rows):
                if row[j] is not None:
                    column[i], seen[i] = row[j], 1
            out[f"{block}.{name}"], out[f"{block}.{name}.observed"] = column, seen
