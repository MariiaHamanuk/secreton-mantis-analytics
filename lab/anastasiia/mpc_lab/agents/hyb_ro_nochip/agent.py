"""anastasiia_hybrid_chiplp plus a weekly rollout chooser on the fuel entries (rollout_part).

The base agent unchanged (rules for wafers, fuel and straits, the window LP's first week for the chip entries); then
``rollout_part.Rollout`` builds 2-4 candidates of the week's action that differ only in the fuel entries, plays each
forward in the vendored simulator on the persistence forecast and plays the cheapest. Any failure, or a week whose CPU
already passed ``time_share`` of the budget, plays the as-is action. A ``params.json`` beside this file replaces
entries of ``PARAMS`` (and of ``ROLLOUT``, under the key "rollout_params", or directly by its names).
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
KIND = {"wafer": 1, "raw": 2, "pack": 3}  # chip_part's slot kinds

PARAMS = {
    "take": ["raw", "pack"],  # the entries the program decides
    "lp": {},  # replaces entries of lp_part.LP
}
_RP: dict = {}
if (HERE / "params.json").is_file():
    _j = json.loads((HERE / "params.json").read_text())
    PARAMS |= {k: v for k, v in _j.items() if k in PARAMS}
    _RP = {k: v for k, v in _j.items() if k not in PARAMS}


def _part(name: str):
    """A module beside this file, under a name of its own (two agents may each ship a ``fuel_part``)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chip, _fuel, _strait, _lp = _part("chip_part"), _part("fuel_part"), _part("strait_part"), _part("lp_part")
_roll = _part("rollout_part")


class Agent:
    def __init__(self, config=None):
        t0 = time.process_time()
        self.chips = _chip.Agent(config)
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL)
        self.strait = _strait.StraitRules(config)
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]
        self.take = np.flatnonzero(np.isin(self.chips.kind, [KIND[name] for name in PARAMS["take"]]))
        try:
            self.planner = _lp.Planner(config, PARAMS["lp"])
        except Exception:  # the rules alone are a complete agent
            self.planner = None
        self.planned_weeks = 0  # weeks whose chip entries came from the program (diagnostics)
        self.roll = None
        rp = _roll.ROLLOUT | {k: v for k, v in _RP.items() if k in _roll.ROLLOUT}
        if rp["rollout"] and self.planner is not None:
            try:
                self.roll = _roll.Rollout(config, self.fuel, self.planner, rp)
            except Exception:
                self.roll = None
        self.init_cpu = time.process_time() - t0  # counts toward week 1

    def _planned(self, observation) -> np.ndarray | None:
        """The program's flows for this week, or None: it has no plan, or it failed."""
        if self.planner is None:
            return None
        try:
            return self.planner.flows(observation)
        except Exception:
            return None

    def act(self, observation):
        t0 = time.process_time() - (self.init_cpu if int(observation["week"][0]) == 1 else 0.0)
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # wafers, raw chips, packaged chips by the rules
        planned = self._planned(observation)
        if planned is not None:
            flows[self.take] = planned[self.take]
            self.planned_weeks += 1
        self.fuel.fill(observation, flows)  # the fuel entries (reads the wafer entries)
        extra = self.strait.fill(observation)  # tanker cargo waiting at straits
        if self.roll is not None:
            chosen = flows
            try:
                if time.process_time() - t0 < self.roll.p["time_share"] * self.roll.budget:
                    _name, f = self.roll.choose(observation, flows, extra, t0)
                    if f.shape == flows.shape and np.isfinite(f).all() and (f >= 0).all():
                        chosen = f
            except Exception:
                chosen = flows
            flows = chosen
            try:
                self.roll.remember(flows)
            except Exception:
                pass
        return {"flows": flows, **extra}
