"""Rules only, no linear program: fuel by ``fuel_part``; the chip chain by ``chip_part``, then shared out within what
the routes carry by ``strait_part`` (lab variant v3).
"""

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _part(name: str):
    """A module beside this file, under a name of its own (two agents may each ship a ``fuel_part``)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chip, _fuel, _strait = _part("chip_part"), _part("fuel_part"), _part("strait_part")


class Agent:
    def __init__(self, config=None):
        self.chips = _chip.Agent(config)
        self.strait = _strait.Agent(config)
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL)
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]

    def act(self, observation):
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # wafers, raw chips, packaged chips
        flows = np.nan_to_num(self.strait.adjust(flows, observation), nan=0.0, posinf=0.0, neginf=0.0)
        flows = np.maximum(flows, 0.0) * (observation["action_mask"] == 1)
        self.fuel.fill(observation, flows)  # the fuel entries
        return {"flows": flows}
