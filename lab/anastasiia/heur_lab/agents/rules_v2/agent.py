"""Rules only, no linear program: fuel by ``fuel_part``, the chip chain by ``chip_part`` (lab variant v2).

The two fill disjoint entries of ``flows`` (tanker and bulk cargo against containers). Both read every size and table
from ``config``.
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


_chip, _fuel = _part("chip_part"), _part("fuel_part")


class Agent:
    def __init__(self, config=None):
        self.chips = _chip.Agent(config)
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL)
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]

    def act(self, observation):
        flows = np.zeros(self.n_slots)
        self.fuel.fill(observation, flows)  # the fuel entries
        self.chips.fill_chip_flows(observation, flows)  # wafers, raw chips, packaged chips
        return {"flows": flows}
