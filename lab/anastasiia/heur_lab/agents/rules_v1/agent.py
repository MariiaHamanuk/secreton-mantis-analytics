"""Rules only, no linear program: two rule sets on disjoint slots.

- Chip chain (wafers, raw chips, packaged chips): ``strait_part`` takes send-the-maximum requests and shares them out
  within what every route can carry this week, sends a packaging plant only what it can ship on, a fab only what its
  storage takes, and nothing that cannot be sold before the end.
- Fuel (lng, crude, nuclear fuel): ``fuel_part`` orders up to each grid's need lane by lane within what the lanes carry,
  serves the grids in the order of what completing them is worth, and runs a grid that cannot be kept full in on and
  off weeks (fuel waits at the terminal while it is off).

The two never share an edge or a strait pool (tanker and bulk cargo against containers), so each fills its own entries
of ``flows``. Both read every size and table from ``config``.
"""

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _part(name: str):
    """A module beside this file, under a name of its own (two agents may each ship a ``fuel_part``)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_strait, _fuel = _part("strait_part"), _part("fuel_part")


class Agent:
    def __init__(self, config=None):
        self.chips = _strait.Agent(config)
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL)

    def act(self, observation):
        flows = self.chips.act(observation)["flows"]
        self.fuel.fill(observation, flows)  # overwrites the fuel entries, touches no other
        return {"flows": flows}
