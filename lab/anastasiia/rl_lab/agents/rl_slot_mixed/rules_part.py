"""Rules only, no linear program: the chip chain by ``chip_part``, fuel by ``fuel_part``, straits by ``strait_part``.

- Chip chain (wafers, raw chips, packaged chips): feasible flows by small max-flows. Raw chips go to a packaging plant
  only up to what its open routes to markets can ship on; packaged chips go to markets by what each can still sell;
  a fab holds three weeks of wafers, less where its chips cannot reach a market, never more than its store takes, and
  the wafers of a source whose supply is cut are kept for the fabs whose outlets will need them.
- Fuel (lng, crude, nuclear fuel): orders up to each grid's need, lane by lane, within what the lanes carry this week.
  A fab runs only on what its grid delivers above the base load, so a grid short of a fuel gets it in whole weeks: its
  lng waits at the terminal while it is off, and a fuel whose weekly burn exceeds that sliver (crude at some grids)
  moves a full week's burn or nothing. The grid is offered the load and not more (the throttle): the lng level follows
  what the fabs will ask. A terminal that fills up passes its surplus to the grid's own storage, and nuclear fuel is
  brought as early as the lanes carry it, because a lane open today may be closed when the stock runs low.
- Straits: tanker cargo waiting behind an out-edge whose capacity was cut is released onto another out-edge with room.

The chip part runs first: the fuel part reads the wafer entries of ``flows`` to know what the fabs will ask of their
grids next week. The parts fill disjoint entries of the action. All read every size and table from ``config``; the
same code runs on Tiny, Small and Full.
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
        self.fuel = _fuel.FuelRules(config, _fuel.FUEL)
        self.strait = _strait.StraitRules(config)
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]

    def act(self, observation):
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # wafers, raw chips, packaged chips
        self.fuel.fill(observation, flows)  # the fuel entries (reads the wafer entries just filled)
        return {"flows": flows, **self.strait.fill(observation)}  # tanker cargo waiting at straits
