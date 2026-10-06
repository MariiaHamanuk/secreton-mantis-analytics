"""A diagnostic, not a submission: the rule agent with some chip entries taken from the weekly linear program.

``agents/anastasiia_rules_v2`` decides everything; then the entries named in ``PARAMS["take"]`` (``wafer``: source ->
fab, ``raw``: fab -> plant, ``pack``: plant -> market) are replaced by the first week of the plan of
``agents/anastasiia_mpc_baseload``, and the fuel rules run on the result (they read the wafer entries). The question it
answers: would an exact program of the chip chain ship more than the rules do, with the same fuel and power?

Both agents are loaded from the repository by path: ``PARAMS["root"]`` names it (the scorer plays a copy of the folder,
so the path cannot be found from this file), None looks for it above this file. ``PARAMS["lp"]`` replaces entries of
the program's own ``PARAMS`` (``power_weeks`` 0: a linear program, no yes/no decisions, about ten times faster).
A ``params.json`` beside this file replaces ``PARAMS``.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
KIND = {"wafer": 1, "raw": 2, "pack": 3}  # chip_part's slot kinds

PARAMS = {"take": ["pack"], "lp": {"power_weeks": 0}, "root": None}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())
ROOT = (
    Path(PARAMS["root"])
    if PARAMS["root"]
    else next(p for p in HERE.parents if (p / "agents" / "anastasiia_rules_v2" / "agent.py").is_file())
)


def _agent(folder: str):
    """The agent module of ``agents/<folder>``, under a name of this folder's own (variants stay apart)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{folder}", ROOT / "agents" / folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rules, _lp = _agent("anastasiia_rules_v2"), _agent("anastasiia_mpc_baseload")
_lp.PARAMS.update(PARAMS["lp"])


class Agent:
    def __init__(self, config=None):
        self.rules = _rules.Agent(config)
        self.lp = _lp.Agent(config)
        kinds = [KIND[name] for name in PARAMS["take"]]
        self.take = np.flatnonzero(np.isin(self.rules.chips.kind, kinds))

    def act(self, observation):
        rules = self.rules
        flows = np.zeros(rules.n_slots)
        rules.chips.fill_chip_flows(observation, flows)
        planned = self.lp.act(observation)["flows"]
        flows[self.take] = planned[self.take]
        rules.fuel.fill(observation, flows)
        return {"flows": flows, **rules.strait.fill(observation)}
