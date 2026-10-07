"""Rules for fuel and power, a window program for the chips that already exist and for the wafers.

The rules of ``lab/anastasiia/heur_lab4/agents/rules_v3`` unchanged (``chip_part``, ``fuel_part``, ``strait_part``),
except that the entries of the action
named in ``PARAMS["take"]`` come from the first week of a linear program of the whole network (``lp_part``): ``raw``
(raw chips, fab -> plant) and ``pack`` (packaged chips, plant -> market); ``wafer`` (source -> fab) can be taken too.

Why this split. After the fab the chain is linear: transport with delays and capacities, and what matures in the next
weeks is already in process, so the program plans it exactly, both chips and both stages together, where the rules
decide stage by stage and the dearer chip first. Power is not linear (a grid serves its base load first, lng is
rationed by yesterday's stock), the program gets it wrong, and the rules decide fuel, wafers and strait releases.

This copy (``lab/anastasiia/mpc_lab/agents/hyb_waferlp``) lets the program decide the wafer entries too
(``PARAMS["wafer_lp"]``, which adds ``wafer`` to ``take``): which fab gets the wafers is a lever the rules miss (the
full-knowledge plan starts a quarter fewer lots at fab_kr and sells more chip_le, hub/FINDINGS.md). To trust its lots,
``lp_part`` adds the base-load-first rule of ``agents/anastasiia_mpc_baseload`` as yes/no decisions (``power_*``) and
``mpc_fuelval``'s end-of-window credit for fuel (``fuel_value``), all inside ``solve_share`` of the week's CPU budget.
``chip_part`` still runs its wafer rule every week (its records of overflow and shipments stay whole) and the
program's values overwrite it; a week without a plan keeps the rule's wafers, as it keeps its chips. The fuel rules are
unchanged and now read the program's wafer entries. ``{"wafer_lp": false}`` in ``params.json`` is the ablation;
``{"lp": {"power_weeks": 0, "fuel_value": 0}}`` turns off the program's two fixes.

The rules fill the chip entries first; a week whose program does not solve in time keeps them, and an error in the
planner costs that week's plan, not the week. The fuel rules run last: they read the wafer entries to know what the
fabs will ask of their grids. Every size and table comes from ``config``. A ``params.json`` beside this file replaces
``PARAMS`` (and, as in the rule agent, any number of the parts that it names).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
KIND = {"wafer": 1, "raw": 2, "pack": 3}  # chip_part's slot kinds

PARAMS = {
    "take": ["raw", "pack"],  # the entries the program decides
    "wafer_lp": True,  # the program decides the wafer entries (source -> fab) too
    "lp": {},  # replaces entries of lp_part.LP
}
if (HERE / "params.json").is_file():
    PARAMS |= {k: v for k, v in json.loads((HERE / "params.json").read_text()).items() if k in PARAMS}
if PARAMS["wafer_lp"] and "wafer" not in PARAMS["take"]:
    PARAMS["take"] = [*PARAMS["take"], "wafer"]


def _part(name: str):
    """A module beside this file, under a name of its own (two agents may each ship a ``fuel_part``)."""
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_chip, _fuel, _strait, _lp = _part("chip_part"), _part("fuel_part"), _part("strait_part"), _part("lp_part")


class Agent:
    def __init__(self, config=None):
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

    def _planned(self, observation) -> np.ndarray | None:
        """The program's flows for this week, or None: it has no plan, or it failed."""
        if self.planner is None:
            return None
        try:
            return self.planner.flows(observation)
        except Exception:
            return None

    def act(self, observation):
        flows = np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)  # wafers, raw chips, packaged chips by the rules
        planned = self._planned(observation)
        if planned is not None:  # the program's entries replace the rules' (wafers too with ``wafer_lp``)
            flows[self.take] = planned[self.take]
            self.planned_weeks += 1
        self.fuel.fill(observation, flows)  # the fuel entries (reads the wafer entries)
        return {"flows": flows, **self.strait.fill(observation)}  # tanker cargo waiting at straits
