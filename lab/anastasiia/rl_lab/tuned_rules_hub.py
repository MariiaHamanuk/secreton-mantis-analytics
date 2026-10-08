"""``agents/anastasiia_hybrid_hub`` with the numbers of its rules supplied from outside, not from a ``params.json``.

The hub agent is the rules of ``rules_v3`` (``chip_part``, ``fuel_part``, ``strait_part``) plus a linear program
(``lp_part``) that decides the raw- and packaged-chip entries. The parts take their numbers as in
``tuned_rules.py``: ``FuelRules``, ``StraitRules`` and ``Planner`` take a dict in the constructor, ``chip_part`` reads
its module-level ``PARAMS`` while it runs, so a candidate is put in place without writing a file.

``Composed.act`` is ``agents/anastasiia_hybrid_hub/agent.py``'s ``act`` line for line; with the shipped numbers it
plays the folder to the cent (checked by ``hub_cem.py --check``).

``lp=False`` drops the program (the rule entries for raw and packaged chips stay): the same rules at a twentieth of
the CPU. The first search (``hub_cem.py search --nolp``) ran this way, because with the program a candidate on
Small x32 + Full x12 costs ~2000 CPU s; its result is then judged on the real hub agent.

``KNOBS`` are the numbers worth searching. Not in it: the on/off flags; the integers; the chip knobs that only
shape the raw and packaged entries (the program overwrites them: ``pack_*``, ``raw_*``, ``sink_pull``); and the
program's horizon and CPU share (they decide its time, and a search should not buy cost with CPU).
"""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HUB = ROOT / "agents" / "anastasiia_hybrid_hub"
KIND = {"wafer": 1, "raw": 2, "pack": 3}
TAKE = ["raw", "pack"]  # agent.PARAMS["take"] as shipped

# (name, which part, default, low, high)
KNOBS = [
    # fuel orders
    ("top_weeks", "fuel", 8.0, 4.0, 16.0),
    ("share_first", "fuel", 0.5, 0.2, 0.8),
    ("queue_frac", "fuel", 0.15, 0.05, 0.40),
    ("queue_horizon", "fuel", 4.0, 2.0, 8.0),
    ("fleet_frac", "fuel", 1.0, 0.92, 1.08),  # 1.0 is a sharp optimum by hand (0.5 lost 0.04)
    ("grid_buffer", "fuel", 0.6, 0.3, 0.95),
    ("nuc_buffer", "fuel", 6.0, 2.0, 12.0),
    ("nuc_safety", "fuel", 1.5, 1.0, 2.5),
    ("ss_weeks", "fuel", 0.3, 0.0, 1.0),
    ("grid_cover_weeks", "fuel", 0.5, 0.2, 1.0),
    # fuel: time concentration
    ("pulse_min", "fuel", 0.1, 0.02, 0.3),
    ("on_ratio", "fuel", 0.9, 0.70, 1.0),
    ("prime_weeks", "fuel", 2.0, 1.0, 4.0),
    ("run_band", "fuel", 0.9, 0.70, 1.0),
    ("end_weeks", "fuel", 16.0, 10.0, 22.0),
    ("overflow_frac", "fuel", 0.92, 0.70, 1.0),
    # fuel: gate and bank
    ("gate_ratio", "fuel", 1.0, 0.6, 1.3),
    ("gate_on", "fuel", 0.98, 0.90, 1.0),
    ("gate_ask", "fuel", 0.2, 0.05, 0.5),
    ("gate_cover", "fuel", 0.9, 0.6, 1.1),
    ("bank_ratio", "fuel", 0.9, 0.7, 1.0),
    ("gate_keep", "fuel", 3.0, 1.5, 6.0),
    # fuel: throttle
    ("thr_useful", "fuel", 0.5, 0.2, 0.8),
    ("thr_full", "fuel", 0.6, 0.3, 0.9),
    ("thr_margin", "fuel", 0.0, 0.0, 0.3),
    ("thr_floor", "fuel", 0.0, 0.0, 0.5),
    ("thr_min", "fuel", 0.0, 0.0, 0.5),
    # fuel: straits as forward stores, pipeline cap
    ("hub_queue", "fuel", 3.0, 1.5, 6.0),
    ("hub_step", "fuel", 0.5, 0.25, 1.0),
    ("hub_cut", "fuel", 0.5, 0.25, 0.8),
    ("pipe_cap", "fuel", 0.9, 0.6, 1.0),
    # chips: wafers (the program does not decide them)
    ("w_weeks", "chip", 3.0, 1.5, 5.0),
    ("w_fill", "chip", 0.9, 0.6, 1.0),
    ("w_u_min", "chip", 0.02, 0.005, 0.1),
    ("pull_buffer", "chip", 12.0, 6.0, 20.0),
    ("scarce_frac", "chip", 0.5, 0.3, 0.8),
    ("cap_gain", "chip", 16.0, 6.0, 32.0),
    ("cap_floor", "chip", 0.8, 0.5, 1.0),
    # straits
    ("need_weeks", "strait", 8.0, 4.0, 14.0),
    ("room_frac", "strait", 0.9, 0.7, 1.0),
    ("min_extra", "strait", 0.02, 0.005, 0.1),
    ("strait_fleet", "strait", 0.95, 0.8, 1.0),
    # the program
    ("demand_scale", "lp", 1.1, 0.9, 1.4),
]
NAMES = [k[0] for k in KNOBS]
DEFAULTS = [k[2] for k in KNOBS]
LOW = [k[3] for k in KNOBS]
HIGH = [k[4] for k in KNOBS]
PART = {k[0]: k[1] for k in KNOBS}


def _module(name, folder=HUB):
    spec = importlib.util.spec_from_file_location(f"tunedhub_{folder.name}_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Parts:
    """The four modules, imported once per process."""

    def __init__(self, folder=HUB):
        if (folder / "params.json").is_file():
            raise RuntimeError(f"{folder} has a params.json: the defaults here would not be the folder's numbers")
        self.chip = _module("chip_part", folder)
        self.fuel = _module("fuel_part", folder)
        self.strait = _module("strait_part", folder)
        self.lp = _module("lp_part", folder)
        self.chip_defaults = dict(self.chip.PARAMS)
        for name, part, default, _, _ in KNOBS:  # the defaults above are the folder's numbers
            table = {"fuel": self.fuel.FUEL, "chip": self.chip_defaults, "strait": self.strait.STRAIT, "lp": self.lp.LP}
            if table[part][name] != default:
                raise RuntimeError(f"{name}: {default} here, {table[part][name]} in {folder}")

    def agent(self, config, values=None, lp=True):
        values = values or {}

        def of(part):
            return {n: v for n, v in values.items() if PART.get(n) == part}

        self.chip.PARAMS.clear()
        self.chip.PARAMS.update(self.chip_defaults)
        self.chip.PARAMS.update(of("chip"))
        planner = None
        if lp:
            try:
                # solve_share is not searched, but a check may lift the program's CPU limit (hub_cem.py check)
                extra = {"solve_share": values["solve_share"]} if "solve_share" in values else {}
                planner = self.lp.Planner(config, of("lp") | extra)
            except Exception:  # as agent.py: the rules alone are a complete agent
                planner = None
        return Composed(
            self.chip.Agent(config),
            self.fuel.FuelRules(config, dict(self.fuel.FUEL, **of("fuel"))),
            self.strait.StraitRules(config, of("strait")),
            planner,
            config,
        )


class Composed:
    """``agents/anastasiia_hybrid_hub/agent.py``'s ``Agent.act``."""

    def __init__(self, chips, fuel, strait, planner, config):
        import numpy as np

        self._np = np
        self.chips, self.fuel, self.strait, self.planner = chips, fuel, strait, planner
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]
        self.take = np.flatnonzero(np.isin(self.chips.kind, [KIND[name] for name in TAKE]))
        self.planned_weeks = 0

    def act(self, observation):
        flows = self._np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)
        planned = None
        if self.planner is not None:
            try:
                planned = self.planner.flows(observation)
            except Exception:
                planned = None
        if planned is not None:
            flows[self.take] = planned[self.take]
            self.planned_weeks += 1
        self.fuel.fill(observation, flows)
        return {"flows": flows, **self.strait.fill(observation)}


def clip(values):
    """A candidate vector held inside the ranges, as a dict the parts understand."""
    return {name: float(min(max(float(v), lo), hi)) for (name, _, _, lo, hi), v in zip(KNOBS, values)}
