"""The rules of ``anastasiia_rules_v2`` with their numbers supplied from outside, not from a ``params.json``.

The three parts take their numbers differently: ``FuelRules`` and ``StraitRules`` take a dict in the constructor,
``chip_part`` reads its module-level ``PARAMS`` when it runs. So a candidate is applied by building the first two
with the dict and updating the third's module dict in place — no file is written and nothing is re-imported, which
is what makes a search of thousands of candidates affordable.

``KNOBS`` lists the numbers worth searching, with the range each is allowed. The defaults are the ones the model
ships with; the ranges come from what ``hub/tried/heuristics.md`` reports as having been tried by hand, widened
where the hand search only moved one number at a time (a joint search may want more room than a single-knob one).
"""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RULES = ROOT / "agents" / "anastasiia_rules_v2"

# (name, which part, default, low, high). "fuel" goes to FuelRules, "chip" to chip_part.PARAMS, "strait" to
# StraitRules. Only numbers: the on/off flags stay as the model ships them.
KNOBS = [
    # fuel: the 64 % of the cost that is shed load lives here
    ("ss_weeks", "fuel", 0.3, 0.0, 1.0),
    ("top_weeks", "fuel", 8.0, 4.0, 16.0),
    ("grid_buffer", "fuel", 0.6, 0.3, 0.95),
    ("overflow_frac", "fuel", 0.92, 0.70, 1.0),
    ("on_ratio", "fuel", 0.9, 0.70, 1.0),
    ("prime_weeks", "fuel", 2.0, 1.0, 4.0),
    ("end_weeks", "fuel", 16.0, 10.0, 22.0),
    ("run_band", "fuel", 0.9, 0.70, 1.0),
    ("gate_ratio", "fuel", 1.0, 0.6, 1.3),
    ("gate_on", "fuel", 0.98, 0.90, 1.0),
    ("gate_ask", "fuel", 0.2, 0.05, 0.5),
    ("gate_cover", "fuel", 0.9, 0.6, 1.1),
    ("thr_full", "fuel", 0.6, 0.3, 0.9),
    ("thr_useful", "fuel", 0.5, 0.2, 0.8),
    ("queue_frac", "fuel", 0.15, 0.05, 0.40),
    ("queue_horizon", "fuel", 4.0, 2.0, 8.0),
    ("fleet_frac", "fuel", 1.0, 0.85, 1.15),
    ("nuc_safety", "fuel", 1.5, 1.0, 2.5),
    ("nuc_buffer", "fuel", 6.0, 2.0, 12.0),
    ("share_first", "fuel", 0.5, 0.2, 0.8),
    ("bank_ratio", "fuel", 0.9, 0.7, 1.0),
    ("grid_cover_weeks", "fuel", 0.5, 0.2, 1.0),
    ("pulse_min", "fuel", 0.1, 0.02, 0.3),
    # chips: the 35 % that is unmet demand
    ("sink_pull", "chip", 1.3, 1.0, 2.0),
    ("w_weeks", "chip", 3.0, 1.5, 5.0),
    ("w_fill", "chip", 0.9, 0.6, 1.0),
    ("w_u_min", "chip", 0.02, 0.005, 0.1),
    ("pack_margin", "chip", 1.3, 1.0, 2.0),
    ("pack_cover", "chip", 6.0, 3.0, 10.0),
    ("raw_extra", "chip", 1.0, 0.5, 2.0),
    ("pull_buffer", "chip", 12.0, 6.0, 20.0),
    ("scarce_frac", "chip", 0.5, 0.3, 0.8),
    # straits
    ("need_weeks", "strait", 8.0, 4.0, 14.0),
    ("room_frac", "strait", 0.9, 0.7, 1.0),
    ("min_extra", "strait", 0.02, 0.005, 0.1),
    ("strait_fleet", "strait", 0.95, 0.8, 1.0),
]
NAMES = [k[0] for k in KNOBS]
DEFAULTS = [k[2] for k in KNOBS]
LOW = [k[3] for k in KNOBS]
HIGH = [k[4] for k in KNOBS]
PART = {k[0]: k[1] for k in KNOBS}


def _module(name, folder=RULES):
    spec = importlib.util.spec_from_file_location(f"tuned_{folder.name}_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Parts:
    """The three modules, imported once per process. Building an agent from them costs a few milliseconds."""

    def __init__(self, folder=RULES):
        self.chip = _module("chip_part", folder)
        self.fuel = _module("fuel_part", folder)
        self.strait = _module("strait_part", folder)
        self.chip_defaults = dict(self.chip.PARAMS)

    def agent(self, config, values=None, schedule=None):
        """The composed rule agent on ``values``, or on ``schedule`` — one set of numbers per segment of the
        episode, put in place by the week. A schedule of one segment is the same thing as ``values``.
        """
        sets = schedule if schedule else [values or {}]
        first = sets[0]
        chip_over = {n: v for n, v in first.items() if PART.get(n) == "chip"}
        fuel_over = {n: v for n, v in first.items() if PART.get(n) == "fuel"}
        strait_over = {n: v for n, v in first.items() if PART.get(n) == "strait"}
        # chip_part reads its module dict while it runs, so it is set here and restored by the next call
        self.chip.PARAMS.clear()
        self.chip.PARAMS.update(self.chip_defaults)
        self.chip.PARAMS.update(chip_over)
        return Composed(
            self.chip.Agent(config),
            self.fuel.FuelRules(config, dict(self.fuel.FUEL, **fuel_over)),
            self.strait.StraitRules(config, strait_over),
            config,
            self.chip.PARAMS,
            self.chip_defaults,
            sets,
        )


class Composed:
    """The same order as ``agents/anastasiia_rules_v2/agent.py``: chips, then fuel, then the straits.

    With more than one set of numbers the episode is cut into equal segments and the week's set is put in place
    before the parts run. All three read their numbers while they work, so this costs nothing but a dict update.
    """

    def __init__(self, chip, fuel, strait, config, chip_params, chip_defaults, sets):
        import numpy as np

        self._np = np
        self.chips, self.fuel, self.strait = chip, fuel, strait
        self.n_slots = config["spaces"]["action"]["flows"]["shape"][0]
        self.T = int(config["T"])
        self.chip_params, self.chip_defaults = chip_params, chip_defaults
        self.sets = sets

    def _apply(self, week):
        if len(self.sets) < 2:
            return
        i = min(int(week * len(self.sets) / max(self.T, 1)), len(self.sets) - 1)
        values = self.sets[i]
        self.chip_params.clear()
        self.chip_params.update(self.chip_defaults)
        self.chip_params.update({n: v for n, v in values.items() if PART.get(n) == "chip"})
        self.fuel.P.update({n: v for n, v in values.items() if PART.get(n) == "fuel"})
        self.strait.P.update({n: v for n, v in values.items() if PART.get(n) == "strait"})

    def act(self, observation):
        self._apply(int(self._np.asarray(observation["week"]).reshape(-1)[0]) - 1)
        flows = self._np.zeros(self.n_slots)
        self.chips.fill_chip_flows(observation, flows)
        self.fuel.fill(observation, flows)
        return {"flows": flows, **self.strait.fill(observation)}


def clip(values):
    """A candidate vector held inside the ranges, as a dict the parts understand."""
    out = {}
    for (name, _, _, lo, hi), v in zip(KNOBS, values):
        out[name] = float(min(max(float(v), lo), hi))
    return out
