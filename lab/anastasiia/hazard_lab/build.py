"""Assemble a variant of ``agents/anastasiia_plan_hull3`` with this lab's sources and other numbers.

    uv run python lab/anastasiia/hazard_lab/build.py base_s --preset=small
    uv run python lab/anastasiia/hazard_lab/build.py ends_all_s --preset=small --truth_events='{"types": "all"}'

Writes ``outputs/hazard_lab/agents/<name>/``: a copy of the committed model's folder with the files of ``src/``
(the model's own ``agent.py``, ``plan_core.py`` and ``sim_model.py`` plus this lab's switches, all off by default,
and this lab's ``watch.py``) and a ``regime.json`` that is the model's, then the preset's entries, then the named ones
(a JSON value each). A folder a run may still be reading is never rebuilt: build under a new name.

``--terms='{"model": {}, "hold": {"RATE": 0.01}}'`` adds terms to the objective of the week's program (the hook of
``plan_core.py``, from evolve_lab): ``model`` is the ``terms.py`` of ``agents/anastasiia_plan_hazard3``, any other
name a file of this lab's ``terms/``; the numbers are set on that file's module. The folder then holds one
``terms.py`` that sums them.

``--rules='{"chip_part": {"rate_cap": 1.5}, "fuel_part": {"thr_full": 0.75}}'`` writes other numbers into the
``PARAMS`` of the folder's copies of the rules' files (they have no file of numbers of their own).

Presets (the model has a clock that fills the week's CPU budget, so its plays depend on the machine's load; a lever
is sized without it):

- ``model``: the model's numbers as they are.
- ``small``: no clock, the search over sets of whole weeks at every week (what the model does on Small).
- ``full``: no clock, no search (on Full the model's clock lets the search into one week in five, and a search at
  every week takes 8 s a week).
"""

import json
import re
import shutil
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "agents" / "anastasiia_plan_hull3"
OUT = ROOT / "outputs" / "hazard_lab" / "agents"
NO_CLOCK = {"share": 0, "fit": 0.0, "carry_debt": False, "episode_share": 0.0, "fit_rules": 0, "fit_fresh": 0,
            "fit_horizon": 0}  # fmt: skip
PRESETS = {"model": {}, "small": NO_CLOCK | {"search": 8, "search_room": 1}, "full": NO_CLOCK | {"search": 0}}
MODEL_TERMS = ROOT / "agents" / "anastasiia_plan_hazard3" / "terms.py"
SUM = '''"""The sum of this folder's terms of the week's objective (written by hazard_lab/build.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARTS = {parts!r}  # the file of each part and the numbers set on its module


def _part(stem: str, numbers: dict):
    name = f"{{HERE.name}}_{{stem}}"
    spec = importlib.util.spec_from_file_location(name, HERE / f"{{stem}}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for key, value in numbers.items():
        setattr(module, key, value)
    return module


_PARTS = [_part(stem, numbers) for stem, numbers in PARTS.items()]


def add(ep, mode, ref):
    out = np.zeros(ep.N)
    for part in _PARTS:
        v = part.add(ep, mode, ref)
        if v is not None:
            out = out + v
    return out
'''


def _words(value):
    """fire reads a JSON object as a Python literal and leaves true, false and null as strings: put them back."""
    if isinstance(value, dict):
        return {k: _words(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_words(v) for v in value]
    return {"true": True, "false": False, "null": None}.get(value, value) if isinstance(value, str) else value


def main(name: str, preset: str = "model", terms: dict | None = None, rules: dict | None = None, **numbers) -> None:
    out = OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists: a run may be reading it; build under a new name or remove it yourself")
    shutil.copytree(BASE, out, ignore=shutil.ignore_patterns("__pycache__"))
    # the model's files with this lab's switches, this lab's own, and the files of the exact speed-ups
    shutil.copytree(HERE / "src", out, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    regime = json.loads((BASE / "regime.json").read_text()) | PRESETS[preset]
    for key, value in numbers.items():
        regime[key] = _words(json.loads(value) if isinstance(value, str) and value[:1] in "[{\"" else value)
    (out / "regime.json").write_text(json.dumps(regime))
    if terms and list(_words(terms).items()) == [("model", {})]:  # the model's terms alone: its own file as it is
        shutil.copy(MODEL_TERMS, out / "terms.py")
    elif terms:
        parts = {}
        for part, values in _words(terms).items():
            shutil.copy(MODEL_TERMS if part == "model" else HERE / "terms" / f"{part}.py", out / f"terms_{part}.py")
            parts[f"terms_{part}"] = dict(values or {})
        (out / "terms.py").write_text(SUM.format(parts=parts))
    for part, values in _words(rules or {}).items():
        code = (out / f"{part}.py").read_text()
        for key, value in values.items():
            line = rf'^(\s*"{re.escape(key)}": )[^,]+,'
            code, found = re.subn(line, lambda m: f"{m.group(1)}{value!r},", code, count=1, flags=re.M)  # noqa: B023
            if found != 1:
                raise SystemExit(f"{part}.py has no number {key}")
        (out / f"{part}.py").write_text(code)
    print(out, {k: regime[k] for k in numbers}, terms or "", rules or "")


if __name__ == "__main__":
    fire.Fire(main)
