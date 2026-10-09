"""Assemble a variant of ``agents/anastasiia_plan_hull3`` with this lab's sources and other numbers.

    uv run python lab/anastasiia/hazard_lab/build.py base_s --preset=small
    uv run python lab/anastasiia/hazard_lab/build.py ends_all_s --preset=small --truth_events='{"types": "all"}'

Writes ``outputs/hazard_lab/agents/<name>/``: a copy of the committed model's folder with the files of ``src/``
(the model's own ``agent.py``, ``plan_core.py`` and ``sim_model.py`` plus this lab's switches, all off by default,
and this lab's ``watch.py``) and a ``regime.json`` that is the model's, then the preset's entries, then the named ones (a JSON value
each). A folder a run may still be reading is never rebuilt: build under a new name.

Presets (the model has a clock that fills the week's CPU budget, so its plays depend on the machine's load; a lever
is sized without it):

- ``model``: the model's numbers as they are.
- ``small``: no clock, the search over sets of whole weeks at every week (what the model does on Small).
- ``full``: no clock, no search (on Full the model's clock lets the search into one week in five, and a search at
  every week takes 8 s a week).
"""

import json
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


def _words(value):
    """fire reads a JSON object as a Python literal and leaves true, false and null as strings: put them back."""
    if isinstance(value, dict):
        return {k: _words(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_words(v) for v in value]
    return {"true": True, "false": False, "null": None}.get(value, value) if isinstance(value, str) else value


def main(name: str, preset: str = "model", **numbers) -> None:
    out = OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists: a run may be reading it; build under a new name or remove it yourself")
    shutil.copytree(BASE, out, ignore=shutil.ignore_patterns("__pycache__"))
    for source in sorted((HERE / "src").iterdir()):  # the model's files with this lab's switches, and this lab's own
        if source.is_file():
            shutil.copy(source, out / source.name)
    regime = json.loads((BASE / "regime.json").read_text()) | PRESETS[preset]
    for key, value in numbers.items():
        regime[key] = _words(json.loads(value) if isinstance(value, str) and value[:1] in "[{\"" else value)
    (out / "regime.json").write_text(json.dumps(regime))
    print(out, {k: regime[k] for k in numbers})


if __name__ == "__main__":
    fire.Fire(main)
