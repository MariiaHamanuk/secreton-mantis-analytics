"""Assemble a folder of the model from a copy of the lab's sources, for a speed study.

    uv run python lab/anastasiia/frontier_lab/speed/make.py t_ref --src=outputs/speed_lab/t_base
    uv run python lab/anastasiia/frontier_lab/speed/make.py v_alt --src=outputs/speed_lab/t_work --hull_every=2

Writes ``outputs/speed_lab/<name>/``: ``agents/anastasiia_plan_hull3`` with the files of ``src`` over it (a copy of
``lab/anastasiia/hazard_lab/src``, changed or not), and the objective terms and the ``regime.json`` of
``outputs/hazard_lab/agents/h3_f`` (the model's numbers without its clock and without the search), the named settings
over them. ``--preset=model`` takes the terms and the numbers of ``agents/anastasiia_plan_hazard4`` instead (the clock
and the search as submitted) and ``--budget_scale=0.54`` aims that clock at the server's speed. A folder that exists
is never rebuilt.
"""

import json
import shutil
from pathlib import Path

import fire


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "agents" / "anastasiia_plan_hull3"
TERMS = {"full": ROOT / "outputs" / "hazard_lab" / "agents" / "h3_f",  # the sum's file and the model's terms under it
         "model": ROOT / "agents" / "anastasiia_plan_hazard4"}  # the model's terms as its own file
REGIME = {"full": ROOT / "outputs" / "hazard_lab" / "agents" / "h3_f" / "regime.json",
          "model": ROOT / "agents" / "anastasiia_plan_hazard4" / "regime.json"}  # fmt: skip
OUT = ROOT / "outputs" / "speed_lab"


def _words(value):
    """fire leaves true, false and null as strings: put them back."""
    return {"true": True, "false": False, "null": None}.get(value, value) if isinstance(value, str) else value


def main(name: str, src: str = "outputs/speed_lab/t_base", preset: str = "full", **numbers) -> None:
    out = OUT / name
    if out.exists():
        raise SystemExit(f"{out} exists: build under a new name or remove it yourself")
    skip = shutil.ignore_patterns("__pycache__", "*.orig", "*.rej")
    shutil.copytree(BASE, out, ignore=skip)
    shutil.copytree((ROOT / src).resolve(), out, dirs_exist_ok=True, ignore=skip)
    for file in TERMS[preset].glob("terms*.py"):  # the same files as the folder the preset's numbers come from
        shutil.copy(file, out / file.name)
    regime = json.loads(REGIME[preset].read_text())
    for key, value in numbers.items():
        regime[key] = _words(json.loads(value) if isinstance(value, str) and value[:1] in "[{\"" else value)
    (out / "regime.json").write_text(json.dumps(regime))
    print(out, {k: regime[k] for k in numbers})


if __name__ == "__main__":
    fire.Fire(main)
