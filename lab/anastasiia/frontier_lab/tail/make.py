"""Build a variant folder of this lab: ``h3_f``'s files, this lab's patch (``tw.diff``), other numbers.

    uv run python lab/anastasiia/frontier_lab/tail/make.py tw_e20 --horizon=20 --hull_until=6 --end_reach
    uv run python lab/anastasiia/frontier_lab/tail/make.py diff     # write tw.diff from the folder being worked on

A variant is ``outputs/hazard_lab/agents/<name>/``: a copy of ``outputs/hazard_lab/agents/h3_f`` (the settings of
``anastasiia_plan_hazard3`` without the clock, built by ``hazard_lab/build.py``), ``agent.py`` and ``plan_core.py``
patched with ``tw.diff`` beside this file, and ``h3_f``'s ``regime.json`` with the named numbers (a JSON value each).
Every switch of the patch is off by default, so a variant without numbers plays as ``h3_f`` does. ``diff`` writes the
patch from ``outputs/hazard_lab/agents/tw_dev``, the folder the patch is worked on in.
"""

import json
import shutil
import subprocess
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AGENTS = ROOT / "outputs" / "hazard_lab" / "agents"
BASE, DEV, PATCH = AGENTS / "h3_f", AGENTS / "tw_dev", HERE / "tw.diff"
FILES = ("agent.py", "plan_core.py")


def _words(value):
    """fire leaves true, false and null of a JSON object as strings: put them back."""
    if isinstance(value, dict):
        return {k: _words(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_words(v) for v in value]
    return {"true": True, "false": False, "null": None}.get(value, value) if isinstance(value, str) else value


def diff() -> None:
    parts = []
    for name in FILES:
        run = subprocess.run(["diff", "-u", "--label", f"a/{name}", "--label", f"b/{name}", str(BASE / name), str(DEV / name)],
                             capture_output=True, text=True, check=False)  # fmt: skip
        parts.append(run.stdout)
    PATCH.write_text("".join(parts))
    print(PATCH, f"{sum(p.count(chr(10)) for p in parts)} lines")


def main(name: str, **numbers) -> None:
    if name == "diff":
        return diff()
    if not name.startswith("tw_"):
        raise SystemExit("this lab's folders start with tw_")
    out = AGENTS / name
    if out.exists():
        raise SystemExit(f"{out} exists: a run may be reading it; build under a new name")
    shutil.copytree(BASE, out, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["patch", "-s", "-p1", "-d", str(out), "-i", str(PATCH)], check=True)
    regime = json.loads((BASE / "regime.json").read_text())
    for key, value in numbers.items():
        regime[key] = _words(json.loads(value) if isinstance(value, str) and value[:1] in "[{\"" else value)
    (out / "regime.json").write_text(json.dumps(regime))
    print(out, {k: regime[k] for k in numbers})


if __name__ == "__main__":
    fire.Fire(main)
