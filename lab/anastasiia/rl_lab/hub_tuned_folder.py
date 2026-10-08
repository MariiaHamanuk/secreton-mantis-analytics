"""Write a CEM result for the hub agent as an agent folder: a copy of ``agents/anastasiia_hybrid_hub`` plus a
``params.json`` the parts read at import (fuel, chip and strait numbers flat; the program's under ``"lp"``).

    uv run python lab/anastasiia/rl_lab/hub_tuned_folder.py outputs/rl_lab/hub_cem/run_nolp/params.json \
        lab/anastasiia/rl_lab/agents/hub_tuned

``--solve_share=100`` lifts the program's CPU limit (only for a deterministic local comparison, never to submit).
Only numbers that differ from the shipped ones are written.
"""

import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import tuned_rules_hub as T  # noqa: E402


def main(params, dest, solve_share=None):
    src = Path(params) if Path(params).is_absolute() else ROOT / params
    values = json.loads(src.read_text()) if params != "none" else {}
    dest = Path(dest) if Path(dest).is_absolute() else ROOT / dest
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(T.HUB, dest, ignore=shutil.ignore_patterns("__pycache__"))
    default = dict(zip(T.NAMES, T.DEFAULTS))
    out, lp = {}, {}
    for name, v in values.items():
        if v == default[name]:
            continue
        (lp if T.PART[name] == "lp" else out)[name] = v
    if solve_share is not None:
        lp["solve_share"] = float(solve_share)
    if lp:
        out["lp"] = lp
    if out:
        (dest / "params.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{dest}: {len(out) - (1 if lp else 0)} rule numbers, program {lp or 'as shipped'}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
