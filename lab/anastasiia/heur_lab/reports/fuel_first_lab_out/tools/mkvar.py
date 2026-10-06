"""Snapshot agents/fuel_first/agent.py into scratch/variants/<name>/ with a params.json (scratch).

    python scratch/mkvar.py <name> '{"pulse": false, "cover_weeks": 3}' [<name2> '<json2>' ...]

Prints the absolute folder of each variant, one per line.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "agents" / "fuel_first" / "agent.py"
args = sys.argv[1:]
for name, js in zip(args[::2], args[1::2]):
    folder = ROOT / "scratch" / "variants" / name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC, folder / "agent.py")
    (folder / "params.json").write_text(json.dumps(json.loads(js), indent=1))
    print(folder)
