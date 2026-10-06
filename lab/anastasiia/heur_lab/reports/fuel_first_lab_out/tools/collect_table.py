"""Collect scratch/table_runs/*.txt (screen.py outputs against pull) into scratch/table_all.json (scratch)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = {}
for f in sorted((ROOT / "scratch" / "table_runs").glob("*.txt")):
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 9 and re.match(r"^[+-]?\d\.\d{4}$", parts[0]) and parts[1].startswith(("+", "-")) and parts[-1] != "(base)":
            rows[parts[-1]] = line
(ROOT / "scratch" / "table_all.json").write_text(json.dumps(rows, indent=1))
print(len(rows), "variants")
