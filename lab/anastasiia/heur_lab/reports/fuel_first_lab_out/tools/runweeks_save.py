"""Run scratch/runweeks.py for the final agent and save the table to lab_out/records/run_weeks_final.txt (scratch driver)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.venv/bin/python"
res = subprocess.run(
    [PY, "scratch/runweeks.py", str(ROOT / "agents" / "fuel_first"), "0", "24"], capture_output=True, text=True, cwd=str(ROOT)
)
text = "weeks 1-40 of episodes 0..23 of root 111; lots/cap = lots started that week as a share of the fab's nominal capacity; "
text += "'grid mode' is the lng valve mode of the fab's grid that week (on/run = fed, hold/prime = held back)\n"
text += res.stdout + res.stderr
(ROOT / "lab_out" / "records" / "run_weeks_final.txt").write_text(text)
print(text)
