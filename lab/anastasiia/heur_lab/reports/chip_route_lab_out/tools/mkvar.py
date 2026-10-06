"""Scratch: make a variant folder: current agents/chip_route/agent.py plus a params.json.

    python lab_scratch/mkvar.py NAME '{"pack": false}'  -> lab_scratch/var/NAME/
"""
import json
import shutil
import sys
from pathlib import Path

root = Path("lab_scratch/var") / sys.argv[1]
root.mkdir(parents=True, exist_ok=True)
shutil.copy("agents/chip_route/agent.py", root / "agent.py")
params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
(root / "params.json").write_text(json.dumps(params))
print(root.resolve())
