"""Scratch: make ablation variants of a base parameter set: python mkabl.py base_json name1 '{"k": v}' name2 '{...}' ..."""
import json
import subprocess
import sys

base = json.loads(sys.argv[1])
args = sys.argv[2:]
for name, change in zip(args[0::2], args[1::2]):
    params = dict(base)
    params.update(json.loads(change))
    subprocess.run([sys.executable, "lab_scratch/mkvar.py", name, json.dumps(params)], check=True, capture_output=True)
    print(name, change)
