"""Run the supporting measurements and write them to lab_out/records/measurements.txt (scratch driver)."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.venv/bin/python"
FF = str(ROOT / "agents" / "fuel_first")
PULL = "/Users/anastasiiamazur/Projects/secreton-mantis-analytics/agents/pull"
FULL_CACHE = str(ROOT / "scratch" / "refcache_copy")
out = []


def run(title, args, env=None, tail=1):
    res = subprocess.run([PY, *args], capture_output=True, text=True, cwd=str(ROOT), env={**os.environ, **(env or {})})
    lines = (res.stdout + res.stderr).strip().splitlines()
    out.append(f"== {title}")
    out.extend(lines[-tail:])
    out.append("")
    print(title, "done", flush=True)


run("observed capacity (graph_now) vs the simulator's week value, 12 episodes of root 111", ["scratch/obs_vs_sim.py", "0", "12"], tail=5)
run("fuel waiting at straits, pull (12 episodes)", ["scratch/queue_size.py", PULL, "0", "12"])
run("fuel waiting at straits, fuel_first (12 episodes)", ["scratch/queue_size.py", FF, "0", "12"])
run("throttled / contested strait-weeks, pull", ["scratch/contested_straits.py", PULL, "0", "12"])
run("throttled / contested strait-weeks, fuel_first", ["scratch/contested_straits.py", FF, "0", "12"])
run("fuel stuck behind a prohibited next edge, fuel_first (16 episodes; last line is the mean)", ["scratch/stuck_fuel.py", FF, "0", "16"])
run("non-fuel entries of flows equal pull's, Small", ["scratch/same_nonfuel.py", "small", "0", "1", "2", "3"])
run("non-fuel entries of flows equal pull's, Full", ["scratch/same_nonfuel.py", "full", "0"], env={"SBF_CACHE_DIR": FULL_CACHE})
(ROOT / "lab_out" / "records" / "measurements.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
