"""Run the next batch of variants that have no line yet in scratch/table_runs/*.txt against pull (scratch).

    python scratch/run_rest.py [batch_size]
Each call scores up to batch_size undone variants on 64 episodes of root 111 and saves the output as rest_<k>.txt.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "/Users/anastasiiamazur/Projects/secreton-mantis-analytics/.venv/bin/python"
PULL = "/Users/anastasiiamazur/Projects/secreton-mantis-analytics/agents/pull"
size = int(sys.argv[1]) if len(sys.argv) > 1 else 12
out = ROOT / "scratch" / "table_runs"
done = set()
for f in out.glob("*.txt"):
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 9 and re.match(r"^[+-]?\d\.\d{4}$", parts[0]) and parts[-1] != "(base)" and parts[1].startswith(("+", "-")):
            done.add(parts[-1])
allv = sorted(p.name for p in (ROOT / "scratch" / "variants").iterdir() if (p / "agent.py").is_file())
todo = [n for n in allv if n not in done]
print(f"{len(done)} done, {len(todo)} to do", flush=True)
if not todo:
    sys.exit(0)
batch = todo[:size]
k = len(list(out.glob("rest_*.txt")))
cmd = [PY, str(ROOT / "scratch" / "screen.py"), "--episodes=64", "--n_jobs=3", f"--base={PULL}"] + [str(ROOT / "scratch" / "variants" / n) for n in batch]
res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
(out / f"rest_{k}.txt").write_text(res.stdout + res.stderr)
print("batch", k, "done:", " ".join(batch), flush=True)
print(res.stdout[-300:])
