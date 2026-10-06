"""Run variants of agents/strait_wise (same agent.py, different params.json) through compare.py and log every line.

    PY lab_out/runvars.py --episodes=32 --tag=v0 --variants='{"a": {"queue_w1": 2}, "b": {}}'
    PY lab_out/runvars.py --episodes=32 --src=agents/strait_wise --variants_file=lab_out/vars_x.json

Each variant gets a folder lab_out/variants/<name>/ (agent.py copied from --src, params.json written). The compare.py
lines are appended to lab_out/variants.log with the variant's params, so every try, helpful or not, is on record.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import fire

MAIN = Path("/Users/anastasiiamazur/Projects/secreton-mantis-analytics")
PY = MAIN / ".venv" / "bin" / "python"
COMPARE = MAIN / "outputs" / "heur_lab" / "tools" / "compare.py"
WT = Path(__file__).resolve().parent.parent


def main(variants=None, variants_file=None, episodes=32, tag="", src="agents/strait_wise", base=None, n_jobs=3, log="lab_out/variants.log"):
    if variants_file:
        variants = json.loads(Path(variants_file).read_text())
    if isinstance(variants, str):
        variants = json.loads(variants)
    for name, params in variants.items():  # fire turns `false` on the command line into the string "false": refuse it
        bad = [k for k, v in params.items() if isinstance(v, str) and v.lower() in ("true", "false")]
        if bad:
            raise ValueError(f"variant {name}: {bad} are strings; use --variants_file with real JSON booleans")
    base = base or str(MAIN / "agents" / "pull")
    src_dir = WT / src
    paths = {}
    for name, params in variants.items():
        d = WT / "lab_out" / "variants" / f"{tag}{name}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / "agent.py", d / "agent.py")
        (d / "params.json").write_text(json.dumps(params, indent=1))
        paths[f"{tag}{name}"] = d
    cmd = [str(PY), str(COMPARE), f"--base={base}", f"--episodes={episodes}", f"--n_jobs={n_jobs}"] + [str(p) for p in paths.values()]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WT))
    out = res.stdout
    err = res.stderr[-2000:] if res.returncode else ""
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    with open(WT / log, "a") as f:
        f.write(f"\n## {stamp} episodes={episodes} src={src} tag={tag} ({time.time() - t0:.0f}s)\n")
        for ln in lines:
            for name, p in paths.items():
                if ln.endswith(str(p)):
                    ln = ln[: -len(str(p))] + name + "   params=" + json.dumps(variants[name[len(tag):]])
            f.write(ln + "\n")
        if err:
            f.write("STDERR: " + err + "\n")
    print(out)
    if err:
        print(err)


if __name__ == "__main__":
    fire.Fire(main)
