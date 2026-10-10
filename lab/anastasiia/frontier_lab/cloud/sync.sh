#!/bin/zsh
# pull every machine's kept episodes and merge them into the local play folders (hazard plays: one dict a tag; scenario
# plays: one file an episode)
P=${SBF_GCP_PROJECT:?set the cloud project}
ROOT=$(git rev-parse --show-toplevel)
HERE=${0:A:h}
typeset -A Z; Z=(sbf-s5 us-west1-a sbf-s7 us-west4-a sbf-s8 us-west4-a sbf-s9 us-west4-a sbf-s10 europe-west1-b sbf-s11 europe-west1-b)
for m in ${@:-${(ko)Z}}; do
  ( mkdir -p $HERE/pull/$m; gcloud compute ssh $m --project=$P --zone=${Z[$m]} --quiet --command='cd ~/repo/outputs && tar czf - $(ls -d hazard_lab/play 2>/dev/null) $(find frontier_lab/scen/play -name "*.pkl" 2>/dev/null) 2>/dev/null' 2>/dev/null > $HERE/pull/$m.tgz && tar xzf $HERE/pull/$m.tgz -C $HERE/pull/$m 2>/dev/null ) &
done; wait
cd $ROOT && uv run python - $HERE/pull <<'PY'
import glob, pickle, shutil, sys
from pathlib import Path
pull, out = Path(sys.argv[1]), Path("outputs")
merged = {}
for f in sorted(pull.glob("*/hazard_lab/play/*.pkl")):
    try:
        merged.setdefault(f.name, {}).update(pickle.loads(f.read_bytes()))
    except Exception as e:
        print("skipped", f, e)
for name, d in merged.items():
    path = out / "hazard_lab" / "play" / name
    old = pickle.loads(path.read_bytes()) if path.is_file() else {}
    path.write_bytes(pickle.dumps(old | d))
n = 0
for f in sorted(pull.glob("*/frontier_lab/scen/play/*/*.pkl")):
    dst = out / "frontier_lab" / "scen" / "play" / f.parent.name / f.name
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(f, dst); n += 1
print("hazard tags", {k[:-4]: len(v) for k, v in merged.items()}, "new scenario files", n)
PY
