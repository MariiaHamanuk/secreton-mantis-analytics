"""Out-of-sample paired check of a hub-agent variant on chosen episodes, one episode at a time, resumable.

Plays ``base`` and ``agent`` folders on each episode of ``episodes`` (root ``entropy``), appends one JSON row per
(episode, agent) to ``rows`` (a restart skips what is there) and one line per finished episode to ``hub_cem.log``.
When every episode is done it reads the references from the episode set and prints the RSS of both, the paired
difference with the package's own bootstrap (as ``hub_paired.py`` / ``hub/eval/compare.py``) and the harm levels.

Workers: 1 while other Python processes on the machine number 4 or more, else 2 (checked before every submit).

    uv run python lab/anastasiia/rl_lab/hub_oos.py --task=full --first=16 --last=47 \
        --base=outputs/rl_lab/hub_cem/eval/hub_nolimit --agent=outputs/rl_lab/hub_cem/eval/hub_tuned_nolimit
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import hub_cem as H  # noqa: E402


def _others():
    """Python interpreters running on the machine that are not this run (this process and its workers)."""
    me = os.getpid()
    out = subprocess.run(["ps", "-eo", "pid,ppid,comm"], capture_output=True, text=True).stdout.splitlines()[1:]
    n = 0
    for line in out:
        pid, ppid, comm = line.split(None, 2)
        if comm.strip().startswith("python") and int(pid) != me and int(ppid) != me:
            n += 1
    return n


def _job(payload):
    task, entropy, n, i, folder = payload
    t0 = time.process_time()
    c = H.cost_of_folder(task, entropy, n, i, ROOT / folder)
    return i, folder, c, time.process_time() - t0


def main(
    task="full",
    first=16,
    last=47,
    entropy=111,
    base="outputs/rl_lab/hub_cem/eval/hub_nolimit",
    agent="outputs/rl_lab/hub_cem/eval/hub_tuned_nolimit",
    rows="outputs/rl_lab/hub_cem/oos_full_16_47.jsonl",
    max_workers=2,
):
    rows = ROOT / rows
    done = {}
    if rows.is_file():
        for line in rows.read_text().splitlines():
            r = json.loads(line)
            done[(r["episode"], r["folder"])] = r["cost"]
    eps = list(range(first, last + 1))
    n = last + 1  # the pool's size: episode i is pool index i (as in the search)
    todo = [(task, entropy, n, i, f) for i in eps for f in (base, agent) if (i, f) not in done]
    H._log(f"[{time.strftime('%H:%M:%S')}] out-of-sample {task} {first}..{last} root {entropy}: {agent} vs {base}, "
           f"{len(todo)} plays left")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        running = set()
        while todo or running:
            limit = max_workers if _others() < 4 else 1
            while todo and len(running) < limit:
                running.add(pool.submit(_job, todo.pop(0)))
            fin, running = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            for f in fin:
                i, folder, c, sec = f.result()
                done[(i, folder)] = c
                with rows.open("a") as fh:
                    fh.write(json.dumps({"episode": i, "folder": folder, "cost": c, "cpu_s": sec}) + "\n")
                if (i, base) in done and (i, agent) in done:
                    b, a = done[(i, base)], done[(i, agent)]
                    H._log(f"  {task} ep {i:3d}: base {b:.2f} agent {a:.2f} -> {(b - a) / b:+.4%} "
                           f"({'win' if a < b else 'loss' if a > b else 'tie'}), workers {limit}")

    # the score, with the package's references and bootstrap
    from shockbench_flow_agent import EpisodeSet

    mod = sys.modules[EpisodeSet.__module__]
    es = EpisodeSet.build(task, eps, entropy=entropy, n_jobs=max_workers if _others() < 4 else 1, verbose=False)
    refs = {int(r["episode"]): r for r in es.references}
    jb = [int(round(done[(i, base)] * 100)) for i in eps]
    ja = [int(round(done[(i, agent)] * 100)) for i in eps]
    rb, ra = es.rss(jb), es.rss(ja)
    pooled = ra["pooled"] and rb["pooled"]
    sb, sa = (rb["rss"], ra["rss"]) if pooled else (rb["rss_all"], ra["rss_all"])
    ba, bb = mod._boot_stats(es.references, [ja, jb], pooled, mod.N_BOOT, 0)
    d = ba - bb
    lo, hi = mod._interval(d[np.isfinite(d)], mod.LEVEL)
    def fmt(r):
        return " ".join(f"{k}:{'-' if v is None else f'{v:.3f}'}" for k, v in sorted(r["rss_by_stratum"].items()))

    levels = {}
    for i in eps:
        levels.setdefault(refs[i]["stratum"], []).append(i)
    wins = sum(a < b for a, b in zip(ja, jb))
    lines = [
        f"[{time.strftime('%H:%M:%S')}] out-of-sample {task} {first}..{last}: base {sb:.4f}, agent {sa:.4f} "
        f"({'pooled over harm levels' if pooled else 'not pooled: a harm level is missing'})",
        f"  by harm level: base {fmt(rb)} | agent {fmt(ra)}",
        f"  paired {float(sa) - float(sb):+.4f} ({lo:+.4f} to {hi:+.4f}), cheaper in {wins}/{len(eps)}, "
        f"episodes per harm level {dict(sorted((k, len(v)) for k, v in levels.items()))}",
    ]
    for line in lines:
        H._log(line)
    return {"base": sb, "agent": sa, "lo": lo, "hi": hi, "wins": wins}


if __name__ == "__main__":
    import fire

    fire.Fire(main)
