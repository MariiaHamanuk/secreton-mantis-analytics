"""Screen agent folders on root-111 episodes: one world per episode, every variant rolled out on it (scratch).

    python scratch/screen.py --episodes=32 --n_jobs=3 --base=<folder> <folder_a> <folder_b> ...

Same references, same score and the same paired bootstrap as compare.py (the simulator is deterministic and the rule
agents draw nothing), only the episode worlds are shared between the variants, so it is several times faster.
"""
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


def _play_chunk(spec, episodes, folders):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _policy_seed, _world
    from shockbench_flow_agent.shim import AgentShim, load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    out = {}
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        for folder in folders:
            with without_secret_like():
                factory = load_agent_class(folder, f"screen_{Path(folder).name}")
                shim = AgentShim(factory)
                traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            unload_agent()
            fell = sum(1 for r in traj.records if getattr(r, "invalid", ()) and False)
            out[(folder, n)] = (int(traj.J_cents), len(getattr(shim, "errors", []) or []))
    return out


def main(*folders: str, base: str, task: str = "small", episodes: int = 32, entropy: int = 111, n_jobs: int = 3, first: int = 0, dump: str = "") -> None:
    from shockbench_flow_agent.scoring import EpisodeSet, _boot_stats, _interval

    t0 = time.time()
    ns = list(range(first, first + episodes))
    es = EpisodeSet.build(task, ns, entropy=entropy, n_jobs=n_jobs, verbose=False)
    allf = [base, *folders]
    chunks = [c for c in np.array_split(np.array(ns), n_jobs) if c.size]
    parts = Parallel(n_jobs=n_jobs)(delayed(_play_chunk)(es._spec, [int(x) for x in c], allf) for c in chunks)
    res = {}
    for p in parts:
        res.update(p)
    J = {f: [res[(f, n)][0] for n in ns] for f in allf}
    errs = {f: sum(res[(f, n)][1] for n in ns) for f in allf}
    if dump:
        import json

        refs = [{k: r[k] for k in ("episode", "stratum", "J_naive_cents", "J_oracle_cents", "excluded", "harm_usd")} for r in es.references]
        Path(dump).write_text(json.dumps({"refs": refs, "J": {Path(f).name: J[f] for f in allf}}))
    tab = {f: es.rss(J[f]) for f in allf}
    pooled = all(t["pooled"] for t in tab.values())
    boots = _boot_stats(es.references, [J[f] for f in allf], pooled, 2000, 0)
    print(f"{task}, {episodes} episodes of root {entropy} from {first} ({time.time() - t0:.0f} s)")
    print(f"{'score':>7} {'vs base':>9} {'90% interval of the difference':>32} {'by harm level 1..4':>26} {'errs':>5}  agent")
    for i, f in enumerate(allf):
        lv = " ".join("  n/a" if v is None else f"{v:.3f}" for v in tab[f]["rss_by_stratum"].values())
        if i == 0:
            print(f"{tab[f]['rss']:7.4f} {'':>9} {'':>32} {lv:>26} {errs[f]:5d}  {Path(f).name} (base)")
            continue
        d = boots[i] - boots[0]
        lo, hi = _interval(d, 0.9)
        print(f"{tab[f]['rss']:7.4f} {tab[f]['rss'] - tab[base]['rss']:+9.4f} {f'{lo:+.4f} to {hi:+.4f}':>32} {lv:>26} {errs[f]:5d}  {Path(f).name}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
