"""``hub/eval/compare.py`` without playing the base twice: the same scores, the same paired bootstrap, the same seed.

compare.py scores the base once for its own line and again inside ``EpisodeSet.compare``; with the hub agent at ~14 s
a Small episode and ~100 s a Full one that is a third of the run. This scores each agent once and calls the package's
own bootstrap (``EpisodeSet.compare``'s body), so the numbers are compare.py's.

    uv run python lab/anastasiia/rl_lab/hub_paired.py lab/anastasiia/rl_lab/agents/hub_tuned \
        --base=agents/anastasiia_hybrid_hub --task=small --episodes=64 --n_jobs=2
"""

import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def main(*agents, base="agents/anastasiia_hybrid_hub", task="small", episodes=64, entropy=111, n_jobs=2):
    from shockbench_flow_agent import EpisodeSet

    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    mod = sys.modules[EpisodeSet.__module__]
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    sb = es.score(str(resolve(base)), n_jobs=n_jobs, cpu_budget=False)

    def levels(score):
        return " ".join("-" if v is None else f"{v:.3f}" for v in score.rss_by_stratum.values())

    print(f"{task}, {episodes} episodes of root {entropy}")
    print(f"{sb.rss:7.4f} {'':>9} {'':>32} {levels(sb):>26} {sb.fallback_weeks:>11}  {base} (base)", flush=True)
    for a in agents:
        sa = es.score(str(resolve(a)), n_jobs=n_jobs, cpu_budget=False)
        pooled = sa.pooled and sb.pooled
        ba, bb = mod._boot_stats(es.references, [sa.J_cents, sb.J_cents], pooled, mod.N_BOOT, 0)
        d = ba - bb
        lo, hi = mod._interval(d[np.isfinite(d)], mod.LEVEL)
        print(
            f"{sa.rss:7.4f} {sa.rss - sb.rss:+9.4f} {f'{lo:+.4f} to {hi:+.4f}':>32} {levels(sa):>26} "
            f"{sa.fallback_weeks:>11}  {a}",
            flush=True,
        )
        rel = (np.array(sb.J_cents, float) - np.array(sa.J_cents, float)) / np.array(sb.J_cents, float)
        print(f"        cost {rel.mean():+.4%} per episode (mean), cheaper in {(rel > 0).sum()}/{len(rel)}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
