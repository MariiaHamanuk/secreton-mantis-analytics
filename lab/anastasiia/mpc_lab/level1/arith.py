"""Level-1 arithmetic: is the hub's low level-1 RSS a bigger absolute loss or a smaller denominator?

    uv run python lab/anastasiia/mpc_lab/level1/arith.py --episodes=64 --n_jobs=2

Plays the agent on root 111 (cached references), keeps the rows in outputs/level1/<date_time>/rows.json and prints,
per harm level: episodes, naive - clairvoyant gap, agent loss to clairvoyant, saved, RSS, and the marginal board value
of 1 bn USD/episode on each level.
"""

import datetime
import json
from pathlib import Path

import fire
import numpy as np

import sbf_starter  # noqa: F401 - sets the team's reference cache before the wheel is imported
from sbf_starter import ROOT, scoring

W = (0.50, 0.30, 0.15, 0.05)


def main(agent: str = "agents/anastasiia_hybrid_hub", episodes: int = 64, entropy: int = 111, n_jobs: int = 2,
         rows: str = "") -> None:
    if rows:
        data = json.loads(Path(rows).read_text())
    else:
        es = scoring.episode_set("small", episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
        s = es.score(str(ROOT / agent), n_jobs=n_jobs, cpu_budget=False)
        keys = ("episode", "stratum", "harm_usd", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents", "excluded")
        data = [{k: r.get(k) for k in keys} for r in s.rows]
        out = ROOT / "outputs" / "level1" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        (out / "rows.json").write_text(json.dumps(data, indent=1))
        print(f"rows -> {out / 'rows.json'};  package RSS {s.rss:.4f}, by level {s.rss_by_stratum}")
    data = [r for r in data if r["excluded"] is None]
    lv = np.array([r["stratum"] for r in data])
    jn = np.array([r["J_naive_cents"] for r in data]) / 1e11  # bn USD
    jc = np.array([r["J_clairvoyant_cents"] for r in data]) / 1e11
    ja = np.array([r["J_policy_cents"] for r in data]) / 1e11
    harm = np.array([r["harm_usd"] for r in data]) / 1e9
    gap, loss = jn - jc, ja - jc
    present = [s for s in (1, 2, 3, 4) if (lv == s).any()]
    den = sum(W[s - 1] * gap[lv == s].mean() for s in present)
    rss = sum(W[s - 1] * (gap - loss)[lv == s].mean() for s in present) / den
    print(f"board-weighted RSS {rss:.4f}; weighted gap sum_s w_s*gap_s = {den:.1f} bn")
    print(f"{'lvl':>3} {'n':>3} {'harm':>7} {'naive':>8} {'clairv':>8} {'gap':>7} {'loss':>7} {'loss/gap':>8} "
          f"{'RSS_s':>6} {'RSS_s SE':>8} {'1 bn->RSS':>9} {'loss share of board deficit':>27}")
    deficit = sum(W[s - 1] * loss[lv == s].mean() for s in present) / den
    for s in present:
        m = lv == s
        n = m.sum()
        r_ep = 1 - loss[m] / gap[m]
        # ratio-estimator SE of loss/gap
        q = loss[m].mean() / gap[m].mean()
        se = np.sqrt(np.var(loss[m] - q * gap[m], ddof=1) / n) / gap[m].mean()
        share = W[s - 1] * loss[m].mean() / den / deficit
        print(f"{s:3d} {n:3d} {harm[m].mean():7.0f} {jn[m].mean():8.1f} {jc[m].mean():8.1f} {gap[m].mean():7.1f} "
              f"{loss[m].mean():7.1f} {q:8.4f} {1 - q:6.4f} {se:8.4f} {W[s - 1] / den:9.5f} {share:27.3f}")
    # counterfactual: level-1 loss with the level-2/3 gap
    m1, m23 = lv == 1, (lv == 2) | (lv == 3)
    print(f"level 1: loss {loss[m1].mean():.1f}, gap {gap[m1].mean():.1f};  levels 2-3: loss {loss[m23].mean():.1f}, "
          f"gap {gap[m23].mean():.1f}")
    print(f"  level-1 loss over level-2/3 gap: RSS would be {1 - loss[m1].mean() / gap[m23].mean():.4f}")
    print(f"  level-2/3 loss over level-1 gap: RSS would be {1 - loss[m23].mean() / gap[m1].mean():.4f}")
    print("  per-episode level 1 (episode, harm, gap, loss, rss):")
    for i in np.flatnonzero(m1):
        print(f"    {data[i]['episode']:3d} {harm[i]:6.0f} {gap[i]:7.1f} {loss[i]:7.1f} {1 - loss[i] / gap[i]:6.3f}")
    print("  per-episode level 2-3:")
    for i in np.flatnonzero(m23):
        print(f"    {data[i]['episode']:3d} L{lv[i]} {harm[i]:6.0f} {gap[i]:7.1f} {loss[i]:7.1f} {1 - loss[i] / gap[i]:6.3f}")
    c = np.corrcoef(gap[m1], loss[m1])[0, 1]
    print(f"  corr(gap, loss) on level 1: {c:.2f}; corr(harm, gap) all: {np.corrcoef(harm, gap)[0, 1]:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
