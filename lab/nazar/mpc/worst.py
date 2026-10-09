"""Where an agent is least successful: its score per harm level and its worst episodes.

    uv run python lab/nazar/mpc/worst.py lab/nazar/agents/nazar_rules_lpraw --episodes=32 --entropy=555

Per episode: cost, naive's cost, the clairvoyant plan's, and the episode's own RSS; per harm level: the score and
how much of the board's gap (1 - score, weighted by the level's share) each level holds. Read-only.
"""

import fire
import numpy as np


WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}


def main(
    agent: str, task: str = "small", episodes: int = 32, entropy: int = 555, n_jobs: int = 4, worst: int = 8
) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    s = es.score(str(resolve(agent)), n_jobs=n_jobs, cpu_budget=False)
    rows = [r for r in s.rows if r.get("excluded") is None]
    lvl = np.array([r["stratum"] for r in rows])
    J = np.array([r["J_policy_cents"] for r in rows], dtype=float) / 100
    N = np.array([r["J_naive_cents"] for r in rows], dtype=float) / 100
    C = np.array([r["J_clairvoyant_cents"] for r in rows], dtype=float) / 100
    ep = np.array([r["episode"] for r in rows])
    rss = (N - J) / np.maximum(N - C, 1e-9)
    print(f"{agent}: {task} root {entropy} x{episodes}: board score {s.rss:.4f}")
    print("\n  level  episodes   score   weight   share of the gap (weight x (1-score) / sum)")
    gaps = {}
    for L in (1, 2, 3, 4):
        m = lvl == L
        if m.any():
            sc = (N[m] - J[m]).sum() / (N[m] - C[m]).sum()
            gaps[L] = WEIGHTS[L] * (N[m] - C[m]).mean() * (1 - sc)
            print(f"  {L:>5} {m.sum():>9} {sc:8.3f} {WEIGHTS[L]:8.2f}")
    tot = sum(gaps.values())
    for L, g in gaps.items():
        print(f"  level {L}: {100 * g / tot:5.1f} % of the weighted gap to the clairvoyant")
    print(f"\n  worst {worst} episodes by saving lost (J - J_clairvoyant), USD bn: episode, level, RSS, J, naive, plan")
    lost = (J - C) / 1e9
    for i in np.argsort(-lost)[:worst]:
        print(f"  {ep[i]:>4} {lvl[i]:>2} {rss[i]:7.3f} {J[i] / 1e9:8.0f} {N[i] / 1e9:8.0f} {C[i] / 1e9:8.0f}")
    print(
        f"\n  RSS per episode: min {rss.min():.3f}, 10th pct {np.percentile(rss, 10):.3f}, median {np.median(rss):.3f}"
    )


if __name__ == "__main__":
    fire.Fire(main)
