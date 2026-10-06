"""Per-episode costs of two agents on cached episodes (read-only research tool; same episodes as compare.py).

    PY lab_out/compare_rows.py --a=<agent folder> --b=<agent folder> --episodes=32
"""
import fire
import numpy as np


def main(a: str, b: str, task: str = "small", episodes: int = 32, entropy: int = 111, n_jobs: int = 3) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    sa = es.score(str(resolve(a)), n_jobs=n_jobs, cpu_budget=False)
    sb = es.score(str(resolve(b)), n_jobs=n_jobs, cpu_budget=False)
    print(f"a {a}: rss {sa.rss:.4f}   b {b}: rss {sb.rss:.4f}")
    print(f"{'ep':>3} {'level':>5} {'J_naive':>9} {'J_clair':>9} {'J_a':>9} {'J_b':>9} {'a-b (bn)':>9} {'rss_a':>7} {'rss_b':>7}")
    diffs = []
    for ra, rb in zip(sa.rows, sb.rows):
        jn, jc = ra["J_naive_cents"] / 1e11, (ra["J_clairvoyant_cents"] or float("nan")) / 1e11
        ja, jb = ra["J_policy_cents"] / 1e11, rb["J_policy_cents"] / 1e11
        head = jn - jc
        diffs.append(ja - jb)
        print(f"{ra['episode']:3d} {ra['stratum']:5d} {jn:9.1f} {jc:9.1f} {ja:9.1f} {jb:9.1f} {ja - jb:9.2f} {(jn - ja) / head:7.3f} {(jn - jb) / head:7.3f}")
    print("episodes where a is worse than b:", int(np.sum(np.array(diffs) > 0)), "of", len(diffs))


if __name__ == "__main__":
    fire.Fire(main)
