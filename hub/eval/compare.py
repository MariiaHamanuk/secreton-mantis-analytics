"""Paired comparison of agent folders on cached episodes, one line per agent (the board's score and its 90% interval).

    uv run python hub/eval/compare.py --base=agents/anastasiia_rules_fuelchip --episodes=64 path/to/agentA path/to/B

The difference and its interval are paired (the same episodes); an interval that holds 0 cannot tell the two apart.
"""

import fire


def main(
    *agents: str, base: str = "template", task: str = "small", episodes: int = 64, entropy: int = 111, n_jobs: int = 2
) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    b = es.score(str(resolve(base)), n_jobs=n_jobs, cpu_budget=False)

    def levels(score):
        return " ".join("-" if v is None else f"{v:.3f}" for v in score.rss_by_stratum.values())

    lv = levels(b)
    print(f"{task}, {episodes} episodes of root {entropy}")
    print(
        f"{'score':>7} {'vs base':>9} {'90% interval of the difference':>32} "
        f"{'by harm level 1..4':>26} {'naive weeks':>11}  agent"
    )
    print(f"{b.rss:7.4f} {'':>9} {'':>32} {lv:>26} {b.fallback_weeks:>11}  {base} (base)", flush=True)
    for a in agents:
        cmp = es.compare(str(resolve(a)), str(resolve(base)), n_jobs=n_jobs, cpu_budget=False)
        lo, hi = cmp.interval or (float("nan"), float("nan"))
        lv = levels(cmp.a)
        print(
            f"{cmp.a.rss:7.4f} {cmp.diff:+9.4f} {f'{lo:+.4f} to {hi:+.4f}':>32} {lv:>26} "
            f"{cmp.a.fallback_weeks:>11}  {a}",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
