"""Score an agent on many episodes in batches, with progress after every batch (as the private board would: Full, 400).

    uv run python lab/nazar/mpc/full_run.py lab/nazar/agents/nazar_rules_lpraw_v2 --task=full --episodes=400 --batch=8

Each batch is ``batch`` consecutive episodes of the root, played by the agent and by the base agent on the same episodes
(``cpu_budget`` on: a week over the task's CPU budget is played by the naive rule, as the server does). After each batch:
the episodes done, the cumulative score of both, their paired difference and the weeks the scorer gave to naive. The rows
are appended to ``outputs/full_run/<date_time>/rows.jsonl``; ``--resume=<that folder>`` continues a stopped run.
References of episodes not in the cache are computed on first use (about a minute of CPU each on Full).
"""

import json
import sys
import time
from pathlib import Path

import fire


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "hub" / "eval"))
from formal_eval import arrays, paired, pooled  # noqa: E402


KEYS = ("episode", "stratum", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents", "excluded")


def rss(rows: list[dict]) -> float:
    level, saved, room = arrays(rows)
    return pooled(level, saved, room)


def main(
    agent: str,
    base: str = "agents/anastasiia_rules_v2",
    task: str = "full",
    episodes: int = 400,
    entropy: int = 222,
    batch: int = 8,
    n_jobs: int = 8,
    resume: str | None = None,
) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    out = Path(resume) if resume else ROOT / "outputs" / "full_run" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "rows.jsonl"
    rows = {"agent": [], "base": []}
    if log.is_file():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            rows[d["who"]].append(d["row"])
    done = {r["episode"] for r in rows["agent"]} & {r["episode"] for r in rows["base"]}
    naive = {"agent": 0, "base": 0}
    folders = {"agent": str(resolve(agent)), "base": str(resolve(base))}
    print(
        f"{Path(agent).name} vs {Path(base).name}: {task} root {entropy}, {episodes} episodes in batches of {batch}; log {out}",
        flush=True,
    )
    start = time.perf_counter()
    for lo in range(0, episodes, batch):
        ids = [i for i in range(lo, min(lo + batch, episodes)) if i not in done]
        if not ids:
            continue
        t0 = time.perf_counter()
        es = scoring.episode_set(task, ids, entropy=entropy, n_jobs=n_jobs, verbose=False)
        for who in ("agent", "base"):
            s = es.score(folders[who], n_jobs=n_jobs, cpu_budget=True)
            naive[who] += int(s.fallback_weeks or 0)
            with log.open("a") as f:
                for r in s.rows:
                    row = {k: r.get(k) for k in KEYS}
                    rows[who].append(row)
                    f.write(json.dumps({"who": who, "row": row}) + "\n")
        n = len({r["episode"] for r in rows["agent"]})
        a, b = rss(rows["agent"]), rss(rows["base"])
        elapsed = time.perf_counter() - start
        left = elapsed / max(n - len(done), 1) * (episodes - n)
        print(
            f"[{n:>3}/{episodes}] episodes {ids[0]}..{ids[-1]} in {time.perf_counter() - t0:4.0f} s | "
            f"cumulative: agent {a:.4f}, base {b:.4f}, diff {a - b:+.4f} | naive weeks: agent {naive['agent']}, base {naive['base']} | "
            f"~{left / 60:.0f} min left",
            flush=True,
        )
    d, lo_, hi_ = paired(rows["agent"], rows["base"])
    level = arrays(rows["agent"])[0]
    print(
        f"\nFINAL {len(rows['agent'])} episodes: agent {rss(rows['agent']):.4f}, base {rss(rows['base']):.4f}; paired diff {d:+.4f} (90% {lo_:+.4f} to {hi_:+.4f}); episodes per harm level {[int((level == s).sum()) for s in (1, 2, 3, 4)]}",
        flush=True,
    )


if __name__ == "__main__":
    fire.Fire(main)

# ruff: noqa: E501 (a diagnostic script: long lines in docstrings and tables are kept as written)
