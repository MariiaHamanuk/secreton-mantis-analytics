"""Score an agent and append the result to the team's table, team/results.csv.

    uv run python team/log_result.py heuristic --task=small --note="the kit's heuristic, as shipped"
    uv run python team/log_result.py mine --task=small --entropy=111 --episodes=64 --note="tuning root"
    uv run python team/log_result.py mine --board=0.4321 --note="public board, submission 123456"

One row per run. The references come from the shared cache (team/refcache), so everyone's scores share the same naive
and clairvoyant costs. ``--board`` records a score the server gave instead of running anything.
"""

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import fire

from sbf_starter import ROOT, scoring
from sbf_starter.agents import resolve


TABLE = ROOT / "team" / "results.csv"
COLUMNS = (
    "date_utc",
    "who",
    "agent",
    "commit",
    "source",  # local, or board (the server's score)
    "task",
    "episodes",  # dev, or entropy-<root>x<count>
    "score",
    "lo90",
    "hi90",
    "level1",
    "level2",
    "level3",
    "level4",
    "cost_musd",  # mean cost per episode, millions of USD
    "fallback_weeks",
    "note",
)


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return done.stdout.strip()


def _commit(folder: Path) -> str:
    """The commit the agent was scored at; ``+dirty`` when its folder holds uncommitted changes."""
    dirty = bool(_git("status", "--porcelain", "--", str(folder)))
    return (_git("rev-parse", "--short", "HEAD") or "none") + ("+dirty" if dirty else "")


def _num(x: float | None, digits: int = 4) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def append(row: dict) -> None:
    new = not TABLE.is_file()
    with TABLE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(
    agent: str,
    task: str = "small",
    episodes: str | int = "dev",
    entropy: int = 0,
    cpu_budget: bool = True,
    board: float | None = None,
    note: str = "",
    n_jobs: int = -1,
) -> None:
    """Append one row to team/results.csv.

    Args:
        agent: an agent's name (a folder of agents/) or a submission folder.
        task: tiny, small (the public board's) or full.
        episodes: dev (the 20 public dev episodes) or a count k of a root of your own (with ``entropy``).
        entropy: 0 for the dev episodes; any other integer for a tuning root.
        cpu_budget: a week over the task's CPU budget is played by the naive rule, as on the server.
        board: a score the server gave: recorded as it is, nothing runs.
        note: what changed, in a few words.
        n_jobs: workers of a first run's reference computation (-1: all cores).

    """
    folder = resolve(agent)
    row = dict.fromkeys(COLUMNS, "")
    row |= {
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "who": _git("config", "user.name"),
        "agent": folder.name,
        "commit": _commit(folder),
        "task": task,
        "note": note,
    }
    if board is not None:
        row |= {"source": "board", "episodes": "private", "score": _num(float(board))}
    else:
        result = scoring.evaluate(folder, task, episodes, entropy=entropy, cpu_budget=cpu_budget, n_jobs=n_jobs)
        print(result)
        lo, hi = result.interval or (None, None)
        levels = result.rss_by_stratum if result.pooled else {}
        row |= {
            "source": "local",
            "episodes": "dev" if (episodes, entropy) == ("dev", 0) else f"entropy-{entropy}x{result.episodes}",
            "score": _num(result.rss),
            "lo90": _num(lo),
            "hi90": _num(hi),
            "cost_musd": _num(result.cost_usd / 1e6, 1),
            "fallback_weeks": result.fallback_weeks,
        }
        row |= {f"level{s}": _num(levels.get(s)) for s in (1, 2, 3, 4)}
    append(row)
    print(f"appended to {TABLE.relative_to(ROOT)}: {row['agent']} on {task}, score {row['score']}")


if __name__ == "__main__":
    fire.Fire(main)
