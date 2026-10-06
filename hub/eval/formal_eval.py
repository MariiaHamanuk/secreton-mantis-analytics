"""Score an agent on the team's approved sets and print its row for hub/FORMAL_RESULTS.md.

    uv run python hub/eval/formal_eval.py agents/anastasiia_rules_fuelchip
    uv run python hub/eval/formal_eval.py lab/nazar/agents/mine --tuning --against=anastasiia_rules_fuelchip
    uv run python hub/eval/formal_eval.py agents/mine --record          # also keep its per-episode costs in git

In order: the scorer's checks of the zip (a refused zip stops here); the score on the formal sets of ``sets.py``
(Small root 222 x 256, Full root 222 x 128), with its 90% interval, the score per harm level and on the first 64 and
128 episodes; a timed episode of Small and of Full in the scoring container (``--nodocker``: in a plain process): the
median, the 95th percentile and the largest CPU seconds of a week.

The 90% interval is the package's: episodes are drawn again with replacement inside each harm level, 2,000 times, and
the interval runs from the 5th to the 95th percentile of the scores so obtained. It is the noise of the choice of
episodes: another set of the same size would land inside it nine times in ten.
``--tuning`` adds Small root 111 x 256. ``--against=<model>`` adds the paired difference with a model of
hub/FORMAL_RESULTS.md on the same episodes, from its kept costs when there are any, else by playing it.

No CPU budget is applied to the score: a strong model that is still too slow must show in the table. Whether it fits
the budget is what the container's seconds say.

Nothing is written to hub/FORMAL_RESULTS.md: the row is printed and its author pastes it. The whole run is kept under
``outputs/formal_eval/<model>/<date_time>/``. ``--record`` writes the model's cost per episode on the formal sets to
``hub/eval/records/<model>.json``: 256 + 128 numbers, so that a later model is compared with this one episode by
episode without playing this one again.
"""

import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import fire
import numpy as np
from sets import FORMAL, LEVEL_WEIGHTS, PREFIXES, TUNING

from sbf_starter import ROOT, scoring
from sbf_starter.agents import resolve


RECORDS = ROOT / "hub" / "eval" / "records"
KEYS = ("episode", "stratum", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents", "excluded")
HEADER = (
    "| модель | коміт | sha256 | Small 222 x256 | Full 222 x128 | Small 111 x64 / x128 / x256 "
    "| докер Small, с: медіана / p95 / макс | докер Full, с: медіана / p95 / макс | борд | дата |"
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _commit(folder: Path) -> str:
    """The commit the agent was scored at; ``+dirty`` when its folder holds uncommitted changes."""
    dirty = bool(_git("status", "--porcelain", "--", str(folder)))
    return (_git("rev-parse", "--short", "HEAD") or "none") + ("+dirty" if dirty else "")


def pooled(level: np.ndarray, saved: np.ndarray, room: np.ndarray) -> float:
    """The board's score: the levels' mean savings over their mean attainable savings, weighted by level."""
    present = [s for s in (1, 2, 3, 4) if (level == s).any()]
    w = {s: LEVEL_WEIGHTS[s - 1] for s in present}
    return float(
        sum(w[s] * saved[level == s].mean() for s in present) / sum(w[s] * room[level == s].mean() for s in present)
    )


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(harm level, the agent's saving, the attainable saving) per scored episode, USD cents."""
    rows = [r for r in rows if r["excluded"] is None]
    naive = np.array([r["J_naive_cents"] for r in rows], dtype=float)
    return (
        np.array([r["stratum"] for r in rows]),
        naive - np.array([r["J_policy_cents"] for r in rows], dtype=float),
        naive - np.array([r["J_clairvoyant_cents"] for r in rows], dtype=float),
    )


def paired(rows_a: list[dict], rows_b: list[dict], draws: int = 2000) -> tuple[float, float, float]:
    """a's score minus b's on the same episodes and its 90% interval (episodes resampled inside their harm level)."""
    level, saved_a, room = arrays(rows_a)
    _, saved_b, _ = arrays(rows_b)
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(level == s) for s in (1, 2, 3, 4) if (level == s).any()]
    diffs = np.empty(draws)
    for i in range(draws):
        pick = np.concatenate([rng.choice(g, len(g)) for g in groups])
        diffs[i] = pooled(level[pick], saved_a[pick], room[pick]) - pooled(level[pick], saved_b[pick], room[pick])
    diff = pooled(level, saved_a, room) - pooled(level, saved_b, room)
    return diff, float(np.quantile(diffs, 0.05)), float(np.quantile(diffs, 0.95))


def score_set(folder: Path, task: str, entropy: int, episodes: int, n_jobs: int) -> dict:
    """The agent's score on episodes 0 .. episodes - 1 of a root, with its rows and the score of each prefix."""
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    start = time.perf_counter()
    s = es.score(str(folder), n_jobs=n_jobs, cpu_budget=False)
    rows = [{k: r.get(k) for k in KEYS} for r in s.rows]
    level, saved, room = arrays(rows)
    return {
        "task": task,
        "entropy": entropy,
        "episodes": episodes,
        "rss": s.rss,
        "interval": s.interval,
        "by_level": {int(k): v for k, v in s.rss_by_stratum.items()},
        "prefix": {n: pooled(level[:n], saved[:n], room[:n]) for n in PREFIXES if n <= len(level)},
        "naive_weeks": s.fallback_weeks,
        "seconds": time.perf_counter() - start,
        "rows": rows,
    }


def timed(folder: Path, docker: bool) -> tuple[str, dict]:
    """(the zip's sha256, per task the CPU seconds of a timed dev episode); raises when the scorer refuses the zip."""
    from shockbench_flow_agent.submission import build_submission, check_zip, extract_submission

    from sbf_starter import check as checks
    from sbf_starter import container

    out = {}
    with tempfile.TemporaryDirectory(prefix="formal-eval-") as tmp:
        zip_path = Path(build_submission(folder, Path(tmp) / f"{folder.name}.zip"))
        sub = check_zip(zip_path)  # SubmissionError: the server would refuse it
        root = extract_submission(zip_path, Path(tmp) / "sub").root
        for line in checks.import_warnings(root):
            print("WARNING:", line)
        in_docker = docker and container.docker_available()
        if docker and not in_docker:
            print("NOTE: Docker is not running, the seconds below are this machine's, not the container's")
        for task in ("small", "full"):
            row = (container.play(root, task, 1, echo=lambda s: None) if in_docker else checks.timed_run(root, task))[0]
            if not row.get("imported", True):
                raise SystemExit(f"agent.py does not import with the server's packages:\n{row['stderr'][-800:]}")
            cpu = [c for c in row["cpu_s"] if c is not None]
            p95 = float(np.percentile(cpu, 95)) if cpu else 0.0
            out[task] = checks.summary(row) | {
                "p95_s": p95,
                "naive_weeks": len(row["substitutions"]),
                "docker": in_docker,
            }
        return sub.sha256, out


def fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def main(
    agent: str,
    tuning: bool = False,
    against: str | None = None,
    docker: bool = True,
    record: bool = False,
    small_episodes: int | None = None,
    full_episodes: int | None = None,
    n_jobs: int = 8,
) -> None:
    """Print the agent's formal scores, its seconds in the container and its row for hub/FORMAL_RESULTS.md.

    Args:
        agent: a folder with agent.py (or a name of agents/). The folder's name is the model's name.
        tuning: also score Small root 111 x 256, the tuning set.
        against: a model of hub/FORMAL_RESULTS.md to compare with on the same episodes (a paired interval).
        docker: time an episode in the scoring container (--nodocker: in a plain process on this machine).
        record: write the model's cost per episode on the formal sets to hub/eval/records/<model>.json.
        small_episodes: fewer episodes of the Small set, for a trial run: the row is then marked NOT FORMAL.
        full_episodes: fewer (or 256) episodes of the Full set; 0 skips Full. Other than 128 or 256: NOT FORMAL.
        n_jobs: workers.

    """
    folder = resolve(agent).resolve()
    model = folder.name
    load = os.getloadavg()[0]
    if load > 1.5 * (os.cpu_count() or 1):
        print(f"NOTE: the machine is busy (load {load:.0f}): the seconds below are inflated, the scores are not")
    sha, seconds = timed(folder, docker)
    counts = {"small": small_episodes or FORMAL["small"]["episodes"], "full": FORMAL["full"]["episodes"]}
    if full_episodes is not None:
        counts["full"] = full_episodes
    formal = counts["small"] == FORMAL["small"]["episodes"] and counts["full"] in (FORMAL["full"]["episodes"], 256)
    results = {}
    for task in ("small", "full"):
        if counts[task]:
            results[task] = score_set(folder, task, FORMAL[task]["entropy"], counts[task], n_jobs)
    if tuning:
        t = TUNING["small"]
        results["tuning"] = score_set(folder, "small", t["entropy"], t["episodes"], n_jobs)

    print(f"\n{model}: commit {_commit(folder)}, zip sha256 {sha[:8]}" + ("" if formal else "   ** NOT FORMAL **"))
    for name, r in results.items():
        lo, hi = r["interval"] or (None, None)
        levels = " / ".join(fmt(r["by_level"].get(s)) for s in (1, 2, 3, 4))
        firsts = ", ".join(f"first {n}: {v:.3f}" for n, v in r["prefix"].items() if n < r["episodes"]) or "no prefix"
        print(
            f"  {name:6s} {r['task']} root {r['entropy']} x{r['episodes']}: {r['rss']:.4f} ({fmt(lo)} to {fmt(hi)}); "
            f"by level {levels}; {firsts}; weeks played by naive {r['naive_weeks']}; {r['seconds']:.0f} s"
        )
    where = (
        "the scoring container" if seconds["small"]["docker"] else f"a process on this machine ({platform.machine()})"
    )
    for task, s in seconds.items():
        budget = {"small": 2, "full": 4}[task]
        fits = "fits" if s["max_s"] <= budget and not s["naive_weeks"] else "OVER"
        print(
            f"  seconds per week in {where}, {task}: week 1 {s['week1_s']:.2f}, median {s['median_s']:.2f}, "
            f"95th percentile {s['p95_s']:.2f}, max {s['max_s']:.2f} (budget {budget} s: {fits}); "
            f"weeks the scorer would give to naive {s['naive_weeks']}"
        )

    if against:
        other = resolve(against).resolve() if not (RECORDS / f"{against}.json").is_file() else None
        kept = json.loads((RECORDS / f"{against}.json").read_text()) if other is None else None
        for task in ("small", "full"):
            if task not in results:
                continue
            mine = results[task]
            key = f"{task}-{mine['entropy']}"
            if kept and len(kept["sets"].get(key, {}).get("rows", [])) >= mine["episodes"]:
                theirs = kept["sets"][key]["rows"][: mine["episodes"]]
            else:
                source = other or resolve(against).resolve()
                theirs = score_set(source, task, mine["entropy"], mine["episodes"], n_jobs)["rows"]
            diff, lo, hi = paired(mine["rows"], theirs)
            print(f"  against {against}, {task} x{mine['episodes']}: {diff:+.4f} ({lo:+.4f} to {hi:+.4f})")

    def cell(name: str) -> str:
        r = results.get(name)
        if r is None:
            return "—"
        lo, hi = r["interval"] or (None, None)
        return f"{r['rss']:.3f} ({fmt(lo)}–{fmt(hi)})"

    tune = results.get("tuning")
    tune_cell = " / ".join(fmt(tune["prefix"].get(n)) for n in PREFIXES) if tune else "—"
    times = [
        f"{seconds[t]['median_s']:.2f} / {seconds[t]['p95_s']:.2f} / {seconds[t]['max_s']:.2f}"
        for t in ("small", "full")
    ]
    row = (
        f"| {model} | {_commit(folder)} | {sha[:8]} | {cell('small')} | {cell('full')} | {tune_cell} "
        f"| {times[0]} | {times[1]} | | {time.strftime('%Y-%m-%d')} |"
    )
    print("\nThe row for hub/FORMAL_RESULTS.md" + ("" if formal else " (NOT FORMAL: a trial run, do not paste)") + ":")
    print(HEADER)
    print(row)

    out = ROOT / "outputs" / "formal_eval" / model / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    full = {"model": model, "commit": _commit(folder), "sha256": sha, "seconds": seconds, "results": results}
    (out / "record.json").write_text(json.dumps(full, indent=1, default=str))
    print(f"\nwritten {out.relative_to(ROOT)}/record.json")
    if record:
        if not formal:
            raise SystemExit("--record keeps formal runs only (the approved counts of episodes)")
        sets = {f"{r['task']}-{r['entropy']}": {"episodes": r["episodes"], "rows": r["rows"]} for r in results.values()}
        kept = {
            "model": model,
            "commit": _commit(folder),
            "sha256": sha,
            "date": time.strftime("%Y-%m-%d"),
            "sets": sets,
        }
        RECORDS.mkdir(parents=True, exist_ok=True)
        (RECORDS / f"{model}.json").write_text(json.dumps(kept, indent=1) + "\n")
        print(f"written {(RECORDS / f'{model}.json').relative_to(ROOT)} (commit it with the row)")


if __name__ == "__main__":
    fire.Fire(main)
