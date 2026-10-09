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

A run may be stopped at any moment (Ctrl-C, ``kill``, a closed terminal, the machine switched off) and the same
command goes on from where it stopped: the zip is built once and kept, every episode's row is written the moment the
episode ends, and the container's seconds are kept too, all under
``outputs/formal_eval/<model>/progress_<the first 8 of the zip's sha256>/``. Only the episodes that were being played
are played again. A changed agent folder has another sha256 and starts its own progress. An episode during which the
machine slept (a closed lid) is played again as well: the solvers' time limits are wall clock, so such an episode is
not what the agent plays.

Nothing is written to hub/FORMAL_RESULTS.md: the row is printed and its author pastes it. The whole run is kept under
``outputs/formal_eval/<model>/<date_time>/``. ``--record`` writes the model's cost per episode on the formal sets to
``hub/eval/records/<model>.json``: 256 + 128 numbers, so that a later model is compared with this one episode by
episode without playing this one again.
"""

import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import fire
import numpy as np
from sets import FORMAL, LEVEL_WEIGHTS, PREFIXES, TUNING

from sbf_starter import ROOT, scoring
from sbf_starter.agents import resolve


RECORDS = ROOT / "hub" / "eval" / "records"
PROGRESS = ROOT / "outputs" / "formal_eval"
ASLEEP_S = 2.0  # the wall clock ahead of the steady one by more than this: the machine slept during the episode
KEYS = ("episode", "stratum", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents", "excluded")
HEADER = (
    "| модель | коміт | sha256 | Small 222 x256 | Full 222 x128 | Small 111 x64 / x128 / x256 "
    "| докер Small, с: медіана / p95 / макс | докер Full, с: медіана / p95 / макс | борд | дата |"
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _commit(folder: Path, sha: str | None = None) -> str:
    """The commit the agent was scored at; ``+dirty`` when its folder holds uncommitted changes or is not in git.

    ``sha``: the sha256 of the zip that was scored. A run may last hours and be stopped and started again, so the
    folder is zipped once more here: ``+changed`` when it is no longer what was scored.
    """
    dirty = bool(_git("status", "--porcelain", "--", str(folder))) or not _git("ls-files", "--", str(folder))
    mark = (_git("rev-parse", "--short", "HEAD") or "none") + ("+dirty" if dirty else "")
    if sha is not None:
        from shockbench_flow_agent.submission import build_submission, check_zip

        with tempfile.TemporaryDirectory(prefix="formal-eval-") as tmp:
            now = check_zip(Path(build_submission(folder, Path(tmp) / f"{folder.name}.zip"))).sha256
        mark += "" if now == sha else "+changed"
    return mark


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


def _shown(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _write(path: Path, value: dict) -> None:
    """Whole or not at all: a run killed while writing leaves the file as it was."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=int))
    os.replace(tmp, path)


def _worker() -> None:
    """A worker leaves Ctrl-C to the parent, which stops the pool, and does not outlive a parent that was killed."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(5)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def _play_one(job: tuple) -> dict:
    """One episode's row as the scorer plays it; when it was played, under what load, and how long the machine slept."""
    zip_path, task, entropy, n = job
    wall, steady = time.time(), time.monotonic()
    es = scoring.episode_set(task, [n], entropy=entropy, n_jobs=1, verbose=False)
    row = es.play(zip_path, cpu_budget=False, n_jobs=1)[0]
    row["started"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall))
    row["load"] = round(os.getloadavg()[0], 1)
    row["asleep_s"] = round((time.time() - wall) - (time.monotonic() - steady), 3)
    return row


def score_set(zip_path: Path, progress: Path, task: str, entropy: int, episodes: int, n_jobs: int) -> dict:
    """The agent's score on episodes 0 .. episodes - 1 of a root, with its rows and the score of each prefix.

    The episodes are played one at a time in ``n_jobs`` processes, and each row is written under ``progress`` when its
    episode ends: a run stopped at any moment plays only the episodes it has no row of. The score, its interval and
    the levels are the package's own, reckoned from the rows as ``EpisodeSet.score`` reckons them.
    """
    from shockbench_flow_agent.scoring import LEVEL, N_BOOT, _boot_stats

    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    kept = progress / f"{task}-{entropy}"
    kept.mkdir(parents=True, exist_ok=True)
    played = {n: json.loads((kept / f"{n}.json").read_text()) for n in es.episodes if (kept / f"{n}.json").is_file()}
    todo = [n for n in es.episodes if n not in played]
    name = f"{task} root {entropy}"
    print(f"{name}: {len(played)} of {len(es.episodes)} episodes kept, {len(todo)} to play", flush=True)
    start, took = time.perf_counter(), []
    try:
        while todo:
            again = []
            workers = min(n_jobs, len(todo))
            with multiprocessing.get_context("spawn").Pool(workers, initializer=_worker) as pool:
                jobs = [(str(zip_path), task, entropy, n) for n in todo]
                for row in pool.imap_unordered(_play_one, jobs):
                    n = row["episode"]
                    if row["asleep_s"] > ASLEEP_S:
                        again.append(n)
                        print(
                            f"{name}: episode {n} is played again, the machine slept {row['asleep_s']:.0f} s",
                            flush=True,
                        )
                        continue
                    _write(kept / f"{n}.json", row)
                    played[n] = row
                    took.append(row["seconds"])
                    left = (len(es.episodes) - len(played)) * float(np.mean(took)) / workers
                    print(
                        f"{name}: {len(played)} of {len(es.episodes)} (episode {n}: {row['seconds']:.0f} s), "
                        f"{(time.perf_counter() - start) / 60:.0f} min so far, about {left / 60:.0f} min left",
                        flush=True,
                    )
            todo = again
    except KeyboardInterrupt:
        print(
            f"\n{name}: stopped with {len(played)} of {len(es.episodes)} episodes kept under "
            f"{_shown(progress)}: the same command goes on from here"
        )
        raise SystemExit(130) from None
    played_rows = [played[n] for n in es.episodes]
    for ref, row in zip(es.references, played_rows):
        if ref["omega_hash"] != row["omega_hash"]:
            raise SystemExit(f"episode {ref['episode']} under {kept} was played on another scenario: remove {progress}")
    costs = [r["J_policy_cents"] for r in played_rows]
    (boot,) = _boot_stats(es.references, [costs], es.rss(costs)["pooled"], N_BOOT, 0)
    s = es._score(str(zip_path), played_rows, None, LEVEL, boot)
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
        "seconds": float(sum(r["seconds"] for r in played_rows)),  # of the episodes, one after another
        "rows": rows,
    }


def packed(folder: Path) -> tuple[str, Path, Path]:
    """(the zip's sha256, the zip every episode is played from, the folder of the run's progress).

    The zip is built once and kept beside the progress, so that a run stopped and started again plays the same bytes
    with the same seeds whatever has happened to the folder since; the scorer refusing the zip stops here.
    """
    import shockbench_flow
    from shockbench_flow_agent.submission import build_submission, check_zip

    with tempfile.TemporaryDirectory(prefix="formal-eval-") as tmp:
        built = Path(build_submission(folder, Path(tmp) / f"{folder.name}.zip"))
        sha = check_zip(built).sha256  # SubmissionError: the server would refuse it
        progress = PROGRESS / folder.name / f"progress_{sha[:8]}"
        progress.mkdir(parents=True, exist_ok=True)
        zip_path = progress / built.name
        if not zip_path.is_file():
            shutil.copy2(built, progress / (built.name + ".tmp"))
            os.replace(progress / (built.name + ".tmp"), zip_path)
    if hashlib.sha256(zip_path.read_bytes()).hexdigest() != sha:
        raise SystemExit(f"{zip_path} is not the zip of {folder}: remove {progress}")
    about = {"package": shockbench_flow.__version__, "python": platform.python_version(), "sha256": sha}
    if (progress / "about.json").is_file():
        was = json.loads((progress / "about.json").read_text())
        if was != about:  # other scenarios or other references: the rows kept are of another benchmark
            raise SystemExit(
                f"{progress} was started with {was}, this run is {about}: remove the folder to start again"
            )
    else:
        _write(progress / "about.json", about)
    return sha, zip_path, progress


def timed(zip_path: Path, docker: bool) -> dict:
    """Per task the CPU seconds of a timed dev episode of the zip, in the scoring container when ``docker``."""
    from shockbench_flow_agent.submission import extract_submission

    from sbf_starter import check as checks
    from sbf_starter import container

    out = {}
    with tempfile.TemporaryDirectory(prefix="formal-eval-") as tmp:
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
    return out


def _stop(signum: int, frame: object) -> None:
    """``kill`` and a closed terminal stop the run as Ctrl-C does: the pool is taken down, the rows written stay."""
    raise KeyboardInterrupt


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
        n_jobs: workers. A model that watches its own clock wants no more of them than the machine has fast cores,
            and nothing else running: a slow week changes what it plays.

    """
    folder = resolve(agent).resolve()
    model = folder.name
    load = os.getloadavg()[0]
    if load > 1.5 * (os.cpu_count() or 1):
        print(
            f"NOTE: the machine is busy (load {load:.0f}): the seconds below are inflated, and so is the score of a "
            "model that watches its own clock"
        )
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGHUP, _stop)
    if platform.system() == "Darwin" and shutil.which("caffeinate"):  # no idle sleep while this process lives
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    sha, zip_path, progress = packed(folder)
    print(f"{model}: zip sha256 {sha[:8]}, progress under {_shown(progress)}", flush=True)
    seconds = json.loads((progress / "seconds.json").read_text()) if (progress / "seconds.json").is_file() else None
    if seconds is None or (docker and not seconds["small"]["docker"]):
        try:
            seconds = timed(zip_path, docker)
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        _write(progress / "seconds.json", seconds)
    else:
        print("seconds per week: kept from the run this one goes on with", flush=True)
    counts = {"small": small_episodes or FORMAL["small"]["episodes"], "full": FORMAL["full"]["episodes"]}
    if full_episodes is not None:
        counts["full"] = full_episodes
    formal = counts["small"] == FORMAL["small"]["episodes"] and counts["full"] in (FORMAL["full"]["episodes"], 256)
    results = {}
    for task in ("small", "full"):
        if counts[task]:
            results[task] = score_set(zip_path, progress, task, FORMAL[task]["entropy"], counts[task], n_jobs)
    if tuning:
        t = TUNING["small"]
        results["tuning"] = score_set(zip_path, progress, "small", t["entropy"], t["episodes"], n_jobs)

    commit = _commit(folder, sha)
    print(f"\n{model}: commit {commit}, zip sha256 {sha[:8]}" + ("" if formal else "   ** NOT FORMAL **"))
    for name, r in results.items():
        lo, hi = r["interval"] or (None, None)
        levels = " / ".join(fmt(r["by_level"].get(s)) for s in (1, 2, 3, 4))
        firsts = ", ".join(f"first {n}: {v:.3f}" for n, v in r["prefix"].items() if n < r["episodes"]) or "no prefix"
        print(
            f"  {name:6s} {r['task']} root {r['entropy']} x{r['episodes']}: {r['rss']:.4f} ({fmt(lo)} to {fmt(hi)}); "
            f"by level {levels}; {firsts}; weeks played by naive {r['naive_weeks']}; "
            f"{r['seconds']:.0f} s of play"
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
                _, their_zip, their_progress = packed(other or resolve(against).resolve())
                theirs = score_set(their_zip, their_progress, task, mine["entropy"], mine["episodes"], n_jobs)["rows"]
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
        f"| {model} | {commit} | {sha[:8]} | {cell('small')} | {cell('full')} | {tune_cell} "
        f"| {times[0]} | {times[1]} | | {time.strftime('%Y-%m-%d')} |"
    )
    print("\nThe row for hub/FORMAL_RESULTS.md" + ("" if formal else " (NOT FORMAL: a trial run, do not paste)") + ":")
    print(HEADER)
    print(row)

    out = ROOT / "outputs" / "formal_eval" / model / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    full = {"model": model, "commit": commit, "sha256": sha, "seconds": seconds, "results": results}
    (out / "record.json").write_text(json.dumps(full, indent=1, default=str))
    print(f"\nwritten {out.relative_to(ROOT)}/record.json")
    if record:
        if not formal:
            raise SystemExit("--record keeps formal runs only (the approved counts of episodes)")
        sets = {f"{r['task']}-{r['entropy']}": {"episodes": r["episodes"], "rows": r["rows"]} for r in results.values()}
        kept = {
            "model": model,
            "commit": commit,
            "sha256": sha,
            "date": time.strftime("%Y-%m-%d"),
            "sets": sets,
        }
        RECORDS.mkdir(parents=True, exist_ok=True)
        (RECORDS / f"{model}.json").write_text(json.dumps(kept, indent=1) + "\n")
        print(f"written {(RECORDS / f'{model}.json').relative_to(ROOT)} (commit it with the row)")


if __name__ == "__main__":
    fire.Fire(main)
