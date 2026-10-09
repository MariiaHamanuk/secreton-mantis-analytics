"""An agent folder played in the scoring container on a machine like the bench's, several episodes side by side.

    uv run python lab/anastasiia/regime_lab/bench_vm.py run agents/anastasiia_plan_hull3 --tag=hull3 --task=small \
        --entropy=111 --episodes=64 --side=8
    uv run python lab/anastasiia/regime_lab/bench_vm.py show hull3 --task=small --entropy=111 --episodes=64

``run`` plays each episode with the package's ``play_container`` (the scoring image, the server's caps of one CPU
and 4 GB, its CPU meter read from Docker), ``side`` episodes at a time, each from a process of its own, and keeps
what the meter read in ``outputs/regime_lab/bench/<tag>_<task>_<entropy>.json``: every week's CPU seconds, the weeks
over the budget (the server plays them by the naive rule), the other substitutions, the start-up and the wall
clock. ``filler`` more episodes are played after the asked ones, so that the last of them still have neighbours;
``show`` reads the first ``episodes``. The container gives no cost: the score under the meter is ``budget.py``'s.
``budget`` other than the task's stands for a server that much slower than this machine; the folder then has to
carry the same share in its clock (``budget_scale``).

The bench (the organisers, 9 October): Google Cloud c3d-standard-16, AMD EPYC 9B14, 16 vCPU as 8 cores of 2 threads,
15 episodes side by side, each in a container of 1 CPU and 4 GB. On a c3d-standard-8 (4 cores of 2 threads) ``side``
7 or 8 keeps the other thread of a core as busy.
"""

import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab")]
import plan  # noqa: E402


OUT = plan.OUT / "bench"


def _path(tag: str, task: str, entropy: int) -> Path:
    return OUT / f"{tag}_{task}_{entropy}.json"


def _one(folder: str, image: str, task: str, episode: int, entropy: int, budget: float) -> dict:
    from shockbench_flow_agent import play_container
    from shockbench_flow_agent.isolated import LIMITS

    limits = dataclasses.replace(LIMITS, cpu_budget_s={**LIMITS.cpu_budget_s, task: budget}) if budget else LIMITS
    row = play_container(folder, image, task, episode, entropy=entropy, limits=limits)
    row["stderr"] = row["stderr"][-2000:] if row.get("stderr") else ""
    return row


def run(
    agent: str,
    tag: str,
    task: str = "small",
    entropy: int = 111,
    episodes: int = 64,
    side: int = 8,
    filler: int = -1,
    budget: float = 0.0,
) -> None:
    from shockbench_flow_agent import build_image

    from sbf_starter.agents import resolve

    folder = str(resolve(agent).resolve())
    image = build_image()
    filler = side - 1 if filler < 0 else filler
    OUT.mkdir(parents=True, exist_ok=True)
    rows, start = {}, time.time()
    with ProcessPoolExecutor(max_workers=side) as pool:
        jobs = {pool.submit(_one, folder, image, task, n, entropy, budget): n for n in range(episodes + filler)}
        for job in as_completed(jobs):
            n = jobs[job]
            try:
                rows[n] = job.result()
            except Exception as error:  # the episode is kept as failed, the others go on
                rows[n] = {"episode": n, "error": repr(error)}
            over = len(rows[n].get("over_budget", []))
            print(f"episode {n}: weeks over the budget {over}, {time.time() - start:.0f} s so far", flush=True)
            kept = {
                "agent": agent,
                "task": task,
                "entropy": entropy,
                "side": side,
                "episodes": episodes,
                "filler": filler,
                "rows": [rows[k] for k in sorted(rows)],
            }
            _path(tag, task, entropy).write_text(json.dumps(kept))
    show(tag, task=task, entropy=entropy, episodes=episodes)


def show(tag: str, task: str = "small", entropy: int = 111, episodes: int = 64, first: int = 0) -> None:
    kept = json.loads(_path(tag, task, entropy).read_text())
    rows = [r for r in kept["rows"] if first <= r["episode"] < first + episodes]
    failed = [r["episode"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    budget = rows[0]["cpu_budget_s"]
    weeks = sum(r["weeks"] for r in rows)
    over = [len(r["over_budget"]) for r in rows]
    causes: dict[str, int] = {}
    for r in rows:
        for _, code in r["substitutions"]:
            causes[code] = causes.get(code, 0) + 1
    cpu = np.array([s for r in rows for s in r["cpu_s"] if s is not None])
    week1 = np.array([r["cpu_s"][0] for r in rows if r["cpu_s"] and r["cpu_s"][0] is not None])
    unread = sum(1 for r in rows for s in r["cpu_s"] if s is None)
    print(
        f"{tag}: {task}, root {entropy}, {len(rows)} episodes of {kept['side']} side by side, {weeks} weeks, "
        f"a budget of {budget:g} CPU s a week"
    )
    print(
        f"  weeks over the budget: {sum(over)} ({100 * sum(over) / weeks:.2f} %), in {sum(1 for o in over if o)} "
        f"episodes; most in one episode: {sorted(over, reverse=True)[:8]}"
    )
    print(f"  substitutions by cause: {causes or 'none'}")
    print(
        f"  CPU seconds a week: median {np.median(cpu):.2f}, p95 {np.percentile(cpu, 95):.2f}, "
        f"p99 {np.percentile(cpu, 99):.2f}, max {cpu.max():.2f}; week 1: median {np.median(week1):.2f}, "
        f"max {week1.max():.2f}"
    )
    print(
        f"  the meter's reads that failed: {unread}; episodes cut by the kill timer: "
        f"{sum(1 for r in rows if r['watchdog_fired'])}; containers gone: "
        f"{sum(1 for r in rows if r['container_gone'])}; episodes that failed to play: {failed or 'none'}"
    )
    print(
        f"  wall clock of an episode: median {np.median([r['wall_s'] for r in rows]):.0f} s, "
        f"start-up: median {np.median([r['ready_s'] for r in rows if r['ready_s'] is not None]):.1f} s"
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
