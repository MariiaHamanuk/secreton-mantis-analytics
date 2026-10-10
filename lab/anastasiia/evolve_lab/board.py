"""A model played as the board plays it, as near as a rented machine allows, with its cost and its clock's journal.

    uv run python lab/anastasiia/evolve_lab/board.py run outputs/evolve_lab/agents/hazard3_j --tag=hazard3 --task=full \
        --entropy=111 --episodes=64 --side=15
    uv run python lab/anastasiia/evolve_lab/board.py show hazard3 --task=full --entropy=111 --episodes=64

The bench (the organisers, 9 October): Google Cloud c3d-standard-16, AMD EPYC 9B14, 16 vCPU as 8 cores of 2 threads,
15 episodes side by side, each in a container of 1 CPU and 4 GB, the CPU of the whole container counted, 2 s a week
on Small and 4 s on Full, a week over the budget played by the naive rule. ``run`` does the same on such a machine:
every episode in the package's scoring container with the server's caps and its meter (as
``shockbench_flow_agent.isolated.play_container``), ``side`` episodes at a time, each from a process of its own. Two
things are added to the package's own run. The environment has the scorer's naive fallback (the package's local
container run plays an empty action in a week the meter hands over), so the episode's cost is the one the board
would count. And the agent's whole stderr is kept: a folder assembled with ``assemble.py --journal`` writes one
line a week there, what its clock chose and every run of the solver, which ``show`` counts.

``run`` keeps per episode: the cost, each week's CPU seconds as the meter read them, the weeks over the budget, the
substitutions by cause, the journal. It also writes the costs to ``outputs/evolve_lab/costs/<tag>_<task>_<root>.json``
in ``evolve.py``'s form, so that ``evolve.py score`` pairs two such runs (a budget twice the board's against the
board's: what the week's time costs). ``budget`` other than the task's needs the same share in the model's clock
(``assemble.py --params`` with ``budget_scale``); ``timeout`` replaces the episode's kill timer (a longer budget
needs a longer one).
"""

import dataclasses
import json
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "evolve_lab"
sys.path[:0] = [str(HERE)]


def _one(
    folder: str,
    image: str,
    task: str,
    episode: int,
    entropy: int,
    budget: float,
    timeout: float,
    fq_replications: int,
    cache: str | None,
) -> dict:
    from shockbench_flow.dynamics.env import Env, took_fallback
    from shockbench_flow.hosting import metering
    from shockbench_flow.hosting.docker import DockerTransport
    from shockbench_flow.information.runner import play_wire_episode
    from shockbench_flow.information.wire import WireLimits
    from shockbench_flow_agent import isolated, scoring
    from shockbench_flow_agent.local_eval import SPLIT, check_local_run

    import sbf_starter  # noqa: F401 - points the package at the team's reference cache

    limits = isolated.LIMITS
    if budget:
        limits = dataclasses.replace(limits, cpu_budget_s={**limits.cpu_budget_s, task: float(budget)})
    check_local_run(SPLIT, entropy)
    spend = limits.cpu_budget_s[task]
    with tempfile.TemporaryDirectory(prefix="sbf-board-") as tmp:
        work = Path(tmp)
        checked = isolated._prepare(folder, work, limits)
        inst, omega, marks, fallback = scoring._world(task, entropy, episode, fq_replications, cache)
        seed = scoring._policy_seed(entropy, episode, checked.sha256)
        timer = float(timeout) if timeout else float(limits.episode_timeout_s[task])
        cid, errlog = work / "cid", work / "stderr.txt"

        def resolve():
            return metering.cpu_reader("docker_stats", metering.read_cidfile(cid), socket_path=isolated.docker_socket())

        meter = metering.WeekCpu(resolve, budget_s=spend, poll_s=metering.POLL_S["docker_stats"])
        start = time.perf_counter()
        with errlog.open("wb") as err:
            transport = DockerTransport(
                image, checked.clean_dir, deadline_s=limits.deadline_s, startup_s=limits.startup_s,
                max_reply_bytes=limits.max_reply_bytes, limits=limits.container, docker=("docker",),
                stderr=err.fileno(), episode_timeout_s=timer, cidfile=cid,
            )  # fmt: skip
            transport.interrupt, transport.interrupt_poll_s = meter.interrupt, meter.poll_s
            wired = play_wire_episode(
                Env(fallback=fallback), inst, transport,
                limits=WireLimits(max_reply_bytes=limits.max_reply_bytes, bank_seconds=limits.bank_seconds),
                regime="standard", omega=omega, policy_seed=seed, marks=marks, policy_name="submission", meter=meter,
            )  # fmt: skip
        text = errlog.read_text(errors="replace") if errlog.exists() else ""
        journal = [json.loads(line[8:]) for line in text.splitlines() if line.startswith("JOURNAL ")]
        other = [line for line in text.splitlines() if not line.startswith("JOURNAL ")]
        traj = wired.trajectory
        return {
            "episode": episode, "weeks": inst.T, "J": int(traj.J_cents), "cpu_s": list(meter.weeks),
            "cpu_budget_s": spend, "over_budget": list(meter.over),
            "naive": int(sum(took_fallback(r) for r in traj.records)),
            "substitutions": [[int(w), str(code)] for w, code in wired.substitutions],
            "ready_s": transport.ready_s, "wall_s": time.perf_counter() - start,
            "watchdog_fired": transport.watchdog_fired,
            "meter_errors": list(meter.errors), "journal": journal, "stderr": "\n".join(other)[-3000:],
        }  # fmt: skip


def _path(tag: str, task: str, entropy: int) -> Path:
    return OUT / "board" / f"{tag}_{task}_{entropy}.json"


def run(
    agent: str,
    tag: str,
    task: str = "full",
    entropy: int = 111,
    episodes: int = 64,
    first: int = 0,
    side: int = 15,
    filler: int = -1,
    budget: float = 0.0,
    timeout: float = 0.0,
) -> None:
    """``filler`` more episodes are played after the asked ones so that the last still have neighbours (default: one
    less than ``side``); ``show`` and the cost file read the asked ones."""
    from shockbench_flow_agent import build_image

    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    folder = str(resolve(agent).resolve())
    image = build_image()
    filler = side - 1 if filler < 0 else filler
    ns = list(range(first, first + episodes + filler))
    es = scoring.episode_set(
        task, ns, entropy=entropy, verbose=False
    )  # the references and naive's demand model, cached
    cache = None if es.cache_dir is None else str(es.cache_dir)
    path = _path(tag, task, entropy)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, start = {}, time.time()
    with ProcessPoolExecutor(max_workers=side) as pool:
        jobs = {
            pool.submit(_one, folder, image, task, n, entropy, budget, timeout, es.fq_replications, cache): n
            for n in ns
        }
        for job in as_completed(jobs):
            n = jobs[job]
            try:
                rows[n] = job.result()
            except Exception as error:  # the episode is kept as failed, the others go on
                rows[n] = {"episode": n, "error": repr(error)[:300]}
            print(
                f"episode {n}: weeks naive {rows[n].get('naive', '?')}, {time.time() - start:.0f} s so far", flush=True
            )
            kept = {"agent": agent, "task": task, "entropy": entropy, "side": side, "first": first,
                    "episodes": episodes,
                    "rows": [rows[k] for k in sorted(rows)]}  # fmt: skip
            path.write_text(json.dumps(kept))
    played = {n: r for n, r in rows.items() if "J" in r and first <= n < first + episodes}
    costs = {
        "sha": f"board:{tag}",
        "rows": {
            str(n): {"J": r["J"], "naive": r["naive"], "over": len(r["over_budget"]), "weeks": r["weeks"]}
            for n, r in played.items()
        },
    }
    (OUT / "costs").mkdir(parents=True, exist_ok=True)
    (OUT / "costs" / f"{tag}_{task}_{entropy}.json").write_text(json.dumps(costs))
    show(tag, task=task, entropy=entropy)


def show(tag: str, task: str = "full", entropy: int = 111) -> None:
    kept = json.loads(_path(tag, task, entropy).read_text())
    first, episodes = kept["first"], kept["episodes"]
    rows = [r for r in kept["rows"] if first <= r["episode"] < first + episodes]
    failed = [r["episode"] for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    budget, weeks = rows[0]["cpu_budget_s"], sum(r["weeks"] for r in rows)
    naive = [r["naive"] for r in rows]
    over = [len(r["over_budget"]) for r in rows]
    causes: dict[str, int] = {}
    for r in rows:
        for _, code in r["substitutions"]:
            causes[code] = causes.get(code, 0) + 1
    cpu = np.array([s for r in rows for s in r["cpu_s"] if s is not None])
    print(
        f"{tag}: {task}, root {entropy}, episodes {first}..{first + episodes - 1} ({len(rows)} played), "
        f"{kept['side']} side by side, a budget of {budget:g} CPU s a week"
    )
    print(
        f"  weeks the naive rule played: {sum(naive)} of {weeks} ({100 * sum(naive) / weeks:.3f} %), in "
        f"{sum(1 for x in naive if x)} episodes; weeks over the budget by the meter: {sum(over)}; "
        f"by cause: {causes or 'none'}"
    )
    print(
        f"  CPU seconds a week by the container's meter: median {np.median(cpu):.2f}, "
        f"p95 {np.percentile(cpu, 95):.2f}, "
        f"p99 {np.percentile(cpu, 99):.2f}, max {cpu.max():.2f}; weeks over 0.9 of the budget: "
        f"{int((cpu > 0.9 * budget).sum())} ({100 * (cpu > 0.9 * budget).mean():.2f} %), "
        f"over 0.95: {int((cpu > 0.95 * budget).sum())}"
    )
    print(
        f"  episodes cut by the kill timer: {sum(1 for r in rows if r['watchdog_fired'])}; "
        f"failed to play: {failed or 'none'}; wall clock of an episode: "
        f"median {np.median([r['wall_s'] for r in rows]):.0f} s, longest {max(r['wall_s'] for r in rows):.0f} s"
    )
    lines = [w for r in rows for w in r["journal"]]
    if not lines:
        print("  no journal: the folder was not assembled with --journal")
        return
    n = len(lines)

    def share(test) -> str:
        k = sum(1 for w in lines if test(w))
        return f"{k} ({100 * k / n:.1f} %)"

    cut = lambda w, kind: any(s[2] == "Time limit reached" and kind(s[0]) for s in w["solves"])  # noqa: E731
    exact = ("exact", "plain", "no hint", "no tweak")
    text = lambda w: " ".join(w["note"])  # noqa: E731
    tries = [sum(1 for s in w["solves"] if s[0].startswith("search")) for w in lines]
    print(f"  the clock, of {n} weeks with a journal:")
    print(
        f"    a solve stopped at the clock's limit: {share(lambda w: cut(w, lambda k: True))}; of the hull: "
        f"{share(lambda w: cut(w, lambda k: k == 'hull'))}; of the exact cell: {
            share(lambda w: cut(w, lambda k: k in exact))
        }; "
        f"of a search try: {share(lambda w: cut(w, lambda k: k.startswith('search')))}"
    )
    print(
        f"    the exact cell first, the hull after it or not at all: {share(lambda w: 'exact first' in text(w))}; "
        f"the hull left out that week: {share(lambda w: 'exact first' in text(w) and 'hull after' not in text(w))}"
    )
    print(
        f"    the window of 20 weeks: {share(lambda w: w['short'])}, in "
        f"{sum(1 for r in rows if any(w['short'] for w in r['journal']))} of {len(rows)} episodes; "
        f"the rules rolled out for a part of the window: {share(lambda w: w['fresh'] > 0)}"
    )
    print(
        f"    the search ran: {sum(1 for t in tries if t)} weeks ({100 * np.mean([t > 0 for t in tries]):.1f} %), "
        f"{np.mean([t for t in tries if t] or [0]):.1f} tries a week when it did; no plan of its own that week "
        f"(the rules or an error): {share(lambda w: w['note'][0] in ('no model', 'error') or 'rules alone' in text(w))}"
    )
    any_clock = lambda w: cut(w, lambda k: True) or "exact first" in text(w) or w["short"] or w["fresh"] > 0  # noqa: E731
    print(f"    weeks the clock changed in any of these ways: {share(any_clock)}")


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show})
