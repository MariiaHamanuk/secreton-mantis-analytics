"""The judge of the search: candidates played on fresh episodes, and their paired difference with the base.

    uv run python lab/anastasiia/evolve_lab/evolve.py play --names=base,a1,a2 --first=0 --episodes=32 --n_jobs=32
    uv run python lab/anastasiia/evolve_lab/evolve.py refs --episodes=64
    uv run python lab/anastasiia/evolve_lab/evolve.py score --names=a1,a2 --first=0 --episodes=32

``play`` assembles each name's folder (``assemble.py``; ``base`` has no terms, another name takes
``terms/<name>.py``) and plays the episodes ``first`` .. ``first + episodes - 1`` of the root under gymnasium, all the
names' episodes in one pool, without a CPU limit. An episode's cost goes to ``outputs/evolve_lab/costs/<name>.json``
the moment the pool hands it back, with the weeks the agent's planner raised in (it then plays the rules: such a
week is no part of a comparison) and the week's CPU seconds. An episode already there is not played again; a file
made from other terms or settings under the same name is refused, so a name is one candidate for good. An episode
whose candidate raised is kept with the error instead of a cost. ``refs`` writes the episodes' reference costs and
harm levels as JSON (computing those not yet cached: about two CPU minutes an episode of Small). ``score`` reads
both files and prints each name's score, its paired difference with the base on the episodes both played and an
interval (episodes drawn again inside the harm levels; a set without one of the four levels is scored as ``sbf
evaluate`` scores it). The JSON files are all ``score`` needs, so it runs wherever they are copied.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "evolve_lab"
WEIGHTS = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05}  # the board's weights of the harm levels


def _names(names) -> list[str]:
    return [str(n) for n in (names.split(",") if isinstance(names, str) else names) if str(n)]


def _costs(name: str, task: str, entropy: int) -> Path:
    return OUT / "costs" / f"{name}_{task}_{entropy}.json"


def _one(folder: str, name: str, task: str, entropy: int, n: int) -> tuple:
    sys.path[:0] = [str(HERE.parent / "regime_lab"), str(HERE.parent / "mpc_lab")]
    import play

    try:
        r = play.episode(folder, task, entropy, n)
    except BaseException as error:  # the candidate's terms raised (``TermsError``): it is out, the others go on
        if type(error).__name__ != "TermsError" and not isinstance(error, Exception):
            raise
        return name, n, {"error": repr(error)[:300]}
    cpu = np.asarray(r["cpu"], dtype=float)
    raised = sum(1 for e in r["log"] if len(e) > 1 and e[1] == "error")
    limits = sum(1 for d in r["detail"] for s in d.get("solves") or () if s.get("status") == "Time limit reached")
    return (
        name,
        n,
        {
            "J": int(r["J"]),
            "raised": raised,
            "limits": limits,
            "cpu": [float(np.median(cpu)), float(np.percentile(cpu, 95)), float(cpu.max())],
        },
    )


def play(
    names,
    first: int = 0,
    episodes: int = 32,
    task: str = "small",
    entropy: int = 777,
    n_jobs: int = 32,
    clock: bool = False,
    suffix: str = "",
    speed: bool = False,
    soft: bool = False,
) -> None:
    """``soft``: the shipped hook (a failed term is no term). ``speed``: with lab/nazar's speedups. ``clock``: the
    model's own settings (a winner's check); ``suffix`` then names those folders and files apart."""
    from joblib import Parallel, delayed

    todo, kept = [], {}
    for name in _names(names):
        terms = [] if name == "base" else [f"--terms={HERE / 'terms' / (name + '.py')}"]
        made = subprocess.run(
            [sys.executable, str(HERE / "assemble.py"), f"--name={name}{suffix}", *terms]
            + (["--clock"] if clock else [])
            + (["--speed"] if speed else [])
            + (["--soft"] if soft else []),
            capture_output=True,
            text=True,
        )
        if made.returncode:
            print(f"{name}: not assembled: {made.stderr.strip()[-300:]}")
            continue
        sha, folder = made.stdout.strip().splitlines()[-2:]
        path = _costs(name + suffix, task, entropy)
        kept[name] = json.loads(path.read_text()) if path.is_file() else {"sha": sha, "rows": {}}
        if kept[name]["sha"] != sha:
            print(f"{name}: {path.name} was played from other terms or settings: take a new name")
            del kept[name]
            continue
        todo += [(folder, name, n) for n in range(first, first + episodes) if str(n) not in kept[name]["rows"]]
    (OUT / "costs").mkdir(parents=True, exist_ok=True)
    start = time.time()
    jobs = (delayed(_one)(folder, name, task, entropy, n) for folder, name, n in todo)
    for done, (name, n, row) in enumerate(Parallel(n_jobs=n_jobs, return_as="generator_unordered")(jobs), start=1):
        kept[name]["rows"][str(n)] = row
        _costs(name + suffix, task, entropy).write_text(json.dumps(kept[name]))
        if "error" in row:
            print(f"{name} episode {n}: {row['error']}", flush=True)
        if done % 32 == 0 or done == len(todo):
            print(f"{done} of {len(todo)} episodes, {time.time() - start:.0f} s", flush=True)
    for name, file in kept.items():
        rows = [file["rows"][str(n)] for n in range(first, first + episodes) if str(n) in file["rows"]]
        bad = sum(1 for r in rows if "error" in r)
        print(
            f"{name}: {len(rows) - bad} episodes played, {bad} raised; weeks the planner raised in: "
            f"{sum(r.get('raised', 0) for r in rows)}; solves cut by a time limit: "
            f"{sum(r.get('limits', 0) for r in rows)}"
        )


def meter(
    names,
    first: int = 0,
    episodes: int = 128,
    task: str = "small",
    entropy: int = 888,
    n_jobs: int = 32,
    budget: float = 0.0,
    speed: bool = False,
    soft: bool = False,
) -> None:
    """Names in the model's own settings (clocks and all), played as the scorer plays under its CPU meter: a week
    over the budget is the naive rule's and what it took over is charged to the next. One name after another, each
    alone on the machine, so that no arm's weeks are slowed by another's. Costs go to ``<name>_clock`` files."""
    from sbf_starter import scoring

    limit = float(budget) or {"small": 2.0, "full": 4.0}[task]
    es = scoring.episode_set(task, list(range(first, first + episodes)), entropy=entropy, n_jobs=n_jobs, verbose=False)
    suffix = "_clock" + ("_s" if speed else "") + ("_soft" if soft else "")
    (OUT / "costs").mkdir(parents=True, exist_ok=True)
    for name in _names(names):
        terms = [] if name == "base" else [f"--terms={HERE / 'terms' / (name + '.py')}"]
        made = subprocess.run(
            [sys.executable, str(HERE / "assemble.py"), f"--name={name}{suffix}", "--clock", *terms]
            + (["--speed"] if speed else [])
            + (["--soft"] if soft else []),
            capture_output=True,
            text=True,
        )
        if made.returncode:
            print(f"{name}: not assembled: {made.stderr.strip()[-300:]}")
            continue
        sha, folder = made.stdout.strip().splitlines()[-2:]
        path = _costs(name + suffix, task, entropy)
        kept = json.loads(path.read_text()) if path.is_file() else {"sha": sha, "rows": {}}
        if kept["sha"] != sha:
            print(f"{name}: {path.name} was played from other terms or settings: take a new name")
            continue
        start = time.time()
        for r in es.play(folder, cpu_budget=limit, n_jobs=n_jobs):
            kept["rows"][str(r["episode"])] = {
                "J": int(r["J_policy_cents"]),
                "naive": int(r["fallback_weeks"]),
                "over": int(r["cpu_weeks"]),
                "weeks": int(r["weeks"]),
                "error": r.get("first_error") or "",
            }
            if not kept["rows"][str(r["episode"])]["error"]:
                del kept["rows"][str(r["episode"])]["error"]
        path.write_text(json.dumps(kept))
        rows = [kept["rows"][str(n)] for n in range(first, first + episodes) if str(n) in kept["rows"]]
        print(
            f"{name}{suffix}: {len(rows)} episodes of {task} {entropy} from {first}, budget {limit:g} s, "
            f"{time.time() - start:.0f} s; weeks naive {sum(r['naive'] for r in rows)} of "
            f"{sum(r['weeks'] for r in rows)}",
            flush=True,
        )


def refs(episodes: int = 64, task: str = "small", entropy: int = 777, first: int = 0) -> None:
    """The reference costs and harm levels of the episodes ``first`` .. ``first + episodes - 1``, added to the
    file of the task and root (a block played on one machine has its references cached there)."""
    from sbf_starter import scoring

    start = time.time()
    ns = list(range(first, first + episodes))
    rows = scoring.episode_set(task, ns, entropy=entropy, verbose=False).references
    path = OUT / f"refs_{task}_{entropy}.json"
    kept = _refs(path) if path.is_file() else {}
    kept.update({str(n): {k: r[k] for k in ("J_naive_cents", "J_oracle_cents", "stratum")} for n, r in zip(ns, rows)})
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept))
    levels = [sum(1 for n in ns if kept[str(n)]["stratum"] == s) for s in (1, 2, 3, 4)]
    print(f"{len(ns)} episodes from {first}, {time.time() - start:.0f} s; by level {levels}; {len(kept)} in the file")


def _refs(path: Path) -> dict:
    """References by episode (an older file is a list from episode 0)."""
    kept = json.loads(path.read_text())
    return {str(n): r for n, r in enumerate(kept)} if isinstance(kept, list) else kept


def _rss(refs: list, costs: list) -> float:
    """The board's score; a set without one of the four levels: all that was saved over all that could be."""
    saved = np.array([r["J_naive_cents"] - j for r, j in zip(refs, costs)], dtype=float)
    room = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float)
    level = np.array([r["stratum"] for r in refs])
    if set(level.tolist()) != {1, 2, 3, 4}:
        return float(saved.sum() / room.sum())
    # the package's formula (``mpc_lab/package_baselines.rss``): the levels' mean savings over their mean room
    return float(
        sum(w * saved[level == s].mean() for s, w in WEIGHTS.items())
        / sum(w * room[level == s].mean() for s, w in WEIGHTS.items())
    )


def _episodes(first: int, episodes: int, ranges: str) -> list[int]:
    if not ranges:
        return list(range(first, first + episodes))
    return [n for part in str(ranges).split(",") for n in range(int(part.split("-")[0]), int(part.split("-")[1]) + 1)]


def score(
    names,
    first: int = 0,
    episodes: int = 32,
    task: str = "small",
    entropy: int = 777,
    base: str = "base",
    draws: int = 2000,
    suffix: str = "",
    ranges: str = "",
    level: float = 0.90,
    costs: str = "",
) -> None:
    """``ranges``: stretches of episodes instead of one, "0-31,96-191" (ends included). ``level``: the interval's.
    ``costs``: another folder of cost files for the base (the same base played on another machine: an A/A check)."""
    all_refs = _refs(OUT / f"refs_{task}_{entropy}.json")
    wanted = _episodes(first, episodes, ranges)
    based = json.loads(_costs(base + suffix, task, entropy).read_text())["rows"]
    tail = 100 * (1 - level) / 2
    print(
        f"{task}, root {entropy}, episodes {wanted[0]}..{wanted[-1]} ({len(wanted)}); paired with {base}{suffix}; "
        f"{100 * level:g} % intervals"
    )
    for name in _names(names):
        path = (
            Path(costs) / _costs(name + suffix, task, entropy).name if costs else _costs(name + suffix, task, entropy)
        )
        rows = json.loads(path.read_text())["rows"]
        failed = [n for n in wanted if "error" in rows.get(str(n), {})]
        ns = [
            n
            for n in wanted
            if "J" in rows.get(str(n), {})
            and "J" in based.get(str(n), {})
            and str(n) in all_refs
            and all_refs[str(n)]["J_oracle_cents"] is not None
        ]
        if failed or not ns:
            print(
                f"{name:10s} raised in {len(failed)} episodes, played {len(ns)}: "
                f"{rows[str(failed[0])]['error'][:160] if failed else ''}"
            )
            continue
        rs = [all_refs[str(n)] for n in ns]
        a = [rows[str(n)]["J"] for n in ns]
        b = [based[str(n)]["J"] for n in ns]
        levels = np.array([r["stratum"] for r in rs])
        groups = [np.flatnonzero(levels == s) for s in sorted(set(levels.tolist()))]
        rng = np.random.default_rng(0)
        drawn = []
        for _ in range(draws):
            p = np.concatenate([rng.choice(g, len(g)) for g in groups])
            drawn.append(_rss([rs[i] for i in p], [a[i] for i in p]) - _rss([rs[i] for i in p], [b[i] for i in p]))
        lo, hi = np.percentile(drawn, [tail, 100 - tail])
        if "cpu" in rows[str(ns[0])]:  # the form without clocks: the week's CPU seconds, weeks the planner raised in
            cpu = [np.median([r[str(n)]["cpu"][i] for n in ns]) for r in (rows, based) for i in (0, 1)]
            raised = sum(rows[str(n)].get("raised", 0) for n in ns), sum(based[str(n)].get("raised", 0) for n in ns)
            note = (
                f"week {cpu[0]:.2f}/{cpu[1]:.2f} s against {cpu[2]:.2f}/{cpu[3]:.2f}; "
                f"raised weeks {raised[0]} against {raised[1]}"
            )
        else:  # under the meter: the weeks it handed to the naive rule
            naive = sum(rows[str(n)]["naive"] for n in ns), sum(based[str(n)]["naive"] for n in ns)
            note = f"weeks naive {naive[0]} against {naive[1]}; base {_rss(rs, b):.4f}"
        print(
            f"{name:10s} {_rss(rs, a):.4f}  {_rss(rs, a) - _rss(rs, b):+.4f} ({lo:+.4f} to {hi:+.4f})  cheaper "
            f"{sum(1 for x, y in zip(a, b) if x < y)}, dearer {sum(1 for x, y in zip(a, b) if x > y)}, same "
            f"{sum(1 for x, y in zip(a, b) if x == y)} of {len(ns)} "
            f"(levels {[int((levels == s).sum()) for s in (1, 2, 3, 4)]}); "
            f"{note}"
        )


if __name__ == "__main__":
    fire.Fire({"play": play, "meter": meter, "refs": refs, "score": score})
