"""Pick the planner's clock share for the machine the agent will run on: the same agent at several shares, each one
scored under the server's CPU meter.

    uv run python lab/nazar/mpc/share_sweep.py agents/anastasiia_plan_hull2 --task=small --episodes=16 --n_jobs=4
    uv run python lab/nazar/mpc/share_sweep.py agents/anastasiia_plan_hull2 --task=full --episodes=6 --n_jobs=4 --shares=0.6,0.75,0.85,0.9

The agent's ``regime.json`` gets ``share`` (the share of the week's CPU budget the planner may reach, from the start of
``act``; 0 is no clock); the variants are copied to ``outputs/share_sweep/<name>/`` and removed with it. Each is scored on
the same episodes with ``cpu_budget`` on, so a week over the budget (with the debt of the weeks before it) is played by
the naive rule. Prints per share: the score, its 90% interval, the weeks played by naive and the seconds it took.

``--speed=f`` plays a server f times faster than this machine without having one: the weekly budget, which the meter
and the agent both read, is f times longer (2 s becomes 2f s; every limit of the agent is a share of that budget, so
the agent behaves as it would on the faster machine). The server is said to be 1.6 to 1.9 times slower than the
developer's machine, and this machine is about 4 times slower than it, so a server is about 2 to 2.5 times faster
than this one: ``--speed=2`` and ``--speed=2.5`` are the ones to read.

Run it on the machine the agent will be judged on, with nothing else running there and at most as many processes as
physical cores (a vCPU is often one thread of a core: eight processes on sixteen vCPUs are not eight cores each).
The clock reads CPU time, so a loaded machine makes every share look worse. The rule for the pick: the largest share
with no weeks (or the fewest) played by naive, as the score rises with the share until the weeks over the budget begin;
leave room for a server slower than this machine (a share one step below the best).
"""

import json
import re
import shutil
import time
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[3]


def main(
    agent: str,
    task: str = "small",
    episodes: int = 16,
    entropy: int = 111,
    shares: str = "0,0.3,0.45,0.6,0.75,0.9",
    n_jobs: int = 4,
    speed: float = 1.0,
) -> None:
    from sbf_starter import scoring
    from sbf_starter.agents import resolve

    base = resolve(agent).resolve()
    out = ROOT / "outputs" / "share_sweep" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    print(f"{base.name}: {task} root {entropy} x{episodes}, {n_jobs} processes, the server's CPU meter on, speed x{speed:g}")
    print(f"  {'share':>6} {'score':>8} {'90% interval':>16} {'weeks naive':>12} {'seconds':>8}")
    # fire turns ``0,0.45`` into a tuple and a single number into a number
    items = [shares] if isinstance(shares, (int, float)) else list(shares) if isinstance(shares, (list, tuple)) else str(shares).split(",")
    for share in [float(x) for x in items]:
        folder = out / f"share_{share:g}"
        shutil.copytree(base, folder, ignore=shutil.ignore_patterns("__pycache__"))
        if speed != 1.0:  # the agent's own idea of the weekly budget
            code = (folder / "agent.py").read_text(encoding="utf-8")
            budgets = f'BUDGET_S = {{"small": {2.0 * speed:g}, "full": {4.0 * speed:g}}}'
            new = re.sub(r'BUDGET_S = \{"small": [\d.]+, "full": [\d.]+\}', budgets, code, count=1)
            if new == code:
                raise SystemExit("agent.py has no BUDGET_S = {...} line to scale")
            (folder / "agent.py").write_text(new, encoding="utf-8")
        path = folder / "regime.json"
        regime = json.loads(path.read_text()) if path.is_file() else {}
        regime["share"] = share
        path.write_text(json.dumps(regime))
        t0 = time.perf_counter()
        s = es.score(str(folder), n_jobs=n_jobs, cpu_budget=speed * (2.0 if task == "small" else 4.0))
        lo, hi = s.interval or (float("nan"), float("nan"))
        total = len(s.rows) * (52 if task == "small" else 104)
        print(f"  {share:6g} {s.rss:8.4f} {lo:7.3f}-{hi:<7.3f} {int(s.fallback_weeks or 0):>5} of {total:<5} {time.perf_counter() - t0:8.0f}", flush=True)
    shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    fire.Fire(main)
