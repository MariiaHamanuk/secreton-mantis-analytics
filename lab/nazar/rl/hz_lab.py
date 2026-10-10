"""Variants of ``anastasiia_plan_hazard`` scored under the server's CPU meter, and a search over its numbers.

    uv run python lab/nazar/rl/hz_lab.py compare --preset=rooms --episodes=16 --entropy=111 --speed=2 --n_jobs=8
    uv run python lab/nazar/rl/hz_lab.py es --room=1 --episodes=24 --entropy=777 --speed=2 --generations=6 --pairs=5

The planner's clock reads CPU time, so a variant is only comparable to another one played at the same ``--speed``: a
server ``f`` times faster than this machine is emulated by a weekly budget ``f`` times longer, which both the meter and
the agent's own ``BUDGET_S`` read (as in ``lab/nazar/mpc/share_sweep.py``). The server is about 2 to 2.5 times faster
than this machine, so ``--speed=2`` is the one to read. ``es`` is an antithetic evolution strategy over five numbers of
``regime.json`` (the three quantiles of the watch, ``watch_ask`` and ``fit``) on a training root; root 777 is taken for it.
"""

import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "hub" / "eval"))
from formal_eval import paired  # noqa: E402

BASE = ROOT / "agents" / "anastasiia_plan_hazard"
KEYS = ("episode", "stratum", "excluded", "J_policy_cents", "J_naive_cents", "J_clairvoyant_cents")
PRESETS = {
    "rooms": {"hazard": {}, "room1": {"search_room": 1}, "room2": {"search_room": 2}, "room3": {"search_room": 3}},
    "search": {"room1": {"search_room": 1}, "room1_s4": {"search_room": 1, "search": 4}, "room1_s12": {"search_room": 1, "search": 12}},
    "esbest": {"room2": {"search_room": 2}, "es_best": {"search_room": 2, "watch": {"weather_closure": 0.39, "port_strike_stoppage": 0.3846, "port_strike_slowdown": 0.488}, "watch_ask": 0.2543, "fit": 1.2901}},
    "asks": {"room1": {"search_room": 1}, "ask0": {"search_room": 1, "watch_ask": 0.0}, "ask5": {"search_room": 1, "watch_ask": 0.5}},
}


def variant(folder: Path, over: dict, speed: float, task: str) -> Path:
    shutil.copytree(BASE, folder, ignore=shutil.ignore_patterns("__pycache__"))
    code = (folder / "agent.py").read_text(encoding="utf-8")
    budgets = f'BUDGET_S = {{"small": {2.0 * speed:g}, "full": {4.0 * speed:g}}}'
    new = re.sub(r'BUDGET_S = \{"small": [\d.]+, "full": [\d.]+\}', budgets, code, count=1)
    if new == code and speed != 1.0:
        raise SystemExit("agent.py has no BUDGET_S line to scale")
    (folder / "agent.py").write_text(new, encoding="utf-8")
    regime = json.loads((folder / "regime.json").read_text())
    for key, value in over.items():
        if key == "watch":
            regime["watch"] = {**regime["watch"], **value}
        else:
            regime[key] = value
    (folder / "regime.json").write_text(json.dumps(regime))
    return folder


def play(es, over: dict, speed: float, task: str, n_jobs: int):
    with tempfile.TemporaryDirectory(prefix="hz-") as tmp:
        folder = variant(Path(tmp) / "agent", over, speed, task)
        return es.score(str(folder), n_jobs=n_jobs, cpu_budget=speed * (2.0 if task == "small" else 4.0))


def rows_of(score) -> list[dict]:
    return [{k: r.get(k) for k in KEYS} for r in score.rows]


def compare(preset: str = "rooms", task: str = "small", episodes: int = 16, entropy: int = 111, speed: float = 2.0, n_jobs: int = 8) -> None:
    from sbf_starter import scoring

    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    print(f"hazard variants '{preset}': {task} root {entropy} x{episodes}, speed x{speed:g}, {n_jobs} processes", flush=True)
    first = None
    for name, over in PRESETS[preset].items():
        t0 = time.perf_counter()
        s = play(es, over, speed, task, n_jobs)
        rows = rows_of(s)
        first = first or rows
        d, lo, hi = paired(rows, first)
        total = len(s.rows) * (52 if task == "small" else 104)
        print(f"  {name:>10} {s.rss:.4f}  vs first {d:+.4f} ({lo:+.4f}..{hi:+.4f})  naive weeks {int(s.fallback_weeks or 0)} of {total}  {time.perf_counter() - t0:.0f} s", flush=True)


def knob_values(theta: np.ndarray, room: int) -> dict:
    sig = lambda x: float(np.clip(1 / (1 + np.exp(-x)), 0.1, 0.9))  # noqa: E731 (a quantile, 0.5 at 0)
    ask = float(np.clip(0.3 * np.exp(0.7 * np.tanh(theta[3])), 0.05, 1.0))
    return {
        "search_room": room,
        "watch": {"weather_closure": sig(theta[0]), "port_strike_stoppage": sig(theta[1]), "port_strike_slowdown": sig(theta[2])},
        "watch_ask": ask,
        "fit": float(1.25 * np.exp(0.3 * np.tanh(theta[4]))),
    }


def es(room: int = 1, task: str = "small", episodes: int = 24, entropy: int = 777, speed: float = 2.0, generations: int = 6, pairs: int = 5, sigma: float = 0.6, lr: float = 1.0, seed: int = 0, n_jobs: int = 8) -> None:
    from sbf_starter import scoring

    es_set = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    out = ROOT / "outputs" / "es_hazard" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    def score(theta: np.ndarray) -> float:
        return float(play(es_set, knob_values(theta, room), speed, task, n_jobs).rss)

    rng = np.random.default_rng(seed)
    theta, dim = np.zeros(5), 5
    base = score(theta)
    best = (base, theta.copy())
    print(f"hazard room {room}: {base:.4f} on {task} root {entropy} x{episodes}, speed x{speed:g}", flush=True)
    for gen in range(generations):
        noise = rng.standard_normal((pairs, dim))
        gains = np.empty((pairs, 2))
        for i, eps in enumerate(noise):
            for j, sign in enumerate((1.0, -1.0)):
                cand = theta + sign * sigma * eps
                gains[i, j] = score(cand)
                if gains[i, j] > best[0]:
                    best = (gains[i, j], cand.copy())
        diff = gains[:, 0] - gains[:, 1]
        theta = theta + lr * (diff[:, None] * noise).mean(0) / (2 * max(gains.std(), 1e-6))
        now = score(theta)
        if now > best[0]:
            best = (now, theta.copy())
        print(json.dumps({"generation": gen, "theta_rss": round(now, 4), "best_rss": round(best[0], 4), "base_rss": round(base, 4), "pairs_mean": round(float(gains.mean()), 4), "pairs_max": round(float(gains.max()), 4)}), flush=True)
        (out / "best.json").write_text(json.dumps({"theta": best[1].tolist(), "regime": knob_values(best[1], room), "rss": best[0], "base": base}))
    print(f"best {best[0]:.4f} vs {base:.4f}: {out / 'best.json'}", flush=True)


if __name__ == "__main__":
    fire.Fire({"compare": compare, "es": es})
