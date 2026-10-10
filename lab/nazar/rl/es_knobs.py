"""Evolution strategy for ``lab/nazar/agents/rl_knobs``: a small network that moves eight numbers of the rules every week.

    uv run python lab/nazar/rl/es_knobs.py --episodes=24 --entropy=777 --generations=8 --pairs=12 --n_jobs=8

Antithetic pairs around theta (98 weights, start 0 = the rules unchanged); every candidate is scored on the same episodes of
the training root (common random numbers). Root 777 is taken here for training: add it to CLAUDE.md before using the result.
The best theta so far is written to ``outputs/es_knobs/<date_time>/best_policy.json`` after every generation.
"""

import importlib.util
import json
import shutil
import tempfile
import time
from pathlib import Path

import fire
import numpy as np

from sbf_starter import ROOT, scoring

AGENT = ROOT / "lab" / "nazar" / "agents" / "rl_knobs"


def knobs():
    spec = importlib.util.spec_from_file_location("rl_knobs_def", AGENT / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand(theta, mod, static: bool, span: float):
    """The full weight vector: with ``static`` only the eight output biases are searched (a constant factor each)."""
    if theta is None or not static:
        return theta
    full = np.zeros(mod.n_weights())
    full[-len(mod.KNOBS):] = theta
    return full


def score(es, theta, mod, n_jobs: int, static: bool = False, span: float = 0.3) -> float:
    theta = expand(theta, mod, static, span)
    with tempfile.TemporaryDirectory(prefix="es-knobs-") as tmp:
        folder = Path(tmp) / AGENT.name
        shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns("__pycache__", "policy.json"))
        if theta is not None:
            (folder / "policy.json").write_text(json.dumps({**mod.unpack(theta), "span": span}))
        return float(es.score(str(folder), n_jobs=n_jobs, cpu_budget=False).rss)


def main(episodes: int = 24, entropy: int = 777, generations: int = 8, pairs: int = 12, sigma: float = 0.15, lr: float = 0.6, seed: int = 0, n_jobs: int = 8, static: bool = False, span: float = 0.3) -> None:
    mod = knobs()
    dim = len(mod.KNOBS) if static else mod.n_weights()
    es = scoring.episode_set("small", episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    out = ROOT / "outputs" / "es_knobs" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    base = score(es, None, mod, n_jobs, static, span)
    print(f"rules_v3 (theta 0): {base:.4f} on Small root {entropy} x{episodes}; {dim} weights", flush=True)
    rng = np.random.default_rng(seed)
    theta, best = np.zeros(dim), (base, np.zeros(dim))
    for gen in range(generations):
        noise = rng.standard_normal((pairs, dim))
        gains = np.empty((pairs, 2))
        for i, eps in enumerate(noise):
            for j, sign in enumerate((1.0, -1.0)):
                gains[i, j] = score(es, theta + sign * sigma * eps, mod, n_jobs, static, span)
                if gains[i, j] > best[0]:
                    best = (gains[i, j], theta + sign * sigma * eps)
        diff = gains[:, 0] - gains[:, 1]
        theta = theta + lr * sigma * (diff[:, None] * noise).mean(0) / (2 * sigma) / max(gains.std(), 1e-6)
        now = score(es, theta, mod, n_jobs, static, span)
        if now > best[0]:
            best = (now, theta.copy())
        print(json.dumps({"generation": gen, "theta_rss": round(now, 4), "best_rss": round(best[0], 4), "base_rss": round(base, 4), "pairs_mean": round(float(gains.mean()), 4), "pairs_max": round(float(gains.max()), 4)}), flush=True)
        (out / "best_policy.json").write_text(json.dumps({**mod.unpack(expand(best[1], mod, static, span)), "span": span}))
    print(f"best {best[0]:.4f} vs rules_v3 {base:.4f}: {out / 'best_policy.json'}")


if __name__ == "__main__":
    fire.Fire(main)
