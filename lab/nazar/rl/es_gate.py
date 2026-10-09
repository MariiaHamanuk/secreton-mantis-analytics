"""Evolution strategy for the learned gate of ``lab/nazar/agents/rules_v2_dev`` (fuel_part.py, ``policy.json``).

    uv run python lab/nazar/rl/es_gate.py --task=small --episodes=16 --generations=15
    uv run python lab/nazar/rl/es_gate.py --smoke      # identity checks only

The policy shifts three numbers of the hold / prime / run decision (on_ratio, run_band, prime_weeks) by
scale x tanh(w . [1, inflow/burn, cover, weeks left]): 12 weights. Antithetic evolution strategy; every candidate is
scored on the same episodes of the training root (common random numbers), so the differences are paired.

Training root 555 (the roots 111, 222, 333, 444 are taken, see CLAUDE.md): never the tuning or formal sets.
Output: outputs/es_gate/<date_time>/log.jsonl and best_policy.json (local, not in git).
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import fire
import numpy as np

from sbf_starter import ROOT, scoring


AGENT = ROOT / "lab" / "nazar" / "agents" / "rules_v2_dev"
OUTPUTS = ("on", "run", "prime")
N_FEATURES = 3  # inflow / burn, cover above the threshold, weeks left


def to_policy(theta: np.ndarray) -> dict:
    w = theta.reshape(len(OUTPUTS), N_FEATURES + 1)
    return {name: [float(x) for x in row] for name, row in zip(OUTPUTS, w)}


def score(es, theta: np.ndarray | None, n_jobs: int) -> float:
    """RSS of the agent with the policy ``theta`` (None: the rules as they are) on the episode set."""
    with tempfile.TemporaryDirectory(prefix="es-gate-") as tmp:
        folder = Path(tmp) / AGENT.name
        shutil.copytree(AGENT, folder, ignore=shutil.ignore_patterns("__pycache__", "policy.json"))
        if theta is not None:
            (folder / "policy.json").write_text(json.dumps(to_policy(theta)))
        return float(es.score(str(folder), n_jobs=n_jobs, cpu_budget=False).rss)


def main(
    task: str = "small",
    episodes: int = 16,
    entropy: int = 555,
    generations: int = 15,
    pairs: int = 6,
    sigma: float = 0.4,
    lr: float = 0.5,
    seed: int = 0,
    n_jobs: int = 8,
    smoke: bool = False,
) -> None:
    es = scoring.episode_set(task, episodes if not smoke else 4, entropy=entropy, n_jobs=n_jobs, verbose=False)
    dim = len(OUTPUTS) * (N_FEATURES + 1)
    out = ROOT / "outputs" / "es_gate" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    base = score(es, None, n_jobs)
    zero = score(es, np.zeros(dim), n_jobs)
    print(f"rules {base:.4f}   zero policy {zero:.4f}   (must be equal)", flush=True)
    if smoke:
        return
    rng = np.random.default_rng(seed)
    theta, best = np.zeros(dim), (base, np.zeros(dim))
    for gen in range(generations):
        noise = rng.standard_normal((pairs, dim))
        gains = np.empty((pairs, 2))
        for i, eps in enumerate(noise):
            for j, sign in enumerate((1.0, -1.0)):
                gains[i, j] = score(es, theta + sign * sigma * eps, n_jobs)
                if gains[i, j] > best[0]:
                    best = (gains[i, j], theta + sign * sigma * eps)
        diff = gains[:, 0] - gains[:, 1]
        step = (diff[:, None] * noise).mean(0) / (2 * sigma) * lr / max(gains.std(), 1e-6)
        theta = theta + step * sigma
        now = score(es, theta, n_jobs)
        if now > best[0]:
            best = (now, theta.copy())
        row = {"generation": gen, "theta_rss": now, "best_rss": best[0], "rules_rss": base, "theta": theta.tolist()}
        print(json.dumps({k: v for k, v in row.items() if k != "theta"}), flush=True)
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        (out / "best_policy.json").write_text(json.dumps(to_policy(best[1])))
    print(f"best {best[0]:.4f} vs rules {base:.4f}: {out / 'best_policy.json'}")


if __name__ == "__main__":
    fire.Fire(main)
