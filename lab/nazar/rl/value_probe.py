"""Does the state at a week explain the cost still to come? A diagnostic for a learned terminal value of the plan's window.

    uv run python lab/nazar/rl/value_probe.py play --episodes=16 --entropy=777 --n_jobs=4
    uv run python lab/nazar/rl/value_probe.py fit

``play`` scores a copy of ``anastasiia_plan_hazard`` (search_room 2, speed x2) whose ``act`` also writes, once a week, the
state (week, stock by commodity group, backlog, queued lots, last week's shed and cost components) to
``outputs/value_probe/rec/<policy_seed>.npz``. ``fit`` turns each file into a cost to go (the sum of the cost components
of the weeks that follow) and compares, on whole held-out episodes, a model that knows only the week against one that
also knows the state (ridge on log-scaled aggregates). A state model that does not beat the week-only one by a margin
has nothing to give a terminal value; one that does is a candidate for ``plan_core``'s end-of-window term.
"""

import shutil
import sys
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
REC = ROOT / "outputs" / "value_probe" / "rec"
HOOK = '''

# --- value_probe: write the state once a week (appended; no effect on the actions) ---
import os as _os

_REC_DIR = Path(_os.environ.get("VALUE_PROBE_DIR", "/tmp/value_probe_rec"))
_orig_init, _orig_act = Agent.__init__, Agent.act


def _init(self, config=None):
    _orig_init(self, config)
    self._rec_seed, self._rec = int(config["policy_seed"]), []


def _act(self, observation):
    o = observation
    week = float(np.asarray(o["week"]).ravel()[0])
    self._rec.append(np.concatenate([
        [week], np.asarray(o["stock.qty"], float), np.asarray(o["backlog.qty"], float),
        [float(np.asarray(o["queue_lots.qty"], float).sum())], np.asarray(o["last_week.shed.qty"], float),
        np.asarray(o["last_week.cost_components"], float),
    ]))
    try:
        _REC_DIR.mkdir(parents=True, exist_ok=True)
        np.save(_REC_DIR / f"{self._rec_seed}.npy", np.array(self._rec))
    except OSError:
        pass
    return _orig_act(self, observation)


Agent.__init__, Agent.act = _init, _act
'''


def play(episodes: int = 16, entropy: int = 777, task: str = "small", speed: float = 2.0, n_jobs: int = 4) -> None:
    import os
    import tempfile

    from hz_lab import variant
    from sbf_starter import scoring

    shutil.rmtree(REC, ignore_errors=True)
    REC.mkdir(parents=True, exist_ok=True)
    os.environ["VALUE_PROBE_DIR"] = str(REC)
    es = scoring.episode_set(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=False)
    with tempfile.TemporaryDirectory(prefix="vp-") as tmp:
        folder = variant(Path(tmp) / "agent", {"search_room": 2}, speed, task)
        with (folder / "agent.py").open("a", encoding="utf-8") as f:
            f.write(HOOK)
        s = es.score(str(folder), n_jobs=n_jobs, cpu_budget=speed * (2.0 if task == "small" else 4.0))
    print(f"played {len(s.rows)} episodes, rss {s.rss:.4f}, naive weeks {int(s.fallback_weeks or 0)}; {len(list(REC.glob('*.npy')))} files in {REC}")


def fit(folds: int = 4) -> None:
    files = sorted(REC.glob("*.npy"))
    eps = []
    for p in files:
        a = np.load(p)
        if len(a) < 10:
            continue
        cost = a[:, -8:].sum(axis=1)  # the cost of the week before each row
        to_go = np.cumsum(cost[::-1])[::-1] - cost  # the weeks that follow the row's week
        so_far = np.cumsum(cost)
        recent = np.array([cost[max(0, k - 3) : k + 1].mean() for k in range(len(cost))])
        stock, backlog, queue, shed = a[:, 1:60].sum(1), a[:, 60:68].sum(1), a[:, 68], a[:, 69:73].sum(1)
        eps.append((a[:, 0], to_go, {
            "aggregates": np.column_stack([np.log1p(stock), np.log1p(backlog), np.log1p(queue), np.log1p(shed)]),
            "aggregates + cost so far": np.column_stack([np.log1p(stock), np.log1p(backlog), np.log1p(queue), np.log1p(shed), np.log1p(so_far), np.log1p(recent)]),
        }))
    t = np.concatenate([e[0] for e in eps])
    y = np.log1p(np.maximum(np.concatenate([e[1] for e in eps]), 0.0))
    group = np.concatenate([[i] * len(e[0]) for i, e in enumerate(eps)])
    print(f"{len(eps)} episodes, {len(t)} weeks; the target is log(1 + cost still to come)")

    def cv(D: np.ndarray, mask: np.ndarray, alpha: float) -> float:
        pred = np.zeros_like(y)
        for k in range(folds):
            test = group % folds == k
            A = D[~test]
            reg = alpha * np.eye(D.shape[1])
            reg[0, 0] = 0
            w = np.linalg.solve(A.T @ A + reg, A.T @ y[~test])
            pred[test] = D[test] @ w
        r = y[mask] - pred[mask]
        return float(1 - (r**2).sum() / ((y[mask] - y[mask].mean()) ** 2).sum())

    week = np.column_stack([np.ones_like(t), t / 52, (t / 52) ** 2])
    masks = (("all weeks", np.ones(len(y), bool)), ("weeks 20-34", (t >= 20) & (t <= 34)))
    for name, mask in masks:
        print(f"{name}: week only {cv(week, mask, 1.0):.3f}")
    for feats in eps[0][2]:
        F = np.vstack([e[2][feats] for e in eps])
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        for alpha in (10.0, 100.0, 1000.0):
            row = ", ".join(f"{name} {cv(np.column_stack([week, F]), mask, alpha):.3f}" for name, mask in masks)
            print(f"week + {feats} (ridge {alpha:g}): {row}")


if __name__ == "__main__":
    fire.Fire({"play": play, "fit": fit})
