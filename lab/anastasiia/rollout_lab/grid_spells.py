"""How long a spell of low output at a grid lasts: the table ``model.Model`` forecasts the grids' output with.

    uv run python lab/anastasiia/rollout_lab/grid_spells.py --episodes=160

A grid's deliverable output drops to a level, stays there for some weeks and comes back in one step; nothing announces
either end. "It stays as it is" therefore keeps a spell alive for the whole forecast. This script measures, on
scenarios of root 333 (the generator's statistics root: no agent is played or tuned on it; Small and Full pooled),
what an agent that sees a grid below its calm output at the start of a week should expect of that week and of the
weeks after: the share of the missing output that is back in each week's average, by the age of the spell as the agent
counts it (weeks in a row seen low; a spell already running in week 1 has no known age and goes to the last row).

No episode is played: the output is the generator's (``marks.G_bar`` is a week's average, ``G_bar_now`` what the
observation shows at its start). Writes ``grid_recovery.json`` beside this file: ``back[age - 1][j]`` is the mean share
back in the week j weeks ahead (0: this week), ``gone[age - 1][j]`` the share of cases where the spell is over by
then, ``weeks[age - 1]`` the number of grid-weeks behind the row.
"""

import json
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
LOW = 0.999  # a grid below this share of its calm output is in a spell (the planner's own threshold)
OLD = 10**6  # the age of a spell running in week 1 (the planner's own mark)


def spells(task: str, entropy: int, n: int, ages: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sums of the share back and of "the spell is over", and counts, by (age row, weeks ahead), of one scenario."""
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks, event_free_marks

    inst, params = task_generator(task)
    calm = np.asarray(event_free_marks(inst).G_bar)[0]
    marks = compute_marks(inst, sample_omega(inst, params, entropy, n, "train"))
    week, now = np.asarray(marks.G_bar)[: inst.T] / calm, np.asarray(marks.G_bar_now)[: inst.T] / calm
    back, gone, count = (np.zeros((ages, horizon)) for _ in range(3))
    age = np.zeros(week.shape[1], dtype=np.int64)
    for t in range(inst.T):
        low = now[t] < LOW
        age = np.where(low, age + (OLD if t == 0 else 1), 0)
        for g in np.flatnonzero(low):
            row, ahead = min(int(age[g]), ages) - 1, week[t : t + horizon, g]
            share = (ahead - now[t, g]) / (1.0 - now[t, g])
            back[row, : len(ahead)] += np.clip(share, -1.0, 1.0)
            gone[row, : len(ahead)] += ahead >= LOW
            count[row, : len(ahead)] += 1
    return back, gone, count


def main(episodes: int = 160, entropy: int = 333, ages: int = 12, horizon: int = 24, n_jobs: int = 2) -> None:
    """Measure the spells and write ``grid_recovery.json``.

    Args:
        episodes: scenarios per network (Small and Full).
        entropy: the root of the scenarios.
        ages: rows of the table; the last holds every older spell and those of unknown age.
        horizon: weeks ahead.
        n_jobs: worker processes.

    """
    jobs = [(task, n) for task in ("small", "full") for n in range(episodes)]
    parts = Parallel(n_jobs=n_jobs)(delayed(spells)(task, entropy, n, ages, horizon) for task, n in jobs)
    back, gone, count = (sum(p[i] for p in parts) for i in range(3))
    seen = np.maximum(count, 1)
    table = {
        "entropy": entropy,
        "episodes": episodes,
        "weeks": count[:, 0].astype(int).tolist(),
        "back": np.round(back / seen, 4).tolist(),
        "gone": np.round(gone / seen, 4).tolist(),
    }
    (HERE / "grid_recovery.json").write_text(json.dumps(table))
    print(f"{2 * episodes} scenarios of root {entropy}; grid-weeks in a spell by age: {table['weeks']}")
    print("share of the missing output back, by age (rows) and weeks ahead 0, 1, 2, 3, 4, 6, 8, 12, 16, 23:")
    for a, row in enumerate(table["back"]):
        cells = " ".join(f"{row[j]:5.2f}" for j in (0, 1, 2, 3, 4, 6, 8, 12, 16, 23) if j < horizon)
        print(f"  age {a + 1:2d}{'+' if a == ages - 1 else ' '} {cells}")


if __name__ == "__main__":
    fire.Fire(main)
