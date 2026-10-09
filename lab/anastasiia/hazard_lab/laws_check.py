"""The durations of the short disruptions as the generator's scenarios have them, against the laws it draws from.

    uv run python lab/anastasiia/hazard_lab/laws_check.py --episodes_small=2000 --episodes_full=600

For every weather closure and port strike that starts inside an episode of root 333 (no choice by length: the event
list keeps each of them whatever its duration): the quantiles of its duration, the lognormal fitted to them (mean and
standard deviation of the logarithm), and the law of ``src/watch.py``. A scenario is not a free draw of the laws: the
generator keeps or drops whole scenarios (its gates and harm levels), so the kept events may run longer.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
KINDS = ("weather_closure", "port_strike_stoppage", "port_strike_slowdown")


def one(task: str, entropy: int, n: int) -> list:
    from shockbench_flow import marks as M
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.omega import codes

    inst, params = task_generator(task)
    omega = sample_omega(inst, params, entropy, n, "train")
    inst = inst.at_digest(str(omega.arrays["meta_instance_digest"]))
    out = []
    for q in M.read_events(inst, omega.arrays):
        kind = codes.EVENT_TYPES[q.type]
        if kind == "port_strike":
            kind += "_stoppage" if q.severity >= 0.9 else "_slowdown"
        if kind in KINDS:
            out.append((kind, float(q.onset), float(q.duration)))
    return out


def main(episodes_small: int = 2000, episodes_full: int = 600, entropy: int = 333, n_jobs: int = 3) -> None:
    import watch as W

    for task, N in (("small", episodes_small), ("full", episodes_full)):
        rows = [r for rs in Parallel(n_jobs=n_jobs)(delayed(one)(task, entropy, n) for n in range(N)) for r in rs]
        print(f"{task}, root {entropy}, {N} episodes; events that start inside the episode")
        for kind in KINDS:
            d = np.array([dur for k, onset, dur in rows if k == kind and onset >= 0])
            _name, mu, sigma, per_week = W.LAWS[kind]
            logs = np.log(d * per_week)
            law = np.exp(mu + sigma * np.array([-1.2816, -0.6745, 0.0, 0.6745, 1.2816])) / per_week
            print(f"  {kind:22s} n {len(d):5d}  quantiles 10/25/50/75/90 {np.round(np.percentile(d, [10, 25, 50, 75, 90]), 2)}  "
                  f"law {np.round(law, 2)}  fitted ln: mean {logs.mean():.3f} (law {mu}), sd {logs.std():.3f} (law {sigma}), "
                  f"se of the mean {logs.std() / np.sqrt(len(d)):.3f}")


if __name__ == "__main__":
    fire.Fire(main)
