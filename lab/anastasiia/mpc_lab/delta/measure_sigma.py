"""Weekly drift of edge capacities u in the true marks: sigma per pool (fuel = 'tb', chip/container = 'ct').

    uv run python lab/anastasiia/mpc_lab/delta/measure_sigma.py --task=small --entropy=555 --episodes=8

For every finite-capacity edge and week t: r = (u[t+1] - u[t]) / u[t] (u[t] > 0; clipped to [-1, 1]) and
d = (u[t+1] - u[t]) / u0 (nominal). sigma = RMS over (edge, week). Also the h-week spread RMS of
(u[t+h] - u[t]) / u[t] for h = 1, 4, 9, 16 (does it grow like sqrt(h)?), split into first edges of action slots
("first") and downstream edges ("down").
"""

import fire
import numpy as np


def marks_of(task, entropy, n):
    from shockbench_flow.disruption.sampler import sample_omega
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.marks import compute_marks
    from shockbench_flow_agent.scoring import _label

    inst, params = task_generator(task)
    omega = sample_omega(inst, params, entropy, n, _label(entropy))
    return inst, compute_marks(inst, omega)


def rms(a):
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.mean(a**2))) if a.size else float("nan")


def main(task="small", entropy=555, episodes=8, start=0):
    acc = {}
    per_ep = []
    for n in range(start, start + episodes):
        inst, m = marks_of(task, entropy, n)
        u = np.asarray(m.u, dtype=float)  # (T, E)
        first = {e for e, _, _ in inst.action_slots}
        ep_row = {}
        for e, ed in enumerate(inst.edges):
            if ed.u0 is None or not np.all(np.isfinite(u[:, e])) or ed.pool is None:
                continue
            where = "first" if e in first else "down"
            for h in (1, 4, 9, 16):
                a, b = u[:-h, e], u[h:, e]
                ok = a > 0
                r = np.clip((b[ok] - a[ok]) / a[ok], -1, 1)
                acc.setdefault((ed.pool, where, h), []).append(r)
                acc.setdefault((ed.pool, "all", h), []).append(r)
                if h == 1:
                    ep_row.setdefault(ed.pool, []).append(r)
                    acc.setdefault((ed.pool, "all", "d0"), []).append((b - a) / ed.u0)
                    acc.setdefault((ed.pool, "all", "nz"), []).append(r[np.abs(r) > 1e-9])
        per_ep.append({p: rms(np.concatenate(v)) for p, v in ep_row.items()})
        print(f"ep {n}: " + "  ".join(f"{p} {s:.4f}" for p, s in sorted(per_ep[-1].items())), flush=True)
    print(f"\n{task} root {entropy} eps {start}..{start + episodes - 1}")
    print("pool where   h   RMS(rel)   n      frac_nonzero")
    for (pool, where, h), v in sorted(acc.items(), key=lambda x: str(x[0])):
        if h in ("d0", "nz"):
            continue
        a = np.concatenate(v)
        print(f"{pool:4} {where:6} {h:3} {rms(a):9.4f} {a.size:6d}  {np.mean(np.abs(a) > 1e-9):.3f}"
              f"  mean {a.mean():+.4f}  down-RMS {rms(np.minimum(a, 0)):.4f}")
    for pool in ("tb", "ct"):
        d0 = np.concatenate(acc[(pool, "all", "d0")])
        nz = np.concatenate(acc[(pool, "all", "nz")])
        print(f"{pool}: RMS(du/u0)={rms(d0):.4f}  median|r| among nonzero={np.median(np.abs(nz)) if nz.size else 0:.4f}"
              f"  per-episode RMS(rel) {', '.join(f'{e.get(pool, float('nan')):.3f}' for e in per_ep)}")


if __name__ == "__main__":
    fire.Fire(main)
