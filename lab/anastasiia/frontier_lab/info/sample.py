"""Stage 1: draw episodes of the public generator on a free root and keep what the filter study needs.

python sample.py <task> <first episode> <episodes> <folder of the draws> [root]
"""

import pickle
import sys
import time

import numpy as np
from shockbench_flow.disruption import announce, sampler
from shockbench_flow.hosting.tasks import task_generator


ROOT = int(sys.argv[5]) if len(sys.argv) > 5 else 1001  # the root of the draws (the study's own: 1001)
PRE = 60.0  # weeks of history before the instant 0 kept for the excitation state


def one(task, inst, p, ep):
    s = sampler.sample_events(inst, p, ROOT, ep)
    ev_lead, _V = announce.sample_leads(inst, p, s.stored, ROOT, ep)
    cap = {}
    orig = announce.shadow_cascade

    def wrap(*a, **k):
        cap["c"] = orig(*a, **k)
        return cap["c"]

    announce.shadow_cascade = wrap
    try:
        sh = announce.sample_shadows(inst, p, s.regimes, ROOT, ep, rules=s.rules)
    finally:
        announce.shadow_cascade = orig
    r, B, T = s.regimes, p.burn_in, inst.T
    cols = slice(B - 1, B + T + 1)  # weeks -1 .. T
    out = dict(task=task, ep=ep, T=T)
    out["X"], out["W"] = r.X[:, cols].astype(np.float32), r.W[:, cols].astype(np.float32)
    out["z_c"], out["z_p"], out["z_dyad"], out["z_own"] = (
        np.asarray(a[:, cols]) for a in (r.z_c, r.z_p, r.z_dyad, r.z_c_own)
    )
    stored_keys = {e.key for e in s.stored}
    # the cluster process: every raw event from PRE weeks before 0 (no-op included)
    cl = []
    for raw, m in zip(s.raw, s.marked):
        if raw.onset < -PRE:
            continue
        par = s.raw[raw.parent] if raw.parent >= 0 else None
        cl.append(
            (
                raw.block,
                raw.region,
                raw.onset,
                -1 if m is None else m.type,
                -1 if par is None else par.block,
                -1 if par is None else par.region,
                np.nan if par is None else par.onset,
                -1 if m is None else m.target_kind,
                -1 if m is None else m.target,
                -1 if m is None else m.commodity,
                -1 if m is None else m.counterpart,
                np.nan if m is None else m.duration,
                np.nan if m is None else m.severity,
                0 if m is None else int(m.persistent),
                int(m is not None and m.key in stored_keys),
            )
        )
    out["cluster"] = np.array(cl, dtype=np.float64).reshape(-1, 15)
    lead_of = {e.key: ev_lead[i] for i, e in enumerate(s.stored)}
    der_keys = {e.key for e in s.derived}
    st = []
    for e in s.stored:
        st.append(
            (
                e.type,
                e.block,
                e.region,
                e.counterpart,
                e.target_kind,
                e.target,
                e.commodity,
                e.onset,
                e.duration,
                e.severity,
                int(e.persistent),
                int(e.key in der_keys),
                e.T0,
                e.tau_rho,
                *lead_of[e.key],
            )
        )
    out["stored"] = np.array(st, dtype=np.float64).reshape(-1, 20)
    out["shadows"] = np.array(
        [
            (
                x.event.type,
                x.event.block,
                x.event.region,
                x.event.target_kind,
                x.event.target,
                x.event.commodity,
                x.event.onset,
                x.channel,
                x.u_decoy,
                *x.lead,
            )
            for x in sh
        ],
        dtype=np.float64,
    ).reshape(-1, 15)
    c = cap["c"]
    out["shadow_raw"] = np.array(
        [(e.block, e.region, e.onset) for e in c.raw if e.onset >= -PRE], dtype=np.float64
    ).reshape(-1, 3)
    out["shadow_scale"] = c.scale
    return out


def main(task, start, n):
    inst, p = task_generator(task)
    t0 = time.process_time()
    eps = [one(task, inst, p, ep) for ep in range(start, start + n)]
    path = f"{sys.argv[4]}/{task}_{ROOT}_{start}_{start + n}.pkl"
    with open(path, "wb") as f:
        pickle.dump(eps, f, protocol=4)
    print(task, start, n, "cpu", round(time.process_time() - t0, 1), "->", path)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
