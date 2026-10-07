"""Closed-loop check of probe B: the hub plays the episode in gym, but its KR gas slots (and lng override releases at
Hormuz / Malacca in the weeks probe B changed them) are replaced by probe B's final schedule; everything else (wafers,
chips, other fuel) is the hub reacting to the state. Shows whether the extra KR gas turns into lots and sales.

    uv run python lab/anastasiia/mpc_lab/level1/closed_kr.py --eps=26,45
"""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parents[3] / "outputs" / "level1" / "probeB"


def one(n, wide=False):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401
    from shockbench_flow_agent.convert import agent_config

    from probeB import AGENT, kr_slots, setup
    from sbf_starter import env_id
    from sbf_starter.agents import load

    LSF = setup()
    with open(OUT / f"lsb{'w' if wide else ''}_ep{n}.pkl", "rb") as f:
        res = pickle.load(f)
    d, R, rep = LSF.verify_one(n)  # the hub's open-loop replay, to rebuild probe B's final flows
    # probe B's final flows are not stored: re-apply its accepted moves' end values from the move log is not enough
    # (moves overlap), so probe B now stores them; fall back to an error if missing
    flows = res.get("flows")
    if flows is None:
        raise SystemExit("probe B result has no final flows (rerun with the storing version)")
    inst = R.inst
    ks = set(kr_slots(inst)) if not wide else set(res["slots"])
    env = gym.make(env_id("small"), entropy=111)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    ag = load(AGENT)(agent_config(info["static"], info["policy_seed"], u.layout, obs))
    done, t = False, 0
    while not done:
        a = ag.act(obs)
        a = {k: np.array(v, copy=True) for k, v in a.items()}
        for s in ks:
            a["flows"][s] = flows[t].get(s, 0.0)
        obs, _r, term, trunc, _i = env.step(a)
        done = term or trunc
        t += 1
    recs = u.core._ep.traj.records
    J = u.core._ep.traj.J_cents
    comp = {c: sum(getattr(r.costs, c) for r in recs) for c in ("shortage", "shed")}
    kr = inst.fabs.index(next(f for f in inst.fabs if inst.nodes[f].id == "fab_kr_memory_1"))
    lots = sum(r.lots_started[kr] for r in recs[:40])
    return n, res["before"], res["after"], J, comp, lots, res["comp0"]


def main(eps, wide: bool = False) -> None:
    which = [int(e) for e in eps] if isinstance(eps, (list, tuple)) else [int(e) for e in str(eps).split(",")]
    print("ep  hub J | open-loop probe B J (dJ) | closed loop: hub + probe B KR gas J (dJ), shortage / shed change, "
          "KR lots wk1-40 (M)")
    for n in which:
        n, before, after, J, comp, lots, comp0 = one(n, wide)
        print(f"{n:3d} {before / 1e11:8.1f} | {after / 1e11:8.1f} ({(before - after) / 1e11:6.1f}) | {J / 1e11:8.1f} "
              f"({(before - J) / 1e11:6.1f}), {(comp['shortage'] - comp0['shortage']) / 1e9:+6.1f} / "
              f"{(comp['shed'] - comp0['shed']) / 1e9:+6.1f}, KR lots {lots / 1e6:.2f}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
