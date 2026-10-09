"""What an agent sees about running cuts: play episodes with zero flows and compare the observation with the truth.

    uv run python lab/anastasiia/hazard_lab/observe_check.py --task=small --episodes=12

Root 333. For each played episode it checks (1) that closure_end is never observed in the scored regime, (2) that
graph_now.u / u0 equals the marks' instantaneous u_now / u0, (3) when graph_now.war_risk rises and falls against the
closures and conflicts that set it, (4) which message kinds and channels appear and whether any message ever
refers to the end of a running event, (5) the fab / OSAT restoration factors against a regional conflict's dead time.
"""

import fire
import numpy as np


def main(task: str = "small", episodes: int = 12, entropy: int = 333) -> None:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401  (registers the environments)
    from shockbench_flow import marks as M
    from shockbench_flow.hosting.tasks import scenario, task_generator
    from shockbench_flow.omega import codes

    inst, params = task_generator(task)
    env = gym.make(f"ShockBench/{task.capitalize()}-v0", entropy=entropy)
    u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
    T = inst.T
    closure_seen = 0
    u_err = 0.0
    kinds, chans, withdrawals, msg_rows = set(), set(), 0, 0
    war_checks = []  # (kind, rises at expected week, falls at expected week)
    rf_checks = []
    for seed in range(episodes):
        obs, info = env.reset(seed=seed)
        n = info["episode"]
        om = scenario(task, n, entropy=entropy)
        evs = M.read_events(inst, om)
        gm = M.graph_marks(inst, evs, params.marks)
        Rf, Ro = M.restoration_factors(inst, evs, params.marks)
        wr = M._war_risk_class(inst, evs, params.marks, np.arange(1, T + 1, dtype=float))
        obs_wr, obs_rf = [], []
        for t in range(1, T + 1):
            closure_seen += int(obs["closure_end.chokepoint.observed"].sum() + obs["closure_end.end_week.observed"].sum())
            ob = obs["graph_now.u.observed"].astype(bool)
            if ob.any():
                ratio = obs["graph_now.u"][ob] / u0[ob]
                truth = gm["u_now"][t - 1][ob] / u0[ob]
                u_err = max(u_err, float(np.abs(ratio - truth).max()))
            obs_wr.append(obs["graph_now.war_risk"].copy())
            obs_rf.append(obs["graph_now.fab.R"].copy())
            m = obs["messages.kind.observed"].astype(bool)
            msg_rows += int(m.sum())
            kinds.update(obs["messages.kind"][m].tolist())
            chans.update(obs["messages.channel"][m].tolist())
            withdrawals += int((obs["messages.kind"][m] == 4).sum())
            obs, _, term, trunc, _ = env.step({"flows": np.zeros(env.action_space["flows"].shape),
                                               "override_qty": np.zeros(env.action_space["override_qty"].shape),
                                               "release_mode": np.zeros(env.action_space["release_mode"].shape, dtype=int)})
            if term or trunc:
                break
        obs_wr = np.array(obs_wr)
        war_checks.append(float(np.abs(obs_wr[: len(wr)] - wr[: len(obs_wr)]).max()))
        obs_rf = np.array(obs_rf)
        rf_checks.append(float(np.abs(obs_rf - np.ones_like(obs_rf)).max()))
        for q in evs:
            if q.type == 4 and q.T0 == q.T0 and 0 <= q.onset < T - 25:
                print(f"episode {n}: conflict onset {q.onset:.1f} dur {q.duration:.1f} T0 {q.T0:.1f} tau {q.tau_rho:.1f};"
                      f" min fab R by week: {np.round(obs_rf.min(axis=1)[int(q.onset):int(q.onset) + 30:3], 2).tolist()}")
    print(f"closure_end observed entries over {episodes} episodes: {closure_seen}")
    print(f"max |graph_now.u/u0 - u_now/u0| = {u_err:.2e}")
    print(f"max |obs war_risk - marks war_risk(week t)| per episode: {war_checks}")
    print(f"messages seen: rows {msg_rows}, kinds {sorted(codes.MESSAGE_KINDS[k] for k in kinds)}, "
          f"channels {sorted(codes.CHANNELS[c] for c in chans)}, withdrawals {withdrawals}")


if __name__ == "__main__":
    fire.Fire(main)
