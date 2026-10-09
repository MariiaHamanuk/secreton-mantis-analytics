"""How well ``src/watch.py`` tells the short disruptions from the observations: checked against the scenario's events.

    uv run python lab/anastasiia/hazard_lab/watch_check.py --task=small --entropy=333 --episodes=40

Each episode is played with zero flows (the network's marks do not depend on the action), the watch reads every
week's ``graph_now`` as the agent's would, and its list of running short cuts is compared, week by week and element by
element (an edge for a port strike, a strait for a weather closure), with the events the scenario really has running
at the instant the observation shows. Printed per kind: element-weeks the scenario has, the share the watch names
(recall), element-weeks the watch names, the share that are true (precision), how often the week it first saw a cut
is the event's first week, and how well the law's forecast of the end is calibrated (the share of watched
element-weeks that are over by the week the law gives for a chance q, which should be q).
"""

import sys
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "src")]
KINDS = ("weather_closure", "port_strike_stoppage", "port_strike_slowdown")
QS = (0.25, 0.5, 0.75, 0.9)


def episode(task: str, entropy: int, n: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    import watch as W
    from shockbench_flow_agent.convert import agent_config

    import play
    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    u = env.unwrapped
    inst = u.core._ep.inst
    events = play.events_of(inst, u._omega(n))
    layout = agent_config(info["static"], info["policy_seed"], u.layout, obs)["layout"]
    straits = [inst.chokepoint_ordinal[int(c)] for c in layout["chokepoints"]]
    watch = W.Watch(inst)
    zero = {"flows": np.zeros(env.action_space["flows"].shape),
            "override_qty": np.zeros(env.action_space["override_qty"].shape),
            "release_mode": np.zeros(env.action_space["release_mode"].shape, dtype=int)}  # fmt: skip
    u_seen = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges], dtype=float)
    o_seen, risk_seen = np.ones(len(inst.chokepoints)), np.zeros(len(inst.chokepoints), dtype=int)
    out = {k: {"true": 0, "named": 0, "watched": 0, "right": 0, "first": [0, 0], "over": {q: [0, 0] for q in QS}} for k in KINDS}
    T, done, week = inst.T, False, 1
    while not done:
        shown = obs["graph_now.u.observed"] == 1
        u_seen[shown] = obs["graph_now.u"][shown]
        shown = obs["graph_now.open.observed"] == 1
        o_seen[np.asarray(straits)[shown]] = obs["graph_now.open"][shown]
        shown = obs["graph_now.war_risk.observed"] == 1
        risk_seen[np.asarray(straits)[shown]] = obs["graph_now.war_risk"][shown]
        watch.see(week, u_seen, o_seen, risk_seen)
        row = week - 1
        truth = {k: {} for k in KINDS}  # kind -> element -> (the event's first observed week, the week it is over)
        for ev in events:
            kind = ev["type"]
            if kind == "port_strike":
                kind += "_stoppage" if ev["severity"] >= 0.9 else "_slowdown"
            if kind not in truth:
                continue
            idx, _avg, now = ev["o" if kind == "weather_closure" else "u"]
            if not len(idx) or not (now[row] < 1.0 - 1e-12).any():
                continue
            on = np.flatnonzero((now < 1.0 - 1e-12).any(axis=1))
            for j in np.flatnonzero(now[row] < 1.0 - 1e-12):
                truth[kind][int(idx[j])] = (int(on[0]) + 1, int(on[-1]) + 2)  # weeks: first seen, first seen over
        seen = {k: {} for k in KINDS}
        for e, cuts in watch.cuts.items():
            for kind, _factor, since, _carried in cuts:
                seen[kind][e] = since
        for c, (since, _back, _carried) in watch.shut.items():
            seen["weather_closure"][c] = since
        for kind in KINDS:
            s = out[kind]
            s["true"] += len(truth[kind])
            s["named"] += sum(1 for e in truth[kind] if e in seen[kind])
            s["watched"] += len(seen[kind])
            for e, since in seen[kind].items():
                if e not in truth[kind]:
                    continue
                s["right"] += 1
                first, over = truth[kind][e]
                s["first"][0] += since == first
                s["first"][1] += 1
                elapsed = week - since + 0.5
                for q in QS:  # is it over by the week the law gives for chance q (only where the episode lasts that long)
                    left = W.law_end(W.LAWS[kind], elapsed, q) - elapsed
                    by = week + int(np.ceil(left))  # the first week whose observation would show it over
                    if by <= T:
                        s["over"][q][1] += 1
                        s["over"][q][0] += over <= by
        obs, _r, term, trunc, _i = env.step(zero)
        done, week = term or trunc, week + 1
    return {"n": n, "stats": out, "unread": watch.unread}


def main(task: str = "small", entropy: int = 333, episodes: int = 40, first: int = 0, n_jobs: int = 3) -> None:
    rows = Parallel(n_jobs=n_jobs)(delayed(episode)(task, entropy, n) for n in range(first, first + episodes))
    print(f"{task} {entropy}, episodes {first}..{first + episodes - 1}; changes of an edge's capacity left unread: "
          f"{sum(r['unread'] for r in rows)}")
    print(f"{'kind':22} {'true el-weeks':>13} {'recall':>7} {'watched':>8} {'precision':>9} {'first week right':>16}  "
          + "  ".join(f"over by q{int(100 * q)}" for q in QS))
    for kind in KINDS:
        tot = {k: sum(r["stats"][kind][k] for r in rows) for k in ("true", "named", "watched", "right")}
        first_ok = [sum(r["stats"][kind]["first"][i] for r in rows) for i in (0, 1)]
        over = {q: [sum(r["stats"][kind]["over"][q][i] for r in rows) for i in (0, 1)] for q in QS}
        print(f"{kind:22} {tot['true']:13d} {tot['named'] / max(tot['true'], 1):7.3f} {tot['watched']:8d} "
              f"{tot['right'] / max(tot['watched'], 1):9.3f} {first_ok[0] / max(first_ok[1], 1):16.3f}  "
              + "  ".join(f"{over[q][0] / max(over[q][1], 1):11.2f}" for q in QS))


if __name__ == "__main__":
    fire.Fire(main)
