"""Gate, part 2: how much of the teacher's advantage sits in strait releases vs in flows (played in gym).

On training episodes (root 555) only. Per episode, in the gymnasium env (same as ``teacher_gen.record``):

- ``teacher``        the stored teacher actions, open loop (must reproduce ``J_teacher``);
- ``t_flows+a_rel``  teacher flows (open loop) + the anchor's strait rules, closed loop on the actual observation;
- ``t_flows+def``    teacher flows + the default release everywhere;
- ``anchor``         ``agents/anastasiia_hybrid_hub`` closed loop;
- ``a_flows+t_rel``  the anchor's flows closed loop + the teacher's stored releases (open loop);
- ``t_chips``        the anchor closed loop, but its chip slots (wafer, raw, packaged) replaced by the teacher's;
- ``t_fuel+t_rel``   the anchor closed loop, but its fuel slots and releases replaced by the teacher's.

Also the share of each side's requested flow that the simulator executed (from ``last_week.clip``), per commodity.

    uv run python lab/anastasiia/mpc_lab/student/gate_play.py --episodes=0,1,2,3,4,5,6,7
"""

import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from prep import ANCHOR, load_agent_module, load_config, log  # noqa: E402

import sbf_starter  # noqa: E402,F401  (sets the reference cache dir)


def play(task, entropy, n, policy):
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": n})
    done, t = False, 0
    req, exe = [], []
    while not done:
        act = policy(t, obs)
        obs, _r, term, trunc, inf = env.step(act)
        req.append(np.asarray(obs["last_week.clip.requested"], dtype=np.float64))
        exe.append(np.asarray(obs["last_week.clip.executed"], dtype=np.float64))
        done = term or trunc
        t += 1
    J = int(env.unwrapped.core.trajectory.J_cents)
    return J, np.stack(req), np.stack(exe)


def main(task="small", entropy=555, episodes="0,1,2,3,4,5,6,7", src=None, only=None, tag=""):
    src = Path(src) if src else ROOT / "outputs" / "teacher" / f"{task}_{entropy}"
    eps = [int(x) for x in str(episodes).split(",")] if not isinstance(episodes, (list, tuple)) else list(episodes)
    config = load_config(src)
    module = load_agent_module(ANCHOR)
    slot_k = np.array([int(k) for k in config["static"]["action_slots"]["k"]])
    names = list(config["static"]["commodities"]["id"])
    v = np.array(config["static"]["commodities"]["v"])
    is_fuel = np.isin(slot_k, [names.index(c) for c in ("lng", "crude", "nucfuel") if c in names])
    rows = []
    for n in eps:
        z = np.load(src / f"ep{n}.npz")
        F, OQ, RM = z["act/flows"], z["act/override_qty"], z["act/release_mode"]
        zeros_oq, zeros_rm = np.zeros_like(OQ[0]), np.zeros_like(RM[0])
        t0 = time.time()
        out = {
            "ep": n,
            "J_teacher_stored": int(z["J_teacher"]),
            "J_naive": int(z["J_naive"]),
            "J_oracle": int(z["J_oracle"]),
        }

        if only == "t_fuel+a_rel":  # the flow-delta form's ceiling: teacher fuel flows, anchor releases and chips
            hub = module.Agent(config)

            def t_fuel_arel(t, o, hub=hub):
                a = hub.act(o)
                return {**a, "flows": np.where(is_fuel, F[t], a["flows"])}

            out["t_fuel+a_rel"], _, _ = play(task, entropy, n, t_fuel_arel)
            hub = module.Agent(config)
            out["anchor"], _, _ = play(task, entropy, n, lambda t, o, hub=hub: hub.act(o))
            out["teacher"] = out["J_teacher_stored"]
            rows.append(out)
            log(
                f"gate_play[{only}] ep{n}: bn USD vs teacher: anchor {(out['anchor'] - out['teacher']) / 1e11:+.1f}, "
                f"t_fuel+a_rel {(out['t_fuel+a_rel'] - out['teacher']) / 1e11:+.1f}"
            )
            continue
        out["teacher"], t_req, t_exe = play(
            task, entropy, n, lambda t, o: {"flows": F[t], "override_qty": OQ[t], "release_mode": RM[t]}
        )

        hub = module.Agent(config)

        def tf_arel(t, o, hub=hub):
            return {"flows": F[t], **hub.strait.fill(o)}

        out["t_flows+a_rel"], _, _ = play(task, entropy, n, tf_arel)
        out["t_flows+def"], _, _ = play(
            task, entropy, n, lambda t, o: {"flows": F[t], "override_qty": zeros_oq, "release_mode": zeros_rm}
        )
        hub = module.Agent(config)
        out["anchor"], a_req, a_exe = play(task, entropy, n, lambda t, o, hub=hub: hub.act(o))
        hub = module.Agent(config)

        def af_trel(t, o, hub=hub):
            a = hub.act(o)
            return {"flows": a["flows"], "override_qty": OQ[t], "release_mode": RM[t]}

        out["a_flows+t_rel"], _, _ = play(task, entropy, n, af_trel)

        # split by family: chips (wafer, raw, packaged) vs fuel (lng, crude, nucfuel; releases go with fuel)
        hub = module.Agent(config)

        def t_chips(t, o, hub=hub):
            a = hub.act(o)
            f = np.where(is_fuel, a["flows"], F[t])
            return {**a, "flows": f}

        out["t_chips"], _, _ = play(task, entropy, n, t_chips)
        hub = module.Agent(config)

        def t_fuel(t, o, hub=hub):
            a = hub.act(o)
            f = np.where(is_fuel, F[t], a["flows"])
            return {"flows": f, "override_qty": OQ[t], "release_mode": RM[t]}

        out["t_fuel+t_rel"], _, _ = play(task, entropy, n, t_fuel)
        for side, rq, ex in (("teacher", t_req, t_exe), ("anchor", a_req, a_exe)):
            out[f"exec_share_{side}"] = {
                names[k]: float(ex[:, slot_k == k].sum() / max(rq[:, slot_k == k].sum(), 1e-9))
                for k in range(len(names))
                if rq[:, slot_k == k].sum() > 0
            }
            out[f"exec_share_{side}_valw"] = float((ex * v[slot_k]).sum() / max((rq * v[slot_k]).sum(), 1e-9))
        rows.append(out)
        bn = lambda a, b: (out[a] - out[b]) / 1e2 / 1e9  # noqa: E731 - cents -> bn USD
        log(
            f"gate_play ep{n}: teacher J ok={out['teacher'] == out['J_teacher_stored']}; bn USD vs teacher: "
            f"anchor {bn('anchor', 'teacher'):+.1f}, t_flows+a_rel {bn('t_flows+a_rel', 'teacher'):+.1f}, "
            f"t_flows+def {bn('t_flows+def', 'teacher'):+.1f}, a_flows+t_rel {bn('a_flows+t_rel', 'teacher'):+.1f}, "
            f"t_chips {bn('t_chips', 'teacher'):+.1f}, t_fuel+t_rel {bn('t_fuel+t_rel', 'teacher'):+.1f}; "
            f"exec share teacher {out['exec_share_teacher_valw']:.3f} anchor {out['exec_share_anchor_valw']:.3f} "
            f"({time.time() - t0:.0f}s)"
        )
    if only:
        d = {k: np.array([(r[k] - r["teacher"]) / 1e11 for r in rows]) for k in ("anchor", only)}
        summ = {
            k: {"mean_bn_vs_teacher": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x)))}
            for k, x in d.items()
        }
        (ROOT / "outputs" / "student" / f"gate_play_{task}_{entropy}_{tag or 'only'}.json").write_text(
            json.dumps({"rows": rows, "summary": summ}, indent=2)
        )
        log(f"gate_play[{only}] summary: " + json.dumps(summ))
        return summ
    keys = ["anchor", "t_flows+a_rel", "t_flows+def", "a_flows+t_rel", "t_chips", "t_fuel+t_rel"]
    summ = {}
    for k in keys:
        d = np.array([(r[k] - r["teacher"]) / 1e11 for r in rows])
        summ[k] = {
            "mean_bn_vs_teacher": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
        }
    gap = np.array([(r["anchor"] - r["teacher"]) / 1e11 for r in rows])
    rel = np.array([(r["t_flows+a_rel"] - r["teacher"]) / 1e11 for r in rows])
    summ["release_share_of_gap"] = float(rel.sum() / gap.sum())
    for side in ("teacher", "anchor"):
        summ[f"exec_share_{side}"] = {
            k: float(np.mean([r[f"exec_share_{side}"].get(k, np.nan) for r in rows])) for k in names
        }
    out_dir = ROOT / "outputs" / "student"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"gate_play_{task}_{entropy}.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=2))
    log("gate_play summary (bn USD/ep, + = worse than teacher): " + json.dumps(summ))
    return summ


if __name__ == "__main__":
    import fire

    fire.Fire(main)
