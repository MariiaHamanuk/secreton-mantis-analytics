"""The cell program as a teacher: (observation, action) pairs of full-knowledge plans, for imitation learning.

    uv run python lab/anastasiia/regime_lab/teacher.py generate --task=small --entropy=555 --episodes=16 --n_jobs=3
    uv run python lab/anastasiia/regime_lab/teacher.py report --task=small --entropy=555

Per episode: an agent (the hybrid) plays it once under gymnasium (kept by ``../search_lab/search.py``); the loop
with the whole future known improves that trajectory (``core.switch_on``); the plan is played under gymnasium and
every week's observation and action are kept. The files have the layout of ``../mpc_lab/teacher/`` (its README has
the table of keys), so a student reads either; they go to ``outputs/teacher_regime/<task>_<entropy>/``. Roots 0, 111,
222, 333 and 444 are refused: training episodes must not meet the tuning and the formal sets.

Differences from that layout: ``J_plan_played`` is the start (the agent's own cost), ``J_claimed`` the last cell's
claim, ``meta`` names the start agent, the grid-weeks switched on and the seconds.
"""

import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "mpc_lab"), str(HERE.parent / "search_lab")]
import core  # noqa: E402
import plan  # noqa: E402


REFUSED = (0, 111, 222, 333, 444)
FORMAT_VERSION = 1


def out_dir(task: str, entropy: int) -> Path:
    return plan.ROOT / "outputs" / "teacher_regime" / f"{task}_{entropy}"


def record(task: str, entropy: int, n: int, acts: list, cfg_path: Path) -> dict:
    """Play ``acts`` in the gymnasium environment: weekly observations, flat actions, reward cents, the cost."""
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow.information.flat import flat_from_action
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id

    env = gym.make(env_id(task), entropy=entropy)
    u = env.unwrapped
    obs, info = env.reset(options={"episode": n})
    if not cfg_path.exists():
        tmp = cfg_path.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as f:  # as ../mpc_lab/teacher/teacher_gen.py writes it
            pickle.dump(agent_config(info["static"], info["policy_seed"], u.layout, obs), f)
        os.replace(tmp, cfg_path)
    F, OQ, RM = [], [], []
    for t, act in enumerate(acts, start=1):
        _f, oq, rm = flat_from_action(u.layout, core.Episode.as_wire(act, t))
        f = np.zeros(u.layout.n_slots, dtype=np.float64)
        for s, q in act[0].items():
            f[s] = q
        F.append(f), OQ.append(oq), RM.append(rm)
    seen, rewards, done, t = [], [], False, 0
    while not done:
        seen.append(obs)
        obs, _r, term, trunc, inf = env.step({"flows": F[t], "override_qty": OQ[t], "release_mode": RM[t]})
        rewards.append(int(inf["reward_cents"]))
        done = term or trunc
        t += 1
    traj = u.core.trajectory
    return dict(obs={k: np.stack([o[k] for o in seen]) for k in seen[0]}, flows=np.stack(F), override_qty=np.stack(OQ),
                release_mode=np.stack(RM), reward_cents=np.array(rewards, dtype=np.int64), J=int(traj.J_cents),
                salvage_cents=int(traj.salvage_cents), omega_hash=traj.omega_hash,
                invalid=sum(len(r.invalid) for r in traj.records))


def episode(task: str, entropy: int, n: int, agent: str, passes: int) -> dict:
    import search  # the search lab's record of what an agent played

    from sbf_starter import scoring

    folder = out_dir(task, entropy)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"ep{n}.npz"
    if not path.exists():
        t0 = time.time()
        ep = core.Episode.of(task, entropy, n)
        start = ep.validated(search.played(agent, task, entropy, n)["actions"])
        d = core.switch_on(ep, start, passes=passes, last_week=ep.T - 12)
        rec = record(task, entropy, n, d["acts"], folder / "config.pkl.gz")
        ref = scoring.episode_set(task, [n], entropy=entropy, n_jobs=1, verbose=False).references[0]
        claims = [h[0] for h in d["hist"] if h[1] is not None]
        meta = dict(format=FORMAT_VERSION, task=task, entropy=entropy, episode=n, teacher="regime_lab", start=agent,
                    switched=d["switched"], passes=len(d["hist"]), total_secs=time.time() - t0,
                    gym_invalid_entries=rec["invalid"], omega_hash=rec["omega_hash"], ref_omega_hash=ref["omega_hash"])
        arrays = {f"obs/{k}": v for k, v in rec["obs"].items()}
        arrays.update({
            "act/flows": rec["flows"], "act/override_qty": rec["override_qty"], "act/release_mode": rec["release_mode"],
            "reward_cents": rec["reward_cents"], "J_teacher": np.int64(rec["J"]), "J_replay": np.int64(d["J"]),
            "J_plan_played": np.int64(ep.simulate(start)[1]), "J_claimed": np.int64(round(claims[-1] * 100) if claims else d["J"]),
            "J_naive": np.int64(ref["J_naive_cents"]), "J_oracle": np.int64(ref["J_oracle_cents"]),
            "stratum": np.int64(ref["stratum"] or 0), "salvage_cents": np.int64(rec["salvage_cents"]),
            "meta": np.array(json.dumps(meta, default=str)),
        })
        tmp = folder / f"ep{n}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    z = np.load(path)
    return {k: int(z[k]) for k in ("J_teacher", "J_replay", "J_plan_played", "J_naive", "J_oracle", "stratum")} | {"n": n}


def generate(task: str = "small", entropy: int = 555, episodes: int = 16, start: int = 0,
             agent: str = "agents/anastasiia_hybrid_hub", passes: int = 60, n_jobs: int = 3) -> None:
    if entropy in REFUSED:
        raise SystemExit(f"root {entropy} is a tuning, formal or statistics root: take a training root (555)")
    rows = Parallel(n_jobs=n_jobs)(delayed(episode)(task, entropy, n, agent, passes) for n in range(start, start + episodes))
    _print(rows)


def report(task: str = "small", entropy: int = 555) -> None:
    rows = []
    for path in sorted(out_dir(task, entropy).glob("ep*.npz"), key=lambda p: int(p.stem[2:])):
        z = np.load(path)
        rows.append({k: int(z[k]) for k in ("J_teacher", "J_replay", "J_plan_played", "J_naive", "J_oracle", "stratum")} | {"n": int(path.stem[2:])})
    _print(rows)


def _print(rows: list[dict]) -> None:
    refs = [{"J_naive_cents": r["J_naive"], "J_oracle_cents": r["J_oracle"], "stratum": r["stratum"]} for r in rows]
    for r in rows:
        room = r["J_naive"] - r["J_oracle"]
        print(f"ep {r['n']:3d} lvl {r['stratum']}: start {(r['J_naive'] - r['J_plan_played']) / room:.4f}  teacher {(r['J_naive'] - r['J_teacher']) / room:.4f}"
              f"{'' if r['J_teacher'] == r['J_replay'] else '  WARNING: gymnasium and the replay differ'}")
    print(f"{len(rows)} episodes: start {plan.both(refs, [r['J_plan_played'] for r in rows])}  teacher {plan.both(refs, [r['J_teacher'] for r in rows])}"
          "  (as `sbf evaluate` [levels weighing the same])")


if __name__ == "__main__":
    fire.Fire({"generate": generate, "report": report})
