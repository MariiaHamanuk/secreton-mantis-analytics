"""Where a teacher's gain over an agent sits, by the kind of decision: the agent plays, a teacher's entries are grafted in.

    uv run python lab/anastasiia/plan_lab/graft.py run --channels=wafer --weeks=1-52 --task=small --episodes=8
    uv run python lab/anastasiia/plan_lab/graft.py run --channels=wafer,valve --teacher=outputs/plan_lab/agents/lev/te_everything
    uv run python lab/anastasiia/plan_lab/graft.py table --task=small --episodes=8

The base agent (``--agent``, a folder built like ``agents/anastasiia_plan_hull``) plays the episode in closed loop. In
the weeks ``--weeks`` (first-last, 1-based) the entries of the named channels are replaced by the teacher's of the
same week; everything else stays the agent's, which goes on from the state the mix leaves. Channels: ``order`` (fuel
from the sources), ``valve`` (terminal into grid), ``wafer``, ``raw`` and ``pack`` (chips before and after the
plants), ``strait`` (tanker releases and holds).

The teacher is either a recorded plan (the full-knowledge plan with the hull of ``lab/anastasiia/regime_lab/plan.py
starts --hull``: ``outputs/regime_lab/hull/<task>_<entropy>_<n>_hybrid.pkl``, its weekly actions as played) or, with
``--teacher``, another agent folder played alongside on the same episode in its own environment (it is handed the
scenario if it asks for it, so a planner that knows part of the future can be the teacher).

``run`` keeps each episode's cost under the tag ``graft_<teacher>_<channels>_<weeks>`` in the harness's costs;
``table`` prints every such tag against the base agent's own costs (tag ``graft_base``, written by ``run
--channels=none``): USD saved per episode and the share of the teacher's whole gain.
"""

import json
import pickle
import time
from pathlib import Path

import fire
import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[3]
CHANNELS = ("order", "valve", "wafer", "raw", "pack", "strait")


def _play(agent: str, teacher: str, task: str, entropy: int, n: int, channels: tuple, first: int, last: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    import sbf_starter.agents as agents_mod
    from sbf_starter import env_id

    def start(folder):
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": n})
        u = env.unwrapped
        ag = agents_mod.load(folder)(agent_config(info["static"], info["policy_seed"], u.layout, obs))
        if hasattr(ag, "tell_truth"):
            ag.tell_truth({"inst": u.instance, "marks": u.core._ep.marks, "task": task, "entropy": entropy, "episode": n, "env": u})
        return env, obs, ag

    env, obs, ag = start(agent)
    kinds = np.array(ag.kinds)
    live = None
    if teacher:
        live = start(teacher)
        acts = None
    else:
        acts = pickle.loads((ROOT / "outputs" / "regime_lab" / "hull" / f"{task}_{entropy}_{n}_hybrid.pkl").read_bytes())["acts"]
    week, done = 0, False
    while not done:
        week += 1
        action = ag.act(obs)
        other = None
        if live is not None:
            env_t, obs_t, ag_t = live
            other = ag_t.act(obs_t)
            obs_t, _r, term_t, trunc_t, _i = env_t.step(other)
            live = (env_t, obs_t, ag_t)
        elif week <= len(acts):
            other = ag._arrays(acts[week - 1])
        if other is not None and first <= week <= last:
            flows = np.array(action["flows"], dtype=float)
            for c in channels:
                if c == "strait":
                    action["override_qty"] = np.array(other["override_qty"])
                    action["release_mode"] = np.array(other["release_mode"])
                elif c in CHANNELS:
                    take = kinds == c
                    flows[take] = np.asarray(other["flows"], dtype=float)[take]
            action["flows"] = flows
        obs, _r, term, trunc, _i = env.step(action)
        done = term or trunc
    return {"episode": n, "J_cents": int(env.unwrapped.core._ep.traj.J_cents), "cpu_mean": 0.0, "cpu_p95": 0.0,
            "cpu_max": 0.0, "invalid": 0, "notes": None}


def _tag(teacher: str, channels: tuple, weeks: str) -> str:
    who = Path(teacher).name if teacher else "plan"
    return f"graft_{who}_{'+'.join(channels) if channels else 'none'}_{weeks}"


def run(channels: str | tuple = "none", weeks: str = "", agent: str = "agents/anastasiia_plan_hull", teacher: str = "",
        task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0, n_jobs: int = 4) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import harness

    names = tuple(c for c in (channels.split(",") if isinstance(channels, str) else channels) if c and c != "none")
    T = {"tiny": 26, "small": 52, "full": 104}[task]
    a, b = (int(x) for x in weeks.split("-")) if weeks else (1, T)
    tag = "graft_base" if not names else _tag(teacher, names, f"{a}-{b}")
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(_play)(str(ROOT / agent), str(ROOT / teacher) if teacher else "", task, entropy, n,
                                                  names, a, b) for n in range(first, first + episodes))
    path = harness.COSTS / f"{task}_{entropy}.json"
    book = json.loads(path.read_text()) if path.is_file() else {}
    rows = book.setdefault(tag, {})
    for o in out:
        rows[str(o["episode"])] = o
    path.write_text(json.dumps(book))
    print(f"{tag}: {len(out)} episodes of {task} {entropy} in {time.time() - t0:.0f} s; mean cost "
          f"{np.mean([o['J_cents'] for o in out]) / 1e11:.1f} bn")


def table(task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0, teacher: str = "plan") -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import harness

    book = json.loads((harness.COSTS / f"{task}_{entropy}.json").read_text())
    eps = [str(n) for n in range(first, first + episodes)]
    base = np.array([book["graft_base"][n]["J_cents"] for n in eps], dtype=float) / 1e11
    refs = harness.references(task, entropy, first + episodes)[first:]
    room = np.mean([(r["J_naive_cents"] - r["J_oracle_cents"]) for r in refs]) / 1e11
    whole = None
    if teacher == "plan":
        whole = np.array([pickle.loads((ROOT / "outputs" / "regime_lab" / "hull" / f"{task}_{entropy}_{n}_hybrid.pkl").read_bytes())["J"]
                          for n in eps], dtype=float) / 1e11
    elif f"lev_{teacher}" in book:
        whole = np.array([book[f"lev_{teacher}"][n]["J_cents"] for n in eps], dtype=float) / 1e11
    gap = float(np.mean(base - whole)) if whole is not None else float("nan")
    print(f"{task} {entropy}, episodes {eps[0]}..{eps[-1]}: the base costs {base.mean():.1f} bn an episode, the teacher "
          f"({teacher}) {gap:.1f} bn less ({gap / room:+.4f} of the score); 0.01 of the score = {room / 100:.2f} bn")
    print(f"{'graft':44} {'bn saved':>9} {'90% interval':>18} {'of the gap':>11} {'better':>7}")
    rng = np.random.default_rng(0)
    for tag in sorted(t for t in book if t.startswith(f"graft_{teacher}_")):
        if any(n not in book[tag] for n in eps):
            continue
        saved = base - np.array([book[tag][n]["J_cents"] for n in eps], dtype=float) / 1e11
        boots = [saved[rng.integers(0, len(saved), len(saved))].mean() for _ in range(4000)]
        print(f"{tag[len('graft_' + teacher) + 1:]:44} {saved.mean():9.1f} {np.quantile(boots, 0.05):8.1f}..{np.quantile(boots, 0.95):<8.1f} "
              f"{saved.mean() / gap if gap else float('nan'):11.0%} {int((saved > 0).sum()):4d}/{len(eps)}")


if __name__ == "__main__":
    fire.Fire({"run": run, "table": table})
