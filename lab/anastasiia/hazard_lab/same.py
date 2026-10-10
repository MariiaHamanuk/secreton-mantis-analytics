"""Does a folder of the model act as the reference folder does, week by week, and how much CPU does each take?

    uv run python lab/anastasiia/hazard_lab/same.py record outputs/speed_lab/ref --task=full --episode=3 --weeks=30
    uv run python lab/anastasiia/hazard_lab/same.py check outputs/speed_lab/s1 --task=full --episode=3 --weeks=30

For a change that must not move a single action (a speed-up). Both commands play the first ``weeks`` weeks of one
episode of root 444 in this process, with the folder's files and the model's settings without its clock (the
``regime.json`` of ``outputs/hazard_lab/agents/h3_s`` or ``h3_f``: a clock would make the actions depend on the
machine), and without the solver's time limits, which are wall-clock seconds and on a busy machine send a run down
another path. The folder is copied to a temporary one first, so its own ``regime.json`` is not read and nothing is
written into it.

``record`` keeps each week's action arrays, the agent's note of the week (the model's cost of the reference plan and
of the chosen one, and what the search did) and the CPU seconds, in ``outputs/speed_lab/records/``. ``check`` plays
the same weeks and compares them with the record: the arrays bit for bit, the notes as they are. It prints the first
week that differs (or that all are the same) and the CPU seconds of both, in all and by the model's own kinds of
work. A folder is checked in a process of its own because the folders share module names (``sbfv``).
"""

import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "speed_lab" / "records"
REGIME = {"small": "h3_s", "full": "h3_f"}
NO_LIMITS = {"warm_share": 0, "solve_share": 0, "solve_seconds": 600}  # no run of the solver is cut short


def _play(folder: str, task: str, entropy: int, episode: int, weeks: int) -> dict:
    import gymnasium as gym
    import shockbench_flow_gym  # noqa: F401 - registers the environments
    from shockbench_flow_agent.convert import agent_config

    from sbf_starter import env_id
    from sbf_starter.agents import load

    source = Path(folder).resolve()
    with tempfile.TemporaryDirectory(prefix="same_") as tmp:
        target = Path(tmp) / f"{source.name}_{task}"
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        regime = json.loads((ROOT / "outputs" / "hazard_lab" / "agents" / REGIME[task] / "regime.json").read_text())
        (target / "regime.json").write_text(json.dumps(regime | NO_LIMITS))
        env = gym.make(env_id(task), entropy=entropy)
        obs, info = env.reset(options={"episode": episode})
        t0 = time.process_time()
        agent = load(str(target))(agent_config(info["static"], info["policy_seed"], env.unwrapped.layout, obs))
        rows, done = [], False
        while not done and len(rows) < weeks:
            action = agent.act(obs)
            cpu = time.process_time() - t0
            arrays = {k: np.array(v) for k, v in action.items()}
            rows.append({"action": arrays, "note": tuple(agent.log[-1][1:]), "cpu": cpu})
            obs, _r, term, trunc, _i = env.step(action)
            done = term or trunc
            t0 = time.process_time()
        took = {kind: float(np.sum(v)) for kind, v in getattr(agent, "took", {}).items()}
    return {"rows": rows, "took": took}


def _path(task: str, entropy: int, episode: int, weeks: int) -> Path:
    return OUT / f"{task}_{entropy}_{episode}_{weeks}_nolimit.pkl"


def record(folder: str, task: str = "full", episode: int = 3, weeks: int = 30, entropy: int = 444) -> None:
    played = _play(folder, task, entropy, episode, weeks)
    OUT.mkdir(parents=True, exist_ok=True)
    _path(task, entropy, episode, weeks).write_bytes(pickle.dumps(played))
    print(f"{folder}: {len(played['rows'])} weeks of {task} {entropy} episode {episode} kept, "
          f"{sum(r['cpu'] for r in played['rows']):.1f} CPU s")  # fmt: skip


def check(folder: str, task: str = "full", episode: int = 3, weeks: int = 30, entropy: int = 444) -> None:
    if not _path(task, entropy, episode, weeks).is_file():
        raise SystemExit(f"{_path(task, entropy, episode, weeks).name}: the record is being made, try in a few minutes")
    kept = pickle.loads(_path(task, entropy, episode, weeks).read_bytes())
    played = _play(folder, task, entropy, episode, weeks)
    first = None
    for week, (a, b) in enumerate(zip(kept["rows"], played["rows"]), 1):
        keys = sorted(set(a["action"]) | set(b["action"]))
        moved = [k for k in keys if k not in a["action"] or k not in b["action"]
                 or not np.array_equal(a["action"][k], b["action"][k])]  # fmt: skip
        if moved or a["note"] != b["note"]:
            first = (week, moved, a["note"], b["note"])
            break
    ref, new = sum(r["cpu"] for r in kept["rows"]), sum(r["cpu"] for r in played["rows"])
    if first is None and len(kept["rows"]) == len(played["rows"]):
        print(f"{folder}: the same in all {len(played['rows'])} weeks of {task} {entropy} episode {episode}")
    elif first is None:
        print(f"{folder}: {len(played['rows'])} weeks played, the record has {len(kept['rows'])}")
    else:
        week, moved, a, b = first
        print(f"{folder}: DIFFERS from week {week} of {task} {entropy} episode {episode}: arrays {moved or 'the same'}")
        print(f"  the record's note: {a}\n  this folder's:     {b}")
    print(f"  CPU seconds: the record {ref:.1f}, this folder {new:.1f} ({new / ref - 1:+.1%})")
    for kind in sorted(set(kept["took"]) | set(played["took"])):
        a, b = kept["took"].get(kind, 0.0), played["took"].get(kind, 0.0)
        if a or b:
            print(f"    {kind:8s} {a:7.1f} -> {b:7.1f}")


if __name__ == "__main__":
    fire.Fire({"record": record, "check": check})
