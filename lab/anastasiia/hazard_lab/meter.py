"""Play an agent folder as the scorer plays it, under its CPU meter; the costs go where ``play.py show`` reads them.

    uv run python lab/anastasiia/hazard_lab/meter.py outputs/hazard_lab/agents/m3c --tag=m3c --task=small --scale=0.54

A week over the budget is the naive rule's, and what it took over is charged to the next week. ``scale`` multiplies
the board's budget (2 s on Small, 4 s on Full): a server ``1 / scale`` times slower than this machine is played here
with the budget ``scale`` times the board's, and the folder's ``regime.json`` must hold the same ``budget_scale``, or
the model's own clock aims at another budget (the folder is refused otherwise). A model with a clock is played alone
on the machine, in no more processes than it has fast cores.

Kept per episode in ``outputs/hazard_lab/play/<tag>_<task>_<entropy>.pkl``: the cost ``J``, the weeks the naive rule
played, the weeks over the budget, and the episode's mean CPU seconds a week (the scorer hands back no single week).
"""

import json
import pickle
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs" / "hazard_lab" / "play"
BOARD = {"small": 2.0, "full": 4.0}  # the boards' CPU seconds a week


def main(agent: str, tag: str, task: str = "small", entropy: int = 111, episodes: int = 64, first: int = 0,
         scale: float = 1.0, n_jobs: int = 8) -> None:
    from sbf_starter import scoring

    folder = Path(agent).resolve()
    own = float(json.loads((folder / "regime.json").read_text()).get("budget_scale", 1.0))
    if abs(own - scale) > 1e-9:
        raise SystemExit(f"{folder.name} aims its clock at {own} of the board's budget, the meter is asked for {scale}")
    path = OUT / f"{tag}_{task}_{entropy}.pkl"
    kept = pickle.loads(path.read_bytes()) if path.is_file() else {}
    todo = [n for n in range(first, first + episodes) if n not in kept]
    if todo:
        es = scoring.episode_set(task, todo, entropy=entropy, n_jobs=n_jobs, verbose=False)
        for r in es.play(str(folder), cpu_budget=scale * BOARD[task], n_jobs=n_jobs):
            kept[int(r["episode"])] = {
                "n": int(r["episode"]), "J": int(r["J_policy_cents"]), "naive": int(r["fallback_weeks"]),
                "over": int(r["cpu_weeks"]), "error": r.get("first_error") or "",
                "cpu": np.array([float(r["seconds"]) / max(1, int(r["weeks"]))]),
            }  # fmt: skip
            OUT.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(kept))
    rows = [kept[n] for n in range(first, first + episodes) if n in kept]
    print(f"{tag} {task} {entropy}: {len(rows)} episodes kept, budget {scale * BOARD[task]:g} s, weeks the naive rule "
          f"played {sum(r['naive'] for r in rows)}, weeks over the budget {sum(r['over'] for r in rows)}, "
          f"episodes with an error {sum(1 for r in rows if r['error'])}")  # fmt: skip


if __name__ == "__main__":
    fire.Fire(main)
