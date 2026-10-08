"""Assemble a lab agent that stands alone: the hybrid's parts, the simulator's model, the cell program, the agent.

    uv run python lab/anastasiia/regime_lab/build.py --name=anastasiia_plan_regime --agent=anastasiia_plan_regime
    uv run python lab/anastasiia/regime_lab/build.py --name=regime_x_p1 --agent=regime_x --params='{"passes": 1}'

Writes ``outputs/regime_lab/agents/<name>/``: ``agent.py`` of ``agents/<agent>/``, ``hybrid_agent.py`` and the parts
of ``agents/anastasiia_hybrid_hub`` (with ``sbfv/``), ``sim_model.py`` (the rollout lab's ``model.py``),
``plan_core.py`` (this lab's ``core.py``), and ``regime.json`` when ``--params`` is given. Nothing in it points into
``lab/``. The folder can be scored by path (``hub/eval/compare.py``, ``sbf
evaluate``).
"""

import json
import shutil
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _booleans(x):
    """Fire reads a JSON ``true`` or ``false`` inside ``--params`` as the string: back to the boolean."""
    if isinstance(x, dict):
        return {k: _booleans(v) for k, v in x.items()}
    return {"true": True, "false": False}.get(x, x) if isinstance(x, str) else x


def main(name: str = "regime_x", agent: str = "regime_x", params: str | dict = "", hybrid: str = "anastasiia_hybrid_hub") -> None:
    out = ROOT / "outputs" / "regime_lab" / "agents" / name
    if out.exists():
        shutil.rmtree(out)
    src = ROOT / "agents" / hybrid
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    (out / "agent.py").rename(out / "hybrid_agent.py")
    shutil.copy(HERE / "agents" / agent / "agent.py", out / "agent.py")
    shutil.copy(ROOT / "lab" / "anastasiia" / "rollout_lab" / "model.py", out / "sim_model.py")
    shutil.copy(ROOT / "lab" / "anastasiia" / "rollout_lab" / "grid_recovery.json", out / "grid_recovery.json")
    shutil.copy(HERE / "core.py", out / "plan_core.py")
    if params:
        params = params if isinstance(params, dict) else json.loads(params)
        (out / "regime.json").write_text(json.dumps(_booleans(params)))
        if _booleans(params).get("hull") == "model":  # the trees that stand in for the solve with the hull (``shares.py fit``)
            shutil.copy(HERE / "share_model.npz", out / "share_model.npz")
    print(out)


if __name__ == "__main__":
    fire.Fire(main)
