"""Variants of an agent folder for the lever tests: a copy with other numbers in its ``regime.json``.

    uv run python lab/anastasiia/plan_lab/variants.py make passes2 --passes=2
    uv run python lab/anastasiia/plan_lab/variants.py table --task=full --episodes=16 passes2 switch3

``make`` copies ``--base`` (``agents/anastasiia_plan_hull``) to ``outputs/plan_lab/agents/lev/<name>`` and replaces the
named entries of its ``regime.json`` (a JSON value each: ``--truth='["edges"]'``). ``base`` rebuilds the folder the
variants with code are copied from (``outputs/plan_lab/agents/lev_code``). ``table`` reads the episodes'
costs that ``harness.py run --tag=lev_<name>`` kept and prints each variant against the hybrid and, paired, against
the base model, whose costs are the records of ``lab/anastasiia/regime_lab/play.py`` (tag ``L_hre1``).
"""

import json
import pickle
import shutil
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
LEV = ROOT / "outputs" / "plan_lab" / "agents" / "lev"


def _words(value):
    """fire reads a JSON object as a Python literal and leaves true, false and null as strings: put them back."""
    if isinstance(value, dict):
        return {k: _words(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_words(v) for v in value]
    return {"true": True, "false": False, "null": None}.get(value, value) if isinstance(value, str) else value


def make(name: str, base: str = "agents/anastasiia_plan_hull", **overrides) -> None:
    out = LEV / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / base, out, ignore=shutil.ignore_patterns("__pycache__"))
    path = out / "regime.json"
    numbers = json.loads(path.read_text()) if path.is_file() else {}
    for key, value in overrides.items():
        numbers[key] = _words(json.loads(value) if isinstance(value, str) and value[:1] in "[{\"" else value)
    path.write_text(json.dumps(numbers))
    print(name, {k: numbers[k] for k in overrides})


def base(out: str = "outputs/plan_lab/agents/lev_code") -> None:
    """The folder the code variants are copied from: ``agents/anastasiia_plan_hull`` with this lab's ``agent.py`` and
    ``plan_core.py`` (``agents/plan_levers``: the same model plus switches that are off by default)."""
    target = ROOT / out
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ROOT / "agents" / "anastasiia_plan_hull", target, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("agent.py", "plan_core.py"):
        shutil.copy(Path(__file__).parent / "agents" / "plan_levers" / name, target / name)
    print(f"{out}: agents/anastasiia_plan_hull with agent.py and plan_core.py of plan_levers")


def table(*names: str, task: str = "small", entropy: int = 444, episodes: int = 8, first: int = 0,
          base: str = "L_hre1") -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import harness

    last = first + episodes
    refs = harness.references(task, entropy, last)[first:last]
    level = np.array([r["stratum"] for r in refs])
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    room = naive - np.array([r["J_oracle_cents"] for r in refs], dtype=float)
    book = json.loads((harness.COSTS / f"{task}_{entropy}.json").read_text())
    played = pickle.loads((ROOT / "outputs" / "regime_lab" / "play" / f"{base}_{task}_{entropy}.pkl").read_bytes())
    eps = list(range(first, last))
    hub = np.array([book["hub"][str(n)]["J_cents"] for n in eps], dtype=float)
    fixed = np.array([played[n]["J"] for n in eps], dtype=float)
    d, lo, hi = harness.interval(level, naive - fixed, naive - hub, room)
    print(f"{task} {entropy}, episodes {first}..{last - 1}: the hybrid {harness.pooled(level, naive - hub, room):.4f}, "
          f"the base model {harness.pooled(level, naive - fixed, room):.4f} ({d:+.4f}, {lo:+.4f}..{hi:+.4f})")
    print(f"{'variant':22} {'score':>7} {'to the base model':>32} {'bn/ep':>7} {'better':>7} {'cpu mean':>9}")
    for name in names:
        rows = book.get(f"lev_{name}", {})
        if any(str(n) not in rows for n in eps):
            print(f"{name:22} not played on all of these episodes ({sum(str(n) in rows for n in eps)} of {len(eps)})")
            continue
        J = np.array([rows[str(n)]["J_cents"] for n in eps], dtype=float)
        d, lo, hi = harness.interval(level, naive - J, naive - fixed, room)
        cpu = np.mean([rows[str(n)]["cpu_mean"] for n in eps])
        print(f"{name:22} {harness.pooled(level, naive - J, room):7.4f} {d:+9.4f} ({lo:+.4f}..{hi:+.4f}) "
              f"{np.mean(J - fixed) / 1e11:+7.1f} {int((J < fixed).sum()):4d}/{len(eps)} {cpu:9.2f}")


if __name__ == "__main__":
    fire.Fire({"make": make, "table": table, "base": base})
