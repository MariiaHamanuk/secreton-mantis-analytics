"""The starter kit's tooling behind the ``sbf`` command line."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]  # the repository: this file is src/sbf_starter/__init__.py
# the team's shared reference costs, committed (hub/refcache); SBF_CACHE_DIR in the environment or .env wins
os.environ.setdefault("SBF_CACHE_DIR", str(ROOT / "hub" / "refcache"))
DEFAULT_TASK = "tiny"
TASKS = {"tiny": "ShockBench/Tiny-v0", "small": "ShockBench/Small-v0", "full": "ShockBench/Full-v0"}


def check_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, got {task!r}")
    return task


def env_id(task: str) -> str:
    return TASKS[check_task(task)]


def cpu_budget_s(task: str) -> float:
    """CPU seconds per week on the task's board."""
    from shockbench_flow_agent import LIMITS

    return LIMITS.cpu_budget_s[check_task(task)]
