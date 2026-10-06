"""Copy the part of shockbench-flow that builds a planning window's linear program into an agent's folder.

    uv run python lab/anastasiia/mpc_lab/vendor_mpc.py
    uv run python lab/anastasiia/mpc_lab/vendor_mpc.py --agent=agents/anastasiia_mpc_baseload

The server has numpy and SciPy but an agent may not import shockbench-flow, so the agent ships its own copy of the
modules behind ``policies.lp_common`` (the instance loader, the weekly marks, the LP builder), renamed ``sbfv``. Of
the package's other dependencies the copy drops highspy (the agent solves with ``scipy.optimize.linprog``), joblib,
loguru and fastjsonschema (the scorer's instance needs no validation). Run again after an update of shockbench-flow;
the copy is committed with the agent. shockbench-flow is MIT-licensed; its LICENSE is copied beside the modules.
"""

import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

import fire


PACKAGE, COPY = "shockbench_flow", "sbfv"
ROOT_MODULE = f"{PACKAGE}.policies.lp_common"
INIT = (
    '"""A copy of the shockbench-flow modules the planner needs (lab/anastasiia/mpc_lab/vendor_mpc.py writes it)."""\n'
)
NO_SCHEMA = '''

def _validate_schema(raw: Any) -> None:  # noqa: F811 - the copy's replacement
    """The scorer's instance is the published one: the copy carries neither the schema file nor its validator."""
'''
# (file, text, replacement): each must occur exactly once
PATCHES = (
    (
        "instance/io.py",
        "import fastjsonschema\n",
        "\n\nclass fastjsonschema:  # the copy validates nothing (see _validate_schema at the end)\n"
        "    class JsonSchemaValueException(Exception):\n        pass\n\n",
    ),
    ("parallel.py", "    from joblib import Parallel, delayed\n\n", ""),
    (
        "parallel.py",
        "    return list(Parallel(n_jobs=n_jobs, batch_size=BATCH_SIZE)(delayed(fn)(x) for x in xs))",
        '    raise RuntimeError("the copy runs serially: n_jobs must be 1")',
    ),
)


def main(agent: str = "agents/anastasiia_mpc_baseload") -> None:
    """Write ``<agent>/sbfv/`` from the installed shockbench-flow.

    Args:
        agent: the agent's folder (it holds agent.py), or its name in agents/.

    """
    from sbf_starter import ROOT
    from sbf_starter.agents import resolve

    before = set(sys.modules)
    __import__(ROOT_MODULE)
    names = sorted(n for n in set(sys.modules) - before if n == PACKAGE or n.startswith(PACKAGE + "."))
    source = Path(sys.modules[PACKAGE].__file__).parent
    target = resolve(agent).resolve() / COPY
    if target.exists():
        shutil.rmtree(target)
    lines = 0
    for name in names:
        path = Path(sys.modules[name].__file__)
        rel = path.relative_to(source)
        text = INIT if rel.as_posix() == "__init__.py" else path.read_text()
        for file, old, new in PATCHES:
            if file == rel.as_posix():
                if text.count(old) != 1:
                    raise SystemExit(f"{file}: expected exactly one {old!r}; shockbench-flow changed, update PATCHES")
                text = text.replace(old, new)
        if rel.as_posix() == "instance/io.py":
            text += NO_SCHEMA
        text = re.sub(r"^(\s*)import highspy$", r"\1highspy = None  # the copy solves with scipy", text, flags=re.M)
        text = re.sub(rf"\b{PACKAGE}\b", COPY, text)
        left = re.findall(r"^\s*(?:import|from)\s+(highspy|joblib|loguru|fastjsonschema)\b", text, re.M)
        if left:
            raise SystemExit(f"{rel}: still imports {sorted(set(left))}; add a patch")
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        (target / rel).write_text(text)
        lines += text.count("\n")
    licence = next(p for p in metadata.files("shockbench-flow") if p.name == "LICENSE" and "licenses" in p.parts)
    shutil.copy(licence.locate(), target / "LICENSE")
    version = metadata.version("shockbench-flow")
    (target / "VERSION").write_text(f"shockbench-flow {version}\n")
    print(f"{len(names)} modules ({lines} lines) of shockbench-flow {version} copied to {target.relative_to(ROOT)}")


if __name__ == "__main__":
    fire.Fire(main)
