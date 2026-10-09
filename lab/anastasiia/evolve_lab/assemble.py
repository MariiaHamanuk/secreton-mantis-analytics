"""A frozen copy of ``agents/anastasiia_plan_hull3`` with a hook for more terms of the weekly program's objective.

    uv run python lab/anastasiia/evolve_lab/assemble.py --name=base
    uv run python lab/anastasiia/evolve_lab/assemble.py --name=c001 --terms=lab/anastasiia/evolve_lab/terms/c001.py
    uv run python lab/anastasiia/evolve_lab/assemble.py --name=plain --hook=False

Writes ``outputs/evolve_lab/agents/<name>/`` (the model's folder as committed, two patches, the search's settings)
and ``<name>.json`` beside it: the model's fingerprint, the terms' and the settings, in one ``sha``. The model's
files must be the committed ones (``MODEL_SHA``): the search stands on a frozen base.

The patches (``--hook=False`` leaves them out: the folder the hook is checked against, to the cent):

- ``plan_core.py``: ``Episode.terms`` (None) and ``Episode.info``; ``Episode.cell`` ends by asking ``terms`` for a
  vector of USD per column of the program and adds it to the cell's own prices. The candidate gets read-only views
  of the regimes and of the reference. The columns an end credit of a capped window may pay (stock and queues of the
  last week, cargo, lots and plant starts still out at its end, the fuel pools) get no term, whatever their credit:
  that valuation is another direction's. A candidate that raises, or returns anything but one finite number a
  column, stops the episode (``TermsError``, which the agent's own ``except Exception`` lets through): otherwise the
  week would be played by the rules and the candidate still scored.
- ``agent.py``: ``terms.py`` beside it is loaded when there is one, and every week's window carries it with
  ``info`` (the week of the episode, the weeks left, the window's length).

The terms' source is checked before it is copied (``_checked``): numpy and math only, nothing assigned through
``ep``, ``mode`` or ``ref`` but ``ep._*``, no state kept in the module, no timings read.

The search's settings are the model's without its clocks: no ``share``, no ``fit`` and what belongs to it, no
``chain_share`` (a CPU clock on the re-solves of a failed cell), and limits of one solve and of a warm start long
enough not to bind (``solve_share`` 15, ``warm_share`` 2: both are wall-clock limits in the model). A candidate is
then compared with the base on the work it asks for, not on how long its programs take on this machine. A winner is
played in the model's own settings (``--clock``), under the scorer's meter.
"""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = ROOT / "agents" / "anastasiia_plan_hull3"
MODEL_SHA = "87b1ebc9f16ec9f32d857631e5bd328c583365b5b22e082b9bcbe73b1d33075f"  # the folder of commit 95e2450
# the committed models a folder may be assembled from, by the fingerprint of their files
MODELS = {
    "anastasiia_plan_hull3": MODEL_SHA,
    "anastasiia_plan_hazard": "da52b32f8c0034c8b25323207d9817e030e3f7daa2498c13580214b99fab0a70",
    "anastasiia_plan_hazard2": "e0c1ba729696ddba79ed72ff02c78a779c49609c0ff0904403e4c13717841c23",
}
OUT = ROOT / "outputs" / "evolve_lab" / "agents"
# lab/nazar: speedups that leave every action as it was (caches, an early exit, bound methods), written for the
# model before this one; its files there are the same but ``plan_core.py``, where the patch still applies
SPEED = ROOT / "lab" / "nazar" / "patches" / "speedup_same_actions.diff"
CLOCK = ("fit", "carry_debt", "episode_share", "fit_rules", "fit_fresh", "fit_horizon", "search_room")
FREE = {"share": 0, "chain_share": 0, "warm_share": 2.0, "solve_share": 15.0}

CORE = [
    (
        "\n\nclass Cell:\n",
        "\n\nfrom types import MappingProxyType as _View  # noqa: E402 - evolve_lab\n\n\n"
        "class TermsError(BaseException):\n"
        '    """A candidate\'s terms raised or returned no vector: not an ``Exception``, so the agent does not play on."""\n'  # noqa: E501
        "\n\nclass Cell:\n",
    ),
    (
        "        self.solves: list[dict] = []\n",
        "        self.solves: list[dict] = []\n"
        "        # evolve_lab: more terms of a cell's objective, ``terms(episode, mode, ref)`` -> USD per column or None\n"  # noqa: E501
        "        self.terms = None\n"
        "        self.info: dict = {}\n"
        "        self.credited = None  # the columns an end credit may pay, which take no such term\n",
    ),
    (
        '            C.row([(self.n2 + i, 1.0)] + [(j, -1.0) for j in pool["cols"]], -INF, 0.0)\n        return C\n',
        '            C.row([(self.n2 + i, 1.0)] + [(j, -1.0) for j in pool["cols"]], -INF, 0.0)\n'
        "        if self.terms is not None:\n"
        "            try:\n"
        "                views = [_View({k: _View(v) if isinstance(v, dict) else v for k, v in d.items()}) for d in (mode, ref)]\n"  # noqa: E501
        "                extra = self.terms(self, *views)\n"
        "                if extra is not None:\n"
        "                    extra = np.array(extra, dtype=float)\n"
        "                    if extra.shape != (self.N,) or not np.all(np.isfinite(extra)):\n"
        '                        raise ValueError("one finite number a column of the program")\n'
        "            except Exception as error:\n"
        "                FAILED\n"
        "            if extra is not None:\n"
        "                if self.credited is None:\n"
        "                    self.credited = np.zeros(self.N, dtype=bool)\n"
        "                    if self.end is not None:\n"
        "                        nodes, fabs, osats = self.inst.nodes, self.inst.fabs, self.inst.osats\n"
        "                        self.credited[self.n2 : self.n2 + len(self.pools)] = True\n"
        "                        for key, j in self.tm.items():\n"
        '                            if key[0] in ("I", "Q"):\n'
        "                                out = 1\n"
        '                            elif key[0] == "x":\n'
        "                                out = self.inst.edges[key[1]].tau\n"
        '                            elif key[0] == "p":\n'
        "                                out = nodes[fabs[key[1]]].fab.tau\n"
        '                            elif key[0] == "xi":\n'
        "                                out = nodes[osats[key[1]]].osat.tau\n"
        "                            else:\n"
        "                                continue\n"
        "                            for t in range(max(0, self.T - int(out)), self.T):\n"
        "                                self.credited[t * self.nc + j] = True\n"
        "                extra[self.credited] = 0.0\n"
        "                C.cost = extra if C.cost is None else C.cost + extra\n"
        "        return C\n",
    ),
]
AGENT = [
    (
        '_core = _load(f"{HERE.name}_core", HERE / "plan_core.py")\n',
        '_core = _load(f"{HERE.name}_core", HERE / "plan_core.py")\n'
        '_terms = _load(f"{HERE.name}_terms", HERE / "terms.py") if (HERE / "terms.py").is_file() else None\n',
    ),
    (
        '        ep = _core.Episode(w.inst, w.marks, end=end if capped else None, anchored=bool(p["anchor"]))\n',
        '        ep = _core.Episode(w.inst, w.marks, end=end if capped else None, anchored=bool(p["anchor"]))\n'
        "        if _terms is not None:\n"
        '            ep.terms, ep.info = _terms.add, {"week": week, "left": left, "window": H}\n',
    ),
]


def tree_sha(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith("._") and p.name != ".DS_Store":
            h.update(str(p.relative_to(folder)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def _root(node):
    """The name an attribute or subscript chain hangs from, and the first attribute taken of it."""
    first = None
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        if isinstance(node, ast.Attribute):
            first = node.attr
        node = node.value
    return (node.id if isinstance(node, ast.Name) else None), first


def _checked(source: str) -> str:
    """The terms' source, or SystemExit with what it may not do."""
    tree, bad = ast.parse(source), []
    for node in tree.body:  # the module: a docstring, imports, constants, functions
        holds = isinstance(node, ast.Assign) and not any(
            isinstance(n, (ast.Dict, ast.List, ast.Set, ast.Call, ast.ListComp, ast.DictComp, ast.SetComp))
            for n in ast.walk(node.value)
        )
        docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or holds or docstring):
            bad.append(f"line {node.lineno}: the module keeps only imports, constants and functions")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in ("numpy", "math") for a in node.names):
            bad.append(f"line {node.lineno}: imports numpy and math only")
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in ("numpy", "math"):
            bad.append(f"line {node.lineno}: imports numpy and math only")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            bad.append(f"line {node.lineno}: no global state")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in ("exec", "eval", "open", "__import__", "setattr", "delattr", "globals", "vars", "print", "input")
        ):
            bad.append(f"line {node.lineno}: no {node.func.id}()")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "ep"
            and node.attr in ("solves", "terms")
        ):
            bad.append(f"line {node.lineno}: ep.{node.attr} is not the candidate's")
        targets = (
            node.targets
            if isinstance(node, (ast.Assign, ast.Delete))
            else [node.target]
            if isinstance(node, (ast.AugAssign, ast.AnnAssign))
            else []
        )
        for target in targets:
            for leaf in target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]:
                name, first = _root(leaf)
                if isinstance(leaf, (ast.Attribute, ast.Subscript)) and (
                    name in ("mode", "ref") or (name == "ep" and not str(first).startswith("_"))
                ):
                    bad.append(f"line {leaf.lineno}: nothing is assigned through {name} (but ep._*)")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "add" for n in tree.body):
        bad.append("no function add(ep, mode, ref)")
    if bad:
        raise SystemExit("terms: " + "; ".join(bad))
    return source


STRICT = "raise TermsError(repr(error)[:300]) from None"  # the search: a candidate that fails is out
SOFT = "extra = None  # shipped: a term that fails is no term this week, the week is the model's own"


def _patched(text: str, patches: list, failed: str = STRICT) -> str:
    for old, new in patches:
        new = new.replace("FAILED", failed)
        if text.count(old) != 1:
            raise SystemExit(f"the model's source changed: {old.strip()[:60]!r} is there {text.count(old)} times")
        text = text.replace(old, new)
    return text


def main(
    name: str,
    terms: str = "",
    hook: bool = True,
    clock: bool = False,
    params: str = "",
    speed: bool = False,
    soft: bool = False,
    model: str = "anastasiia_plan_hull3",
) -> None:
    """``clock``: the model's own settings, clocks and all (a winner's check). ``params``: JSON of settings on top.
    ``model``: the committed model the folder is built on (``MODELS``). ``soft``: the folder to ship: terms that
    raise or return no vector are left out that week instead of stopping
    the episode (the scorer's shim does not catch a ``BaseException``)."""
    base = ROOT / "agents" / model
    if tree_sha(base) != MODELS[model]:
        raise SystemExit(f"{base} is not the committed model: {tree_sha(base)}")
    target = OUT / name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(base, target, ignore=shutil.ignore_patterns("__pycache__", "._*", ".DS_Store"))
    if speed:  # before the hook: the patch was written for the committed text
        subprocess.run(["patch", "-p3", "-s", "-i", str(SPEED)], cwd=target, check=True)
        for left in list(target.rglob("*.orig")) + list(target.rglob("*.rej")):
            left.unlink()
    if hook:
        for file, patches in (("plan_core.py", CORE), ("agent.py", AGENT)):
            (target / file).write_text(_patched((target / file).read_text(), patches, SOFT if soft else STRICT))
    settings = json.loads((base / "regime.json").read_text())
    if not clock:
        settings = {k: v for k, v in settings.items() if k not in CLOCK} | FREE
    if params:
        settings.update(params if isinstance(params, dict) else json.loads(params))
    (target / "regime.json").write_text(json.dumps(settings))
    source = ""
    if terms:
        if not hook:
            raise SystemExit("terms need the hook")
        source = _checked((Path(terms) if Path(terms).is_absolute() else ROOT / terms).read_text())
        (target / "terms.py").write_text(source)
    sped = hashlib.sha256(SPEED.read_bytes()).hexdigest() if speed else ""
    sha = hashlib.sha256(
        json.dumps(
            [MODELS[model], bool(hook), source, settings, sped] + (["soft"] if soft else []), sort_keys=True
        ).encode()
    ).hexdigest()
    (OUT / f"{name}.json").write_text(
        json.dumps(
            {
                "sha": sha,
                "model": MODELS[model],
                "hook": bool(hook),
                "settings": settings,
                "terms": hashlib.sha256(source.encode()).hexdigest() if source else "",
            }
        )
    )
    print(sha)
    print(target)


if __name__ == "__main__":
    fire.Fire(main)
