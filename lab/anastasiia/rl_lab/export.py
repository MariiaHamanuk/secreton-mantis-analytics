"""Write a submission folder: the rules, the network, the weights and the glue that joins them.

    .venv/bin/python lab/anastasiia/rl_lab/export.py --run=outputs/rl_lab/<run> --name=lab_rl_slot

The folder holds only what the scoring image allows: the standard library, numpy, SciPy and PyTorch. The training
code stays here; nothing of it travels. The weights are read at import time, so ``Agent(config)`` costs only the
rule agent's own tables.
"""

import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

TEMPLATE = '''"""{title}

The rules of ``{rules_name}`` decide the week; a small network multiplies each slot's request by a number between
{lo} and {hi}. The network's last layer was initialised at zero, so an untrained one plays exactly the rules; a
correction that is not finite or would make a flow negative is dropped and the week is played by the rules alone.

Every size and table is read from ``config``: the same file runs on Tiny, Small and Full. Trained in
``lab/anastasiia/rl_lab`` ({note}).
"""

import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
torch.set_num_threads(1)


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{{HERE.name}}_{{name}}", HERE / f"{{name}}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


META = json.loads((HERE / "meta.json").read_text())
_nets = _module("nets")
_residual = _module("residual")
_rules = _module("rules_part")

# at module level: the weights are read once, before the first week's clock starts
POLICY = _nets.build(META["kind"], META["sizes"], META["width"], META["rounds"])
POLICY.load_state_dict(torch.load(HERE / "policy.pt", map_location="cpu", weights_only=True))
POLICY.eval()


class Agent:
    def __init__(self, config=None):
        self.inner = _residual.Residual(
            config,
            _rules,
            POLICY,
            groups=META["groups"],
            lo=META["lo"],
            hi=META["hi"],
            budget_s=META["budget_s"],
            families=META["families"],
        )

    def act(self, observation):
        return self.inner.act(observation)
'''


def main(run, name, rules=None, policy="policy.pt", budget_s=0.5, note="", into="agents"):
    run = Path(run)
    if not run.is_absolute():
        run = ROOT / run
    settings = json.loads((run / "settings.json").read_text())
    rules_folder = Path(rules) if rules else Path(settings["rules"])
    if not rules_folder.is_absolute():
        rules_folder = ROOT / rules_folder

    out = ROOT / into / name if into else ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    # the rule agent, under a name that cannot clash with this file
    shutil.copy(rules_folder / "agent.py", out / "rules_part.py")
    for extra in rules_folder.glob("*.py"):
        if extra.name != "agent.py":
            shutil.copy(extra, out / extra.name)
    for extra in rules_folder.glob("params.json"):
        shutil.copy(extra, out / extra.name)
    for module in ("features.py", "nets.py", "residual.py"):
        shutil.copy(HERE / module, out / module)
    shutil.copy(run / policy, out / "policy.pt")

    meta = {
        "kind": settings["kind"],
        "sizes": settings_sizes(run, settings),
        "width": settings["width"],
        "rounds": settings["rounds"],
        "groups": settings["groups"],
        "families": settings.get("families"),
        "lo": settings["lo"],
        "hi": settings["hi"],
        "budget_s": budget_s,
        "run": str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run),
        "rules": rules_folder.name,
        "note": note or settings.get("note", ""),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "agent.py").write_text(
        TEMPLATE.format(
            title=f"{name}: the rules of {rules_folder.name} with a learnt correction.",
            rules_name=rules_folder.name,
            lo=settings["lo"],
            hi=settings["hi"],
            note=meta["note"] or meta["run"],
        )
    )
    print(f"wrote {out}")
    return str(out)


def settings_sizes(run, settings):
    """The feature widths the weights were trained with, from the run's log or measured again."""
    if "sizes" in settings:
        return settings["sizes"]
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT / "src"))
    import rollout

    sizes, _, _ = rollout.sizes_for(settings["tasks"][0], settings["rules"], settings["groups"])
    return sizes


if __name__ == "__main__":
    import fire

    fire.Fire(main)
