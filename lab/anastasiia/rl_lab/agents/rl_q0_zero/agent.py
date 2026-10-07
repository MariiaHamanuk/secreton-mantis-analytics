"""rl_q0_zero: the rules of anastasiia_rules_v2 with a learnt correction.

The rules of ``anastasiia_rules_v2`` decide the week; a small network multiplies each slot's request by a number between
0.5 and 2.0. The network's last layer was initialised at zero, so an untrained one plays exactly the rules; a
correction that is not finite or would make a flow negative is dropped and the week is played by the rules alone.

Every size and table is read from ``config``: the same file runs on Tiny, Small and Full. Trained in
``lab/anastasiia/rl_lab`` (Q0: zero head).
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
    spec = importlib.util.spec_from_file_location(f"{HERE.name}_{name}", HERE / f"{name}.py")
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
