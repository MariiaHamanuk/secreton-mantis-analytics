"""The rule agent with a network correction on top, used both in training and in the submission.

The rules decide the week; the network only multiplies each slot by a number between 0.5 and 2. The head of the
network starts at zero, so before any training the product is exactly the rules' own request. Every week the
result is checked: a correction that is not finite, or negative, or that takes longer than its share of the CPU
budget, is dropped and the week is played by the rules alone. The agent therefore cannot score below the rules
because of the network's arithmetic — only because of what the network learnt.

``Residual.act`` is what the submission's ``agent.py`` calls. ``Residual.step`` is the same week for training: it
samples a correction and hands back the tables, the value and the log-probability the update needs.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import torch
from features import Featurizer, History, Layout


def load_rules(folder, tag="rules"):
    """The rule agent in ``folder``, imported under a name of its own (several agents ship a ``fuel_part``)."""
    folder = Path(folder).resolve()
    spec = importlib.util.spec_from_file_location(f"{folder.name}_{tag}_agent", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_week_batch(node, edge, slot, glob, index):
    """The four tables of one week as the networks' batch of one graph."""
    t = torch.from_numpy
    return {
        "node": t(node),
        "edge": t(edge),
        "slot": t(slot),
        "global": t(glob).unsqueeze(0),
        "node_graph": torch.zeros(node.shape[0], dtype=torch.long),
        "edge_graph": torch.zeros(edge.shape[0], dtype=torch.long),
        "slot_graph": torch.zeros(slot.shape[0], dtype=torch.long),
        "n_graphs": 1,
        "n_nodes": index["n_nodes"],
        **{k: v for k, v in index.items() if k != "n_nodes"},
    }


class Residual:
    def __init__(self, config, rules_module, policy, groups=None, lo=0.5, hi=2.0, budget_s=None, families=None):
        self.rules = rules_module.Agent(config)
        self.layout = Layout(config)
        self.feat = Featurizer(self.layout, groups) if groups else Featurizer(self.layout)
        self.history = History(self.layout)
        self.policy = policy
        self.lo, self.hi = float(lo), float(hi)
        self.budget_s = budget_s
        self.fallbacks = 0
        self.weeks = 0
        L = self.layout
        # which slots the correction may touch at all; the rest keep the rules' request exactly
        if families:
            wanted = set(families)
            self.touch = np.array(
                [L.commodities[int(k)] in wanted for k in L.slot_k], dtype=bool
            )
        else:
            self.touch = np.ones(L.n_slots, dtype=bool)
        as_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long()  # noqa: E731
        self.index = {
            "edge_tail": as_t(L.edge_tail),
            "edge_head": as_t(L.edge_head),
            "slot_edge": as_t(L.slot_edge),
            "slot_tail": as_t(L.slot_tail),
            "slot_dest": as_t(L.slot_dest),
            "route_slot": as_t(L.route_slot),
            "route_edge": as_t(L.route_edge),
            "n_nodes": L.n_nodes,
        }

    # ---- one week ---------------------------------------------------------------------------------------------
    def tables(self, obs, flows):
        """The week's four tables. ``history`` is advanced here, once per week, before anything reads it."""
        return self.feat.week(obs, flows, self.history)

    def rules_action(self, obs):
        self.history.update(self.layout, obs)
        return self.rules.act(obs)

    def apply(self, action, raw):
        """The rules' action with the correction applied, or the rules' own action if anything is off."""
        flows = np.asarray(action["flows"], dtype=np.float64)
        a = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            self.fallbacks += 1
            return action
        t = np.tanh(a)
        factor = np.where(t >= 0.0, 1.0 + t * (self.hi - 1.0), 1.0 + t * (1.0 - self.lo))
        out = flows * np.where(self.touch, factor, 1.0)
        if not np.all(np.isfinite(out)) or np.any(out < 0.0):
            self.fallbacks += 1
            return action
        return {**action, "flows": out}

    def act(self, obs):
        """The submission's week: rules, then the network's mean correction, with a timer and a fall-back."""
        self.weeks += 1
        action = self.rules_action(obs)
        start = time.process_time()
        try:
            tables = self.tables(obs, action["flows"])
            with torch.inference_mode():
                raw, _ = self.policy(one_week_batch(*tables, self.index))
            if self.budget_s is not None and time.process_time() - start > self.budget_s:
                self.fallbacks += 1
                return action
            return self.apply(action, raw.numpy())
        except Exception:  # a week played by the rules is always better than a week played by naive
            self.fallbacks += 1
            return action

    def step(self, obs, generator=None):
        """Training's week: sample a correction, and report what the update needs."""
        self.weeks += 1
        action = self.rules_action(obs)
        flows = np.asarray(action["flows"], dtype=np.float64)
        node, edge, slot, glob = self.tables(obs, flows)
        batch = one_week_batch(node, edge, slot, glob, self.index)
        with torch.no_grad():
            raw, value = self.policy(batch)
            std = self.policy.log_std.exp()
            sample = raw + torch.randn(raw.shape, generator=generator) * std
            # a zero slot cannot be scaled, and a slot outside `families` is not ours to move: neither carries
            # any probability mass, or the update would spend its variance on actions that change nothing
            live = torch.from_numpy(((flows > 0) & self.touch).astype(np.float32))
            logp = float((self.policy.log_prob(sample, raw) * live).sum())
        return action_with(self, action, sample), {
            "node": node,
            "edge": edge,
            "slot": slot,
            "global": glob,
            "sample": sample.numpy().astype(np.float32),
            "live": live.numpy(),
            "logp": logp,
            "value": float(value),
        }


def action_with(residual, action, sample):
    return residual.apply(action, sample.numpy())
