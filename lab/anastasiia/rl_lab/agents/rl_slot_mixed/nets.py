"""Two networks over the week's tables, both of which read any network size.

``SlotNet`` looks at each action slot on its own (question 1 of the brief: is there anything to learn at all).
``GraphNet`` adds message passing along the edges first (question 2: does talking between nodes buy anything).
Both end in the same head, whose last layer starts at zero, so an untrained agent plays exactly the rules.

A batch is one week or many weeks at once: the weeks are laid end to end with their node and edge numbers shifted,
and ``slot_graph``/``node_graph`` say which week each row belongs to. One week is simply a batch of one, so the
same code runs in play and in the update.

Only ``torch`` is used, no graph library: the messages are an ``index_add_``.
"""

import math

import torch
import torch.nn as nn


def mlp(sizes, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def segment_sum(values, index, n):
    total = torch.zeros(n, values.shape[1], dtype=values.dtype, device=values.device)
    total.index_add_(0, index, values)
    return total


def segment_mean(values, index, n):
    total = segment_sum(values, index, n)
    count = torch.zeros(n, 1, dtype=values.dtype, device=values.device)
    count.index_add_(0, index, torch.ones(index.shape[0], 1, dtype=values.dtype, device=values.device))
    return total / count.clamp(min=1.0)


def segment_max(values, index, n):
    out = values.new_full((n, values.shape[1]), float("-inf"))
    out = out.scatter_reduce(0, index.unsqueeze(1).expand(-1, values.shape[1]), values, "amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


class Head(nn.Module):
    """One number per slot, starting at exactly zero: the correction is 1.0 until something is learnt."""

    def __init__(self, width):
        super().__init__()
        self.body = mlp([width, width, width])
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h):
        return self.out(self.body(h)).squeeze(-1)


class Critic(nn.Module):
    """The value of a week: its slots summed up, plus the week's global features."""

    def __init__(self, width, n_global):
        super().__init__()
        self.body = mlp([2 * width + n_global, width, width])
        self.out = nn.Linear(width, 1)

    def forward(self, slot_h, slot_graph, n_graphs, global_feat):
        pooled = torch.cat(
            [segment_mean(slot_h, slot_graph, n_graphs), segment_max(slot_h, slot_graph, n_graphs), global_feat],
            dim=-1,
        )
        return self.out(self.body(pooled)).squeeze(-1)


class SlotNet(nn.Module):
    """Question 1: every slot on its own, no message passing."""

    kind = "slot"

    def __init__(self, sizes, width=64, rounds=0):
        super().__init__()
        self.width = width
        self.enc = mlp([sizes["slot"] + sizes["global"], width, width])
        self.head = Head(width)
        self.critic = Critic(width, sizes["global"])

    def forward(self, b):
        g = b["global"][b["slot_graph"]]
        h = self.enc(torch.cat([b["slot"], g], dim=-1))
        return self.head(h), self.critic(h, b["slot_graph"], b["n_graphs"], b["global"])


class GraphNet(nn.Module):
    """Question 2: the same head, but the nodes and edges talk first."""

    kind = "graph"

    def __init__(self, sizes, width=64, rounds=2):
        super().__init__()
        self.width = width
        self.rounds = rounds
        self.node_enc = mlp([sizes["node"] + sizes["global"], width, width])
        self.edge_enc = mlp([sizes["edge"] + sizes["global"], width, width])
        self.msg = nn.ModuleList([mlp([3 * width, width, width]) for _ in range(rounds)])
        self.upd = nn.ModuleList([mlp([3 * width, width, width]) for _ in range(rounds)])
        self.slot_enc = mlp([sizes["slot"] + sizes["global"] + 4 * width, width, width])
        self.head = Head(width)
        self.critic = Critic(width, sizes["global"])

    def forward(self, b):
        tail, head, n_nodes = b["edge_tail"], b["edge_head"], b["n_nodes"]
        h_n = self.node_enc(torch.cat([b["node"], b["global"][b["node_graph"]]], dim=-1))
        h_e = self.edge_enc(torch.cat([b["edge"], b["global"][b["edge_graph"]]], dim=-1))
        for msg, upd in zip(self.msg, self.upd):
            m = msg(torch.cat([h_n[tail], h_n[head], h_e], dim=-1))
            h_e = h_e + m
            into = segment_mean(m, head, n_nodes)  # what arrives at a node
            outof = segment_mean(m, tail, n_nodes)  # what leaves it
            h_n = h_n + upd(torch.cat([h_n, into, outof], dim=-1))

        route = segment_mean(h_e[b["route_edge"]], b["route_slot"], b["slot"].shape[0])
        h = self.slot_enc(
            torch.cat(
                [
                    b["slot"],
                    b["global"][b["slot_graph"]],
                    h_e[b["slot_edge"]],
                    h_n[b["slot_tail"]],
                    h_n[b["slot_dest"]],
                    route,
                ],
                dim=-1,
            )
        )
        return self.head(h), self.critic(h, b["slot_graph"], b["n_graphs"], b["global"])


class Policy(nn.Module):
    """The network plus the spread of the correction while it is learning. In play only the mean is taken."""

    def __init__(self, net, log_std=-1.0):
        super().__init__()
        self.net = net
        self.log_std = nn.Parameter(torch.full((1,), float(log_std)))

    def forward(self, b):
        return self.net(b)

    def log_prob(self, sample, mean):
        std = self.log_std.exp()
        return -0.5 * ((sample - mean) ** 2 / (std * std) + 2.0 * self.log_std + math.log(2.0 * math.pi))

    def entropy(self):
        return 0.5 * math.log(2.0 * math.pi * math.e) + self.log_std


def correction(raw, lo=0.5, hi=2.0):
    """``raw`` in R to a multiplier in ``[lo, hi]``, equal to 1 at 0 — where ``Head`` starts."""
    t = torch.tanh(raw)
    return torch.where(t >= 0, 1.0 + t * (hi - 1.0), 1.0 + t * (1.0 - lo))


def build(kind, sizes, width=64, rounds=2, log_std=-1.0):
    net = SlotNet(sizes, width) if kind == "slot" else GraphNet(sizes, width, rounds)
    return Policy(net, log_std)
