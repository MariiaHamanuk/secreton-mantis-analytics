"""Many weeks as one batch: the graphs are laid end to end and their node and edge numbers shifted.

Weeks of Small and weeks of Full can sit in the same batch — that is the point: the brief asks for the two networks
mixed from the start, and nothing in the model knows how many nodes a network has.
"""

import numpy as np
import torch


def collate(weeks, indices):
    """``weeks`` is a list of ``(task, node, edge, slot, global)``; ``indices`` maps a task to its index tensors."""
    node, edge, slot, glob = [], [], [], []
    node_graph, edge_graph, slot_graph = [], [], []
    e_tail, e_head, s_edge, s_tail, s_dest, r_slot, r_edge = [], [], [], [], [], [], []
    n_off = e_off = s_off = 0
    for g, (task, nd, ed, sl, gl) in enumerate(weeks):
        ix = indices[task]
        node.append(nd)
        edge.append(ed)
        slot.append(sl)
        glob.append(gl)
        node_graph.append(np.full(nd.shape[0], g, dtype=np.int64))
        edge_graph.append(np.full(ed.shape[0], g, dtype=np.int64))
        slot_graph.append(np.full(sl.shape[0], g, dtype=np.int64))
        e_tail.append(ix["edge_tail"] + n_off)
        e_head.append(ix["edge_head"] + n_off)
        s_edge.append(ix["slot_edge"] + e_off)
        s_tail.append(ix["slot_tail"] + n_off)
        s_dest.append(ix["slot_dest"] + n_off)
        r_slot.append(ix["route_slot"] + s_off)
        r_edge.append(ix["route_edge"] + e_off)
        n_off += nd.shape[0]
        e_off += ed.shape[0]
        s_off += sl.shape[0]
    cat = lambda xs: torch.from_numpy(np.concatenate(xs))  # noqa: E731
    return {
        "node": cat(node),
        "edge": cat(edge),
        "slot": cat(slot),
        "global": torch.from_numpy(np.stack(glob)),
        "node_graph": cat(node_graph),
        "edge_graph": cat(edge_graph),
        "slot_graph": cat(slot_graph),
        "n_graphs": len(weeks),
        "n_nodes": n_off,
        "edge_tail": cat(e_tail),
        "edge_head": cat(e_head),
        "slot_edge": cat(s_edge),
        "slot_tail": cat(s_tail),
        "slot_dest": cat(s_dest),
        "route_slot": cat(r_slot),
        "route_edge": cat(r_edge),
    }


def index_arrays(layout):
    """The index tensors of a task as plain int64 arrays, so they travel between processes cheaply."""
    return {
        "edge_tail": layout.edge_tail,
        "edge_head": layout.edge_head,
        "slot_edge": layout.slot_edge,
        "slot_tail": layout.slot_tail,
        "slot_dest": layout.slot_dest,
        "route_slot": layout.route_slot,
        "route_edge": layout.route_edge,
        "n_nodes": layout.n_nodes,
    }
