"""Graph data augmentation + batch replication.

Used by inference decoding and POMO multi-start:

* :func:`repeat_graphs` — replicate each graph ``k`` times into one disjoint-union
  batch (optionally permuting each copy via :func:`_permute_single`, an isomorphic
  node relabelling that leaves every environment's objective invariant), returning
  a ``group`` vector that maps every copy back to its source graph — so N
  independent rollouts per graph can be reduced with a per-group max / mean.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from graco.buffers.base import SingleGraph, collate_single_graphs, to_single_graphs
from graco.data.batch import BatchedGraph


def _permute_single(sg: SingleGraph, generator: Optional[torch.Generator]) -> SingleGraph:
    n = sg.num_nodes
    perm = torch.randperm(n, generator=generator)
    newid = torch.empty(n, dtype=torch.long)
    newid[perm] = torch.arange(n)  # old id -> new id
    ei = newid[sg.edge_index] if sg.edge_index.numel() else sg.edge_index
    return SingleGraph(
        edge_index=ei,
        num_nodes=n,
        edge_weight=sg.edge_weight,  # per-edge; order unchanged
        node_type=sg.node_type[perm] if sg.node_type is not None else None,
        edge_type=sg.edge_type,
        node_pos=sg.node_pos[perm] if sg.node_pos is not None else None,
        inc_node=newid[sg.inc_node] if sg.inc_node is not None else None,
        inc_hyperedge=sg.inc_hyperedge,
        num_hyperedges=sg.num_hyperedges,
        meta_scalars=dict(sg.meta_scalars),
    )


def repeat_graphs(
    bg: BatchedGraph,
    k: int,
    permute: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tuple[BatchedGraph, Tensor]:
    """Replicate each graph ``k`` times; return (batch, group[B*k] -> source id)."""
    sgs = to_single_graphs(bg)
    repeated: List[SingleGraph] = []
    group: List[int] = []
    for gi, sg in enumerate(sgs):
        for _ in range(k):
            repeated.append(_permute_single(sg, generator) if permute else sg)
            group.append(gi)
    out, _ = collate_single_graphs(repeated, bg.device)
    return out, torch.tensor(group, device=bg.device)
