"""Typed heterogeneous message-passing network.

A general relational MPNN (in the spirit of Gilmer et al., 2017, made
heterogeneous like Schlichtkrull et al., 2018): every relation ``r``
(``edge_type``) owns a message MLP over ``(x_src, x_dst)`` and every node type
``t`` (``node_type``) owns an update MLP over ``(x_i, agg_i)``.  Messages are
summed per destination, then the type-specific update produces the new state::

    m_{j->i} = MSG_{r(j,i)}([h_j, h_i]),   a_i = sum_{j->i} m_{j->i},
    h'_i     = UPD_{t(i)}([h_i, a_i]).

Relation / node-type counts come from ``num_relations`` / ``num_edge_types`` and
``num_node_types`` (init params or ``graph.meta``); the per-type MLPs are built
lazily on the first forward when unknown.  Absent ``edge_type`` / ``node_type``
falls back to a single relation / single node type — an ordinary MPNN.

Reference: J. Gilmer et al., "Neural Message Passing for Quantum Chemistry"
(ICML 2017); M. Schlichtkrull et al., "Modeling Relational Data with GCNs"
(ESWC 2018).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


def _infer_counts(
    graph: BatchedGraph, num_relations: Optional[int], num_node_types: Optional[int]
) -> Tuple[int, int]:
    if num_relations is not None:
        R = int(num_relations)
    elif graph.edge_type is not None and graph.num_edges > 0:
        default = int(graph.edge_type.max().item()) + 1
        R = int(graph.meta.get("num_edge_types", default))
    else:
        R = int(graph.meta.get("num_edge_types", 1))
    if num_node_types is not None:
        T = int(num_node_types)
    elif graph.node_type is not None and graph.num_nodes > 0:
        default = int(graph.node_type.max().item()) + 1
        T = int(graph.meta.get("num_node_types", default))
    else:
        T = int(graph.meta.get("num_node_types", 1))
    return max(1, R), max(1, T)


class _TypedMPNNLayer(nn.Module):
    """Per-relation message MLPs + per-node-type update MLPs, summed."""

    def __init__(self, dim: int, num_relations: int, num_node_types: int):
        super().__init__()
        self.R = num_relations
        self.T = num_node_types
        self.msg = nn.ModuleList(
            MLP(2 * dim, dim, dim, num_layers=2, act="relu") for _ in range(num_relations)
        )
        self.upd = nn.ModuleList(
            MLP(2 * dim, dim, dim, num_layers=2, act="relu") for _ in range(num_node_types)
        )

    def forward(
        self, h: Tensor, src: Tensor, dst: Tensor, etype: Tensor, ntype: Tensor, num_nodes: int
    ) -> Tensor:
        agg = h.new_zeros(h.shape)
        for r in range(self.R):
            mask = etype == r
            if not bool(mask.any()):
                continue
            s, d = src[mask], dst[mask]
            m = self.msg[r](torch.cat([h.index_select(0, s), h.index_select(0, d)], dim=-1))
            agg = agg + scatter_sum(m, d, num_nodes)
        out = h.new_zeros(h.shape)
        for t in range(self.T):
            idx = (ntype == t).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            val = self.upd[t](
                torch.cat([h.index_select(0, idx), agg.index_select(0, idx)], dim=-1)
            )
            out = out.index_copy(0, idx, val)
        return out


@ENCODERS.register("typed_mpnn")
class TypedMPNN(GNNEncoder):
    """Heterogeneous MPNN with per-relation and per-node-type sub-networks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_relations: Optional[int] = None,
        num_edge_types: Optional[int] = None,
        num_node_types: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.num_relations = num_relations if num_relations is not None else num_edge_types
        self.num_node_types = num_node_types
        self.node_in = nn.Linear(self.in_dim, h)
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))
        self.layers: Optional[nn.ModuleList] = None
        self._built: Optional[Tuple[int, int]] = None
        if self.num_relations is not None and self.num_node_types is not None:
            self._build(int(self.num_relations), int(self.num_node_types), torch.device("cpu"))

    def _build(self, num_relations: int, num_node_types: int, device: torch.device) -> None:
        self.layers = nn.ModuleList(
            _TypedMPNNLayer(self.hidden_dim, num_relations, num_node_types)
            for _ in range(self.num_layers)
        ).to(device)
        self._built = (num_relations, num_node_types)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        num_relations, num_node_types = _infer_counts(
            graph, self.num_relations, self.num_node_types
        )
        if self._built != (num_relations, num_node_types):
            self._build(num_relations, num_node_types, h.device)
        if next(self.layers.parameters()).device != h.device:
            self.layers = self.layers.to(h.device)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        if graph.edge_type is not None and graph.num_edges > 0:
            etype = graph.edge_type.to(h.device).clamp(max=num_relations - 1)
        else:
            etype = torch.zeros(graph.num_edges, dtype=torch.long, device=h.device)
        if graph.node_type is not None:
            ntype = graph.node_type.to(h.device).clamp(max=num_node_types - 1)
        else:
            ntype = torch.zeros(graph.num_nodes, dtype=torch.long, device=h.device)

        for i, layer in enumerate(self.layers):
            out = self.act(self.norms[i](layer(h, src, dst, etype, ntype, graph.num_nodes)))
            h = h + out if self.residual else out
        return h
