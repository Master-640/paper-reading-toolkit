"""Heterogeneous attention network (HAN, Wang et al., 2019).

GRACO graphs carry per-edge relation ids rather than explicit meta-paths, so
``han`` implements a **type-aware GAT**: for every relation ``r`` (an
``edge_type``) a relation-specific projection plus GAT-style attention produces a
per-relation node embedding ``z^r``, and a learned *semantic* attention over
relations fuses ``{z^r}`` into one embedding

    beta_r = softmax_r( (1/N) sum_i q . tanh(W z^r_i) ),
    z_i    = sum_r beta_r z^r_i.

Each relation additionally attends over an implicit self-loop so isolated nodes
keep a representation.  The relation count comes from ``num_relations`` /
``num_edge_types`` / ``graph.meta['num_edge_types']`` / ``edge_type.max()+1``;
relation parameters are (re)built lazily on the first forward when unknown.  When
no ``edge_type`` is present this collapses to a single relation, i.e. a plain
multi-head GAT.

Reference: X. Wang et al., "Heterogeneous Graph Attention Network" (WWW 2019).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_softmax, scatter_sum


def _infer_num_relations(graph: BatchedGraph, override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    if graph.edge_type is not None and graph.num_edges > 0:
        default = int(graph.edge_type.max().item()) + 1
        return max(1, int(graph.meta.get("num_edge_types", default)))
    return max(1, int(graph.meta.get("num_edge_types", 1)))


class _HANLayer(nn.Module):
    """One type-aware GAT layer with semantic attention over relations."""

    def __init__(
        self,
        in_dim: int,
        dim: int,
        heads: int,
        num_relations: int,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.R = num_relations
        self.heads = heads
        self.per_head = max(1, dim // heads)
        self.width = self.per_head * heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        # relation-specific node projection + per-relation GAT attention vectors.
        self.rel_lin = nn.ModuleList(
            nn.Linear(in_dim, self.width, bias=False) for _ in range(num_relations)
        )
        self.att_src = nn.Parameter(torch.empty(num_relations, heads, self.per_head))
        self.att_dst = nn.Parameter(torch.empty(num_relations, heads, self.per_head))
        # semantic-level attention that weighs the per-relation representations.
        self.sem_lin = nn.Linear(self.width, self.width)
        self.sem_q = nn.Parameter(torch.empty(self.width))
        self.out_lin = nn.Linear(self.width, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.sem_q)
        for m in (*self.rel_lin, self.sem_lin, self.out_lin):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, h: Tensor, src: Tensor, dst: Tensor, etype: Tensor, num_nodes: int) -> Tensor:
        H, C, N = self.heads, self.per_head, num_nodes
        self_loop = torch.arange(N, device=h.device)
        z_list = []
        for r in range(self.R):
            xr = self.rel_lin[r](h).view(N, H, C)  # [N, H, C]
            al = (xr * self.att_src[r]).sum(-1)  # [N, H]
            ar = (xr * self.att_dst[r]).sum(-1)  # [N, H]
            mask = etype == r
            rs = torch.cat([src[mask], self_loop])  # relation edges + self-loop
            rd = torch.cat([dst[mask], self_loop])
            score = al.index_select(0, rs) + ar.index_select(0, rd)  # [E_r, H]
            score = F.leaky_relu(score, self.negative_slope)
            alpha = scatter_softmax(score, rd, N)  # [E_r, H]
            if self.dropout > 0:
                alpha = F.dropout(alpha, self.dropout, self.training)
            val = xr.index_select(0, rs) * alpha.unsqueeze(-1)  # [E_r, H, C]
            z_list.append(scatter_sum(val, rd, N).reshape(N, self.width))  # [N, width]
        Z = torch.stack(z_list, dim=0)  # [R, N, width]
        # semantic attention: relation importance averaged over nodes.
        w = (torch.tanh(self.sem_lin(Z)) * self.sem_q).sum(-1)  # [R, N]
        beta = torch.softmax(w.mean(dim=1), dim=0)  # [R]
        fused = (beta.view(self.R, 1, 1) * Z).sum(0)  # [N, width]
        return self.out_lin(fused)


@ENCODERS.register("han")
class HAN(GNNEncoder):
    """Type-aware GAT with semantic attention over relations (HAN-style)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        num_relations: Optional[int] = None,
        num_edge_types: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.num_heads = int(num_heads) if h % int(num_heads) == 0 else 1
        # explicit relation count (``num_relations`` wins over ``num_edge_types``).
        self.num_relations = num_relations if num_relations is not None else num_edge_types
        self.node_in = nn.Linear(self.in_dim, h)
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))
        self.layers: Optional[nn.ModuleList] = None
        self._built_relations: Optional[int] = None
        if self.num_relations is not None:
            self._build(int(self.num_relations), torch.device("cpu"))

    def _build(self, num_relations: int, device: torch.device) -> None:
        self.layers = nn.ModuleList(
            _HANLayer(
                self.hidden_dim, self.hidden_dim, self.num_heads, num_relations,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ).to(device)
        self._built_relations = num_relations

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        num_relations = _infer_num_relations(graph, self.num_relations)
        if self._built_relations != num_relations:
            self._build(num_relations, h.device)
        if next(self.layers.parameters()).device != h.device:
            self.layers = self.layers.to(h.device)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        if graph.edge_type is not None and graph.num_edges > 0:
            etype = graph.edge_type.to(h.device).clamp(max=num_relations - 1)
        else:  # plain graph -> single relation -> plain GAT
            etype = torch.zeros(graph.num_edges, dtype=torch.long, device=h.device)
        for i, layer in enumerate(self.layers):
            out = self.act(self.norms[i](layer(h, src, dst, etype, graph.num_nodes)))
            h = h + out if self.residual else out
        return h
