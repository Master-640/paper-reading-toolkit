"""GraphSAGE (Hamilton, Ying & Leskovec, 2017).

Each layer separately transforms the node's own feature and an aggregate of its
neighbours, then combines them either by concatenation (the original
formulation) or addition::

    h_i = W_self x_i  ||  W_neigh AGG_{j -> i} x_j          (combine="concat")
    h_i = W_self x_i  +   W_neigh AGG_{j -> i} x_j          (combine="add")

``AGG`` is a scatter-based mean or max over each destination's incoming edges.
With the default ``norm="l2"`` every layer output is L2-normalized, reproducing
the unit-sphere embeddings of the original paper.

Reference: W. Hamilton, R. Ying, J. Leskovec, "Inductive Representation
Learning on Large Graphs" (NeurIPS 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils import accel
from graco.utils.scatter import scatter_max, scatter_mean, scatter_sum


@ENCODERS.register("graphsage", aliases=["sage"])
class GraphSAGEEncoder(GNNEncoder):
    """Stacked GraphSAGE layers with sum/mean/max aggregation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        aggr: str = "mean",
        combine: str = "concat",
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        if aggr not in ("mean", "max", "sum"):
            raise ValueError(f"GraphSAGE aggr must be 'sum'/'mean'/'max', got {aggr!r}")
        if combine not in ("concat", "add"):
            raise ValueError(f"GraphSAGE combine must be 'concat' or 'add', got {combine!r}")
        self.aggr = aggr
        self.combine = combine
        dims = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        self.lin_self = nn.ModuleList()
        self.lin_neigh = nn.ModuleList()
        for i in range(self.num_layers):
            d_in, d_out = dims[i], dims[i + 1]
            if combine == "concat":
                # split the width so concat([self, neigh]) == d_out exactly
                n_neigh = d_out // 2
                n_self = d_out - n_neigh
            else:
                n_self = n_neigh = d_out
            self.lin_self.append(nn.Linear(d_in, n_self))
            self.lin_neigh.append(nn.Linear(d_in, n_neigh))
        self.norms = nn.ModuleList(
            Normalizer(self.norm_kind, dims[i + 1]) for i in range(self.num_layers)
        )
        self._dims = dims

    def _aggregate(self, x: Tensor, graph: BatchedGraph) -> Tensor:
        src, dst = graph.edge_index[0], graph.edge_index[1]
        # SpMM fast path: fused sparse matmul with a cached adjacency (sum/mean only)
        if accel.spmm_enabled() and self.aggr in ("sum", "mean"):
            return torch.sparse.mm(graph.adj_norm(self.aggr), x)
        neigh = x.index_select(0, src)  # [E, d_in]
        if self.aggr == "sum":
            return scatter_sum(neigh, dst, graph.num_nodes)
        if self.aggr == "max":
            return scatter_max(neigh, dst, graph.num_nodes)
        return scatter_mean(neigh, dst, graph.num_nodes)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        for i in range(self.num_layers):
            agg = self._aggregate(x, graph)
            z_self = self.lin_self[i](x)
            z_neigh = self.lin_neigh[i](agg)
            out = torch.cat([z_self, z_neigh], dim=-1) if self.combine == "concat" else z_self + z_neigh
            out = self.act(out)  # non-linearity on every layer (matches FINDER's S2V)
            if self.dropout > 0 and i < self.num_layers - 1:
                out = F.dropout(out, self.dropout, self.training)
            out = self.norms[i](out)  # L2 (or configured) normalization on every layer
            if self.residual and self._dims[i] == self._dims[i + 1]:
                out = out + x
            x = out
        return x
