"""TAGCN: Topology-Adaptive Graph Convolutional Network (Du et al., 2017).

Each layer applies a learned polynomial of the normalized adjacency, so a single
layer already aggregates information from ``0`` to ``K`` hops with an independent
weight per hop::

    P    = D~^{-1/2} A~ D~^{-1/2}          (renormalized adjacency, A~ = A + I)
    h    = sum_{k=0}^{K} (P^k x) W_k

The ``k=0`` term is the node's own transformed feature and higher powers are
accumulated by repeatedly propagating ``P`` (no explicit powering).  ``K``
defaults to 2 and ``num_layers`` such convolutions are stacked with a
nonlinearity between them.  Aggregation is scatter-based over ``edge_index``
(src=[0], dst=[1]) and folds in the implicit self-loop.

Reference: J. Du et al., "Topology Adaptive Graph Convolutional Networks"
(arXiv:1710.10370, 2017).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer, gcn_norm
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("tagcn")
class TAGCNEncoder(GNNEncoder):
    """Stacked topology-adaptive convolutions; embeddings ``[N, hidden_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        K: int = 2,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.out_dim = self.hidden_dim
        self.K = max(1, int(K))  # polynomial order (hops 0 .. K)
        dims = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        # per layer: one weight per hop k = 0 .. K; only the k=0 term carries a bias
        self.convs = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1], bias=(k == 0)) for k in range(self.K + 1)
            )
            for i in range(self.num_layers)
        )
        self.norms = nn.ModuleList(
            Normalizer(self.norm_kind, dims[i + 1]) for i in range(self.num_layers)
        )
        self._dims = dims

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = gcn_norm(graph)
        n = graph.num_nodes

        for i, convs in enumerate(self.convs):
            cur = x
            out = convs[0](cur)  # k=0: self term x W_0
            for k in range(1, self.K + 1):
                msg = cur.index_select(0, src) * c.unsqueeze(-1)
                cur = scatter_sum(msg, dst, n) + self_coef.unsqueeze(-1) * cur  # P^k x
                out = out + convs[k](cur)
            if i < self.num_layers - 1:
                out = self.act(out)
                out = self.norms[i](out)
                if self.dropout > 0:
                    out = F.dropout(out, self.dropout, self.training)
            if self.residual and self._dims[i] == self._dims[i + 1]:
                out = out + x
            x = out
        return x
