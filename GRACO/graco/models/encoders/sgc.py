"""SGC: Simple Graph Convolution (Wu et al., 2019).

Collapses a ``K``-layer GCN into a single linear model by removing all
intermediate nonlinearities: the fixed ``K``-hop diffusion ``P^K x`` is
precomputed with ``K`` sparse aggregations, then a single learned linear maps it
to the output width::

    P    = D~^{-1/2} A~ D~^{-1/2}      (renormalized adjacency, A~ = A + I)
    out  = (P^K x) W

``K`` defaults to ``num_layers``.  Because propagation is parameter-free the only
learnable weights live in the final linear.  Aggregation is scatter-based over
``edge_index`` (src=[0], dst=[1]) and folds in the implicit self-loop.

Reference: F. Wu et al., "Simplifying Graph Convolutional Networks" (ICML 2019).
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, gcn_norm
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("sgc")
class SGCEncoder(GNNEncoder):
    """Simplified graph convolution; embeddings ``[N, hidden_dim]``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.out_dim = self.hidden_dim
        self.K = self.num_layers  # number of diffusion hops P^K
        self.lin = nn.Linear(self.in_dim, self.hidden_dim)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = gcn_norm(graph)

        h = x
        for _ in range(self.K):  # precompute P^K x, no nonlinearity between hops
            msg = h.index_select(0, src) * c.unsqueeze(-1)
            h = scatter_sum(msg, dst, graph.num_nodes) + self_coef.unsqueeze(-1) * h
        return self.lin(h)
