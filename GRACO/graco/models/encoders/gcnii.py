"""GCNII: deep GCN via Initial residual and Identity mapping (Chen et al., 2020).

Two ingredients let arbitrarily deep stacks avoid over-smoothing.  Every layer
adds an *initial residual* back to the projected input ``h0`` and applies an
*identity mapping* to its weight matrix::

    P        = D~^{-1/2} A~ D~^{-1/2}          (renormalized adjacency, A~ = A + I)
    beta_l   = log(lambda / l + 1)
    inner    = (1 - alpha) P h + alpha h0
    h        = sigma( ((1 - beta_l) I + beta_l W_l) inner )

with teleport probability ``alpha`` (default 0.1) and strength ``lambda``
(default 0.5).  Aggregation is scatter-based over ``edge_index`` (src=[0],
dst=[1]) and folds in the implicit self-loop, exactly like :class:`GCNEncoder`.

Reference: M. Chen et al., "Simple and Deep Graph Convolutional Networks"
(ICML 2020).
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, gcn_norm
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("gcnii")
class GCNIIEncoder(GNNEncoder):
    """Deep GCN with initial residual + identity mapping; embeddings ``[N, hidden_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        alpha: float = 0.1,
        lambda_: float = 0.5,
        **kwargs,
    ):
        # allow the reserved-word alias ``lambda`` to arrive via config kwargs
        lambda_ = float(kwargs.pop("lambda", lambda_))
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.out_dim = self.hidden_dim
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.lin_in = nn.Linear(self.in_dim, self.hidden_dim)
        # one square weight per propagation layer (identity mixing done in forward)
        self.weights = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_layers)
        )

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = gcn_norm(graph)

        h0 = self.act(self.lin_in(x))  # projected input [N, H]
        h = h0
        for layer, weight in enumerate(self.weights, start=1):
            # P h: renormalized-adjacency propagation with implicit self-loop
            msg = h.index_select(0, src) * c.unsqueeze(-1)
            ph = scatter_sum(msg, dst, graph.num_nodes) + self_coef.unsqueeze(-1) * h
            inner = (1.0 - self.alpha) * ph + self.alpha * h0  # initial residual
            beta = math.log(self.lambda_ / layer + 1.0)
            out = (1.0 - beta) * inner + beta * weight(inner)  # identity mapping
            out = self.act(out)
            if self.dropout > 0:
                out = F.dropout(out, self.dropout, self.training)
            h = out
        return h
