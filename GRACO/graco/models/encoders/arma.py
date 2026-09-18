"""ARMA: Auto-Regressive Moving Average graph convolution (Bianchi et al., 2021).

An ARMA filter is approximated by ``num_stacks`` parallel Graph Convolutional
Skip (GCS) recursions, each unrolled for ``T`` iterations and then averaged.
Every GCS step mixes a propagated state with a skip connection to the initial
transformed features::

    P          = D~^{-1/2} A~ D~^{-1/2}        (renormalized adjacency, A~ = A + I)
    X0         = sigma(W_in x)
    X^{t+1}    = sigma( a_s (P X^t) + b_s X0 )   (repeated T times, per stack s)
    out        = mean_s X^T_s

``a_s`` / ``b_s`` are per-stack linear maps shared across the ``T`` iterations
(``T`` defaults to ``num_layers``; ``num_stacks`` defaults to 2).  This is a
compact, robust approximation of the full ARMA_K filter bank.  Aggregation is
scatter-based over ``edge_index`` (src=[0], dst=[1]).

Reference: F. M. Bianchi et al., "Graph Neural Networks with Convolutional ARMA
Filters" (IEEE TPAMI 2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, gcn_norm
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("arma")
class ARMAEncoder(GNNEncoder):
    """Stacked ARMA/GCS filters; embeddings ``[N, hidden_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_stacks: int = 2,
        act: str = "relu",
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, act=act, **kwargs)
        self.out_dim = self.hidden_dim
        self.num_stacks = max(1, int(num_stacks))
        self.T = self.num_layers  # GCS iterations per stack
        self.lin_in = nn.Linear(self.in_dim, self.hidden_dim)
        # per-stack propagation (a) and skip (b) maps, shared across iterations
        self.prop = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for _ in range(self.num_stacks)
        )
        self.skip = nn.ModuleList(
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_stacks)
        )

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = gcn_norm(graph)

        x0 = self.act(self.lin_in(x))  # initial transformed features [N, H]
        outs = []
        for s in range(self.num_stacks):
            h = x0
            for _ in range(self.T):
                msg = h.index_select(0, src) * c.unsqueeze(-1)
                ph = scatter_sum(msg, dst, graph.num_nodes) + self_coef.unsqueeze(-1) * h
                h = self.act(self.prop[s](ph) + self.skip[s](x0))
                if self.dropout > 0:
                    h = F.dropout(h, self.dropout, self.training)
            outs.append(h)
        return torch.stack(outs, dim=0).mean(dim=0)  # average over stacks
