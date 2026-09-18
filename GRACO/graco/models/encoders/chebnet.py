"""ChebNet: Chebyshev spectral graph convolution (Defferrard et al., 2016).

A localized spectral filter of order ``K`` expressed in the Chebyshev basis of
the scaled Laplacian, so it needs only sparse matrix-vector products (no eigen
decomposition)::

    L~     = 2 L / lambda_max - I,   L = I - D^{-1/2} A D^{-1/2}
    T_0    = x
    T_1    = L~ x
    T_k    = 2 L~ T_{k-1} - T_{k-2}
    out    = sum_{k=0}^{K-1} T_k W_k

We take ``lambda_max ~= 2`` (the loose upper bound for a normalized Laplacian),
which gives ``L~ ~= -D^{-1/2} A D^{-1/2}`` — the *symmetric-normalized adjacency
without self-loops*.  ``K`` (default 3) sets the number of Chebyshev terms;
``num_layers`` such convolutions are stacked with a nonlinearity between them.
Aggregation is scatter-based over ``edge_index`` (src=[0], dst=[1]).

Reference: M. Defferrard, X. Bresson, P. Vandergheynst, "Convolutional Neural
Networks on Graphs with Fast Localized Spectral Filtering" (NeurIPS 2016).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer, gcn_norm
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("chebnet")
class ChebNetEncoder(GNNEncoder):
    """Stacked Chebyshev spectral convolutions; embeddings ``[N, hidden_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        K: int = 3,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.out_dim = self.hidden_dim
        self.K = max(1, int(K))  # number of Chebyshev terms T_0 .. T_{K-1}
        dims = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        # per layer: one weight per Chebyshev term; only the T_0 term carries a bias
        self.convs = nn.ModuleList(
            nn.ModuleList(
                nn.Linear(dims[i], dims[i + 1], bias=(k == 0)) for k in range(self.K)
            )
            for i in range(self.num_layers)
        )
        self.norms = nn.ModuleList(
            Normalizer(self.norm_kind, dims[i + 1]) for i in range(self.num_layers)
        )
        self._dims = dims

    def _ahat(self, h: Tensor, src: Tensor, dst: Tensor, c: Tensor, num_nodes: int) -> Tensor:
        """Symmetric-normalized adjacency product ``D^{-1/2} A D^{-1/2} h``."""
        return scatter_sum(h.index_select(0, src) * c.unsqueeze(-1), dst, num_nodes)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c = gcn_norm(graph, add_self_loops=False)[0]
        n = graph.num_nodes

        for i, convs in enumerate(self.convs):
            t0 = x
            out = convs[0](t0)
            if self.K > 1:
                t1 = -self._ahat(t0, src, dst, c, n)  # L~ x ~= -A_hat x
                out = out + convs[1](t1)
                t_prev, t_cur = t0, t1
                for k in range(2, self.K):
                    t_next = -2.0 * self._ahat(t_cur, src, dst, c, n) - t_prev
                    out = out + convs[k](t_next)
                    t_prev, t_cur = t_cur, t_next
            if i < self.num_layers - 1:
                out = self.act(out)
                out = self.norms[i](out)
                if self.dropout > 0:
                    out = F.dropout(out, self.dropout, self.training)
            if self.residual and self._dims[i] == self._dims[i + 1]:
                out = out + x
            x = out
        return x
