"""Graph Isomorphism Network (Xu et al., 2019) and its edge-aware variant GINE.

GIN uses injective **sum** aggregation followed by an MLP::

    h_i = MLP( (1 + eps) x_i + sum_{j -> i} x_j )

with a (optionally learnable) scalar ``eps`` per layer.  GINE (Hu et al., 2020)
folds projected edge features into each message before the sum::

    m_ij = ReLU( x_j + W_e e_ij ),   h_i = MLP( (1 + eps) x_i + sum_{j -> i} m_ij )

Both variants live here: :class:`GIN` (``"gin"``) and :class:`GINE`
(``"gine"``) are thin subclasses of a shared base.  GINE gracefully degrades to
GIN when no edge features are present (``edge_dim == 0``).

References: K. Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019);
W. Hu et al., "Strategies for Pre-training Graph Neural Networks" (ICLR 2020).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


class _GINBase(GNNEncoder):
    """Shared GIN/GINE machinery. Subclasses set ``_use_edge``."""

    _use_edge: bool = False

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        act: str = "relu",
        eps: float = 0.0,
        train_eps: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, act=act, **kwargs)
        self.use_edge = self._use_edge and self.edge_dim > 0
        dims = [self.in_dim] + [self.hidden_dim] * (self.num_layers - 1) + [self.out_dim]
        self._dims = dims
        # per-layer MLP with the configured norm applied inside (BatchNorm-style).
        self.mlps = nn.ModuleList(
            MLP(dims[i], self.hidden_dim, dims[i + 1], num_layers=2, act=act,
                dropout=self.dropout, norm=self.norm_kind)
            for i in range(self.num_layers)
        )
        eps_init = torch.full((self.num_layers,), float(eps))
        if train_eps:
            self.eps = nn.Parameter(eps_init)
        else:
            self.register_buffer("eps", eps_init)
        if self.use_edge:
            # project edge features onto the (per-layer) node feature width
            self.edge_lins = nn.ModuleList(
                nn.Linear(self.edge_dim, dims[i]) for i in range(self.num_layers)
            )

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        use_edge = self.use_edge and graph.edge_attr is not None
        if use_edge:
            edge_attr = graph.edge_attr.float()
        for i, mlp in enumerate(self.mlps):
            if use_edge:
                m = F.relu(x.index_select(0, src) + self.edge_lins[i](edge_attr))  # [E, d_in]
            else:
                m = x.index_select(0, src)
            agg = scatter_sum(m, dst, graph.num_nodes)  # [N, d_in]
            out = mlp((1.0 + self.eps[i]) * x + agg)  # [N, d_out]
            if self.residual and self._dims[i] == self._dims[i + 1]:
                out = out + x
            x = out
        return x


@ENCODERS.register("gin")
class GIN(_GINBase):
    """Graph Isomorphism Network (sum aggregation, edge-agnostic)."""

    _use_edge = False


@ENCODERS.register("gine")
class GINE(_GINBase):
    """Edge-aware GIN: neighbour messages are ``ReLU(x_j + W_e e_ij)``."""

    _use_edge = True
