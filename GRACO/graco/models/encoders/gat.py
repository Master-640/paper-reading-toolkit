"""Graph Attention Networks: GAT (Velickovic et al., 2018) and GATv2 (Brody
et al., 2022).

Multi-head attention with scatter-based segmented softmax over each
destination's incoming edges.

GAT (static attention)::

    e_ij = LeakyReLU( a_l . (W h_i) + a_r . (W h_j) [+ a_e . (W_e e_ij)] )

GATv2 (dynamic attention) applies the attention vector *after* the
nonlinearity, which is strictly more expressive::

    e_ij = a . LeakyReLU( W_l h_i + W_r h_j [+ W_e e_ij] )

In both cases ``alpha = softmax_i(e_ij)`` (segmented over dst) and the output is
``sum_{j -> i} alpha_ij (W h_j)``.  Heads are concatenated in the hidden layers
and averaged in the last one.  Optional edge features are folded into the
pre-nonlinearity score.

References: P. Velickovic et al., "Graph Attention Networks" (ICLR 2018);
S. Brody, U. Alon, E. Yahav, "How Attentive are Graph Attention Networks?"
(ICLR 2022).
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


class _GATLayer(nn.Module):
    """One multi-head (v1 or v2) attention layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,  # per head
        heads: int,
        concat: bool,
        edge_dim: int = 0,
        v2: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.out = out_dim
        self.concat = concat
        self.edge_dim = edge_dim
        self.v2 = v2
        self.negative_slope = negative_slope
        self.dropout = dropout
        if v2:
            self.lin_l = nn.Linear(in_dim, heads * out_dim)  # target side
            self.lin_r = nn.Linear(in_dim, heads * out_dim)  # source side (values)
            self.att = nn.Parameter(torch.empty(1, heads, out_dim))
        else:
            self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
            self.att_l = nn.Parameter(torch.empty(1, heads, out_dim))
            self.att_r = nn.Parameter(torch.empty(1, heads, out_dim))
        if edge_dim > 0:
            self.lin_e = nn.Linear(edge_dim, heads * out_dim, bias=False)
            self.att_e = nn.Parameter(torch.empty(1, heads, out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(
        self, x: Tensor, src: Tensor, dst: Tensor, num_nodes: int, edge_attr: Optional[Tensor]
    ) -> Tensor:
        H, C = self.heads, self.out
        if self.v2:
            xl = self.lin_l(x).view(-1, H, C)  # [N, H, C]  target
            xr = self.lin_r(x).view(-1, H, C)  # [N, H, C]  source / value
            e = xl.index_select(0, dst) + xr.index_select(0, src)  # [E, H, C]
            if self.edge_dim > 0 and edge_attr is not None:
                e = e + self.lin_e(edge_attr).view(-1, H, C)
            e = F.leaky_relu(e, self.negative_slope)
            score = (e * self.att).sum(dim=-1)  # [E, H]
            value = xr.index_select(0, src)  # [E, H, C]
        else:
            xh = self.lin(x).view(-1, H, C)  # [N, H, C]
            al = (xh * self.att_l).sum(dim=-1)  # [N, H]
            ar = (xh * self.att_r).sum(dim=-1)  # [N, H]
            score = al.index_select(0, dst) + ar.index_select(0, src)  # [E, H]
            if self.edge_dim > 0 and edge_attr is not None:
                score = score + (self.lin_e(edge_attr).view(-1, H, C) * self.att_e).sum(dim=-1)
            score = F.leaky_relu(score, self.negative_slope)
            value = xh.index_select(0, src)  # [E, H, C]
        alpha = scatter_softmax(score, dst, num_nodes)  # [E, H] (max-stable)
        if self.dropout > 0:
            alpha = F.dropout(alpha, self.dropout, self.training)
        out = scatter_sum(value * alpha.unsqueeze(-1), dst, num_nodes)  # [N, H, C]
        if self.concat:
            return out.reshape(num_nodes, H * C)
        return out.mean(dim=1)  # [N, C]


class _GATBase(GNNEncoder):
    """Shared GAT/GATv2 stack. Subclasses toggle ``_v2``."""

    _v2: bool = False

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.num_heads = int(num_heads)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        widths = []
        running = self.in_dim
        for li in range(self.num_layers):
            last = li == self.num_layers - 1
            if last:  # average heads down to out_dim
                per_head, concat, out_width = self.out_dim, False, self.out_dim
            else:  # concat heads to (approximately) hidden_dim
                per_head = max(1, self.hidden_dim // self.num_heads)
                concat, out_width = True, per_head * self.num_heads
            self.layers.append(
                _GATLayer(running, per_head, self.num_heads, concat, self.edge_dim,
                          v2=self._v2, dropout=self.dropout)
            )
            self.norms.append(Normalizer(self.norm_kind, out_width))
            widths.append((running, out_width))
            running = out_width
        self._widths = widths

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        edge_attr = graph.edge_attr.float() if graph.edge_attr is not None else None
        for li, layer in enumerate(self.layers):
            out = layer(x, src, dst, graph.num_nodes, edge_attr)
            if li < self.num_layers - 1:
                out = self.act(out)
                out = self.norms[li](out)
                if self.dropout > 0:
                    out = F.dropout(out, self.dropout, self.training)
            in_w, out_w = self._widths[li]
            if self.residual and in_w == out_w:
                out = out + x
            x = out
        return x


@ENCODERS.register("gat")
class GAT(_GATBase):
    """Graph Attention Network (static attention)."""

    _v2 = False


@ENCODERS.register("gatv2")
class GATv2(_GATBase):
    """GATv2 with dynamic (post-nonlinearity) attention."""

    _v2 = True
