"""Residual Gated Graph ConvNet (Bresson & Laurent, 2017).

Unlike most encoders in GRACO, GatedGCN maintains **both** node and edge
features across the layer stack.  Every layer first updates the edge feature
``e`` from its endpoints, derives a soft gate ``eta = sigmoid(e_hat)`` and then
performs a *gated* mean aggregation of the source features into each
destination.  All aggregation is scatter-based over ``edge_index`` (src=[0],
dst=[1]).

Reference: X. Bresson, T. Laurent, "Residual Gated Graph ConvNets" (2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


class _GatedGCNLayer(nn.Module):
    """One residual gated graph-conv layer (node + edge update)."""

    def __init__(self, dim: int, norm: str, act, residual: bool):
        super().__init__()
        self.act = act
        self.residual = residual
        # edge update: e_hat = A x_dst + B x_src + C e
        self.A = nn.Linear(dim, dim)
        self.B = nn.Linear(dim, dim)
        self.C = nn.Linear(dim, dim)
        # node update: U x + gated-mean(V x_src)
        self.U = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.norm_e = Normalizer(norm, dim)
        self.norm_h = Normalizer(norm, dim)

    def forward(self, h: Tensor, e: Tensor, src: Tensor, dst: Tensor, num_nodes: int):
        e_hat = self.A(h.index_select(0, dst)) + self.B(h.index_select(0, src)) + self.C(e)
        e_new = self.act(self.norm_e(e_hat))
        e = e + e_new if self.residual else e_new
        eta = torch.sigmoid(e_hat)  # [E, dim] soft gate
        gated = eta * self.V(h.index_select(0, src))  # [E, dim]
        num = scatter_sum(gated, dst, num_nodes)  # [N, dim]
        den = scatter_sum(eta, dst, num_nodes) + 1e-6  # [N, dim]
        h_hat = self.U(h) + num / den
        h_new = self.act(self.norm_h(h_hat))
        h = h + h_new if self.residual else h_new
        return h, e


@ENCODERS.register("gatedgcn")
class GatedGCN(GNNEncoder):
    """Residual Gated GCN encoder with maintained edge features."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.node_in = nn.Linear(self.in_dim, h)
        # edge feature initialisation: from edge_attr if present else from edge_weight
        self.edge_from_weight = nn.Linear(1, h)
        self.edge_proj = nn.Linear(self.edge_dim, h) if self.edge_dim > 0 else None
        self.layers = nn.ModuleList(
            _GatedGCNLayer(h, self.norm_kind, self.act, self.residual)
            for _ in range(self.num_layers)
        )

    def _init_edges(self, graph: BatchedGraph, device: torch.device) -> Tensor:
        e = graph.num_edges
        if (
            graph.edge_attr is not None
            and self.edge_proj is not None
            and graph.edge_attr.shape[-1] == self.edge_dim
        ):
            return self.edge_proj(graph.edge_attr.float())
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(e, device=device)
        return self.edge_from_weight(w.view(e, 1).float())

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        e = self._init_edges(graph, h.device)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        for layer in self.layers:
            h, e = layer(h, e, src, dst, graph.num_nodes)
        return h
