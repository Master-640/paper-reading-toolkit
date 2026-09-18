"""Generic Message Passing Neural Networks (Gilmer et al., 2017).

Two classic variants live here:

``mpnn``
    The canonical MPNN: a message is produced by an edge MLP over
    ``(x_src, x_dst, e)``, summed into each destination, and the node state is
    updated by a ``GRUCell`` (default) or an MLP.

``nnconv`` (a.k.a. ECC, Simonovsky & Komodakis 2017)
    The edge network maps ``edge_attr`` to a ``F_out x F_in`` weight matrix and
    the message is ``Theta_e @ x_src``, mean-aggregated into the destination.

Reference: J. Gilmer et al., "Neural Message Passing for Quantum Chemistry"
(ICML 2017); M. Simonovsky, N. Komodakis, "Dynamic Edge-Conditioned Filters"
(CVPR 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_mean, scatter_sum


@ENCODERS.register("mpnn")
class MPNN(GNNEncoder):
    """Generic message-passing encoder with a GRU or MLP node update."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        update: str = "gru",
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.update_kind = (update or "gru").lower()
        self.node_in = nn.Linear(self.in_dim, h)
        # edge embedding: from edge_attr if present else from edge_weight (dim 1).
        self.edge_from_weight = nn.Linear(1, h)
        self.edge_proj = nn.Linear(self.edge_dim, h) if self.edge_dim > 0 else None
        self.msg_mlps = nn.ModuleList(
            MLP(3 * h, h, h, num_layers=2, act="relu") for _ in range(self.num_layers)
        )
        if self.update_kind == "gru":
            self.grus = nn.ModuleList(nn.GRUCell(h, h) for _ in range(self.num_layers))
            self.upd_mlps = None
        else:
            self.grus = None
            self.upd_mlps = nn.ModuleList(
                MLP(2 * h, h, h, num_layers=2, act="relu") for _ in range(self.num_layers)
            )
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))

    def _edge_emb(self, graph: BatchedGraph, device: torch.device) -> Tensor:
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
        e = self._edge_emb(graph, h.device)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        for i in range(self.num_layers):
            msg = self.msg_mlps[i](
                torch.cat([h.index_select(0, src), h.index_select(0, dst), e], dim=-1)
            )
            agg = scatter_sum(msg, dst, graph.num_nodes)  # [N, h]
            if self.grus is not None:
                h_new = self.grus[i](agg, h)
            else:
                h_new = self.act(self.upd_mlps[i](torch.cat([h, agg], dim=-1)))
                h_new = h + h_new if self.residual else h_new
            h = self.norms[i](h_new)
        return h


@ENCODERS.register("nnconv")
class NNConv(GNNEncoder):
    """Edge-conditioned convolution (ECC): messages are ``Theta_e @ x_src``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.edge_in_dim = self.edge_dim if self.edge_dim > 0 else 1
        self.node_in = nn.Linear(self.in_dim, h)
        # per-layer edge network -> flattened (h x h) weight matrix, plus a root map.
        self.edge_nets = nn.ModuleList(
            MLP(self.edge_in_dim, h, h * h, num_layers=2, act="relu")
            for _ in range(self.num_layers)
        )
        self.roots = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))

    def _edge_features(self, graph: BatchedGraph, device: torch.device) -> Tensor:
        e = graph.num_edges
        if graph.edge_attr is not None and graph.edge_attr.shape[-1] == self.edge_in_dim:
            return graph.edge_attr.float()
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(e, device=device)
        return w.view(e, 1).float().expand(e, self.edge_in_dim)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        ea = self._edge_features(graph, h.device)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        hd = self.hidden_dim
        for i in range(self.num_layers):
            theta = self.edge_nets[i](ea).view(-1, hd, hd)  # [E, h, h]
            x_src = h.index_select(0, src).unsqueeze(-1)  # [E, h, 1]
            msg = torch.bmm(theta, x_src).squeeze(-1)  # [E, h]
            agg = scatter_mean(msg, dst, graph.num_nodes)  # [N, h]
            out = self.roots[i](h) + agg
            out = self.act(self.norms[i](out))
            h = h + out if self.residual else out
        return h
