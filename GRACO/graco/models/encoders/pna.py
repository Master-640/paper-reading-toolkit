"""Principal Neighbourhood Aggregation (Corso et al., 2020).

PNA combines several *aggregators* (mean, max, min, std) with degree-dependent
*scalers* ``S(d, alpha) = (log(d + 1) / delta) ** alpha`` for ``alpha`` in
``{+1, 0, -1}`` (amplification / identity / attenuation).  Each edge produces a
message via an MLP over ``concat(x_src, x_dst, edge_attr?)``; the messages are
reduced with every aggregator, every aggregate is multiplied by every scaler,
and a post-MLP mixes the resulting statistics back to ``hidden_dim``.

Reference: G. Corso et al., "Principal Neighbourhood Aggregation for Graph
Nets" (NeurIPS 2020).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_max, scatter_mean, scatter_min

_EPS = 1e-5


class _PNALayer(nn.Module):
    """A single PNA message + (aggregators x scalers) + post-MLP layer."""

    def __init__(self, dim: int, edge_dim: int, delta: float, norm: str, act, residual: bool):
        super().__init__()
        self.dim = dim
        self.edge_dim = edge_dim
        self.delta = float(delta)
        self.act = act
        self.residual = residual
        msg_in = 2 * dim + (edge_dim if edge_dim > 0 else 0)
        self.msg_mlp = MLP(msg_in, dim, dim, num_layers=2, act="relu")
        # 4 aggregators x 3 scalers = 12 statistics, plus the node's own state.
        self.post_mlp = MLP(dim + 12 * dim, dim, dim, num_layers=2, act="relu")
        self.norm = Normalizer(norm, dim)

    def forward(self, h: Tensor, ea, src: Tensor, dst: Tensor, log_deg: Tensor, num_nodes: int):
        parts = [h.index_select(0, src), h.index_select(0, dst)]
        if self.edge_dim > 0:
            parts.append(ea)
        m = self.msg_mlp(torch.cat(parts, dim=-1))  # [E, dim]

        mean = scatter_mean(m, dst, num_nodes)
        mx = scatter_max(m, dst, num_nodes)
        mn = scatter_min(m, dst, num_nodes)
        mean_sq = scatter_mean(m * m, dst, num_nodes)
        std = (mean_sq - mean * mean).clamp_min(0.0).add(_EPS).sqrt()
        stats = torch.cat([mean, mx, mn, std], dim=-1)  # [N, 4*dim]

        ident = torch.ones_like(log_deg)
        amp = log_deg / self.delta
        att = self.delta / log_deg.clamp_min(_EPS)
        scaled = torch.cat(
            [stats * ident.unsqueeze(-1), stats * amp.unsqueeze(-1), stats * att.unsqueeze(-1)],
            dim=-1,
        )  # [N, 12*dim]

        out = self.post_mlp(torch.cat([h, scaled], dim=-1))
        out = self.act(self.norm(out))
        return h + out if self.residual else out


@ENCODERS.register("pna")
class PNA(GNNEncoder):
    """Principal Neighbourhood Aggregation encoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        delta: float = 1.0,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.delta = float(delta)
        self.node_in = nn.Linear(self.in_dim, h)
        self.layers = nn.ModuleList(
            _PNALayer(h, self.edge_dim, self.delta, self.norm_kind, self.act, self.residual)
            for _ in range(self.num_layers)
        )

    def _edge_features(self, graph: BatchedGraph, device: torch.device):
        if self.edge_dim <= 0:
            return None
        if graph.edge_attr is not None and graph.edge_attr.shape[-1] == self.edge_dim:
            return graph.edge_attr.float()
        return torch.zeros(graph.num_edges, self.edge_dim, device=device)

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        ea = self._edge_features(graph, h.device)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        log_deg = torch.log(graph.degree() + 1.0)  # [N]
        for layer in self.layers:
            h = layer(h, ea, src, dst, log_deg, graph.num_nodes)
        return h
