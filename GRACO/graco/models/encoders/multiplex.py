"""Multiplex / multilayer graph convolution.

A multiplex graph stacks several relation *layers* over one shared node set; the
layer id of each edge lives in ``edge_type`` and the layer count in
``graph.meta['num_layers']`` (falling back to ``edge_type.max()+1``).  Each
message-passing round runs an **independent symmetric-normalized GCN aggregation
per layer** and then FUSES the per-layer node embeddings with a learned
attention over layers (or a plain mean)::

    z^l_i = sum_{j ->_l i} c^l_ij (W_l h_j) + c^l_ii (W_l h_i),
    alpha^l_i = softmax_l( q . tanh(W_a z^l_i) ),
    h'_i = sum_l alpha^l_i z^l_i + W_0 h_i.

``num_conv_layers`` message-passing rounds are stacked (default: ``num_layers``);
the per-layer weights are built lazily on the first forward once the layer count
is known.  With a single layer (no ``edge_type``) this reduces to a plain GCN.

Reference (multiplex GNNs / layer fusion): mGCN and multiplex network embedding
literature; layer attention follows the HAN semantic-attention scheme.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


def _infer_num_layers(graph: BatchedGraph, override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    if graph.edge_type is not None and graph.num_edges > 0:
        default = int(graph.edge_type.max().item()) + 1
        return max(1, int(graph.meta.get("num_layers", default)))
    return max(1, int(graph.meta.get("num_layers", 1)))


def _gcn_layer_agg(
    wh: Tensor, src: Tensor, dst: Tensor, weight: Optional[Tensor], num_nodes: int
) -> Tensor:
    """Symmetric-normalized GCN aggregation (with self-loop) for one layer."""
    device = wh.device
    sl = torch.arange(num_nodes, device=device)
    s = torch.cat([src, sl])
    d = torch.cat([dst, sl])
    self_w = torch.ones(num_nodes, device=device)
    edge_w = torch.ones(src.numel(), device=device) if weight is None else weight
    w = torch.cat([edge_w, self_w])
    deg = scatter_sum(w, d, num_nodes).clamp_min(1e-12)
    dinv = deg.pow(-0.5)
    norm = w * dinv.index_select(0, s) * dinv.index_select(0, d)
    return scatter_sum(wh.index_select(0, s) * norm.unsqueeze(-1), d, num_nodes)


class _MultiplexLayer(nn.Module):
    """Per-layer GCN convs fused by attention (or mean) across layers."""

    def __init__(self, dim: int, num_graph_layers: int, fusion: str = "attention"):
        super().__init__()
        self.L = num_graph_layers
        self.fusion = fusion
        self.lins = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(num_graph_layers))
        self.self_lin = nn.Linear(dim, dim)
        if fusion == "attention":
            self.att_lin = nn.Linear(dim, dim)
            self.att_q = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        h: Tensor,
        src: Tensor,
        dst: Tensor,
        layer_id: Tensor,
        edge_weight: Optional[Tensor],
        num_nodes: int,
    ) -> Tensor:
        per_layer = []
        for li in range(self.L):
            mask = layer_id == li
            s, d = src[mask], dst[mask]
            wl = edge_weight[mask] if edge_weight is not None else None
            per_layer.append(_gcn_layer_agg(self.lins[li](h), s, d, wl, num_nodes))
        Z = torch.stack(per_layer, dim=1)  # [N, L, dim]
        if self.fusion == "attention" and self.L > 1:
            score = torch.tanh(self.att_lin(Z)) @ self.att_q  # [N, L]
            alpha = torch.softmax(score, dim=1)
            fused = (alpha.unsqueeze(-1) * Z).sum(1)  # [N, dim]
        else:
            fused = Z.mean(dim=1)
        return fused + self.self_lin(h)


@ENCODERS.register("multiplex_gcn")
class MultiplexGCN(GNNEncoder):
    """Multiplex GCN: per-layer aggregation with learned cross-layer fusion."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_graph_layers: Optional[int] = None,
        num_conv_layers: Optional[int] = None,
        fusion: str = "attention",
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.num_conv = int(num_conv_layers) if num_conv_layers is not None else self.num_layers
        self.num_graph_layers = num_graph_layers  # None -> infer from the batch
        self.fusion = str(fusion).lower()
        self.node_in = nn.Linear(self.in_dim, h)
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_conv))
        self.rounds: Optional[nn.ModuleList] = None
        self._built_layers: Optional[int] = None
        if self.num_graph_layers is not None:
            self._build(int(self.num_graph_layers), torch.device("cpu"))

    def _build(self, num_graph_layers: int, device: torch.device) -> None:
        self.rounds = nn.ModuleList(
            _MultiplexLayer(self.hidden_dim, num_graph_layers, self.fusion)
            for _ in range(self.num_conv)
        ).to(device)
        self._built_layers = num_graph_layers

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        num_graph_layers = _infer_num_layers(graph, self.num_graph_layers)
        if self._built_layers != num_graph_layers:
            self._build(num_graph_layers, h.device)
        if next(self.rounds.parameters()).device != h.device:
            self.rounds = self.rounds.to(h.device)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        if graph.edge_type is not None and graph.num_edges > 0:
            layer_id = graph.edge_type.to(h.device).clamp(max=num_graph_layers - 1)
        else:  # single layer -> plain GCN
            layer_id = torch.zeros(graph.num_edges, dtype=torch.long, device=h.device)
        ew = graph.edge_weight
        for i, rnd in enumerate(self.rounds):
            out = self.act(self.norms[i](rnd(h, src, dst, layer_id, ew, graph.num_nodes)))
            h = h + out if self.residual else out
        return h
