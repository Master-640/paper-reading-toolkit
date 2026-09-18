"""Sparse-neighbour graph Transformer (Graphormer-lite).

A pre-LayerNorm Transformer block that attends only over graph neighbours: for
each directed edge ``src -> dst`` a scaled dot-product score is computed between
the destination query and the source key, softmaxed **per destination** with a
segmented (scatter) softmax, and used to aggregate source values.  Two
Graphormer touches are included: a learnable *degree embedding* added to the
input features, and an optional *edge bias* derived from ``edge_attr``.

Reference: C. Ying et al., "Do Transformers Really Perform Bad for Graph
Representation?" (Graphormer, NeurIPS 2021); Dwivedi & Bresson, "A Generalization
of Transformer Networks to Graphs" (2020).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_softmax, scatter_sum


class _GraphAttentionLayer(nn.Module):
    """One pre-LN sparse multi-head attention block with an FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_mult: int, dropout: float, act):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dk = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.dk)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, dim * ffn_mult, dim, num_layers=2, act="gelu")
        self.dropout = dropout

    def forward(
        self,
        h: Tensor,
        src: Tensor,
        dst: Tensor,
        num_nodes: int,
        edge_bias: Optional[Tensor],
    ) -> Tensor:
        n, hd = num_nodes, self.num_heads
        x = self.ln1(h)
        q = self.q(x).view(n, hd, self.dk)
        k = self.k(x).view(n, hd, self.dk)
        v = self.v(x).view(n, hd, self.dk)
        score = (q.index_select(0, dst) * k.index_select(0, src)).sum(-1) * self.scale  # [E, hd]
        if edge_bias is not None:
            score = score + edge_bias
        alpha = scatter_softmax(score, dst, num_nodes)  # [E, hd]
        msg = alpha.unsqueeze(-1) * v.index_select(0, src)  # [E, hd, dk]
        agg = scatter_sum(msg, dst, num_nodes).reshape(n, self.dim)  # [N, dim]
        h = h + self.o(agg)  # residual (pre-LN)
        h = h + self.ffn(self.ln2(h))  # residual FFN
        return h


@ENCODERS.register("graph_transformer", aliases=["transformer"])
class GraphTransformer(GNNEncoder):
    """Graphormer-lite encoder: sparse attention + degree encoding."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_mult: int = 2,
        max_degree: int = 64,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        if h % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.max_degree = int(max_degree)
        self.node_in = nn.Linear(self.in_dim, h)
        # Graphormer-style learnable degree embedding (bucketed in-degree).
        self.degree_emb = nn.Embedding(self.max_degree + 1, h)
        self.edge_bias = nn.Linear(self.edge_dim, num_heads) if self.edge_dim > 0 else None
        self.layers = nn.ModuleList(
            _GraphAttentionLayer(h, num_heads, ffn_mult, self.dropout, self.act)
            for _ in range(self.num_layers)
        )

    def _edge_bias(self, graph: BatchedGraph, device: torch.device) -> Optional[Tensor]:
        if self.edge_bias is None:
            return None
        if graph.edge_attr is not None and graph.edge_attr.shape[-1] == self.edge_dim:
            ea = graph.edge_attr.float()
        else:
            ea = torch.zeros(graph.num_edges, self.edge_dim, device=device)
        return self.edge_bias(ea)  # [E, num_heads]

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        deg_idx = graph.degree().long().clamp(min=0, max=self.max_degree)
        h = self.node_in(x) + self.degree_emb(deg_idx)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        edge_bias = self._edge_bias(graph, h.device)
        for layer in self.layers:
            h = layer(h, src, dst, graph.num_nodes, edge_bias)
        return h
