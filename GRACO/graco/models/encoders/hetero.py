"""Heterogeneous / relational encoders.

``rgcn``
    Relational GCN (Schlichtkrull et al., 2018) with **basis decomposition**:
    every relation's weight ``W_r = sum_b a_{rb} V_b`` shares a small set of
    basis matrices, so parameter count stays bounded in the number of relations.
    Messages are normalised by the per-relation in-degree ``c_{i,r}`` and a
    self-connection ``W_0 x_i`` is added.

``hgt`` (Heterogeneous Graph Transformer, lite)
    Typed multi-head attention: per-relation additive attention priors and value
    biases modulate a shared scaled-dot-product attention over neighbours.

Both read the per-edge relation from ``graph.edge_type`` (``None`` -> all edges
are relation 0) and the relation count from ``num_relations`` /
``graph.meta['num_edge_types']``.

Reference: M. Schlichtkrull et al., "Modeling Relational Data with GCNs" (ESWC
2018); Z. Hu et al., "Heterogeneous Graph Transformer" (WWW 2020).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_softmax, scatter_sum


def _infer_num_relations(graph: BatchedGraph, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    if graph.edge_type is not None:
        default = int(graph.edge_type.max().item()) + 1 if graph.num_edges > 0 else 1
        return int(graph.meta.get("num_edge_types", default))
    return int(graph.meta.get("num_edge_types", 1))


@ENCODERS.register("rgcn")
class RGCN(GNNEncoder):
    """Relational GCN with basis-decomposed relation weights."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_relations: Optional[int] = None,
        num_bases: int = 4,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.num_bases = int(num_bases)
        self.num_relations = num_relations
        self.node_in = nn.Linear(self.in_dim, h)
        self.self_lins = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))
        # Basis matrices + per-relation coefficients are (re)built once R is known.
        self.bases: Optional[nn.ParameterList] = None
        self.coeffs: Optional[nn.ParameterList] = None
        self._built_relations: Optional[int] = None
        if num_relations is not None:
            self._build_relations(int(num_relations), torch.device("cpu"))

    def _build_relations(self, num_relations: int, device: torch.device) -> None:
        h, b = self.hidden_dim, self.num_bases
        bases, coeffs = [], []
        for _ in range(self.num_layers):
            v = nn.Parameter(torch.empty(b, h, h, device=device))
            a = nn.Parameter(torch.empty(num_relations, b, device=device))
            nn.init.xavier_uniform_(v)
            nn.init.xavier_uniform_(a)
            bases.append(v)
            coeffs.append(a)
        self.bases = nn.ParameterList(bases)
        self.coeffs = nn.ParameterList(coeffs)
        self._built_relations = num_relations

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        num_relations = _infer_num_relations(graph, self.num_relations)
        if self._built_relations != num_relations:
            self._build_relations(num_relations, h.device)
        # ensure lazily-built params share the feature device
        if self.bases[0].device != h.device:
            self.bases = self.bases.to(h.device)
            self.coeffs = self.coeffs.to(h.device)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        if graph.edge_type is not None:
            etype = graph.edge_type.to(h.device).clamp(max=num_relations - 1)
        else:
            etype = torch.zeros(graph.num_edges, dtype=torch.long, device=h.device)

        for layer in range(self.num_layers):
            # W_r = sum_b a_{rb} V_b  ->  [R, h, h]
            w = torch.einsum("rb,bij->rij", self.coeffs[layer], self.bases[layer])
            out = self.self_lins[layer](h)  # self-connection W_0 x_i
            for r in range(num_relations):
                mask = etype == r
                if not bool(mask.any()):
                    continue
                s, d = src[mask], dst[mask]
                m = h.index_select(0, s) @ w[r]  # [E_r, h]
                agg = scatter_sum(m, d, graph.num_nodes)  # [N, h]
                deg = scatter_sum(
                    torch.ones(s.numel(), device=h.device), d, graph.num_nodes
                ).clamp_min(1.0)
                out = out + agg / deg.unsqueeze(-1)  # normalise by c_{i,r}
            out = self.act(self.norms[layer](out))
            h = h + out if self.residual else out
        return h


class _HGTLayer(nn.Module):
    """Typed multi-head attention block (HGT-lite)."""

    def __init__(self, dim: int, num_heads: int, num_relations: int, act):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dk = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.dk)
        self.act = act
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # per-relation attention prior (per head) and value bias.
        self.rel_prior = nn.Parameter(torch.zeros(num_relations, num_heads))
        self.rel_msg = nn.Embedding(num_relations, dim)
        self.ffn = MLP(dim, dim * 2, dim, num_layers=2, act="gelu")
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, h, src, dst, etype, num_nodes):
        n, hd = num_nodes, self.num_heads
        x = self.ln1(h)
        q = self.q(x).view(n, hd, self.dk)
        k = self.k(x).view(n, hd, self.dk)
        v = self.v(x).view(n, hd, self.dk)
        score = (q.index_select(0, dst) * k.index_select(0, src)).sum(-1) * self.scale  # [E, hd]
        score = score + self.rel_prior.index_select(0, etype)  # typed prior
        alpha = scatter_softmax(score, dst, num_nodes)  # [E, hd]
        val = v.index_select(0, src) + self.rel_msg(etype).view(-1, hd, self.dk)
        msg = alpha.unsqueeze(-1) * val  # [E, hd, dk]
        agg = scatter_sum(msg, dst, num_nodes).reshape(n, self.dim)
        h = h + self.o(agg)
        h = h + self.ffn(self.ln2(h))
        return h


@ENCODERS.register("hgt")
class HGT(GNNEncoder):
    """Heterogeneous Graph Transformer (lite): typed sparse attention."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        num_relations: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        if h % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.node_in = nn.Linear(self.in_dim, h)
        self.layers: Optional[nn.ModuleList] = None
        self._built_relations: Optional[int] = None
        if num_relations is not None:
            self._build(int(num_relations), torch.device("cpu"))

    def _build(self, num_relations: int, device: torch.device) -> None:
        self.layers = nn.ModuleList(
            _HGTLayer(self.hidden_dim, self.num_heads, num_relations, self.act)
            for _ in range(self.num_layers)
        ).to(device)
        self._built_relations = num_relations

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        num_relations = _infer_num_relations(graph, self.num_relations)
        if self._built_relations != num_relations:
            self._build(num_relations, h.device)
        if next(self.layers.parameters()).device != h.device:
            self.layers = self.layers.to(h.device)

        src, dst = graph.edge_index[0], graph.edge_index[1]
        if graph.edge_type is not None:
            etype = graph.edge_type.to(h.device).clamp(max=num_relations - 1)
        else:
            etype = torch.zeros(graph.num_edges, dtype=torch.long, device=h.device)
        for layer in self.layers:
            h = layer(h, src, dst, etype, graph.num_nodes)
        return h
