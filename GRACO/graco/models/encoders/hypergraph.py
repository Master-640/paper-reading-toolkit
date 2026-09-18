"""Hypergraph convolution (HGNN, Feng et al., 2019).

A hypergraph is stored as an incidence in ``graph.meta``:

``inc_node``       LongTensor ``[nnz]``   node id of each incidence entry.
``inc_hyperedge``  LongTensor ``[nnz]``   hyperedge id of each incidence entry.
``num_hyperedges`` int (optional)         total hyperedges across the batch.
``inc_weight``     FloatTensor ``[nnz]``  optional incidence weights.

A layer is two scatter stages: node -> hyperedge (``f_e``) then hyperedge ->
node.  When no incidence meta is present we **fall back** to an ordinary,
symmetric-normalised GCN pass over ``edge_index`` so the encoder never crashes.

Reference: Y. Feng et al., "Hypergraph Neural Networks" (AAAI 2019); S. Bai et
al., "Hypergraph Convolution and Hypergraph Attention" (2020).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


def _read_incidence(
    graph: BatchedGraph, device: torch.device
) -> Optional[Tuple[Tensor, Tensor, Tensor, int]]:
    """Return ``(inc_node, inc_hyperedge, inc_weight, num_he)`` or ``None``.

    Defensive about meta shapes: any inconsistency falls back to ``None`` so the
    caller uses the plain-graph GCN path instead of crashing.
    """
    meta = graph.meta or {}
    if "inc_node" not in meta or "inc_hyperedge" not in meta:
        return None
    try:
        inc_node = torch.as_tensor(meta["inc_node"], device=device, dtype=torch.long).view(-1)
        inc_he = torch.as_tensor(meta["inc_hyperedge"], device=device, dtype=torch.long).view(-1)
        if inc_node.numel() == 0 or inc_node.numel() != inc_he.numel():
            return None
        if int(inc_node.max().item()) >= graph.num_nodes:
            return None
        if "inc_weight" in meta and meta["inc_weight"] is not None:
            w = torch.as_tensor(meta["inc_weight"], device=device, dtype=torch.float).view(-1)
            if w.numel() != inc_node.numel():
                w = torch.ones_like(inc_node, dtype=torch.float)
        else:
            w = torch.ones_like(inc_node, dtype=torch.float)
        if "num_hyperedges" in meta:
            num_he = int(meta["num_hyperedges"])
        elif "hyperedge_ptr" in meta and meta["hyperedge_ptr"] is not None:
            num_he = int(torch.as_tensor(meta["hyperedge_ptr"]).view(-1)[-1].item())
        else:
            num_he = int(inc_he.max().item()) + 1
        if num_he <= 0 or int(inc_he.max().item()) >= num_he:
            return None
        return inc_node, inc_he, w, num_he
    except Exception:  # pragma: no cover - defensive against odd meta shapes
        return None


@ENCODERS.register("hgnn", aliases=["hypergraph_conv"])
class HGNN(GNNEncoder):
    """Hypergraph neural network with a graph-GCN fallback."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.node_in = nn.Linear(self.in_dim, h)
        self.thetas = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))

    # ------------------------------------------------------------ hypergraph
    def _hyper_step(self, h: Tensor, theta, inc, num_nodes: int) -> Tensor:
        inc_node, inc_he, w, num_he = inc
        xw = theta(h)  # [N, h]
        w = w.unsqueeze(-1)
        # stage 1: node -> hyperedge, normalised by hyperedge degree D_e.
        d_e = scatter_sum(w.squeeze(-1), inc_he, num_he).clamp_min(1.0)
        f_e = scatter_sum(xw.index_select(0, inc_node) * w, inc_he, num_he)
        f_e = f_e / d_e.unsqueeze(-1)  # [num_he, h]
        # stage 2: hyperedge -> node, normalised by sqrt(node degree D_v).
        d_v = scatter_sum(w.squeeze(-1), inc_node, num_nodes).clamp_min(1.0)
        out = scatter_sum(f_e.index_select(0, inc_he), inc_node, num_nodes)
        return out / d_v.sqrt().unsqueeze(-1)

    # ------------------------------------------------------- graph fallback
    def _graph_step(self, h: Tensor, theta, graph: BatchedGraph, device: torch.device) -> Tensor:
        src, dst = graph.edge_index[0], graph.edge_index[1]
        n = graph.num_nodes
        xw = theta(h)
        deg = graph.degree() + 1.0  # +1 for the self-loop
        d_inv_sqrt = deg.pow(-0.5)
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(
            graph.num_edges, device=device
        )
        norm = w * d_inv_sqrt.index_select(0, src) * d_inv_sqrt.index_select(0, dst)
        m = xw.index_select(0, src) * norm.unsqueeze(-1)
        agg = scatter_sum(m, dst, n)
        self_term = xw * (d_inv_sqrt * d_inv_sqrt).unsqueeze(-1)  # self-loop
        return agg + self_term

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        inc = _read_incidence(graph, h.device)
        for i in range(self.num_layers):
            if inc is not None:
                out = self._hyper_step(h, self.thetas[i], inc, graph.num_nodes)
            else:
                out = self._graph_step(h, self.thetas[i], graph, h.device)
            out = self.act(self.norms[i](out))
            h = h + out if self.residual else out
        return h
