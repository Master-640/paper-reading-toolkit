"""HNHN: Hypergraph Networks with Hyperedge Neurons (Dong et al., 2020).

A hypergraph is stored as an incidence in ``graph.meta`` (``inc_node``,
``inc_hyperedge``, ``num_hyperedges``).  HNHN alternates two propagation stages,
each with its own linear map and nonlinearity, and normalizes with **tunable
degree exponents** ``alpha`` (node->hyperedge) and ``beta`` (hyperedge->node)::

    f_e = sigma( W_E sum_{v in e} (D_v^alpha / sum_{u in e} D_u^alpha) h_v ),
    f'_v = W_V sum_{e ni v} (D_e^beta / sum_{e' ni v} D_e'^beta) f_e,

where ``D_v`` is the number of hyperedges containing ``v`` and ``D_e`` the size
of hyperedge ``e``.  Setting ``alpha = beta = 0`` recovers plain mean pooling.

When no incidence meta is present the encoder **falls back** to an ordinary GCN
pass over ``edge_index`` so it never crashes on a plain graph.

Reference: Y. Dong, W. Sawin, Y. Bengio, "HNHN: Hypergraph Networks with
Hyperedge Neurons" (ICML Graph Representation Learning workshop, 2020).
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
) -> Optional[Tuple[Tensor, Tensor, int]]:
    """Return ``(inc_node, inc_hyperedge, num_hyperedges)`` or ``None`` (defensive)."""
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
        if "num_hyperedges" in meta:
            num_he = int(meta["num_hyperedges"])
        else:
            num_he = int(inc_he.max().item()) + 1
        if num_he <= 0 or int(inc_he.max().item()) >= num_he:
            return None
        return inc_node, inc_he, num_he
    except Exception:  # pragma: no cover - defensive against odd meta shapes
        return None


@ENCODERS.register("hnhn")
class HNHN(GNNEncoder):
    """HNHN two-stage node<->hyperedge propagation with a graph-GCN fallback."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        alpha: float = -0.5,
        beta: float = -0.5,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.node_in = nn.Linear(self.in_dim, h)
        self.node_lins = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        self.edge_lins = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))

    # ------------------------------------------------------------ hypergraph
    def _hnhn_step(
        self,
        h: Tensor,
        node_lin: nn.Linear,
        edge_lin: nn.Linear,
        inc: Tuple[Tensor, Tensor, int],
        num_nodes: int,
    ) -> Tensor:
        inc_node, inc_he, num_he = inc
        d_v = scatter_sum(
            torch.ones_like(inc_node, dtype=torch.float), inc_node, num_nodes
        ).clamp_min(1.0)
        d_e = scatter_sum(
            torch.ones_like(inc_he, dtype=torch.float), inc_he, num_he
        ).clamp_min(1.0)
        # stage 1: node -> hyperedge, normalized by D_v^alpha within each hyperedge.
        wv = d_v.pow(self.alpha).index_select(0, inc_node)  # [nnz]
        denom_e = scatter_sum(wv, inc_he, num_he).clamp_min(1e-12)
        norm_ve = wv / denom_e.index_select(0, inc_he)
        f_e = scatter_sum(h.index_select(0, inc_node) * norm_ve.unsqueeze(-1), inc_he, num_he)
        f_e = self.act(edge_lin(f_e))  # hyperedge neuron + nonlinearity
        # stage 2: hyperedge -> node, normalized by D_e^beta within each node.
        we = d_e.pow(self.beta).index_select(0, inc_he)  # [nnz]
        denom_v = scatter_sum(we, inc_node, num_nodes).clamp_min(1e-12)
        norm_ev = we / denom_v.index_select(0, inc_node)
        f_v = scatter_sum(f_e.index_select(0, inc_he) * norm_ev.unsqueeze(-1), inc_node, num_nodes)
        return node_lin(f_v)

    # ------------------------------------------------------- graph fallback
    def _graph_step(
        self, h: Tensor, lin: nn.Linear, graph: BatchedGraph, device: torch.device
    ) -> Tensor:
        src, dst = graph.edge_index[0], graph.edge_index[1]
        xw = lin(h)
        deg = graph.degree() + 1.0  # +1 for the self-loop
        dinv = deg.pow(-0.5)
        w = (
            graph.edge_weight
            if graph.edge_weight is not None
            else torch.ones(graph.num_edges, device=device)
        )
        norm = w * dinv.index_select(0, src) * dinv.index_select(0, dst)
        agg = scatter_sum(xw.index_select(0, src) * norm.unsqueeze(-1), dst, graph.num_nodes)
        return agg + (dinv * dinv).unsqueeze(-1) * xw  # self-loop term

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        inc = _read_incidence(graph, h.device)
        for i in range(self.num_layers):
            if inc is not None:
                out = self._hnhn_step(
                    h, self.node_lins[i], self.edge_lins[i], inc, graph.num_nodes
                )
            else:
                out = self._graph_step(h, self.node_lins[i], graph, h.device)
            out = self.act(self.norms[i](out))
            h = h + out if self.residual else out
        return h
