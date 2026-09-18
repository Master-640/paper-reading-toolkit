"""HyperGCN (Yadati et al., 2019).

A hypergraph is stored as an incidence in ``graph.meta`` (``inc_node``,
``inc_hyperedge``, ``num_hyperedges``).  HyperGCN reduces each hyperedge to a
small set of ordinary edges and then runs a GCN over them.  For every hyperedge
``e`` a data-dependent pair ``(i_e, j_e)`` is chosen as the incident nodes with
the extreme values of a learned scalar projection of the current features; with
**mediators** every other incident node ``k`` is wired to both ``i_e`` and
``j_e``.  Each generated edge is weighted by ``1 / (2|e| - 3)`` (Yadati's
mediator normalization).  The resulting weighted graph is convolved with the
symmetric-normalized GCN propagation (``A~ = A + I``).

When no incidence meta is present the encoder **falls back** to an ordinary GCN
pass over ``edge_index`` so it never crashes on a plain graph.

Reference: N. Yadati et al., "HyperGCN: A New Method of Training Graph
Convolutional Networks on Hypergraphs" (NeurIPS 2019).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_max, scatter_min, scatter_sum


def _read_incidence(
    graph: BatchedGraph, device: torch.device
) -> Optional[Tuple[Tensor, Tensor, int]]:
    """Return ``(inc_node, inc_hyperedge, num_hyperedges)`` or ``None``.

    Defensive about meta shapes so a malformed incidence falls back to the
    plain-graph GCN path instead of crashing.
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
        if "num_hyperedges" in meta:
            num_he = int(meta["num_hyperedges"])
        else:
            num_he = int(inc_he.max().item()) + 1
        if num_he <= 0 or int(inc_he.max().item()) >= num_he:
            return None
        return inc_node, inc_he, num_he
    except Exception:  # pragma: no cover - defensive against odd meta shapes
        return None


def _gcn_norm_agg(wh: Tensor, src: Tensor, dst: Tensor, w: Tensor, num_nodes: int) -> Tensor:
    """Symmetric-normalized aggregation over directed edges ``(src, dst, w)`` + self-loops."""
    sl = torch.arange(num_nodes, device=wh.device)
    src = torch.cat([src, sl])
    dst = torch.cat([dst, sl])
    w = torch.cat([w, torch.ones(num_nodes, device=wh.device)])
    deg = scatter_sum(w, dst, num_nodes).clamp_min(1e-12)
    dinv = deg.pow(-0.5)
    norm = w * dinv.index_select(0, src) * dinv.index_select(0, dst)
    return scatter_sum(wh.index_select(0, src) * norm.unsqueeze(-1), dst, num_nodes)


@ENCODERS.register("hypergcn")
class HyperGCN(GNNEncoder):
    """HyperGCN with a data-dependent mediator expansion and a graph-GCN fallback."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mediators: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.mediators = bool(mediators)
        self.node_in = nn.Linear(self.in_dim, h)
        self.thetas = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        # per-layer scalar projection used to pick each hyperedge's extreme pair.
        self.proj = nn.ModuleList(nn.Linear(h, 1) for _ in range(self.num_layers))
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))

    def _hyper_edges(
        self, h: Tensor, proj: nn.Linear, inc: Tuple[Tensor, Tensor, int], num_nodes: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build directed (both-direction) mediator edges for the current features."""
        inc_node, inc_he, num_he = inc
        # scalar signal per node; detached because the argmax/argmin selection is
        # discrete (non-differentiable) — gradients still flow via the aggregation.
        s = proj(h).squeeze(-1).detach()  # [N]
        s_inc = s.index_select(0, inc_node)  # [nnz]
        hi_val = scatter_max(s_inc, inc_he, num_he).index_select(0, inc_he)  # [nnz]
        lo_val = scatter_min(s_inc, inc_he, num_he).index_select(0, inc_he)
        neg = torch.full_like(inc_node, -1)
        # representative node ids per hyperedge (ties -> largest node id, deterministic).
        i_node = scatter_max(torch.where(s_inc >= hi_val, inc_node, neg).float(), inc_he, num_he)
        j_node = scatter_max(torch.where(s_inc <= lo_val, inc_node, neg).float(), inc_he, num_he)
        i_node, j_node = i_node.long(), j_node.long()
        size_e = scatter_sum(torch.ones_like(inc_node, dtype=torch.float), inc_he, num_he)
        w_e = (2.0 * size_e - 3.0).clamp_min(1.0).reciprocal()  # mediator weight

        i_of = i_node.index_select(0, inc_he)  # [nnz]
        j_of = j_node.index_select(0, inc_he)
        w_of = w_e.index_select(0, inc_he)
        if self.mediators:  # each incident node connects to both endpoints
            u = torch.cat([inc_node, inc_node])
            v = torch.cat([i_of, j_of])
            w = torch.cat([w_of, w_of])
        else:  # single representative simple edge per hyperedge
            u, v, w = i_node, j_node, w_e
        # symmetrize -> directed both-directions
        return torch.cat([u, v]), torch.cat([v, u]), torch.cat([w, w])

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        inc = _read_incidence(graph, h.device)
        for i in range(self.num_layers):
            wh = self.thetas[i](h)
            if inc is not None:
                src, dst, w = self._hyper_edges(h, self.proj[i], inc, graph.num_nodes)
            else:  # plain-graph fallback
                src, dst = graph.edge_index[0], graph.edge_index[1]
                w = (
                    graph.edge_weight
                    if graph.edge_weight is not None
                    else torch.ones(graph.num_edges, device=h.device)
                )
            out = _gcn_norm_agg(wh, src, dst, w, graph.num_nodes)
            out = self.act(self.norms[i](out))
            h = h + out if self.residual else out
        return h
