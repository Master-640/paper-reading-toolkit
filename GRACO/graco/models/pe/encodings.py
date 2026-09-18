"""Positional / structural encodings (PE) as a composable encoder wrapper.

``PEEncoder`` computes cheap, fully-vectorized structural features from the graph
topology, concatenates them to the node features, and runs any inner encoder on
the augmented input.  Use it from YAML by naming ``type: pe`` and giving the
inner encoder::

    encoder:
      type: pe
      pe_types: [degree, neighbor_degree]
      encoder: {type: gat, hidden_dim: 64, num_layers: 3}

The default structural features (degree, mean/max neighbour degree, 2-hop reach)
are scatter-computed in O(E) with no eigendecomposition, so they are safe to
recompute every forward pass. Laplacian / random-walk PE can be added as extra
``pe_types`` for small graphs — see :func:`laplacian_pe`.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_max, scatter_mean, scatter_sum


def structural_pe(graph: BatchedGraph, kinds: List[str]) -> Tensor:
    """Concatenate the requested cheap structural encodings -> ``[N, pe_dim]``."""
    src, dst = graph.edge_index[0], graph.edge_index[1]
    deg = graph.degree()  # [N]
    feats = []
    for k in kinds:
        if k == "degree":
            feats.append(torch.log1p(deg).unsqueeze(-1))
        elif k == "neighbor_degree":
            nbr_mean = scatter_mean(deg.index_select(0, src), dst, graph.num_nodes)
            nbr_max = scatter_max(deg.index_select(0, src), dst, graph.num_nodes)
            feats.append(torch.log1p(nbr_mean).unsqueeze(-1))
            feats.append(torch.log1p(nbr_max).unsqueeze(-1))
        elif k == "two_hop":
            two_hop = scatter_sum(deg.index_select(0, src), dst, graph.num_nodes)
            feats.append(torch.log1p(two_hop).unsqueeze(-1))
        else:
            raise ValueError(f"Unknown structural PE '{k}'")
    return torch.cat(feats, dim=-1) if feats else deg.new_zeros(graph.num_nodes, 0)


@torch.no_grad()
def laplacian_pe(graph: BatchedGraph, k: int = 4) -> Tensor:
    """Laplacian positional encoding: k smallest non-trivial eigenvectors ``[N, k]``.

    Computed per graph via dense ``eigh`` (fine for small graphs / evaluation).
    Sign is arbitrary; randomly flip during training if used as a learned input.
    """
    out = graph.edge_index.new_zeros(graph.num_nodes, k, dtype=torch.float32)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    for g in range(graph.num_graphs):
        lo, hi = int(graph.ptr[g]), int(graph.ptr[g + 1])
        n = hi - lo
        if n <= 1:
            continue
        m = (src >= lo) & (src < hi)
        a = torch.zeros(n, n, device=graph.edge_index.device)
        a[src[m] - lo, dst[m] - lo] = 1.0
        deg = a.sum(-1).clamp_min(1e-8)
        dinv = deg.pow(-0.5)
        lap = torch.eye(n, device=a.device) - dinv.unsqueeze(1) * a * dinv.unsqueeze(0)
        _, evecs = torch.linalg.eigh(lap)
        take = evecs[:, 1 : k + 1]  # skip the trivial component
        out[lo:hi, : take.shape[1]] = take.float()
    return out


@ENCODERS.register("pe", aliases=["pe_wrapper"])
class PEEncoder(GNNEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 0,
        encoder: Optional[dict] = None,
        pe_types: Optional[List[str]] = None,
        lap_k: int = 0,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, edge_dim=edge_dim, **kwargs)
        self.pe_types = pe_types or ["degree", "neighbor_degree"]
        self.lap_k = int(lap_k)
        # pe dimension: degree(1) + neighbor_degree(2) + two_hop(1) + lap_k
        dim_map = {"degree": 1, "neighbor_degree": 2, "two_hop": 1}
        self.pe_dim = sum(dim_map[k] for k in self.pe_types) + self.lap_k

        inner_cfg = dict(encoder or {"type": "gcn"})
        inner_cfg.setdefault("hidden_dim", hidden_dim)
        inner_cfg.setdefault("num_layers", num_layers)
        self.inner = ENCODERS.build(inner_cfg, in_dim=in_dim + self.pe_dim, edge_dim=edge_dim)
        self.out_dim = self.inner.out_dim

    def forward(self, graph: BatchedGraph) -> Tensor:
        pe = structural_pe(graph, self.pe_types)
        if self.lap_k > 0:
            pe = torch.cat([pe, laplacian_pe(graph, self.lap_k)], dim=-1)
        x = torch.cat([self.input_x(graph), pe], dim=-1)
        aug = dataclasses.replace(graph, x=x, _cache={})
        return self.inner(aug)
