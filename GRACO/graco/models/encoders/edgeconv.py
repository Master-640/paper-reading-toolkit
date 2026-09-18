"""Dynamic Edge Convolution (EdgeConv / DGCNN, Wang et al., 2019).

For each directed edge ``j -> i`` (``src -> dst``) EdgeConv forms the message
``MLP(concat(x_i, x_j - x_i))`` — an explicitly *relative* edge feature — and
aggregates messages into the central node ``i`` with a max reduction.  Here the
graph is treated as static (no dynamic k-NN recomputation): we message-pass over
the provided ``edge_index``.

Reference: Y. Wang et al., "Dynamic Graph CNN for Learning on Point Clouds"
(DGCNN, ACM ToG 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_max


@ENCODERS.register("edgeconv")
class EdgeConv(GNNEncoder):
    """EdgeConv encoder: ``max_j MLP(x_i, x_j - x_i)``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.node_in = nn.Linear(self.in_dim, h)
        self.mlps = nn.ModuleList(
            MLP(2 * h, h, h, num_layers=2, act="relu") for _ in range(self.num_layers)
        )
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        h = self.node_in(x)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        for i in range(self.num_layers):
            x_i = h.index_select(0, dst)  # central node
            x_j = h.index_select(0, src)  # neighbour
            msg = self.mlps[i](torch.cat([x_i, x_j - x_i], dim=-1))  # [E, h]
            agg = scatter_max(msg, dst, graph.num_nodes)  # [N, h]
            out = self.act(self.norms[i](agg))
            h = h + out if self.residual else out
        return h
