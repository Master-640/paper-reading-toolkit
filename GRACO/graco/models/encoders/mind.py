"""MIND encoder — "Learning Network Dismantling Without Handcrafted Inputs".

A message-passing GNN designed to dismantle from **constant (all-ones) node inputs**,
combining three ingredients the paper relies on instead of hand-crafted features:

1. **Learnable-normalized attention** — GATv2-style edge messages whose weights are
   gated by a learned sigmoid (an MLP normalization) rather than a softmax.
2. **Omni global node** — a graph-level embedding (mean pool) broadcast back into
   every node each layer, giving each node a view of the whole (shrinking) network.
3. **Jumping knowledge** — the per-layer node states are concatenated and projected,
   so shallow and deep structure are both retained.

Pairs with discrete SAC (``graco/configs/mind.yaml``); reuses the whole GRACO stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, Normalizer
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum

_EPS = 1e-8


@ENCODERS.register("mind")
class MINDEncoder(GNNEncoder):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        h = self.hidden_dim
        self.inp = nn.Linear(self.in_dim, h)
        self.msg = nn.ModuleList(nn.Linear(2 * h, h) for _ in range(self.num_layers))
        self.att = nn.ModuleList(nn.Linear(h, 1) for _ in range(self.num_layers))
        self.upd = nn.ModuleList(nn.Linear(2 * h, h) for _ in range(self.num_layers))
        self.omni = nn.ModuleList(nn.Linear(h, h) for _ in range(self.num_layers))
        self.norms = nn.ModuleList(Normalizer(self.norm_kind, h) for _ in range(self.num_layers))
        self.jk = MLP(h * self.num_layers, h, self.out_dim, num_layers=2)  # jumping knowledge

    def forward(self, graph: BatchedGraph) -> Tensor:
        h = self.act(self.inp(self.input_x(graph)))
        src, dst = graph.edge_index[0], graph.edge_index[1]
        n = graph.num_nodes
        outs = []
        for i in range(self.num_layers):
            if src.numel():
                e = F.leaky_relu(self.msg[i](torch.cat([h.index_select(0, src), h.index_select(0, dst)], -1)))
                w = torch.sigmoid(self.att[i](e))  # learnable-normalized attention (no softmax)
                num = scatter_sum(w * e, dst, n)
                den = scatter_sum(w, dst, n).clamp_min(_EPS)
                agg = num / den
            else:
                agg = torch.zeros_like(h)
            omni = graph.broadcast_to_nodes(graph.pool(h, reduce="mean"))  # global (omni) node
            h = self.act(self.upd[i](torch.cat([h, agg], -1)) + self.omni[i](omni))
            h = self.norms[i](h)
            if self.dropout > 0 and i < self.num_layers - 1:
                h = F.dropout(h, self.dropout, self.training)
            outs.append(h)
        return self.jk(torch.cat(outs, dim=-1))  # [N, out_dim]
