"""Structure2Vec (structure2vec / S2V), the message-passing core of DIRAC.

Faithful re-implementation of the DIRAC node+edge alternating scheme.  Node and
edge states are refined together over ``num_layers`` (parameter-tied)
iterations.  Every ReLU is followed by L2 normalization, exactly as in the
reference.  A raw edge embedding ``edge_init`` (from ``edge_attr`` if present,
otherwise from the scalar ``edge_weight``) is kept fixed across iterations.

Per iteration::

    msg      = cur_node @ p_node_conv1                       # [N, H]
    n2e      = msg[src]                                       # [E, H]
    cur_edge = l2( relu( [n2e W_e1 || edge_init W_e2] ) )     # [E, H]
    e2n      = scatter_sum(cur_edge, dst, N)                  # [N, H]
    cur_new  = l2( relu( [e2n W_n1 || cur_node W_n2] ) )      # [N, H]
    cur_node = [cur_new || cur_node_prev] @ w_l               # [N, H]

Reference: H. Dai et al., "Learning Combinatorial Optimization Algorithms over
Graphs" (structure2vec, NeurIPS 2017); DIRAC (Fan et al.) reuses the same core.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("s2v", aliases=["dirac_s2v"])
class Structure2Vec(GNNEncoder):
    """DIRAC-style structure2vec encoder; embeddings are ``[N, hidden_dim]``."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        self.out_dim = self.hidden_dim  # S2V embeddings live at hidden width
        h = self.hidden_dim
        edge_in = self.edge_dim if self.edge_dim > 0 else 1  # edge_weight is a 1-d feature
        h1 = h // 2
        h2 = h - h1  # split so concat == h for any h
        self.w_n2l = nn.Linear(self.in_dim, h, bias=False)
        self.w_e2l = nn.Linear(edge_in, h, bias=False)
        self.p_node_conv1 = nn.Linear(h, h, bias=False)
        self.trans_edge_1 = nn.Linear(h, h1, bias=False)
        self.trans_edge_2 = nn.Linear(h, h2, bias=False)
        self.trans_node_1 = nn.Linear(h, h1, bias=False)
        self.trans_node_2 = nn.Linear(h, h2, bias=False)
        self.w_l = nn.Linear(2 * h, h, bias=False)

    @staticmethod
    def _l2(t: Tensor) -> Tensor:
        return F.normalize(t, p=2.0, dim=1, eps=1e-12)

    def _edge_features(self, graph: BatchedGraph) -> Tensor:
        if self.edge_dim > 0 and graph.edge_attr is not None:
            return graph.edge_attr.float()
        w = graph.edge_weight
        if w is None:
            w = torch.ones(graph.num_edges, device=graph.device)
        return w.view(graph.num_edges, 1).float()

    def forward(self, graph: BatchedGraph) -> Tensor:
        src, dst, n, edge_init, cur_node = self._init_propagation(graph)
        for _ in range(self.num_layers):
            cur_node = self._round(cur_node, edge_init, src, dst, n)
        return cur_node

    def _init_propagation(self, graph: BatchedGraph):
        """Shared setup: input projection, fixed edge embedding, initial node state."""
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        n = graph.num_nodes
        edge_init = F.relu(self.w_e2l(self._edge_features(graph)))  # [E, H] fixed across rounds
        cur_node = self._l2(F.relu(self.w_n2l(x)))  # [N, H]
        return src, dst, n, edge_init, cur_node

    def _round(self, cur_node: Tensor, edge_init: Tensor, src: Tensor, dst: Tensor, n: int) -> Tensor:
        """One structure2vec node+edge refinement round -> new node state ``[N, H]``."""
        prev = cur_node
        msg = self.p_node_conv1(cur_node)  # [N, H]
        n2e = msg.index_select(0, src)  # [E, H]
        cur_edge = self._l2(
            F.relu(torch.cat([self.trans_edge_1(n2e), self.trans_edge_2(edge_init)], dim=-1))
        )  # [E, H]
        e2n = scatter_sum(cur_edge, dst, n)  # [N, H]
        cur_node_new = self._l2(
            F.relu(torch.cat([self.trans_node_1(e2n), self.trans_node_2(cur_node)], dim=-1))
        )  # [N, H]
        return self.w_l(torch.cat([cur_node_new, prev], dim=-1))  # [N, H]
