"""VNS encoder — "Vital Node Searcher" (Du et al., Connection Science 2022).

VNS finds critical nodes for dismantling by (1) a graph-embedding module, (2) a
**Long-Short-Term (LSTM) module** that "fully exploits the historical information
contained in the sequence data", and (3) a dueling-Q module — trained with DQN on
the ANC objective.  In GRACO the dueling-Q module is the existing ``dueling_q``
head and the DQN loop is the ``dqn`` algo, so this encoder supplies exactly the
first two pieces:

* the struct2vec message-passing core (reused from :class:`Structure2Vec`), which
  produces a **sequence** of node embeddings ``μ^(1), …, μ^(T)`` over
  ``T = num_layers`` propagation rounds, and
* an ``nn.LSTM`` run over that per-node round sequence, whose final hidden state
  is the node embedding — the "historical information in the sequence data" VNS
  refers to being the successive embedding iterations.

Pair with ``head: dueling_q`` + ``algo: dqn`` (see ``graco/configs/vital_node_searcher.yaml``)
to reproduce the full Vital Node Searcher.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.encoders.s2v import Structure2Vec
from graco.registries import ENCODERS


@ENCODERS.register("vns", aliases=["vital_node_searcher"])
class VNSEncoder(Structure2Vec):
    """Struct2vec message passing + an LSTM over the propagation-round sequence."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, **kwargs)
        # Long-Short-Term module over the sequence of per-round node embeddings.
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)

    def forward(self, graph: BatchedGraph) -> Tensor:
        src, dst, n, edge_init, cur_node = self._init_propagation(graph)
        seq = []  # per-round node embeddings μ^(t)
        for _ in range(self.num_layers):
            cur_node = self._round(cur_node, edge_init, src, dst, n)
            seq.append(cur_node)
        # LSTM over the round dimension: input [N, T, H] (N as batch, T as time)
        out, _ = self.lstm(torch.stack(seq, dim=1))
        return out[:, -1, :]  # final hidden state per node -> [N, H]
