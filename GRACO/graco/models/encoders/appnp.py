"""APPNP: Approximate Personalized Propagation of Neural Predictions
(Klicpera et al., 2019).

Predict-then-propagate.  A small MLP produces per-node predictions ``H0``, which
are then diffused by ``K`` power-iteration steps of *personalized PageRank* on
the renormalized adjacency, teleporting back to ``H0`` with probability
``alpha``::

    P     = D~^{-1/2} A~ D~^{-1/2}            (renormalized adjacency, A~ = A + I)
    H0    = MLP(x)
    H     = (1 - alpha) P H + alpha H0        (repeated K times)

``K`` defaults to ``num_layers`` and the teleport probability ``alpha`` to 0.1.
Propagation carries no learnable weights, so depth is decoupled from width.
Aggregation is scatter-based over ``edge_index`` (src=[0], dst=[1]).

Reference: J. Klicpera, A. Bojchevski, S. Guennemann, "Predict then Propagate:
Graph Neural Networks meet Personalized PageRank" (ICLR 2019).
"""

from __future__ import annotations

from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP, GNNEncoder, gcn_norm
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_sum


@ENCODERS.register("appnp")
class APPNPEncoder(GNNEncoder):
    """Predict-then-propagate encoder; embeddings ``[N, hidden_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        alpha: float = 0.1,
        act: str = "relu",
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, act=act, **kwargs)
        self.out_dim = self.hidden_dim
        self.alpha = float(alpha)
        self.K = self.num_layers  # number of power-iteration (propagation) steps
        self.mlp = MLP(
            self.in_dim, self.hidden_dim, self.hidden_dim,
            num_layers=2, act=act, dropout=self.dropout, norm=self.norm_kind,
        )

    def forward(self, graph: BatchedGraph) -> Tensor:
        x = self.input_x(graph)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        c, self_coef = gcn_norm(graph)

        h0 = self.mlp(x)  # per-node predictions [N, H]
        h = h0
        for _ in range(self.K):
            msg = h.index_select(0, src) * c.unsqueeze(-1)
            ph = scatter_sum(msg, dst, graph.num_nodes) + self_coef.unsqueeze(-1) * h
            h = (1.0 - self.alpha) * ph + self.alpha * h0
        return h
