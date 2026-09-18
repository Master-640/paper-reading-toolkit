"""Output heads.

Heads turn per-node embeddings ``h [N, H]`` (and a pooled graph context) into
task outputs: per-node Q-values, dueling Q, distributional/quantile Q, actor
logits, and a per-graph critic value.  All heads that score nodes optionally
concatenate a **broadcast graph embedding** (the ``y_potential`` context used by
both DIRAC and FINDER), so ``Q(s, a)`` depends on the whole state, not just the
node.  Graph-level ``aux`` features (e.g. FINDER's 4-dim readout) may be appended.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import MLP
from graco.registries import HEADS


class _NodeScoreHead(nn.Module):
    """Shared machinery for heads that emit a per-node score vector."""

    def __init__(
        self,
        embed_dim: int,
        out_dim: int = 1,
        hidden_dim: Optional[int] = None,
        use_graph_context: bool = True,
        pool: str = "sum",
        aux_dim: int = 0,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.use_graph_context = use_graph_context
        self.pool = pool
        self.aux_dim = aux_dim
        in_dim = embed_dim * (2 if use_graph_context else 1) + aux_dim
        hidden_dim = hidden_dim or embed_dim
        self.mlp = MLP(in_dim, hidden_dim, out_dim, num_layers=num_layers)

    def node_input(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor]) -> Tensor:
        parts = [h]
        if self.use_graph_context:
            g = graph.pool(h, reduce=self.pool)  # [B, H]
            parts.append(graph.broadcast_to_nodes(g))
        if self.aux_dim and aux is not None:
            parts.append(graph.broadcast_to_nodes(aux))
        return torch.cat(parts, dim=-1)


@HEADS.register("q_node")
class QHead(_NodeScoreHead):
    """Scalar Q-value per node ``[N]``."""

    def __init__(self, embed_dim: int, **kw):
        super().__init__(embed_dim, out_dim=1, **kw)

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        return self.mlp(self.node_input(h, graph, aux)).squeeze(-1)


@HEADS.register("dueling_q")
class DuelingQHead(nn.Module):
    """Dueling Q: ``Q_i = V(graph) + A_i - mean_{valid} A``."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        pool: str = "sum",
        aux_dim: int = 0,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.pool = pool
        self.aux_dim = aux_dim
        hidden_dim = hidden_dim or embed_dim
        self.value = MLP(embed_dim + aux_dim, hidden_dim, 1, num_layers=num_layers)
        self.adv = MLP(embed_dim * 2 + aux_dim, hidden_dim, 1, num_layers=num_layers)

    def forward(
        self,
        h: Tensor,
        graph: BatchedGraph,
        aux: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        from graco.utils.scatter import scatter_mean, scatter_sum

        g = graph.pool(h, reduce=self.pool)  # [B, H]
        v_in = g if not self.aux_dim or aux is None else torch.cat([g, aux], -1)
        v = self.value(v_in).squeeze(-1)  # [B]
        a_parts = [h, graph.broadcast_to_nodes(g)]
        if self.aux_dim and aux is not None:
            a_parts.append(graph.broadcast_to_nodes(aux))
        adv = self.adv(torch.cat(a_parts, -1)).squeeze(-1)  # [N]
        # mean advantage over VALID nodes only
        if mask is not None:
            m = mask.to(adv.dtype)
            denom = scatter_sum(m, graph.batch, graph.num_graphs).clamp_min(1.0)
            adv_mean = scatter_sum(adv * m, graph.batch, graph.num_graphs) / denom
        else:
            adv_mean = scatter_mean(adv, graph.batch, graph.num_graphs)
        return graph.broadcast_to_nodes(v) + adv - graph.broadcast_to_nodes(adv_mean)


@HEADS.register("distributional_q")
class DistributionalQHead(_NodeScoreHead):
    """C51: categorical distribution over ``num_atoms`` per node ``[N, atoms]``."""

    def __init__(self, embed_dim: int, num_atoms: int = 51, **kw):
        super().__init__(embed_dim, out_dim=num_atoms, **kw)
        self.num_atoms = num_atoms

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        logits = self.mlp(self.node_input(h, graph, aux))  # [N, atoms]
        return logits.log_softmax(dim=-1)  # log-probabilities


@HEADS.register("quantile_q")
class QuantileQHead(_NodeScoreHead):
    """QR-DQN: ``num_quantiles`` quantile values per node ``[N, quantiles]``."""

    def __init__(self, embed_dim: int, num_quantiles: int = 51, **kw):
        super().__init__(embed_dim, out_dim=num_quantiles, **kw)
        self.num_quantiles = num_quantiles

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        return self.mlp(self.node_input(h, graph, aux))  # [N, quantiles]


@HEADS.register("actor")
class ActorHead(_NodeScoreHead):
    """Per-node policy logits ``[N]``."""

    def __init__(self, embed_dim: int, **kw):
        super().__init__(embed_dim, out_dim=1, **kw)

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        return self.mlp(self.node_input(h, graph, aux)).squeeze(-1)


@HEADS.register("q_label")
class QLabelHead(_NodeScoreHead):
    """Per-node **per-label** Q-values ``[N, k]`` for k-label action heads.

    Used by the ``label_dqn`` algorithm, which flattens the ``[N, k]`` scores to a
    joint ``(node, color)`` action space of size ``N·k`` per graph. ``needs_mask``
    stays False (masking is applied on the flattened scores by the algorithm).
    """

    def __init__(self, embed_dim: int, num_labels: int = 3, **kw):
        super().__init__(embed_dim, out_dim=int(num_labels), **kw)
        self.num_labels = int(num_labels)

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        return self.mlp(self.node_input(h, graph, aux))  # [N, k]


@HEADS.register("critic")
class CriticHead(nn.Module):
    """Per-graph state value ``[B]`` from a pooled graph embedding."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        pool: str = "mean",
        aux_dim: int = 0,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.pool = pool
        self.aux_dim = aux_dim
        hidden_dim = hidden_dim or embed_dim
        self.mlp = MLP(embed_dim + aux_dim, hidden_dim, 1, num_layers=num_layers)

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        g = graph.pool(h, reduce=self.pool)
        if self.aux_dim and aux is not None:
            g = torch.cat([g, aux], -1)
        return self.mlp(g).squeeze(-1)
