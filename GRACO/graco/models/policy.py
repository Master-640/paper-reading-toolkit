"""Policies: compose an encoder with output head(s).

Policies are deliberately thin — they run the encoder once and apply head(s).
*Action selection* (epsilon-greedy, sampling, masking) and *learning* live in the
algorithms, which use the segmented ops in :mod:`graco.utils.segment_ops`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder
from graco.utils import accel


class QPolicy(nn.Module):
    """Encoder + a node-scoring Q head (plain / dueling / distributional / quantile)."""

    def __init__(self, encoder: GNNEncoder, head: nn.Module, needs_mask: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.needs_mask = needs_mask  # dueling head needs the valid mask
        self._compiled_enc: list = []  # opt-in torch.compile alias (not a registered submodule)

    def _encode(self, graph: BatchedGraph) -> Tensor:
        # compiled-encoder alias kept out of state_dict / deepcopy so target-net sync
        # and checkpoints stay clean; built lazily, falls back to eager on failure.
        if accel.compile_enabled():
            if not self._compiled_enc:
                self._compiled_enc.append(accel.maybe_compile(self.encoder, enabled=True))
            return self._compiled_enc[0](graph)
        return self.encoder(graph)

    def forward(self, graph: BatchedGraph, mask: Optional[Tensor] = None) -> Tensor:
        h = self._encode(graph)
        aux = graph.meta.get("aux_feat") if graph.meta else None
        if self.needs_mask:
            return self.head(h, graph, aux=aux, mask=mask)
        return self.head(h, graph, aux=aux)


class ActorCriticPolicy(nn.Module):
    """Shared-encoder actor + critic (critic can use a separate encoder)."""

    def __init__(
        self,
        encoder: GNNEncoder,
        actor_head: nn.Module,
        critic_head: nn.Module,
        critic_encoder: Optional[GNNEncoder] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.critic_encoder = critic_encoder  # None -> share encoder

    def forward(self, graph: BatchedGraph) -> Tuple[Tensor, Tensor]:
        h = self.encoder(graph)
        aux = graph.meta.get("aux_feat") if graph.meta else None
        logits = self.actor_head(h, graph, aux=aux)  # [N]
        hc = h if self.critic_encoder is None else self.critic_encoder(graph)
        value = self.critic_head(hc, graph, aux=aux)  # [B]
        return logits, value

    def logits(self, graph: BatchedGraph) -> Tensor:
        h = self.encoder(graph)
        aux = graph.meta.get("aux_feat") if graph.meta else None
        return self.actor_head(h, graph, aux=aux)

    def value(self, graph: BatchedGraph) -> Tensor:
        hc = self.encoder(graph) if self.critic_encoder is None else self.critic_encoder(graph)
        aux = graph.meta.get("aux_feat") if graph.meta else None
        return self.critic_head(hc, graph, aux=aux)
