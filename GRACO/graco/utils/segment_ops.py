"""Per-graph (segmented) selection ops used by policies.

A policy produces one score per node in the flat disjoint-union layout ``[N]``;
choosing an action means picking exactly **one node per graph**.  These helpers
do masked per-graph argmax and categorical sampling fully vectorized, returning
**global** node ids (already offset), which feed straight into env scatter.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graco.utils.scatter import scatter_logsumexp, scatter_max, scatter_min

NEG_INF = -1e9  # masking constant (not -inf: keeps softmax/gradients finite)


def apply_mask(scores: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return scores
    return scores.masked_fill(~mask, NEG_INF)


def segment_argmax(
    scores: Tensor, batch: Tensor, num_graphs: int, mask: Optional[Tensor] = None
) -> Tensor:
    """Return the global node id of the max-scoring node per graph ``[B]``.

    Graphs with no valid node (all masked / empty) return ``-1`` (a no-op action).
    """
    from graco.utils.scatter import scatter_sum

    scores = apply_mask(scores, mask)
    seg_max = scatter_max(scores, batch, num_graphs, fill_value=NEG_INF)  # [B]
    gathered = seg_max.index_select(0, batch)  # [N]
    is_max = scores >= gathered  # ties -> pick smallest index below
    n = scores.shape[0]
    idx = torch.arange(n, device=scores.device)
    cand = torch.where(is_max, idx, torch.full_like(idx, n))
    first = scatter_min(cand, batch, num_graphs, fill_value=float(n)).long()  # [B]
    valid = first < n
    if mask is not None:
        has_valid = scatter_sum(mask.to(scores.dtype), batch, num_graphs) > 0
        valid = valid & has_valid
    return torch.where(valid, first, torch.full_like(first, -1))


def segment_log_softmax(logits: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
    """Log-softmax within each graph ``[N]`` (numerically stable)."""
    lse = scatter_logsumexp(logits, batch, num_graphs)  # [B]
    return logits - lse.index_select(0, batch)


def segment_gumbel_sample(
    logits: Tensor,
    batch: Tensor,
    num_graphs: int,
    mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample one node per graph ~ Categorical(softmax(logits over valid)).

    Uses the Gumbel-max trick so sampling is a masked segment-argmax — fully
    vectorized. Returns global node ids ``[B]`` (``-1`` if a graph has no valid
    node).
    """
    logits = apply_mask(logits, mask)
    u = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-12, 1.0)
    gumbel = -torch.log(-torch.log(u))
    return segment_argmax(logits + gumbel, batch, num_graphs, mask=mask)


def gather_action_logprob(
    logits: Tensor,
    batch: Tensor,
    num_graphs: int,
    action: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Log-prob of chosen ``action`` (global ids ``[B]``) under masked softmax."""
    logits = apply_mask(logits, mask)
    log_probs = segment_log_softmax(logits, batch, num_graphs)  # [N]
    valid = action >= 0
    safe_action = torch.where(valid, action, torch.zeros_like(action))
    lp = log_probs.index_select(0, safe_action)
    return torch.where(valid, lp, torch.zeros_like(lp))


def segment_entropy(
    logits: Tensor, batch: Tensor, num_graphs: int, mask: Optional[Tensor] = None
) -> Tensor:
    """Entropy of the per-graph categorical over valid nodes ``[B]``."""
    logits = apply_mask(logits, mask)
    log_probs = segment_log_softmax(logits, batch, num_graphs)  # [N]
    probs = log_probs.exp()
    from graco.utils.scatter import scatter_sum

    ent_contrib = -(probs * log_probs)
    if mask is not None:
        ent_contrib = ent_contrib * mask.to(ent_contrib.dtype)
    return scatter_sum(ent_contrib, batch, num_graphs)
