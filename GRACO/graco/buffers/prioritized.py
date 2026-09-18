"""Prioritized n-step replay with a fully-vectorized sum-tree.

The classic PER sum-tree is a sequential CPU structure; here it is a flat 1-D
tensor binary heap descended for all ``B`` queries in lock-step (one gather per
level -> ``O(B log C)``, no python loop over the batch).  Priorities and IS
weights follow FINDER's ``nstep_replay_mem_prioritized`` exactly:
``p = (min(|δ|+ε, 1))^α``, ``w = (P(i)/min P)^{-β}`` with ``β`` annealed to 1.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from graco.buffers.base import Buffer, SampledBatch, Transition, collate
from graco.registries import BUFFERS


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


@BUFFERS.register("prioritized_replay", aliases=["per", "prioritized"])
class PrioritizedNStepReplay(Buffer):
    def __init__(
        self,
        capacity: int = 500000,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-7,
        abs_err_upper: float = 1.0,
        state_spec: Optional[Dict[str, str]] = None,
        device: str = "cpu",
    ):
        self.capacity = _next_pow2(int(capacity))
        self.depth = self.capacity.bit_length() - 1  # log2(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta0 = beta
        self.eps = eps
        self.abs_err_upper = abs_err_upper
        self.tree_device = torch.device(device)
        # 1-indexed heap: root at 1, leaves at [capacity, 2*capacity)
        self.tree = torch.zeros(2 * self.capacity, dtype=torch.float64, device=self.tree_device)
        self.data: List[Optional[Transition]] = [None] * self.capacity
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        if state_spec is not None:
            self.state_spec = state_spec

    # ---------------------------------------------------------------- writing
    def add(self, transition: Transition) -> None:
        leaf = self.pos
        self.data[leaf] = transition
        self._set_leaves(
            torch.tensor([leaf], device=self.tree_device),
            torch.tensor([self.max_priority], dtype=torch.float64, device=self.tree_device),
        )
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_many(self, transitions: List[Transition]) -> None:
        if not transitions:
            return
        leaves = torch.arange(len(transitions), device=self.tree_device)
        leaves = (leaves + self.pos) % self.capacity
        for off, t in enumerate(transitions):
            self.data[int(leaves[off])] = t
        p = torch.full(
            (len(transitions),), self.max_priority, dtype=torch.float64, device=self.tree_device
        )
        self._set_leaves(leaves, p)
        self.pos = (self.pos + len(transitions)) % self.capacity
        self.size = min(self.size + len(transitions), self.capacity)

    def _set_leaves(self, leaf_idx: Tensor, new_p: Tensor) -> None:
        # dedup leaves (keep max priority) so delta propagation stays exact
        uniq, inv = torch.unique(leaf_idx, return_inverse=True)
        agg = torch.zeros(uniq.numel(), dtype=torch.float64, device=self.tree_device)
        agg.scatter_reduce_(0, inv, new_p.to(torch.float64), reduce="amax", include_self=False)
        node = uniq + self.capacity
        old = self.tree[node].clone()
        self.tree[node] = agg
        delta = agg - old
        while int(node[0]) > 1:
            node = node // 2
            self.tree.index_add_(0, node, delta)

    # ---------------------------------------------------------------- sampling
    def anneal_beta(self, progress: float) -> None:
        self.beta = min(1.0, self.beta0 + (1.0 - self.beta0) * float(progress))

    def sample(self, batch_size: int, device, beta: Optional[float] = None) -> SampledBatch:
        if beta is not None:
            self.beta = beta
        total = self.tree[1]
        seg = total / batch_size
        i = torch.arange(batch_size, device=self.tree_device, dtype=torch.float64)
        v = seg * i + torch.rand(batch_size, device=self.tree_device, dtype=torch.float64) * seg

        node = torch.ones(batch_size, dtype=torch.long, device=self.tree_device)
        for _ in range(self.depth):
            left = 2 * node
            left_sum = self.tree[left]
            go_right = v > left_sum
            v = v - go_right.to(v.dtype) * left_sum
            node = left + go_right.long()
        leaf = node - self.capacity  # data indices
        leaf = leaf.clamp_(0, self.size - 1)

        leaf_prio = self.tree[leaf + self.capacity]
        probs = leaf_prio / total
        min_prob = (self.tree[self.capacity : self.capacity + self.size].min() / total).clamp_min(1e-12)
        weights = (probs / min_prob).pow(-self.beta)
        weights = (weights / weights.max()).to(torch.float32).to(device)

        batch = [self.data[int(j)] for j in leaf]
        graph, state, next_state, action, reward, terminal = collate(batch, self.state_spec, device)
        return SampledBatch(
            graph=graph,
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminal=terminal,
            indices=leaf.to(device),
            is_weights=weights,
        )

    def update_priorities(self, indices: Tensor, td_errors: Tensor) -> None:
        td = td_errors.detach().abs().to(self.tree_device).to(torch.float64)
        p = (td + self.eps).clamp_max(self.abs_err_upper).pow(self.alpha)
        self.max_priority = max(self.max_priority, float(p.max()))
        self._set_leaves(indices.to(self.tree_device).long(), p)

    def __len__(self) -> int:
        return self.size
