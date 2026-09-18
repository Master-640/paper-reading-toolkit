"""Implicit Quantile Networks (IQN).

IQN represents the return distribution implicitly: instead of a fixed set of
quantiles (QR-DQN) it maps *sampled* quantile fractions ``τ ~ U(0, 1)`` to
quantile values through a cosine embedding of ``τ`` fused (Hadamard product)
into the per-node state-action features.  The learning step samples ``N`` online
and ``N'`` target fractions and minimizes the quantile-Huber loss over the
pairwise TD matrix, exactly like QR-DQN but with sampled ``τ``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.algos.dqn import DQNAlgorithm
from graco.data.batch import BatchedGraph
from graco.models.heads import _NodeScoreHead
from graco.registries import ALGOS, HEADS
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import segment_argmax


@HEADS.register("implicit_quantile", aliases=["iqn"])
class ImplicitQuantileHead(_NodeScoreHead):
    """IQN head: quantile values ``[N, N_tau]`` for sampled fractions ``τ [N_tau]``.

    The state-action features ``ψ(s, a)`` (the usual node embedding + broadcast
    graph context) are modulated by a cosine embedding ``φ(τ)`` of the sampled
    quantile fractions before the scoring MLP: ``q(s, a, τ) = mlp(ψ ⊙ φ(τ))``.
    ``τ`` is shared across nodes within a forward pass (one set of fractions per
    batch), which keeps the whole thing vectorized.
    """

    def __init__(self, embed_dim: int, num_cos: int = 64, **kw):
        super().__init__(embed_dim, out_dim=1, **kw)
        self.num_cos = int(num_cos)
        feat_dim = self.mlp.layers[0].in_features  # dim of ψ(s, a)
        self.cos_embed = nn.Linear(self.num_cos, feat_dim)

    def forward(
        self,
        h: Tensor,
        graph: BatchedGraph,
        aux: Optional[Tensor] = None,
        taus: Optional[Tensor] = None,
    ) -> Tensor:
        psi = self.node_input(h, graph, aux)  # [N, D]
        if taus is None:
            taus = torch.rand(8, device=psi.device)
        taus = taus.reshape(-1)  # [T]
        idx = torch.arange(1, self.num_cos + 1, device=psi.device, dtype=psi.dtype)  # [num_cos]
        cos = torch.cos(math.pi * taus.unsqueeze(-1) * idx)  # [T, num_cos]
        phi = F.relu(self.cos_embed(cos))  # [T, D]
        combined = psi.unsqueeze(1) * phi.unsqueeze(0)  # [N, T, D]
        return self.mlp(combined).squeeze(-1)  # [N, T]


@ALGOS.register("iqn", aliases=["implicit_quantile_dqn"])
class IQNAlgorithm(DQNAlgorithm):
    """Implicit Quantile Network agent."""

    def __init__(
        self,
        env,
        num_tau: int = 8,
        num_tau_prime: int = 8,
        num_cos: int = 64,
        kappa: float = 1.0,
        model=None,
        **kw,
    ):
        model = dict(model or {})
        model["head"] = dict(model.get("head") or {})
        model["head"].setdefault("type", "implicit_quantile")
        model["head"]["num_cos"] = num_cos
        self.num_tau = int(num_tau)
        self.num_tau_prime = int(num_tau_prime)
        self.kappa = float(kappa)
        super().__init__(env, model=model, **kw)

    # --------------------------------------------------------------- scoring
    def _quantiles(self, policy, graph, taus: Tensor) -> Tensor:
        h = policy.encoder(graph)
        aux = graph.meta.get("aux_feat") if graph.meta else None
        return policy.head(h, graph, aux=aux, taus=taus)  # [N, T]

    def _node_scores(self, policy, graph, mask) -> Tensor:
        # mean quantile value over K sampled τ -> scalar per node (acting/argmax)
        taus = torch.rand(self.num_tau, device=self.device)
        return self._quantiles(policy, graph, taus).mean(dim=-1)  # [N]

    # ------------------------------------------------------------- learning
    def _learn(self) -> Dict[str, float]:
        self.policy.train()
        if hasattr(self.buffer, "anneal_beta"):
            self.buffer.anneal_beta(self.progress)
        batch = self.buffer.sample(self.batch_size, self.device)
        graph, state, next_state = batch.graph, batch.state, batch.next_state
        b = graph.num_graphs

        obs_t = self.env_class.obs_graph(graph, state)
        taus = torch.rand(self.num_tau, device=self.device)  # [N_tau] online
        theta_all = self._quantiles(self.policy, obs_t, taus)  # [N, N_tau]
        theta_sa = theta_all.index_select(0, batch.action)  # [K, N_tau]

        with torch.no_grad():
            obs_tn = self.env_class.obs_graph(graph, next_state)
            mask_tn = self.env_class.valid_mask(graph, next_state)
            has_valid = scatter_sum(mask_tn.float(), graph.batch, b) > 0
            # greedy next action by mean quantile (double if enabled)
            scorer = self.policy if self.double else self.target
            q_next_mean = self._node_scores(scorer, obs_tn, mask_tn)  # [N]
            a_star = segment_argmax(q_next_mean, graph.batch, b, mask_tn).clamp_min(0)  # [K]
            taus_prime = torch.rand(self.num_tau_prime, device=self.device)  # [N_tau']
            theta_next = self._quantiles(self.target, obs_tn, taus_prime).index_select(0, a_star)
            theta_next = theta_next * has_valid.float().unsqueeze(-1)  # [K, N_tau']
            disc = self.gamma ** self.n_step
            target = (
                batch.reward.unsqueeze(-1)
                + disc * (~batch.terminal).float().unsqueeze(-1) * theta_next
            )  # [K, N_tau']

        # quantile-Huber loss over the (N_tau x N_tau') pairwise TD matrix
        tau_col = taus.view(1, -1, 1)  # [1, N_tau, 1]
        td = target.unsqueeze(1) - theta_sa.unsqueeze(2)  # [K, N_tau(i), N_tau'(j)]
        huber = torch.where(
            td.abs() <= self.kappa,
            0.5 * td.pow(2),
            self.kappa * (td.abs() - 0.5 * self.kappa),
        )
        rho = (tau_col - (td.detach() < 0).float()).abs() * huber / self.kappa
        loss_per = rho.sum(dim=2).mean(dim=1)  # [K]
        weights = batch.is_weights if batch.is_weights is not None else 1.0
        loss = (weights * loss_per).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        if batch.indices is not None:
            self.buffer.update_priorities(batch.indices, loss_per.detach())
        self._learn_steps += 1
        if self._learn_steps % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return {"loss": float(loss.detach()), "epsilon": self.epsilon()}
