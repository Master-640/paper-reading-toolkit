"""Categorical DQN (C51).

C51 models the return distribution as a categorical over a fixed support of
``num_atoms`` atoms spanning ``[v_min, v_max]``.  It reuses the whole DQN
lifecycle (exploration, replay, n-step returns, target network) and only
overrides per-node scoring (expected value ``Σ z_k p_k``) and the learning step
(the categorical projection ``Φ`` of the Bellman target followed by a
cross-entropy loss).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from graco.algos.dqn import DQNAlgorithm
from graco.registries import ALGOS
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import segment_argmax


@ALGOS.register("c51", aliases=["categorical_dqn"])
class C51Algorithm(DQNAlgorithm):
    """Categorical DQN: distributional value with a fixed atom support."""

    def __init__(
        self,
        env,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        model=None,
        **kw,
    ):
        model = dict(model or {})
        model["head"] = dict(model.get("head") or {})
        model["head"].setdefault("type", "distributional_q")
        model["head"]["num_atoms"] = num_atoms
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        super().__init__(env, model=model, **kw)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_atoms, device=self.device
        )  # [atoms]

    # --------------------------------------------------------------- scoring
    def _node_scores(self, policy, graph, mask) -> Tensor:
        # expected value E[Z] = Σ z_k p_k -> scalar per node (used for acting/argmax)
        log_p = policy(graph)  # [N, atoms] log-probabilities
        return (log_p.exp() * self.support).sum(dim=-1)  # [N]

    # ------------------------------------------------------------- learning
    def _learn(self) -> Dict[str, float]:
        self.policy.train()
        if hasattr(self.buffer, "anneal_beta"):
            self.buffer.anneal_beta(self.progress)
        batch = self.buffer.sample(self.batch_size, self.device)
        graph, state, next_state = batch.graph, batch.state, batch.next_state
        b = graph.num_graphs

        obs_t = self.env_class.obs_graph(graph, state)
        log_p_all = self.policy(obs_t)  # [N, atoms]
        log_p_sa = log_p_all.index_select(0, batch.action)  # [K, atoms]

        with torch.no_grad():
            obs_tn = self.env_class.obs_graph(graph, next_state)
            mask_tn = self.env_class.valid_mask(graph, next_state)
            has_valid = scatter_sum(mask_tn.float(), graph.batch, b) > 0
            # greedy next action by expected value (double if enabled)
            scorer = self.policy if self.double else self.target
            q_next_mean = self._node_scores(scorer, obs_tn, mask_tn)  # [N]
            a_star = segment_argmax(q_next_mean, graph.batch, b, mask_tn).clamp_min(0)  # [K]
            p_next = self.target(obs_tn).exp().index_select(0, a_star)  # [K, atoms]

            # Bellman target support: Tz = clip(r + γ^n (1-term) z, v_min, v_max).
            # A dead/terminal next state collapses to a delta at r (γ_eff = 0), and
            # since Σ p_next = 1 the projection then places all mass on that atom.
            disc = self.gamma ** self.n_step
            gamma_eff = disc * (~batch.terminal).float() * has_valid.float()  # [K]
            tz = batch.reward.unsqueeze(-1) + gamma_eff.unsqueeze(-1) * self.support.view(1, -1)
            tz = tz.clamp(self.v_min, self.v_max)  # [K, atoms]
            b_idx = (tz - self.v_min) / self.delta_z  # [K, atoms] in [0, atoms-1]
            lower = b_idx.floor().long()
            upper = b_idx.ceil().long()
            # fix vanishing mass when l == b == u (b is an integer)
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < self.num_atoms - 1) & (lower == upper)] += 1

            m = torch.zeros_like(p_next)  # [K, atoms]
            m.scatter_add_(1, lower, p_next * (upper.float() - b_idx))
            m.scatter_add_(1, upper, p_next * (b_idx - lower.float()))

        # cross-entropy of projected target vs online log-probs
        ce = -(m * log_p_sa).sum(dim=-1)  # [K]
        weights = batch.is_weights if batch.is_weights is not None else 1.0
        loss = (weights * ce).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        if batch.indices is not None:
            self.buffer.update_priorities(batch.indices, ce.detach())
        self._learn_steps += 1
        if self._learn_steps % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return {"loss": float(loss.detach()), "epsilon": self.epsilon()}
