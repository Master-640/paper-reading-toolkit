"""Munchausen-DQN (M-DQN).

Munchausen RL augments the DQN target with a scaled log-policy bonus and turns
the bootstrap into a soft (maximum-entropy) value.  It only touches the target
computation, so it subclasses :class:`DQNAlgorithm` and overrides ``_learn``.
The implicit stochastic policy is ``π = softmax(Q / τ_m)`` taken **per graph**
over the valid nodes; both the Munchausen bonus and the soft next-state value
use the target network (following Vieillard et al., 2020).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from graco.algos.dqn import DQNAlgorithm
from graco.registries import ALGOS
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import apply_mask, segment_log_softmax


@ALGOS.register("munchausen_dqn", aliases=["m_dqn", "mdqn"])
class MunchausenDQNAlgorithm(DQNAlgorithm):
    """Munchausen-DQN with a scalar Q head."""

    def __init__(
        self,
        env,
        alpha: float = 0.9,
        tau_m: float = 0.03,
        l0: float = -1.0,
        model=None,
        **kw,
    ):
        self.alpha = float(alpha)
        self.tau_m = float(tau_m)
        self.m_l0 = float(l0)
        super().__init__(env, model=model, **kw)

    def _learn(self) -> Dict[str, float]:
        self.policy.train()
        if hasattr(self.buffer, "anneal_beta"):
            self.buffer.anneal_beta(self.progress)
        batch = self.buffer.sample(self.batch_size, self.device)
        graph, state, next_state = batch.graph, batch.state, batch.next_state
        b = graph.num_graphs

        obs_t = self.env_class.obs_graph(graph, state)
        mask_t = self.env_class.valid_mask(graph, state)
        q_all = self._node_scores(self.policy, obs_t, mask_t)  # [N]
        qa = q_all.index_select(0, batch.action)  # [K]

        with torch.no_grad():
            # --- Munchausen bonus at s: α τ_m clamp(log π_target(a|s), l0, 0) ---
            q_t_s = self._node_scores(self.target, obs_t, mask_t)  # [N]
            log_pi_s = segment_log_softmax(apply_mask(q_t_s / self.tau_m, mask_t), graph.batch, b)
            log_pi_a = log_pi_s.index_select(0, batch.action)  # [K]
            munchausen = self.alpha * self.tau_m * log_pi_a.clamp(self.m_l0, 0.0)  # [K]

            # --- soft next value: Σ_a' π(a'|s') (Q_target(s',a') - τ_m log π(a'|s')) ---
            obs_tn = self.env_class.obs_graph(graph, next_state)
            mask_tn = self.env_class.valid_mask(graph, next_state)
            has_valid = scatter_sum(mask_tn.float(), graph.batch, b) > 0
            q_t_n = self._node_scores(self.target, obs_tn, mask_tn)  # [N]
            log_pi_n = segment_log_softmax(apply_mask(q_t_n / self.tau_m, mask_tn), graph.batch, b)
            pi_n = log_pi_n.exp()
            contrib = pi_n * (q_t_n - self.tau_m * log_pi_n) * mask_tn.to(q_t_n.dtype)  # [N]
            v_next = scatter_sum(contrib, graph.batch, b)  # [B]
            v_next = torch.where(has_valid, v_next, torch.zeros_like(v_next))

            disc = self.gamma ** self.n_step
            bootstrap = disc * (~batch.terminal).float() * v_next
            target_q = batch.reward + munchausen + bootstrap  # [K]

        td = target_q - qa
        if self.huber:
            per = F.smooth_l1_loss(qa, target_q, reduction="none")
        else:
            per = td.pow(2)
        weights = batch.is_weights if batch.is_weights is not None else 1.0
        loss = (weights * per).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        if batch.indices is not None:
            self.buffer.update_priorities(batch.indices, td)
        self._learn_steps += 1
        if self._learn_steps % self.target_update_period == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return {
            "loss": float(loss.detach()),
            "q_mean": float(qa.mean().detach()),
            "epsilon": self.epsilon(),
            "target_mean": float(target_q.mean().detach()),
        }
