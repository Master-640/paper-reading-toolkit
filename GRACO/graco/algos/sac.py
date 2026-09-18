"""Discrete Soft Actor-Critic (SAC) for graph construction MDPs.

An off-policy maximum-entropy actor-critic.  Collection follows the exact
DIRAC/FINDER-style lifecycle used by :class:`~graco.algos.dqn.DQNAlgorithm`:
per-graph topologies are snapshotted at episode start, local ``(action, reward)``
sequences are recorded per active env, and n-step transitions are built at
episode end and pushed into a :class:`~graco.buffers.replay.UniformNStepReplay`.

The *discrete* SAC updates use the **expectation form** of the soft targets:
every per-graph quantity (soft value, actor loss, entropy) is summed over the
*valid* actions of the graph rather than a single sampled action, so no extra
sampling is needed at learning time.  Twin critics ``q1``/``q2`` (with Polyak
targets) give the clipped double-Q min, and the temperature ``alpha`` is
auto-tuned against a target entropy scaled to each batch's mean action count.
All per-graph reductions are masked segment ops over the flat ``[N]`` node axis.
"""

from __future__ import annotations

import copy
import itertools
import math
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from graco.algos.base import Algorithm
from graco.buffers.base import to_single_graphs
from graco.buffers.replay import UniformNStepReplay, make_nstep_transitions
from graco.envs.base import Observation, StepResult, VectorizedEnv
from graco.models.build import build_encoder, build_head
from graco.registries import ALGOS, BUFFERS
from graco.utils.accel import build_optimizer
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import (
    apply_mask,
    segment_argmax,
    segment_gumbel_sample,
    segment_log_softmax,
)


@ALGOS.register("sac", aliases=["discrete_sac"])
class SACAlgorithm(Algorithm):
    """Discrete Soft Actor-Critic with twin critics and auto-tuned temperature.

    Networks: an actor encoder + per-node ``actor`` head (logits ``[N]``) and two
    ``q_node`` critics ``q1``/``q2`` (scalar ``[N]`` each).  By default the critics
    use their **own** encoder, separate from the actor's, for stability; set
    ``shared_encoder=False`` to make the critics reuse the actor's encoder.
    """

    learns_online = True

    def __init__(
        self,
        env: VectorizedEnv,
        model: Optional[dict] = None,
        buffer: Optional[Any] = None,
        device: Any = "cpu",
        gamma: float = 0.99,
        n_step: int = 1,
        batch_size: int = 64,
        lr: float = 3e-4,
        alpha_init: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy_ratio: float = 0.7,
        tau: float = 0.005,
        shared_encoder: bool = True,
        learn_start: int = 1000,
        learn_freq: int = 1,
        updates_per_step: int = 1,
        grad_clip: Optional[float] = 10.0,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(env, device=device, gamma=gamma)
        if kwargs:
            import warnings

            warnings.warn(f"{type(self).__name__} ignoring config keys: {sorted(kwargs)}")
        self.env_class = type(env)
        self.state_spec = env.state_spec
        self.n_step = n_step
        self.batch_size = batch_size
        self.autotune_alpha = autotune_alpha
        self.target_entropy_ratio = target_entropy_ratio
        self.tau = tau
        self.shared_encoder = shared_encoder
        self.learn_start = learn_start
        self.learn_freq = learn_freq
        self.updates_per_step = updates_per_step
        self.grad_clip = grad_clip

        model = dict(model or {})
        enc_cfg = model.get("encoder")
        in_dim, edge_dim = env.node_feature_dim, env.edge_feature_dim
        # actor: encoder + per-node logits head
        self.actor_encoder = build_encoder(enc_cfg, in_dim, edge_dim).to(self.device)
        self.actor_head = build_head(
            model.get("actor"), self.actor_encoder.out_dim, default_type="actor"
        ).to(self.device)
        # critics: (shared) separate encoder + two independent q_node heads
        if shared_encoder:
            self.critic_encoder = build_encoder(enc_cfg, in_dim, edge_dim).to(self.device)
        else:
            self.critic_encoder = self.actor_encoder
        q_cfg = model.get("critic") or model.get("q")
        self.q1 = build_head(q_cfg, self.critic_encoder.out_dim, default_type="q_node").to(
            self.device
        )
        self.q2 = build_head(q_cfg, self.critic_encoder.out_dim, default_type="q_node").to(
            self.device
        )
        # Polyak targets of the critic encoder + twin critics
        self.tgt_critic_encoder = copy.deepcopy(self.critic_encoder).to(self.device)
        self.tgt_q1 = copy.deepcopy(self.q1).to(self.device)
        self.tgt_q2 = copy.deepcopy(self.q2).to(self.device)
        for m in (self.tgt_critic_encoder, self.tgt_q1, self.tgt_q2):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

        # optimizers (encoder trained by the actor optimizer when shared)
        self._actor_params = list(self.actor_encoder.parameters()) + list(
            self.actor_head.parameters()
        )
        critic_modules = [self.q1, self.q2]
        if self.critic_encoder is not self.actor_encoder:
            critic_modules.insert(0, self.critic_encoder)
        self._critic_params = [p for m in critic_modules for p in m.parameters()]
        self.actor_optimizer = build_optimizer(
            self._actor_params, lr=lr, weight_decay=weight_decay
        )
        self.critic_optimizer = build_optimizer(
            self._critic_params, lr=lr, weight_decay=weight_decay
        )

        # temperature: log_alpha (auto-tuned) or a fixed constant
        log_alpha0 = math.log(max(alpha_init, 1e-8))
        if autotune_alpha:
            self.log_alpha = torch.tensor(log_alpha0, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(log_alpha0, device=self.device)
            self.alpha_optimizer = None

        if buffer is None:
            buffer = UniformNStepReplay(capacity=50000)
        elif isinstance(buffer, dict):
            buffer = BUFFERS.build(dict(buffer))
        self.buffer = buffer
        self.buffer.state_spec = self.state_spec

        self._learn_steps = 0
        self._env_steps = 0

    # ------------------------------------------------------------- lifecycle
    def on_episode_start(self, env: VectorizedEnv) -> None:
        self._graphs = to_single_graphs(env.graph)
        self._ptr = env.graph.ptr.detach().cpu()
        b = env.num_envs
        self._traj_actions = [[] for _ in range(b)]
        self._traj_rewards = [[] for _ in range(b)]

    # --------------------------------------------------------------- scoring
    def _actor_logits(self, graph) -> Tensor:
        aux = graph.meta.get("aux_feat") if graph.meta else None
        return self.actor_head(self.actor_encoder(graph), graph, aux=aux)

    def _q_values(self, encoder, q1_head, q2_head, graph):
        aux = graph.meta.get("aux_feat") if graph.meta else None
        h = encoder(graph)
        return q1_head(h, graph, aux=aux), q2_head(h, graph, aux=aux)

    @torch.no_grad()
    def act(self, obs: Observation, explore: bool = True) -> Tensor:
        self.actor_encoder.eval()
        self.actor_head.eval()
        g = obs.graph
        logits = self._actor_logits(g)  # [N]
        if explore:
            return segment_gumbel_sample(logits, g.batch, g.num_graphs, obs.action_mask)
        return segment_argmax(logits, g.batch, g.num_graphs, obs.action_mask)

    def observe(
        self, env: VectorizedEnv, obs: Observation, action: Tensor, step: StepResult
    ) -> None:
        active = (~obs.done) & (action >= 0)
        act_cpu = action.detach().cpu()
        rew_cpu = step.reward.detach().cpu()
        ptr = self._ptr
        for i in torch.nonzero(active.cpu(), as_tuple=False).flatten().tolist():
            self._traj_actions[i].append(int(act_cpu[i] - ptr[i]))
            self._traj_rewards[i].append(float(rew_cpu[i]))

    def after_step(self) -> Optional[Dict[str, float]]:
        self._env_steps += 1
        if len(self.buffer) < self.learn_start or self._env_steps % self.learn_freq != 0:
            return None
        metrics = None
        for _ in range(self.updates_per_step):
            metrics = self._learn()
        return metrics

    def after_episode(self, env: VectorizedEnv) -> Optional[Dict[str, float]]:
        for i in range(len(self._graphs)):
            trans = make_nstep_transitions(
                self._graphs[i], self._traj_actions[i], self._traj_rewards[i], self.n_step, self.gamma
            )
            if trans:
                self.buffer.add_many(trans)
        return {"buffer_size": float(len(self.buffer))}

    # ------------------------------------------------------------- learning
    def _learn(self) -> Dict[str, float]:
        self._set_train()
        if hasattr(self.buffer, "anneal_beta"):
            self.buffer.anneal_beta(self.progress)
        batch = self.buffer.sample(self.batch_size, self.device)
        graph, state, next_state = batch.graph, batch.state, batch.next_state
        b = graph.num_graphs
        idx = graph.batch
        is_w = batch.is_weights if batch.is_weights is not None else torch.ones(b, device=self.device)

        obs_t = self.env_class.obs_graph(graph, state)
        mask_t = self.env_class.valid_mask(graph, state)
        obs_tn = self.env_class.obs_graph(graph, next_state)
        mask_tn = self.env_class.valid_mask(graph, next_state)

        alpha = self.log_alpha.exp().detach()

        # ---- soft target value (expectation over valid next actions) ----
        with torch.no_grad():
            logits_n = self._actor_logits(obs_tn)
            logp_n = segment_log_softmax(apply_mask(logits_n, mask_tn), idx, b)  # [N]
            p_n = logp_n.exp()
            q1_n, q2_n = self._q_values(self.tgt_critic_encoder, self.tgt_q1, self.tgt_q2, obs_tn)
            min_qn = torch.min(q1_n, q2_n)
            contrib = (p_n * (min_qn - alpha * logp_n)) * mask_tn.to(min_qn.dtype)
            v_next = scatter_sum(contrib, idx, b)  # [K]
            has_valid = scatter_sum(mask_tn.to(min_qn.dtype), idx, b) > 0
            v_next = torch.where(has_valid, v_next, torch.zeros_like(v_next))
            disc = self.gamma ** self.n_step
            target_q = batch.reward + disc * (~batch.terminal).float() * v_next  # [K]

        # ---- critic loss (twin MSE on the chosen action) ----
        q1_all, q2_all = self._q_values(self.critic_encoder, self.q1, self.q2, obs_t)
        q1_sa = q1_all.index_select(0, batch.action)  # [K]
        q2_sa = q2_all.index_select(0, batch.action)  # [K]
        critic_per = (q1_sa - target_q).pow(2) + (q2_sa - target_q).pow(2)
        critic_loss = (is_w * critic_per).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self._critic_params, self.grad_clip)
        self.critic_optimizer.step()

        # ---- actor loss (expectation over valid actions; critics detached) ----
        logits = self._actor_logits(obs_t)
        logp = segment_log_softmax(apply_mask(logits, mask_t), idx, b)  # [N]
        p = logp.exp()
        with torch.no_grad():
            q1_c, q2_c = self._q_values(self.critic_encoder, self.q1, self.q2, obs_t)
            min_qc = torch.min(q1_c, q2_c)
        actor_contrib = (p * (alpha * logp - min_qc)) * mask_t.to(p.dtype)
        actor_loss = scatter_sum(actor_contrib, idx, b).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self._actor_params, self.grad_clip)
        self.actor_optimizer.step()

        # ---- entropy & temperature ----
        ent_contrib = -(p.detach() * logp.detach()) * mask_t.to(p.dtype)
        entropy = scatter_sum(ent_contrib, idx, b)  # [K]
        if self.autotune_alpha:
            n_valid = scatter_sum(mask_t.to(p.dtype), idx, b).clamp_min(1.0)
            target_entropy = self.target_entropy_ratio * torch.log(n_valid.mean())
            alpha_loss = -(self.log_alpha * (entropy - target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        # ---- Polyak soft-update of the target critics ----
        self._soft_update(self.tgt_critic_encoder, self.critic_encoder, self.tau)
        self._soft_update(self.tgt_q1, self.q1, self.tau)
        self._soft_update(self.tgt_q2, self.q2, self.tau)

        self._learn_steps += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(alpha),
            "alpha_loss": float(alpha_loss.detach()),
            "entropy": float(entropy.mean().detach()),
            "q_mean": float((0.5 * (q1_sa + q2_sa)).mean().detach()),
            "target_mean": float(target_q.mean().detach()),
        }

    # ---------------------------------------------------------------- helpers
    def _set_train(self) -> None:
        for m in (self.actor_encoder, self.actor_head, self.critic_encoder, self.q1, self.q2):
            m.train()

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.mul_(1.0 - tau).add_(sp, alpha=tau)

    # -------------------------------------------------------- checkpointing
    def state_dict(self) -> Dict[str, Any]:
        sd = {
            "actor_encoder": self.actor_encoder.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic_encoder": self.critic_encoder.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "tgt_critic_encoder": self.tgt_critic_encoder.state_dict(),
            "tgt_q1": self.tgt_q1.state_dict(),
            "tgt_q2": self.tgt_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "learn_steps": self._learn_steps,
        }
        if self.alpha_optimizer is not None:
            sd["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.actor_encoder.load_state_dict(sd["actor_encoder"])
        self.actor_head.load_state_dict(sd["actor_head"])
        self.critic_encoder.load_state_dict(sd["critic_encoder"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.tgt_critic_encoder.load_state_dict(sd["tgt_critic_encoder"])
        self.tgt_q1.load_state_dict(sd["tgt_q1"])
        self.tgt_q2.load_state_dict(sd["tgt_q2"])
        self.actor_optimizer.load_state_dict(sd["actor_optimizer"])
        self.critic_optimizer.load_state_dict(sd["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        if self.alpha_optimizer is not None and "alpha_optimizer" in sd:
            self.alpha_optimizer.load_state_dict(sd["alpha_optimizer"])
        self._learn_steps = sd.get("learn_steps", 0)

    def parameters(self) -> Iterator[Tensor]:
        params: Iterator[Tensor] = itertools.chain(self._actor_params, self._critic_params)
        if self.autotune_alpha:
            params = itertools.chain(params, [self.log_alpha])
        return params
