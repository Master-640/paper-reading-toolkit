"""Policy-gradient agents: REINFORCE, A2C, and PPO — with per-node masking.

All three share one on-policy rollout collector (:class:`RolloutBuffer`): each
graph is its own trajectory, actions are sampled from a masked per-graph
categorical over nodes, and advantages/returns come from GAE.  They differ only
in the update: REINFORCE (MC advantage, one pass), A2C (advantage actor-critic,
one pass), PPO (clipped surrogate over several epochs of minibatches).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor

from graco.algos.base import Algorithm
from graco.buffers.base import to_single_graphs
from graco.buffers.rollout import RolloutBuffer
from graco.envs.base import Observation, StepResult, VectorizedEnv
from graco.models.build import build_actor_critic_policy
from graco.registries import ALGOS
from graco.utils.accel import build_optimizer
from graco.utils.segment_ops import (
    gather_action_logprob,
    segment_argmax,
    segment_entropy,
    segment_gumbel_sample,
)


class _PGBase(Algorithm):
    learns_online = False

    def __init__(
        self,
        env: VectorizedEnv,
        model: Optional[dict] = None,
        device: Any = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        grad_clip: Optional[float] = 0.5,
        normalize_adv: bool = True,
        shared_encoder: bool = True,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        super().__init__(env, device=device, gamma=gamma)
        if kwargs:
            import warnings

            warnings.warn(f"{type(self).__name__} ignoring config keys: {sorted(kwargs)}")
        self.env_class = type(env)
        self.state_spec = env.state_spec
        self.policy = build_actor_critic_policy(
            dict(model or {}), env, shared_encoder=shared_encoder
        ).to(self.device)
        self.optimizer = build_optimizer(
            self.policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.normalize_adv = normalize_adv
        self.buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda, state_spec=self.state_spec)

    # ------------------------------------------------------------- collection
    def on_episode_start(self, env: VectorizedEnv) -> None:
        self.buffer.start(to_single_graphs(env.graph))
        self._ptr = env.graph.ptr.detach().cpu()

    @torch.no_grad()
    def act(self, obs: Observation, explore: bool = True) -> Tensor:
        self.policy.eval()
        g = obs.graph
        logits, value = self.policy(g)
        mask = obs.action_mask
        if explore:
            action = segment_gumbel_sample(logits, g.batch, g.num_graphs, mask)
        else:
            action = segment_argmax(logits, g.batch, g.num_graphs, mask)
        if explore:
            logp = gather_action_logprob(logits, g.batch, g.num_graphs, action, mask)
            active = (~obs.done) & (action >= 0)
            ptr = self._ptr
            sel = self.env.state["selected"]
            act_cpu, lp_cpu, val_cpu = action.cpu(), logp.cpu(), value.cpu()
            for i in torch.nonzero(active.cpu(), as_tuple=False).flatten().tolist():
                lo, hi = int(ptr[i]), int(ptr[i + 1])
                self.buffer.record_action(
                    i,
                    {"selected": sel[lo:hi].detach().cpu().clone()},
                    int(act_cpu[i]) - lo,
                    float(lp_cpu[i]),
                    float(val_cpu[i]),
                )
        return action

    @torch.no_grad()
    def score(self, obs: Observation) -> Tensor:
        """Per-node policy logits ``[N]`` for decoding (softmax = the policy)."""
        self.policy.eval()
        return self.policy.logits(obs.graph)

    def observe(
        self, env: VectorizedEnv, obs: Observation, action: Tensor, step: StepResult
    ) -> None:
        active = (~obs.done) & (action >= 0)
        rew = step.reward.detach().cpu()
        for i in torch.nonzero(active.cpu(), as_tuple=False).flatten().tolist():
            self.buffer.record_reward(i, float(rew[i]))

    def after_episode(self, env: VectorizedEnv) -> Optional[Dict[str, float]]:
        self.buffer.finalize()
        if len(self.buffer) == 0:
            return None
        metrics = self._update()
        return metrics

    # ----------------------------------------------------------- evaluation
    def _evaluate(self, graph, state, action):
        logits, value = self.policy(graph)
        mask = self.env_class.valid_mask(graph, state)
        logp = gather_action_logprob(logits, graph.batch, graph.num_graphs, action, mask)
        ent = segment_entropy(logits, graph.batch, graph.num_graphs, mask)
        return logp, value, ent

    def _update(self) -> Dict[str, float]:  # pragma: no cover - overridden
        raise NotImplementedError

    # -------------------------------------------------------- checkpointing
    def state_dict(self) -> Dict[str, Any]:
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.optimizer.load_state_dict(sd["optimizer"])

    def parameters(self):
        return self.policy.parameters()


@ALGOS.register("ppo")
class PPOAlgorithm(_PGBase):
    def __init__(
        self,
        env,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        clip_value: bool = True,
        **kw,
    ):
        super().__init__(env, **kw)
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.clip_value = clip_value

    def _update(self) -> Dict[str, float]:
        self.policy.train()
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": 0}
        for _ in range(self.ppo_epochs):
            for mb in self.buffer.iter_minibatches(
                self.minibatch_size, self.device, normalize_adv=self.normalize_adv
            ):
                logp, value, ent = self._evaluate(mb["graph"], mb["state"], mb["action"])
                ratio = (logp - mb["logp_old"]).exp()
                adv = mb["advantage"]
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.clip_value:
                    v_clipped = mb["value_old"] + (value - mb["value_old"]).clamp(
                        -self.clip_eps, self.clip_eps
                    )
                    vloss = torch.max((value - mb["ret"]).pow(2), (v_clipped - mb["ret"]).pow(2)).mean()
                else:
                    vloss = (value - mb["ret"]).pow(2).mean()
                entropy = ent.mean()
                loss = policy_loss + self.value_coef * vloss - self.entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(vloss.detach())
                stats["entropy"] += float(entropy.detach())
                stats["n"] += 1
        n = max(1, stats.pop("n"))
        return {k: v / n for k, v in stats.items()}


@ALGOS.register("a2c")
class A2CAlgorithm(_PGBase):
    def _update(self) -> Dict[str, float]:
        self.policy.train()
        mb = next(self.buffer.iter_minibatches(len(self.buffer), self.device, normalize_adv=self.normalize_adv))
        logp, value, ent = self._evaluate(mb["graph"], mb["state"], mb["action"])
        policy_loss = -(logp * mb["advantage"]).mean()
        vloss = (value - mb["ret"]).pow(2).mean()
        entropy = ent.mean()
        loss = policy_loss + self.value_coef * vloss - self.entropy_coef * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(vloss.detach()),
            "entropy": float(entropy.detach()),
        }


@ALGOS.register("reinforce")
class REINFORCEAlgorithm(_PGBase):
    """REINFORCE with a learned value baseline (GAE(λ=1) = MC advantage)."""

    def __init__(self, env, **kw):
        kw.setdefault("gae_lambda", 1.0)
        super().__init__(env, **kw)

    def _update(self) -> Dict[str, float]:
        self.policy.train()
        mb = next(self.buffer.iter_minibatches(len(self.buffer), self.device, normalize_adv=self.normalize_adv))
        logp, value, ent = self._evaluate(mb["graph"], mb["state"], mb["action"])
        policy_loss = -(logp * mb["advantage"]).mean()
        vloss = (value - mb["ret"]).pow(2).mean()  # fit baseline
        entropy = ent.mean()
        loss = policy_loss + self.value_coef * vloss - self.entropy_coef * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"policy_loss": float(policy_loss.detach()), "entropy": float(entropy.detach())}
