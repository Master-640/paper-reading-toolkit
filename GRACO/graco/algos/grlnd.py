"""GRLND — Graph Reinforcement Learning for Network Dismantling (single-step).

Faithful re-implementation of GRLND (Qu et al., 2025): unlike the rest of the
framework, GRLND formulates dismantling as a **single-step MDP** — one forward
pass over the whole graph produces a per-node Bernoulli deletion probability, a
binary removal *mask* is sampled, and the policy is trained with REINFORCE on a
reward that balances **connectivity disruption** and **removal sparsity**:

    r = -(LCC(G \\ mask) / N)  -  sparsity_coef · (|mask| / N).

Because a whole set is chosen at once, GRLND captures the *joint* effect of
multiple removals rather than a greedy order.  It plugs into the shared CLI via
the Trainer's ``single_step`` path (:meth:`learn_step`); no change to the
sequential trainer used by every other algo.  For evaluation the learned
probabilities are read out greedily as a per-node score and rolled out through
the standard evaluator (which terminates naturally), so GRLND is directly
comparable to the sequential agents on ANC.

Reference: Hongbo Qu et al., "GRLND: A Graph Reinforcement Learning Framework for
Network Dismantling".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from graco.algos.base import Algorithm
from graco.envs.base import Observation, VectorizedEnv
from graco.models.build import build_q_policy
from graco.registries import ALGOS
from graco.utils.accel import build_optimizer
from graco.utils.graph_algos import largest_component_size
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import segment_argmax


@ALGOS.register("grlnd")
class GRLND(Algorithm):
    """Single-step Bernoulli-mask REINFORCE dismantler."""

    single_step = True
    learns_online = False

    def __init__(
        self,
        env: VectorizedEnv,
        model: Optional[dict] = None,
        device: Any = "cpu",
        lr: float = 3e-4,
        sparsity_coef: float = 0.3,
        entropy_coef: float = 0.01,
        grad_clip: Optional[float] = 1.0,
        num_samples: int = 4,
        weight_decay: float = 0.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(env, device=device, gamma=gamma)
        if kwargs:
            import warnings
            warnings.warn(f"{type(self).__name__} ignoring config keys: {sorted(kwargs)}")
        # per-node logit policy (q_node head outputs a scalar per node)
        self.policy = build_q_policy(dict(model or {}), env, head_default="q_node").to(self.device)
        self.optimizer = build_optimizer(self.policy.parameters(), lr=lr, weight_decay=weight_decay)
        self.sparsity_coef = float(sparsity_coef)
        self.entropy_coef = float(entropy_coef)
        self.grad_clip = grad_clip
        self.num_samples = max(1, int(num_samples))

    def learn_step(self, env: VectorizedEnv, obs: Observation) -> Dict[str, float]:
        self.policy.train()
        g = obs.graph
        b, n = g.num_graphs, g.graph_num_nodes.float().clamp_min(1.0)
        logits = self.policy(g)  # [N]
        p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Bernoulli(probs=p)

        rewards, logps = [], []
        for _ in range(self.num_samples):  # several masks -> mean baseline (variance reduction)
            mask = dist.sample()  # [N] in {0,1}, detached
            lcc = largest_component_size(
                g.edge_index, g.num_nodes, g.batch, b, ~mask.bool()
            )
            frac = scatter_sum(mask, g.batch, b) / n  # removal sparsity per graph
            rewards.append(-(lcc / n) - self.sparsity_coef * frac)  # [B] (maximize)
            logps.append(scatter_sum(dist.log_prob(mask), g.batch, b))  # [B] set log-prob
        R = torch.stack(rewards)   # [S, B]
        LP = torch.stack(logps)    # [S, B]
        adv = (R - R.mean(0, keepdim=True)).detach()  # per-graph mean baseline
        entropy = scatter_sum(dist.entropy(), g.batch, b)  # [B]
        loss = -(adv * LP).mean() - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        # expose the greedy mask so the Trainer's train/objective logging is meaningful
        greedy = p.detach() > 0.5
        env.state["selected"] = greedy
        removed_frac = float((scatter_sum(greedy.float(), g.batch, b) / n).mean())
        return {
            "loss": float(loss.detach()),
            "reward": float(R.mean().detach()),
            "removed_frac": removed_frac,
            "entropy": float(entropy.mean().detach()),
        }

    @torch.no_grad()
    def act(self, obs: Observation, explore: bool = True) -> Tensor:
        # greedy readout for evaluation: remove nodes in descending deletion probability
        self.policy.eval()
        g = obs.graph
        logits = self.policy(g)
        return segment_argmax(logits, g.batch, g.num_graphs, obs.action_mask)

    def observe(self, env, obs, action, step) -> None:  # unused (single-step)
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.policy.load_state_dict(sd["policy"])
        self.optimizer.load_state_dict(sd["optimizer"])

    def parameters(self):
        return self.policy.parameters()
