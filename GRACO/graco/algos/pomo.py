"""POMO multi-start training (F5).

POMO (Kwon et al., 2020) trains a stochastic policy by rolling out *N* diverse
trajectories per problem instance and using their **shared mean return** as the
REINFORCE baseline — no learned critic required.  In GRACO this is realized with
two hooks over the ordinary policy-gradient collector (:class:`_PGBase`):

* :meth:`on_sample` replicates each freshly-sampled graph into ``pomo_size``
  disjoint-union copies (optionally with node-label permutation for extra
  diversity, RL4CO-style augmentation).  The trainer's ``num_envs`` graphs thus
  become ``N * num_envs`` independent rollouts, all collected unchanged by the
  inherited stochastic :meth:`act`.
* :meth:`_update` computes, per source instance, the mean episode return across
  its ``N`` copies and subtracts it as a group baseline, then takes one
  REINFORCE step over every stored transition.

The actor-critic policy's value head built by :class:`_PGBase` is simply unused
here (POMO's baseline is the group mean, not a critic).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from graco.algos.pg import _PGBase
from graco.buffers.base import SingleGraph, collate_single_graphs
from graco.registries import ALGOS


@ALGOS.register("pomo")
class POMOAlgorithm(_PGBase):
    """REINFORCE with a POMO shared (per-instance) multi-start baseline."""

    def __init__(
        self,
        env,
        pomo_size: int = 8,
        augment: bool = True,
        entropy_coef: float = 0.01,
        **pg_kwargs: Any,
    ) -> None:
        super().__init__(env, entropy_coef=entropy_coef, **pg_kwargs)
        self.pomo_size = int(pomo_size)
        self.augment = bool(augment)
        self._group: Optional[Tensor] = None

    # ----------------------------------------------------------- multi-start
    def on_sample(self, graph):
        """Replicate each graph into ``pomo_size`` copies (POMO multi-start)."""
        from graco.data.augment import repeat_graphs

        bg, group = repeat_graphs(graph, self.pomo_size, permute=self.augment)
        self._group = group
        return bg

    # ---------------------------------------------------------------- update
    def _update(self) -> Dict[str, float]:
        self.policy.train()
        buf = self.buffer
        b_envs = len(buf.graphs)

        # (1) per-copy episode return R_b = sum of that copy's rewards.
        returns = torch.tensor(
            [float(sum(buf.reward[b])) for b in range(b_envs)],
            dtype=torch.float32,
            device=self.device,
        )

        # (2) group baseline: mean return over the N copies of each source graph.
        if self._group is not None and self._group.numel() == b_envs:
            group = self._group.to(self.device).long()
        else:  # fallback: every env its own group (baseline == return -> zero adv)
            group = torch.arange(b_envs, device=self.device)
        num_groups = int(group.max().item()) + 1 if b_envs > 0 else 0
        sums = torch.zeros(num_groups, device=self.device).scatter_add_(0, group, returns)
        counts = torch.zeros(num_groups, device=self.device).scatter_add_(
            0, group, torch.ones_like(returns)
        )
        baseline = sums / counts.clamp_min(1.0)
        adv_env = returns - baseline[group]  # [b_envs]

        # (3) flatten stored steps, tagging each with its copy's advantage.
        graphs: List[SingleGraph] = []
        states: List[Dict[str, Tensor]] = []
        actions: List[int] = []
        step_adv: List[float] = []
        for b in range(b_envs):
            T = min(len(buf.action[b]), len(buf.reward[b]))
            for t in range(T):
                graphs.append(buf.graphs[b])
                states.append(buf.state[b][t])
                actions.append(buf.action[b][t])
                step_adv.append(float(adv_env[b]))

        if not graphs:
            return {"policy_loss": 0.0, "entropy": 0.0, "adv_std": 0.0}

        adv = torch.tensor(step_adv, dtype=torch.float32, device=self.device)
        adv_std = float(adv.std().detach()) if adv.numel() > 1 else 0.0
        if self.normalize_adv and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        graph, state, action = self._collate_steps(graphs, states, actions)
        logp, _value, ent = self._evaluate(graph, state, action)
        entropy = ent.mean()
        policy_loss = -(logp * adv).mean()
        loss = policy_loss - self.entropy_coef * entropy

        # (4) one optimizer step (grad clip as in _PGBase).
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.detach()),
            "entropy": float(entropy.detach()),
            "adv_std": adv_std,
        }

    # --------------------------------------------------------------- helpers
    def _collate_steps(
        self,
        graphs: List[SingleGraph],
        states: List[Dict[str, Tensor]],
        actions_local: List[int],
    ) -> Tuple[Any, Dict[str, Tensor], Tensor]:
        """Collate stored steps into ``(graph, state, global_action)`` for ``_evaluate``.

        Mirrors :meth:`RolloutBuffer._collate` (states -> spec-scoped tensors,
        local action ids -> global via node offsets).
        """
        graph, offsets = collate_single_graphs(graphs, self.device)
        out_state: Dict[str, Tensor] = {}
        for name, scope in self.buffer.state_spec.items():
            vals = [s[name] for s in states]
            if scope == "graph":
                out_state[name] = torch.stack([v.reshape(()) for v in vals]).to(self.device)
            else:
                out_state[name] = torch.cat([v.reshape(-1) for v in vals]).to(self.device)
        action = torch.tensor(actions_local, device=self.device) + offsets
        return graph, out_state, action
