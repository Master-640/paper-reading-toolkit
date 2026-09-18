"""REINFORCE with a greedy-rollout baseline (Kool, van Hoof & Welling, 2019).

Instead of a *learned* value baseline (as in :class:`~graco.algos.pg.REINFORCEAlgorithm`),
the advantage of a sampled trajectory is its return minus the return a **frozen
copy of the policy** obtains by acting *greedily* on the very same graphs::

    A(traj on graph g) = R_sample(g) - R_greedy_baseline(g)

This whole-trajectory advantage is applied to every step's log-prob, so the
update is ``-(logp * advantage).mean() - entropy_coef * entropy``.  The baseline
network is refreshed from the current policy periodically, or whenever the
current policy significantly beats it (a one-sided paired t-test over the batch),
exactly as in the Attention-Model ("am") training loop.

Reference: W. Kool, H. van Hoof, M. Welling, "Attention, Learn to Solve Routing
Problems!" (ICLR 2019).
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import Tensor

from graco.algos.pg import _PGBase
from graco.envs.base import VectorizedEnv
from graco.registries import ALGOS
from graco.utils.segment_ops import segment_argmax


@ALGOS.register("reinforce_rollout", aliases=["am"])
class REINFORCERolloutAlgorithm(_PGBase):
    """REINFORCE whose baseline is a frozen policy rolled out greedily.

    Reuses :class:`_PGBase`'s on-policy rollout collector (``act``/``observe``);
    only the update (:meth:`_update`) and the baseline bookkeeping differ.  The
    critic head built by the base is kept but left unused.
    """

    def __init__(
        self,
        env: VectorizedEnv,
        baseline_update_every: int = 25,
        baseline_t_threshold: float = 1.0,
        **kw,
    ):
        # the rollout baseline already centres the advantage; do not renormalize
        kw.setdefault("normalize_adv", False)
        super().__init__(env, **kw)
        self.baseline_update_every = int(baseline_update_every)
        self.baseline_t_threshold = float(baseline_t_threshold)
        # frozen baseline policy (never optimized directly)
        self.baseline_policy = copy.deepcopy(self.policy).to(self.device)
        self.baseline_policy.eval()
        for p in self.baseline_policy.parameters():
            p.requires_grad_(False)
        self._episode_count = 0

    # --------------------------------------------------------- baseline rollout
    @torch.no_grad()
    def _baseline_returns(self) -> Tensor:
        """Greedy episode return per graph ``[B]`` for the frozen baseline.

        Runs on ``self.env``'s current (episode) graph without disturbing the
        live terminal state the trainer logs after ``after_episode``.
        """
        env = self.env
        graph = env.graph
        b = graph.num_graphs
        saved = (env.state, env.step_count, env.done)  # references to terminal state
        self.baseline_policy.eval()
        obs = env.reset(graph)  # rebinds env.state to a fresh dict (saved is intact)
        total = torch.zeros(b, device=self.device)
        guard, max_steps = 0, graph.num_nodes + 1
        while not obs.done.all() and guard < max_steps:
            g = obs.graph
            logits = self.baseline_policy.logits(g)
            action = segment_argmax(logits, g.batch, g.num_graphs, obs.action_mask)
            step = env.step(action)
            total = total + step.reward  # no-op / done envs contribute 0
            obs = step.obs
            guard += 1
        env.state, env.step_count, env.done = saved  # restore the sampled terminal
        return total

    def _sampled_returns(self) -> Tensor:
        """Undiscounted episode return per graph ``[B]`` from the rollout buffer."""
        b = len(self.buffer.graphs)
        out = torch.zeros(b, device=self.device)
        for i in range(b):
            T = min(len(self.buffer.value[i]), len(self.buffer.reward[i]))
            if T:
                out[i] = float(sum(self.buffer.reward[i][:T]))
        return out

    def _assign_advantages(self, adv: Tensor) -> None:
        """Overwrite each stored step's advantage with its trajectory advantage.

        Mirrors :meth:`RolloutBuffer.finalize`'s per-trajectory sample ordering,
        so a single scalar advantage is broadcast to every step of a trajectory.
        """
        adv_cpu = adv.detach().cpu().tolist()
        flat = []
        for i in range(len(self.buffer.graphs)):
            T = min(len(self.buffer.value[i]), len(self.buffer.reward[i]))
            if T == 0:
                continue
            flat.extend([adv_cpu[i]] * T)
        samples = self.buffer._samples
        assert len(flat) == len(samples), (len(flat), len(samples))
        for s, a in zip(samples, flat):
            s["advantage"] = a

    # ------------------------------------------------------------------ update
    def _update(self) -> Dict[str, float]:
        self.policy.train()
        baseline_returns = self._baseline_returns()
        sampled_returns = self._sampled_returns()
        advantage = sampled_returns - baseline_returns  # [B], one scalar per graph
        self._assign_advantages(advantage)

        mb = next(
            self.buffer.iter_minibatches(
                len(self.buffer), self.device, normalize_adv=self.normalize_adv
            )
        )
        logp, _value, ent = self._evaluate(mb["graph"], mb["state"], mb["action"])
        policy_loss = -(logp * mb["advantage"]).mean()
        entropy = ent.mean()
        loss = policy_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        updated = self._maybe_update_baseline(sampled_returns, baseline_returns)
        return {
            "policy_loss": float(policy_loss.detach()),
            "entropy": float(entropy.detach()),
            "advantage": float(advantage.mean().detach()),
            "baseline_return": float(baseline_returns.mean().detach()),
            "baseline_updated": float(updated),
        }

    def _maybe_update_baseline(self, sampled: Tensor, baseline: Tensor) -> bool:
        """Refresh the baseline periodically or on a significant improvement."""
        self._episode_count += 1
        diff = sampled - baseline
        if diff.numel() > 1:
            t_stat = diff.mean() / (diff.std().clamp_min(1e-8) / (diff.numel() ** 0.5))
        else:
            t_stat = diff.mean()
        significant = float(t_stat) > self.baseline_t_threshold
        periodic = self._episode_count % self.baseline_update_every == 0
        if significant or periodic:
            self.baseline_policy.load_state_dict(self.policy.state_dict())
            self.baseline_policy.eval()
            for p in self.baseline_policy.parameters():
                p.requires_grad_(False)
            return True
        return False

    # -------------------------------------------------------- checkpointing
    def state_dict(self) -> Dict[str, object]:
        sd = super().state_dict()
        sd["baseline_policy"] = self.baseline_policy.state_dict()
        sd["episode_count"] = self._episode_count
        return sd

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        super().load_state_dict(sd)
        if "baseline_policy" in sd:
            self.baseline_policy.load_state_dict(sd["baseline_policy"])
        self._episode_count = int(sd.get("episode_count", 0))
