"""Algorithm base class.

An :class:`Algorithm` owns its policy network(s), optimizer, and (optionally) a
buffer.  The :class:`~graco.trainers.trainer.Trainer` drives a single unified
batched-rollout loop and calls the algorithm's hooks, so value-based
(off-policy, replay) and policy-gradient (on-policy, rollout) methods share the
exact same outer loop:

    obs = env.reset(graphs); algo.on_episode_start(env)
    while not obs.done.all():
        a = algo.act(obs, explore=True)
        step = env.step(a)
        algo.observe(env, obs, a, step)
        m = algo.after_step()        # off-policy learns here (from replay)
        obs = step.obs
    m = algo.after_episode(env)      # off-policy pushes n-step tuples; on-policy learns

Schedules (epsilon, PER beta, entropy) read ``self.progress`` in ``[0, 1]``,
updated by the trainer via :meth:`set_progress`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from graco.envs.base import Observation, StepResult, VectorizedEnv


class Algorithm(ABC):
    #: "step" -> Trainer may call after_step every env step; "episode" -> only after episodes
    learns_online: bool = False
    #: single-step algos (e.g. GRLND) select a whole removal set in one forward pass;
    #: the Trainer runs :meth:`learn_step` once per iteration instead of the sequential loop.
    single_step: bool = False

    def __init__(self, env: VectorizedEnv, device: Any = "cpu", gamma: float = 1.0):
        self.env = env
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.progress = 0.0  # in [0,1], set by the trainer for schedules
        self.global_step = 0

    def set_progress(self, progress: float, global_step: int) -> None:
        self.progress = float(max(0.0, min(1.0, progress)))
        self.global_step = int(global_step)

    # --------------------------------------------------------------- lifecycle
    def on_sample(self, graph):
        """Transform freshly-sampled graphs before ``env.reset`` (default: no-op).

        POMO overrides this to replicate each graph into N multi-start copies;
        training-time augmentation can permute node labels here too.
        """
        return graph

    def on_episode_start(self, env: VectorizedEnv) -> None:  # noqa: D401
        """Hook at the start of a batched episode (e.g. snapshot topologies)."""

    @abstractmethod
    def act(self, obs: Observation, explore: bool = True) -> Tensor:
        """Return one global node id per graph ``[B]`` (``-1`` = no-op)."""

    @abstractmethod
    def observe(
        self, env: VectorizedEnv, obs: Observation, action: Tensor, step: StepResult
    ) -> None:
        """Record a transition (per-env)."""

    def after_step(self) -> Optional[Dict[str, float]]:
        """Off-policy update from replay (gated internally). Default: no-op."""
        return None

    def after_episode(self, env: VectorizedEnv) -> Optional[Dict[str, float]]:
        """Push transitions / on-policy update at episode end. Default: no-op."""
        return None

    def learn_step(self, env: VectorizedEnv, obs: Observation) -> Optional[Dict[str, float]]:
        """One-shot training step for :attr:`single_step` algos (default: unused)."""
        return None

    # ---------------------------------------------------------- checkpointing
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, sd: Dict[str, Any]) -> None: ...

    @abstractmethod
    def parameters(self):
        """Iterator over trainable parameters (for logging / grad clipping)."""
