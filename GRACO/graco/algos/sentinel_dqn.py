"""Sentinel predictor DQN (see why.md, redesign).

Reuses the entire DQN family (Double/Dueling/n-step/PER, target network, replay,
epsilon schedule) verbatim and adds exactly two things for the observability
task:

1. A **learned predictor** (:class:`SentinelPredictor`) trained with its *own*
   optimizer on independent replay minibatches inside ``after_step`` — supervised
   regression of the global steady mean from the selected sentinels' trajectories.
   The supervised target does not depend on the TD minibatch, so an independent
   sample is simpler than reusing ``_learn``'s batch and decorrelates the updates.

2. A **per-episode frozen reward predictor** the env uses to score ``E(S)``.  It
   is synced from the online predictor once per episode (``on_episode_start``) and
   never updated mid-episode, so within an episode the reward telescopes to
   ``E(∅) − E(S_final)`` and the stored per-step rewards stay consistent — the RL
   side sees a piecewise-stationary reward, exactly like the target Q-network.
"""

from __future__ import annotations

import copy
from itertools import chain
from typing import Any, Dict, Optional

import torch

from graco.algos.dqn import DQNAlgorithm
from graco.envs.base import VectorizedEnv
from graco.models.predictors import SentinelPredictor
from graco.registries import ALGOS
from graco.utils.accel import build_optimizer


@ALGOS.register("sentinel_dqn", aliases=["sentinel_predictor_dqn"])
class SentinelPredictorDQN(DQNAlgorithm):
    def __init__(
        self,
        env: VectorizedEnv,
        predictor: Optional[dict] = None,
        pred_lr: float = 3e-4,
        pred_grad_clip: Optional[float] = 10.0,
        **kw,
    ):
        super().__init__(env, **kw)
        self.predictor = SentinelPredictor(num_conditions=env.M, **dict(predictor or {})).to(self.device)
        self._reward_predictor = copy.deepcopy(self.predictor).eval()
        self.pred_optimizer = build_optimizer(self.predictor.parameters(), lr=pred_lr)
        self.pred_grad_clip = pred_grad_clip
        env.set_predictor(self._reward_predictor)

    def on_episode_start(self, env: VectorizedEnv) -> None:
        super().on_episode_start(env)
        # freeze the reward predictor for the whole episode (keeps reward telescoping
        # and near-stationary); sync it to the latest online weights at the boundary.
        self._reward_predictor.load_state_dict(self.predictor.state_dict())
        self._reward_predictor.eval()
        env.set_predictor(self._reward_predictor)

    def after_step(self) -> Optional[Dict[str, float]]:
        m = super().after_step()  # DQN update (gated on learn_start/learn_freq)
        if m is not None:
            m.update(self._learn_predictor())
        return m

    def _pred_loss(self, yhat, y, conv):
        err = (yhat - y).abs() / y.abs().clamp_min(self.env.err_floor)  # [K, M]
        if conv is not None:
            cf = conv.to(err.dtype)
            return (err * cf).sum() / cf.sum().clamp_min(1.0)  # masked relative MAE
        return err.mean()

    def _learn_predictor(self) -> Dict[str, float]:
        self.predictor.train()
        batch = self.buffer.sample(self.batch_size, self.device)
        obs_t = self.env_class.obs_graph(batch.graph, batch.state)
        yhat = self.predictor(obs_t, batch.state["selected"])  # [K, M]
        y = obs_t.graph_attr["y"]
        loss = self._pred_loss(yhat, y, obs_t.graph_attr.get("converged"))
        self.pred_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.pred_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.pred_grad_clip)
        self.pred_optimizer.step()
        return {"pred_loss": float(loss.detach())}

    # -------------------------------------------------------- checkpointing
    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd["predictor"] = self.predictor.state_dict()
        sd["reward_predictor"] = self._reward_predictor.state_dict()
        sd["pred_optimizer"] = self.pred_optimizer.state_dict()
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        super().load_state_dict(sd)
        self.predictor.load_state_dict(sd["predictor"])
        self._reward_predictor.load_state_dict(sd["reward_predictor"])
        self.pred_optimizer.load_state_dict(sd["pred_optimizer"])

    def parameters(self):
        return chain(self.policy.parameters(), self.predictor.parameters())
