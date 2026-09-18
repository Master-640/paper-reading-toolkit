"""Reinforcement-learning algorithms.

Importing this package registers every algorithm on the ``ALGOS`` registry:
value-based — DQN (+ Double/Dueling/n-step/PER), QR-DQN, C51, IQN,
Munchausen-DQN (+ NoisyNet head); policy-gradient — REINFORCE (+ rollout
baseline), A2C, PPO; off-policy stochastic — Discrete SAC.
"""

from graco.algos.base import Algorithm  # noqa: F401
from graco.algos.c51 import C51Algorithm  # noqa: F401
from graco.algos.dqn import DQNAlgorithm, QRDQNAlgorithm  # noqa: F401
from graco.algos.grlnd import GRLND  # noqa: F401
from graco.algos.iqn import IQNAlgorithm  # noqa: F401  (also registers the IQN head)
from graco.algos.label_dqn import LabelDQNAlgorithm  # noqa: F401
from graco.algos.munchausen import MunchausenDQNAlgorithm  # noqa: F401
from graco.algos.pg import A2CAlgorithm, PPOAlgorithm, REINFORCEAlgorithm  # noqa: F401
from graco.algos.pomo import POMOAlgorithm  # noqa: F401
from graco.algos.reinforce_rollout import REINFORCERolloutAlgorithm  # noqa: F401
from graco.algos.sac import SACAlgorithm  # noqa: F401
from graco.algos.sentinel_dqn import SentinelPredictorDQN  # noqa: F401

__all__ = [
    "Algorithm",
    "DQNAlgorithm",
    "QRDQNAlgorithm",
    "C51Algorithm",
    "IQNAlgorithm",
    "MunchausenDQNAlgorithm",
    "PPOAlgorithm",
    "A2CAlgorithm",
    "REINFORCEAlgorithm",
    "REINFORCERolloutAlgorithm",
    "POMOAlgorithm",
    "SACAlgorithm",
    "SentinelPredictorDQN",
    "LabelDQNAlgorithm",
    "GRLND",
]
