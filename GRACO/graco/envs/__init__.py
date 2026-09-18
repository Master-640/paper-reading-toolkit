"""Vectorized combinatorial-optimization environments.

Importing this package registers every environment on the ``ENVS`` registry:
MaxCut, Spin-Glass (Ising), MVC, MIS, MWIS, Max-Clique, Minimum Dominating Set,
Set Cover, network dismantling (+ cost-aware), and Influence Maximization.
"""

from graco.envs.balanced_partition import BalancedPartitionEnv  # noqa: F401
from graco.envs.base import Observation, StepResult, VectorizedEnv  # noqa: F401
from graco.envs.dismantling import CostAwareDismantlingEnv, DismantlingEnv  # noqa: F401
from graco.envs.dominating_set import MinDominatingSetEnv  # noqa: F401
from graco.envs.improvement import FlipImprovementEnv, SpinGlassFlipImprovementEnv  # noqa: F401
from graco.envs.influence import InfluenceMaxEnv  # noqa: F401
from graco.envs.label import GraphColoringEnv, LabelEnv, MaxKCutEnv  # noqa: F401
from graco.envs.maxclique import MaxCliqueEnv  # noqa: F401
from graco.envs.maxcut import MaxCutEnv, SpinGlassEnv  # noqa: F401
from graco.envs.mis import MISEnv  # noqa: F401
from graco.envs.modularity import ModularityEnv  # noqa: F401
from graco.envs.mvc import MVCEnv  # noqa: F401
from graco.envs.mwis import MWISEnv  # noqa: F401
from graco.envs.nd_extra import (  # noqa: F401
    CascadeHypergraphDismantlingEnv,
    CombatDismantlingEnv,
    EdgeDismantlingEnv,
    MultiplexDismantlingEnv,
    PairwiseDismantlingEnv,
    SISDismantlingEnv,
)
from graco.envs.qubo import QUBOEnv  # noqa: F401
from graco.envs.sentinel import SentinelSelectionEnv  # noqa: F401
from graco.envs.setcover import SetCoverEnv  # noqa: F401

__all__ = [
    "VectorizedEnv",
    "Observation",
    "StepResult",
    "MaxCutEnv",
    "SpinGlassEnv",
    "MVCEnv",
    "MISEnv",
    "MWISEnv",
    "MaxCliqueEnv",
    "MinDominatingSetEnv",
    "SetCoverEnv",
    "DismantlingEnv",
    "CostAwareDismantlingEnv",
    "InfluenceMaxEnv",
    "FlipImprovementEnv",
    "SpinGlassFlipImprovementEnv",
    "QUBOEnv",
    "ModularityEnv",
    "BalancedPartitionEnv",
    "MaxKCutEnv",
    "GraphColoringEnv",
    "LabelEnv",
    "PairwiseDismantlingEnv",
    "EdgeDismantlingEnv",
    "MultiplexDismantlingEnv",
    "SISDismantlingEnv",
    "CombatDismantlingEnv",
    "CascadeHypergraphDismantlingEnv",
    "SentinelSelectionEnv",
]
