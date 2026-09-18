"""Standalone predictor modules (not GNN encoders / heads).

Predictors map an observation to a supervised target trained *alongside* the RL
policy — e.g. the sentinel observability predictor (global steady-state mean from
the selected sentinels' trajectories).
"""

from graco.models.predictors.sentinel_predictor import SentinelPredictor  # noqa: F401

__all__ = ["SentinelPredictor"]
