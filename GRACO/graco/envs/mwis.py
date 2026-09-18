"""Maximum Weighted Independent Set (MWIS) environment.

Same construction rule as MIS (select a node, forbid its neighbours) but the
reward and objective use per-node weights.  Weights are a deterministic function
of the graph topology (default ``degree + 1``), so they are reproducible when a
transition is rebuilt from replay.  ``weight_mode``: ``degree`` | ``inv_degree``
| ``unit`` (unit reduces exactly to MIS).
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State
from graco.envs.mis import MISEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


def _node_weight(graph: BatchedGraph, mode: str) -> Tensor:
    deg = graph.degree()
    if mode == "unit":
        return torch.ones_like(deg)
    if mode == "inv_degree":
        return 1.0 / (deg + 1.0)
    return deg + 1.0  # "degree"


@ENVS.register("mwis", aliases=["max_weighted_is", "weighted_mis"])
class MWISEnv(MISEnv):
    maximize = True

    def __init__(self, reward_scale: float = 1.0, max_steps_frac: float = 1.0,
                 weight_mode: str = "degree"):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        self.weight_mode = weight_mode

    def _init_state(self, graph: BatchedGraph) -> State:
        self._w = _node_weight(graph, self.weight_mode)
        return super()._init_state(graph)

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        return self._w.index_select(0, action.clamp_min(0))  # base zeroes inactive graphs

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # instance override
        w = _node_weight(graph, self.weight_mode)
        return scatter_sum(w * state["selected"].float(), graph.batch, graph.num_graphs)
