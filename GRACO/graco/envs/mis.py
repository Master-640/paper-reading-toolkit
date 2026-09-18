"""Maximum Independent Set (MIS) environment.

Grow an independent set by selecting nodes; selecting a node forbids all of its
neighbours.  A node is a legal action iff it is neither selected nor adjacent to
a selected node.  The episode ends when no legal action remains, and the
objective (maximized) is the independent-set size.  Reward is ``+1`` per pick.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


@ENVS.register("mis")
class MISEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = True
    state_spec = {"selected": "node"}

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        return torch.ones(self.num_envs, device=self.device)

    @staticmethod
    def _blocked(graph: BatchedGraph, state: State) -> Tensor:
        """Nodes adjacent to a selected node."""
        sel = state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        nbr_selected = scatter_sum(sel[src].float(), dst, graph.num_nodes) > 0
        return nbr_selected

    @classmethod
    def node_features(cls, graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"].float().unsqueeze(-1)
        blocked = cls._blocked(graph, state).float().unsqueeze(-1)
        deg = graph.degree()
        deg_max = graph.broadcast_to_nodes(graph.pool(deg.unsqueeze(-1), reduce="max")).clamp_min(1)
        return torch.cat([sel, blocked, deg.unsqueeze(-1) / deg_max], dim=-1)

    @classmethod
    def valid_mask(cls, graph: BatchedGraph, state: State) -> Tensor:
        return (~state["selected"]) & (~cls._blocked(graph, state))

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        valid = cls.valid_mask(graph, state)
        n_valid = scatter_sum(valid.float(), graph.batch, graph.num_graphs)
        return n_valid == 0

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        return scatter_sum(state["selected"].float(), graph.batch, graph.num_graphs)
