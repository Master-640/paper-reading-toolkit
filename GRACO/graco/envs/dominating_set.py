"""Minimum Dominating Set (MDS) environment.

Select nodes until every node is *dominated* (selected or adjacent to a selected
node).  Reward ``-1`` per pick (minimize the set); the action mask keeps only
nodes that would newly dominate something.  Terminal when all nodes are
dominated.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


@ENVS.register("mds", aliases=["dominating_set", "min_dominating_set"])
class MinDominatingSetEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = False
    state_spec = {"selected": "node"}

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        return -torch.ones(self.num_envs, device=self.device)

    @staticmethod
    def _dominated(graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        nbr_selected = scatter_sum(sel[src].float(), dst, graph.num_nodes) > 0
        return sel | nbr_selected

    @classmethod
    def node_features(cls, graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"].float().unsqueeze(-1)
        undominated = (~cls._dominated(graph, state)).float().unsqueeze(-1)
        deg = graph.degree().clamp_min(1).unsqueeze(-1)
        return torch.cat([sel, undominated, deg / deg.max().clamp_min(1)], dim=-1)

    @classmethod
    def valid_mask(cls, graph: BatchedGraph, state: State) -> Tensor:
        undominated = ~cls._dominated(graph, state)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        has_undom_nbr = scatter_sum(undominated[src].float(), dst, graph.num_nodes) > 0
        # a node is useful if selecting it dominates something not yet dominated
        return (~state["selected"]) & (undominated | has_undom_nbr)

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        undominated = (~cls._dominated(graph, state)).float()
        return scatter_sum(undominated, graph.batch, graph.num_graphs) == 0

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        return scatter_sum(state["selected"].float(), graph.batch, graph.num_graphs)
