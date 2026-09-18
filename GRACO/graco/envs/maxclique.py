"""Maximum Clique environment.

Grow a clique one node at a time: a node is a legal action iff it is adjacent to
**every** node already in the clique (equivalently, its count of selected
neighbours equals the current clique size).  Reward ``+1`` per node; the episode
ends when no node can extend the clique; the objective (maximized) is the clique
size.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


@ENVS.register("maxclique", aliases=["max_clique"])
class MaxCliqueEnv(VectorizedEnv):
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
    def _adjacent_to_all_selected(graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        sel_nbr = scatter_sum(sel[src].float(), dst, graph.num_nodes)  # #selected neighbours
        clique_size = scatter_sum(sel.float(), graph.batch, graph.num_graphs)
        return sel_nbr >= graph.broadcast_to_nodes(clique_size.unsqueeze(-1)).squeeze(-1) - 1e-6

    @classmethod
    def node_features(cls, graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"].float().unsqueeze(-1)
        cand = (cls.valid_mask(graph, state)).float().unsqueeze(-1)
        deg = graph.degree()
        deg_max = graph.broadcast_to_nodes(graph.pool(deg.unsqueeze(-1), reduce="max")).clamp_min(1)
        return torch.cat([sel, cand, deg.unsqueeze(-1) / deg_max], dim=-1)

    @classmethod
    def valid_mask(cls, graph: BatchedGraph, state: State) -> Tensor:
        return (~state["selected"]) & cls._adjacent_to_all_selected(graph, state)

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        valid = cls.valid_mask(graph, state)
        return scatter_sum(valid.float(), graph.batch, graph.num_graphs) == 0

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        return scatter_sum(state["selected"].float(), graph.batch, graph.num_graphs)
