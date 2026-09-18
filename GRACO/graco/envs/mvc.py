"""Minimum Vertex Cover (MVC) environment.

Construct a cover by repeatedly selecting nodes; an edge is *covered* once one of
its endpoints is selected.  The episode ends when every edge is covered
(``num_covered == num_edges``), and the objective is the cover size (minimized).
Reward is ``-1`` per pick (S2V-DQN MVC).  The action mask keeps only *useful*
nodes — those still incident to an uncovered edge (FINDER's usefulness filter).
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


@ENVS.register("mvc")
class MVCEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = False  # minimize cover size
    state_spec = {"selected": "node"}

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        # cover grows by one node per active graph -> reward -1
        return -torch.ones(self.num_envs, device=self.device)

    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"]
        unsel = ~sel
        src, dst = graph.edge_index[0], graph.edge_index[1]
        uncovered = (unsel[src] & unsel[dst]).float()
        uncov_deg = scatter_sum(uncovered, src, graph.num_nodes).unsqueeze(-1)
        deg = graph.degree().clamp_min(1).unsqueeze(-1)
        return torch.cat([sel.float().unsqueeze(-1), uncov_deg / deg, torch.ones_like(deg)], dim=-1)

    @staticmethod
    def _uncovered_node(graph: BatchedGraph, state: State) -> Tensor:
        unsel = ~state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        uncovered_edge = unsel[src] & unsel[dst]
        has_uncov = scatter_sum(uncovered_edge.float(), src, graph.num_nodes) > 0
        return unsel & has_uncov

    @classmethod
    def valid_mask(cls, graph: BatchedGraph, state: State) -> Tensor:
        return cls._uncovered_node(graph, state)

    @staticmethod
    def is_done(graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        unsel = ~state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        uncovered_edge = (unsel[src] & unsel[dst]).long()
        num_uncov = scatter_sum(uncovered_edge, graph.edge_batch, graph.num_graphs)
        return num_uncov == 0

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        return scatter_sum(state["selected"].float(), graph.batch, graph.num_graphs)
