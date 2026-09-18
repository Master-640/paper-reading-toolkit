"""Set Cover environment (bipartite).

On a bipartite graph of *element* nodes (``node_type==0``) and *set* nodes
(``node_type==1``), select set nodes until every element is covered (has a
selected set neighbour).  Reward ``-1`` per set; objective = #sets chosen
(minimized).  Pair with the ``set_cover`` generator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


def _node_type(graph: BatchedGraph) -> Tensor:
    if graph.node_type is not None:
        return graph.node_type
    return torch.zeros(graph.num_nodes, dtype=torch.long, device=graph.device)


@ENVS.register("set_cover", aliases=["setcover"])
class SetCoverEnv(VectorizedEnv):
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
    def _element_covered(graph: BatchedGraph, state: State) -> Tensor:
        """Bool [N]: element nodes that have a selected set neighbour."""
        nt = _node_type(graph)
        is_elem = nt == 0
        src, dst = graph.edge_index[0], graph.edge_index[1]
        sel = state["selected"]
        cov_count = scatter_sum(sel[src].float(), dst, graph.num_nodes)  # selected neighbours
        return is_elem & (cov_count > 0)

    @classmethod
    def node_features(cls, graph: BatchedGraph, state: State) -> Tensor:
        nt = _node_type(graph).float().unsqueeze(-1)
        sel = state["selected"].float().unsqueeze(-1)
        covered = cls._element_covered(graph, state).float().unsqueeze(-1)
        return torch.cat([sel, nt, covered], dim=-1)

    @classmethod
    def valid_mask(cls, graph: BatchedGraph, state: State) -> Tensor:
        nt = _node_type(graph)
        is_set = nt == 1
        is_elem = nt == 0
        uncovered_elem = is_elem & (~cls._element_covered(graph, state))
        src, dst = graph.edge_index[0], graph.edge_index[1]
        # a set is useful if it neighbours an uncovered element
        useful = scatter_sum(uncovered_elem[src].float(), dst, graph.num_nodes) > 0
        return is_set & (~state["selected"]) & useful

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        nt = _node_type(graph)
        is_elem = (nt == 0).float()
        covered = cls._element_covered(graph, state).float()
        n_elem = scatter_sum(is_elem, graph.batch, graph.num_graphs)
        n_cov = scatter_sum(covered, graph.batch, graph.num_graphs)
        return n_cov >= n_elem

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        nt = _node_type(graph)
        sel_sets = (state["selected"] & (nt == 1)).float()
        return scatter_sum(sel_sets, graph.batch, graph.num_graphs)
