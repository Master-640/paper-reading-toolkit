"""Network dismantling environment (FINDER).

Remove nodes to fragment the graph.  After each removal the reward penalizes the
size of the largest connected component (LCC) among the *remaining* nodes,
computed with the batched GPU label-propagation in
:mod:`graco.utils.graph_algos`.  Three cost modes:

* ``unit``   (FINDER_ND): ``r = -LCC / n²``.
* ``degree`` (FINDER_CN_cost): ``r = -(LCC / n) · (deg(a) / Σ deg)``.
* ``random`` (FINDER_CN_cost): same but with i.i.d. U(0,1) node costs.

The episode ends when every edge has a removed endpoint (the graph is fully
disconnected into isolated nodes).  The sum of per-step rewards equals the
(negated) Accumulated Normalized Connectivity — the standard dismantling metric.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.graph_algos import largest_component_size
from graco.utils.scatter import scatter_sum


@ENVS.register("dismantling", aliases=["finder", "network_dismantling"])
class DismantlingEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = True  # maximize episode return (= -ANC); higher return = better dismantling
    eval_metric = "return"
    state_spec = {"selected": "node"}
    #: run the GNN on the residual ACTIVE subgraph (keep only edges whose both
    #: endpoints are still present) — this is FINDER's ``idx_map``: node
    #: embeddings then reflect the shrinking network, which is essential for the
    #: policy to tell apart which node to remove next. Without it, constant node
    #: inputs on the full graph give near-identical embeddings and the agent
    #: cannot learn a good order.
    gnn_active_subgraph = True

    def __init__(
        self,
        reward_scale: float = 1.0,
        max_steps_frac: float = 1.0,
        cost_mode: str = "unit",
        lcc_max_iter: Optional[int] = None,
    ):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        assert cost_mode in ("unit", "degree", "random")
        self.cost_mode = cost_mode
        self.lcc_max_iter = lcc_max_iter
        self._node_weight: Optional[Tensor] = None
        self._total_weight: Optional[Tensor] = None

    def _init_state(self, graph: BatchedGraph) -> State:
        # per-node removal cost (static for the episode; used only in the reward)
        if self.cost_mode == "degree":
            self._node_weight = graph.degree().clamp_min(1.0)
        elif self.cost_mode == "random":
            self._node_weight = torch.rand(graph.num_nodes, device=graph.device)
        else:
            self._node_weight = torch.ones(graph.num_nodes, device=graph.device)
        self._total_weight = scatter_sum(
            self._node_weight, graph.batch, graph.num_graphs
        ).clamp_min(1e-8)
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        self._mark_selected(action, active)
        active_nodes = ~self.state["selected"]
        lcc = largest_component_size(
            g.edge_index, g.num_nodes, g.batch, g.num_graphs, active_nodes, self.lcc_max_iter
        )  # [B] float
        n = g.graph_num_nodes.float().clamp_min(1)
        if self.cost_mode == "unit":
            return -lcc / (n * n)
        w_a = self._node_weight.index_select(0, action.clamp_min(0))  # [B]
        return -(lcc / n) * (w_a / self._total_weight)

    @classmethod
    def obs_graph(cls, graph: BatchedGraph, state: State) -> BatchedGraph:
        """GNN sees the residual active subgraph (FINDER ``idx_map``).

        Removed nodes are kept (so node indexing / action gathering is unchanged)
        but become isolated — only edges with both endpoints still active are
        propagated — so embeddings reflect the shrinking network. Removed/isolated
        nodes are masked out of the action set anyway.
        """
        if not cls.gnn_active_subgraph:
            return super().obs_graph(graph, state)
        active = ~state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        emask = active[src] & active[dst]
        edge_feats = cls.edge_features(graph, state)
        return BatchedGraph(
            edge_index=graph.edge_index[:, emask],
            num_nodes=graph.num_nodes,
            batch=graph.batch,
            ptr=graph.ptr,
            edge_weight=graph.edge_weight[emask] if graph.edge_weight is not None else None,
            edge_attr=edge_feats[emask] if edge_feats is not None else None,
            x=cls.node_features(graph, state),
            node_type=graph.node_type,
            edge_type=graph.edge_type[emask] if graph.edge_type is not None else None,
            meta=graph.meta,
        )

    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        removed = state["selected"]
        active = (~removed).float()
        src, dst = graph.edge_index[0], graph.edge_index[1]
        active_edge = ((~removed)[src] & (~removed)[dst]).float()
        active_deg = scatter_sum(active_edge, src, graph.num_nodes).unsqueeze(-1)
        deg = graph.degree().clamp_min(1).unsqueeze(-1)
        return torch.cat([removed.float().unsqueeze(-1), active_deg / deg, active.unsqueeze(-1)], -1)

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        active = ~state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        active_edge = active[src] & active[dst]
        has_active_nbr = scatter_sum(active_edge.float(), src, graph.num_nodes) > 0
        return active & has_active_nbr

    @staticmethod
    def is_done(graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        active = ~state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        active_edge = (active[src] & active[dst]).long()
        remaining = scatter_sum(active_edge, graph.edge_batch, graph.num_graphs)
        return remaining == 0

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        active = ~state["selected"]
        return largest_component_size(graph.edge_index, graph.num_nodes, graph.batch, graph.num_graphs, active)


@ENVS.register("dismantling_cost", aliases=["finder_cost"])
class CostAwareDismantlingEnv(DismantlingEnv):
    """Cost-aware dismantling (degree costs by default)."""

    def __init__(self, reward_scale: float = 1.0, max_steps_frac: float = 1.0,
                 cost_mode: str = "degree", lcc_max_iter: Optional[int] = None):
        super().__init__(reward_scale, max_steps_frac, cost_mode, lcc_max_iter)
