"""Extra network-dismantling environments reproduced from the ND-paper collection.

All reuse :class:`~graco.envs.dismantling.DismantlingEnv` — same per-node-selection
MDP, replay, active-subgraph GNN and rollout — and only change *what connectivity
is penalized*:

* ``pairwise_dismantling`` — reward penalizes **pairwise connectivity**
  ``Σ_C |C|(|C|-1)/2`` instead of the largest component (the ANC objective used by
  VNS / several critical-node papers).
* ``edge_dismantling`` — runs on the **line graph** ``L(G)`` (so a node action = an
  edge removal) but measures connectivity on the **original** graph, reproducing the
  critical-edge papers (SHEAR: pairwise ANC on ``G``; IKEoN: ``Σ_C edges(C)²``).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State
from graco.envs.dismantling import DismantlingEnv
from graco.registries import ENVS
from graco.utils.graph_algos import (
    combat_operational_capability,
    connected_component_labels,
    hypergraph_cascade_survivors,
    largest_component_size,
    mutually_connected_component_size,
    pairwise_connectivity,
    sis_infection_risk,
)
from graco.utils.scatter import scatter_sum


@ENVS.register("pairwise_dismantling", aliases=["anc_dismantling", "vns"])
class PairwiseDismantlingEnv(DismantlingEnv):
    """Node dismantling that minimizes pairwise connectivity (ANC objective)."""

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        self._mark_selected(action, active)
        active_nodes = ~self.state["selected"]
        sigma = pairwise_connectivity(
            g.edge_index, g.num_nodes, g.batch, g.num_graphs, active_nodes, self.lcc_max_iter
        )
        n = g.graph_num_nodes.float().clamp_min(1)
        norm = (n * (n - 1) / 2).clamp_min(1.0)  # initial pairwise connectivity
        return -sigma / norm  # return = -ANC (higher is better)

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        active = ~state["selected"]
        return pairwise_connectivity(graph.edge_index, graph.num_nodes, graph.batch, graph.num_graphs, active)


@ENVS.register("edge_dismantling", aliases=["critical_edges", "shear", "ikeon"])
class EdgeDismantlingEnv(DismantlingEnv):
    """Critical-edge dismantling on the line graph ``L(G)``.

    A node of ``L(G)`` is an edge of the original graph; selecting it removes that
    edge.  Connectivity is measured on the **original** graph (reconstructed from
    ``node_attr['orig_edge']`` and ``meta['orig_num_nodes']``/``['orig_batch']``,
    produced by the ``line_graph`` generator).  ``metric``: ``pairwise`` (SHEAR /
    ANC) or ``edge_sq`` (IKEoN — ``Σ_C edges(C)²``).
    """

    def __init__(self, reward_scale: float = 1.0, max_steps_frac: float = 1.0,
                 metric: str = "pairwise", lcc_max_iter: Optional[int] = None):
        super().__init__(reward_scale, max_steps_frac, cost_mode="unit", lcc_max_iter=lcc_max_iter)
        assert metric in ("pairwise", "edge_sq")
        self.metric = metric

    def _orig_edge_index(self, kept_lnodes: Tensor) -> Tensor:
        oe = self.graph.node_attr["orig_edge"]  # [M,2] global original endpoints
        rem = oe[kept_lnodes]  # remaining original edges
        if rem.numel() == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=oe.device)
        return torch.stack([torch.cat([rem[:, 0], rem[:, 1]]), torch.cat([rem[:, 1], rem[:, 0]])])

    def _orig_connectivity(self, selected: Tensor) -> Tensor:
        g = self.graph
        n_orig = int(g.meta["orig_num_nodes"])
        obatch = g.meta["orig_batch"]
        ei = self._orig_edge_index(~selected)  # edges still present
        if self.metric == "pairwise":
            return pairwise_connectivity(ei, n_orig, obatch, g.num_graphs, None, self.lcc_max_iter)
        # edge_sq: Σ_C edges(C)² — attribute each remaining edge to its component
        labels = connected_component_labels(ei, n_orig, None, self.lcc_max_iter)
        src = ei[0]
        ecount = scatter_sum(torch.ones(src.numel(), device=ei.device), labels.index_select(0, src), n_orig)
        return scatter_sum(ecount * ecount, obatch, g.num_graphs)  # per-graph Σ_C edges(C)²

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        conn = self._orig_connectivity(self.state["selected"])
        n = torch.as_tensor(int(self.graph.meta["orig_num_nodes"]), device=conn.device).float().clamp_min(1)
        return -conn / (n * n)  # return = -accumulated connectivity (maximize)

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        return self._orig_connectivity(state["selected"])


@ENVS.register("multiplex_dismantling", aliases=["miner", "multiplex_disintegration"])
class MultiplexDismantlingEnv(DismantlingEnv):
    """Multiplex/multilayer disintegration (MINER): reward penalizes the giant
    **mutually connected component** (GMCC) instead of the single-layer LCC."""

    def _gmcc(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        nl = int(graph.meta.get("num_layers", 1))
        return mutually_connected_component_size(
            graph.edge_index, graph.edge_type, nl, graph.num_nodes,
            graph.batch, graph.num_graphs, ~selected, self.lcc_max_iter,
        )

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        gmcc = self._gmcc(self.graph, self.state["selected"])
        n = self.graph.graph_num_nodes.float().clamp_min(1)
        return -gmcc / (n * n)

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        return self._gmcc(graph, state["selected"])


@ENVS.register("sis_dismantling", aliases=["deepele"])
class SISDismantlingEnv(DismantlingEnv):
    """SIS critical-node removal (DeepELE): reward penalizes the accumulated SIS
    infection risk on the residual graph instead of the LCC."""

    def __init__(self, reward_scale: float = 1.0, max_steps_frac: float = 1.0,
                 beta: float = 0.3, sis_iters: int = 20, lcc_max_iter=None):
        super().__init__(reward_scale, max_steps_frac, cost_mode="unit", lcc_max_iter=lcc_max_iter)
        self.beta = float(beta)
        self.sis_iters = int(sis_iters)

    def _risk(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        return sis_infection_risk(
            graph.edge_index, graph.num_nodes, graph.batch, graph.num_graphs,
            ~selected, self.beta, self.sis_iters,
        )

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        risk = self._risk(self.graph, self.state["selected"])
        n = self.graph.graph_num_nodes.float().clamp_min(1)
        return -risk / n

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        return self._risk(graph, state["selected"])


@ENVS.register("combat_dismantling", aliases=["shatter", "hdged", "combat_disintegration"])
class CombatDismantlingEnv(DismantlingEnv):
    """Heterogeneous **combat-network** disintegration (SHATTER / HDGED).

    Remove nodes to destroy ``S → D → I`` combat chains; the reward penalizes the
    residual **operational capability** normalized by its initial value, so the
    episode return is ``-ANOC`` (accumulated normalized operational capability).
    The S/D/I type ids come from ``graph.meta`` (produced by the ``combat``
    generator); pair with a relation-aware encoder (``han`` for SHATTER, ``rgcn``
    for HDGED) plus the FINDER Q-head.
    """

    def _sdi(self, graph: BatchedGraph):
        m = graph.meta
        return (int(m.get("sensor_type", 0)), int(m.get("decision_type", 1)),
                int(m.get("influence_type", 2)))

    def _capability(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        s, d, i = self._sdi(graph)
        return combat_operational_capability(
            graph.edge_index, graph.node_type, graph.num_nodes, graph.batch,
            graph.num_graphs, ~selected, s, d, i,
        )

    def _init_state(self, graph: BatchedGraph) -> State:
        state = super()._init_state(graph)
        # initial operational capability per graph → ANOC normalizer
        self._cap0 = self._capability(graph, state["selected"]).clamp_min(1.0)
        return state

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        cap = self._capability(self.graph, self.state["selected"])
        return -cap / self._cap0  # per-step −(normalized capability); Σ = −ANOC

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        return self._capability(graph, state["selected"])


@ENVS.register("cascade_dismantling", aliases=["hypergraph_cascade", "cost_cascade", "c2hd"])
class CascadeHypergraphDismantlingEnv(DismantlingEnv):
    """Cost-constrained **hypergraph dismantling with dynamic cascading failure**
    (Jiang et al., Physica A 2025).

    Each removal seeds a load-redistribution cascade (see
    :func:`~graco.utils.graph_algos.hypergraph_cascade_survivors`): failed nodes
    shed load through their hyperedges onto functional neighbours, which may in
    turn overload and fail.  The reward penalizes the largest connected component
    among the **survivors** of that cascade (on the clique-expansion topology),
    optionally minus a removal cost — so the agent learns to trigger large
    cascades cheaply.  Reuses the dismantling MDP; the bipartite incidence comes
    from the ``hypergraph`` generator's ``meta``.  Initial load is
    ``α·deg^β`` on the clique expansion (a fast, vectorized stand-in for the
    paper's betweenness load) and capacity is ``(1+tolerance)·load``.
    """

    def __init__(self, reward_scale: float = 1.0, max_steps_frac: float = 1.0,
                 alpha: float = 1.0, beta: float = 1.0, tolerance: float = 0.2,
                 cost_weight: float = 0.0, max_rounds: int = 20, lcc_max_iter=None):
        super().__init__(reward_scale, max_steps_frac, cost_mode="unit", lcc_max_iter=lcc_max_iter)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tolerance = float(tolerance)
        self.cost_weight = float(cost_weight)
        self.max_rounds = int(max_rounds)

    def _init_state(self, graph: BatchedGraph) -> State:
        state = super()._init_state(graph)
        deg = graph.degree().clamp_min(1.0)  # clique-expansion degree ~ load centrality
        self._deg = deg
        self._deg_total = scatter_sum(deg, graph.batch, graph.num_graphs).clamp_min(1e-8)
        self._l0 = self.alpha * deg.pow(self.beta)
        self._cap = (1.0 + self.tolerance) * self._l0
        return state

    def _survivors(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        m = graph.meta
        return hypergraph_cascade_survivors(
            m["inc_node"], m["inc_hyperedge"], graph.num_nodes, int(m["num_hyperedges"]),
            self._l0, self._cap, selected, self.max_rounds,
        )

    def _lcc(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        surv = self._survivors(graph, selected)
        return largest_component_size(
            graph.edge_index, graph.num_nodes, graph.batch, graph.num_graphs, surv, self.lcc_max_iter
        )

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        self._mark_selected(action, active)
        lcc = self._lcc(self.graph, self.state["selected"])
        n = self.graph.graph_num_nodes.float().clamp_min(1)
        reward = -lcc / (n * n)
        if self.cost_weight > 0.0:  # cost-constrained: cheaper removals preferred
            w_a = self._deg.index_select(0, action.clamp_min(0))
            reward = reward - self.cost_weight * (w_a / self._deg_total)
        return reward

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        return self._lcc(graph, state["selected"])
