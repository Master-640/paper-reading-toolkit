"""Balanced graph partitioning / minimum bisection.

Split the nodes into two groups minimizing the number (or weight) of edges cut
**subject to the two sides being balanced** — the core kernel of parallel-mesh
partitioning, VLSI/chip layout and domain decomposition (a graph-partitioning
problem in the spirit of Nature 2021's learned chip-placement work).  With
membership spins ``s_i ∈ {+1,−1}`` the cut is ``½ Σ_{(u,v)} w_uv (1 − s_u s_v)``
and the imbalance is ``(Σ_i s_i)²``; we minimize

    cost(s) = cut(s) + (balance_penalty / n) · (Σ_i s_i)²

The imbalance term is dense but rank-1 → computed in ``O(N+E)`` from the cut
edge-sum plus one per-graph scalar ``(Σ_i s_i)²`` (no dense matrix).  MDP: the
spin-flip construction (flip each node at most once, end one short); per-flip
reward is ``−Δcost`` (telescopes to ``cost(∅) − cost(final)``).  Objective =
``cost`` (minimized).  ``report='cut'`` makes :meth:`objective` report the raw
cut instead of the penalized cost.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


def _spin(selected: Tensor) -> Tensor:
    return torch.where(selected, -1.0, 1.0)


@ENVS.register("balanced_partition", aliases=["min_bisection", "graph_partition"])
class BalancedPartitionEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = False  # minimize cut + imbalance penalty
    eval_metric = "objective"
    state_spec = {"selected": "node"}

    def __init__(self, balance_penalty: float = 1.0, report: str = "cost",
                 reward_scale: float = 1.0, max_steps_frac: float = 1.0):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        self.balance_penalty = float(balance_penalty)
        self.report = report  # "cost" (penalized) or "cut" (raw cut)

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _cut(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        s = _spin(selected)
        b = graph.num_graphs
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=graph.device)
        # Σ_directed w (1 − s_u s_v)/2 = 2·cut  ->  cut = ¼ Σ_directed w (1 − s_u s_v)
        agree = scatter_sum((w * s[src] * s[dst]).double(), graph.edge_batch, b)
        w_tot = scatter_sum(w.double(), graph.edge_batch, b)
        return ((w_tot - agree) / 4.0).to(torch.float32)  # undirected cut weight [B]

    def _cost(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        s = _spin(selected)
        b = graph.num_graphs
        imbalance = scatter_sum(s.double(), graph.batch, b).pow(2).to(torch.float32)  # (Σ s_i)²
        n = graph.graph_num_nodes.clamp_min(1).float()
        return self._cut(graph, selected) + (self.balance_penalty / n) * imbalance

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        c_before = self._cost(self.graph, self.state["selected"])
        self._mark_selected(action, active)
        c_after = self._cost(self.graph, self.state["selected"])
        return c_before - c_after  # −Δcost (reward for reducing cost)

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:
        if self.report == "cut":
            return self._cut(graph, state["selected"])
        return self._cost(graph, state["selected"])

    # ------------------------------------------------------ pure observation fns
    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        spin = _spin(state["selected"]).unsqueeze(-1)
        deg = graph.degree()
        deg_max = graph.broadcast_to_nodes(graph.pool(deg.unsqueeze(-1), reduce="max")).clamp_min(1)
        return torch.cat([spin, deg.unsqueeze(-1) / deg_max, torch.ones_like(spin)], dim=-1)  # [N,3]

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        return ~state["selected"]

    @staticmethod
    def is_done(graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        n_sel = scatter_sum(state["selected"].long(), graph.batch, graph.num_graphs)
        return (n_sel + 1) >= graph.graph_num_nodes
