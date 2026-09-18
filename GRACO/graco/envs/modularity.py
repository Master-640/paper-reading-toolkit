"""Community detection by modularity maximization.

Modularity (Newman & Girvan, PRE 2004; Newman, PNAS 2006) is *the* objective of
community detection in network science — maximizing it is NP-hard.  For a split
into two communities with membership spins ``s_i ∈ {+1, −1}`` it is

    Q(s) = 1/(4m) · Σ_ij [ A_ij − k_i k_j / 2m ] s_i s_j
         = 1/(4m) · [ Σ_ij A_ij s_i s_j − (1/2m)(Σ_i k_i s_i)² ]

i.e. an Ising energy on the **modularity matrix** ``B = A − kkᵀ/2m``.  The
``kkᵀ/2m`` term is dense but rank-1, so ``Q`` is computed in ``O(N+E)`` from an
edge term (``Σ`` over edges of ``w·s_u·s_v``) plus one per-graph scalar
``(Σ_i k_i s_i)²`` — no dense matrix is ever formed, so it scales like every
other GRACO env.

MDP: the DIRAC spin-flip construction — start every node in community ``+1`` and
flip nodes (each at most once) until one short of the whole graph; per-flip
reward is ``ΔQ`` (telescopes to the final ``Q``, since ``Q`` of an all-one
partition is 0).  Weighted graphs use the weighted modularity (``k`` = weighted
degree, ``2m`` = total edge weight).  Best on graphs *with* community structure —
pair it with the ``sbm`` (stochastic block model) generator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum

_EPS = 1e-12


def _spin(selected: Tensor) -> Tensor:
    return torch.where(selected, -1.0, 1.0)


@ENVS.register("modularity", aliases=["community", "community_detection"])
class ModularityEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = True
    eval_metric = "objective"
    state_spec = {"selected": "node"}

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    @staticmethod
    def _modularity(graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Two-community modularity ``Q(s)`` per graph ``[B]`` (weighted)."""
        s = _spin(selected)
        b = graph.num_graphs
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=graph.device)
        k = graph.degree(weighted=graph.edge_weight is not None)  # [N]
        two_m = scatter_sum(w.double(), graph.edge_batch, b)  # = 2m per graph
        agree = scatter_sum((w * s[src] * s[dst]).double(), graph.edge_batch, b)  # Σ_ij A_ij s_i s_j
        ks = scatter_sum((k * s).double(), graph.batch, b)  # Σ_i k_i s_i
        q = (agree - ks.pow(2) / two_m.clamp_min(_EPS)) / (2.0 * two_m.clamp_min(_EPS))
        return torch.where(two_m > 0, q, torch.zeros_like(q)).to(torch.float32)

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        q_before = self._modularity(self.graph, self.state["selected"])
        self._mark_selected(action, active)
        q_after = self._modularity(self.graph, self.state["selected"])
        return q_after - q_before  # ΔQ (telescopes to final Q; Q(all-+1)=0)

    @classmethod
    def objective(cls, graph: BatchedGraph, state: State) -> Tensor:
        return cls._modularity(graph, state["selected"])

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
