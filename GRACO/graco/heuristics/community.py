"""Community-detection baselines for the ``modularity`` env (two-community).

* :class:`ModularityLocalSearch` — 1-flip hill climbing on the two-community
  modularity ``Q(s)`` (greedy local optimum).
* :class:`ModularitySpectral` — Newman's leading-eigenvector method (PNAS 2006):
  the sign of the top eigenvector of the modularity matrix ``B = A − kkᵀ/2m``.

Both produce a ``±1`` split scored by the env's exact modularity, so they land in
the same benchmark column as the learned policy (a real modularity gap).

(Louvain optimizes the *multi-community* modularity — a different, more general
objective than this 2-community env — so it is not a like-for-like baseline here;
it becomes the natural baseline once a ``k``-community modularity env exists on
top of the k-label action head.)
"""

from __future__ import annotations

import time

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import VectorizedEnv
from graco.envs.modularity import ModularityEnv
from graco.heuristics.base import HEURISTICS, Heuristic, HeuristicResult
from graco.utils.scatter import scatter_sum
from graco.utils.segment_ops import segment_argmax


def modularity_delta(graph: BatchedGraph, s: Tensor) -> Tensor:
    """ΔQ for flipping each node's community, vectorized ``[N]``.

    ``Q = 1/(4m)[Σ_ij A_ij s_i s_j − (1/2m)(Σ_i k_i s_i)²]``; flipping node ``i``
    (``s_i→−s_i``) gives ``Δagree = −4 s_i f_i`` with ``f_i = Σ_{j~i} w_ij s_j``
    and ``Δ(ks²) = −4·ks·k_i s_i + 4 k_i²``.
    """
    b = graph.num_graphs
    src, dst = graph.edge_index[0], graph.edge_index[1]
    w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=graph.device)
    k = graph.degree(weighted=graph.edge_weight is not None).double()
    sd = s.double()
    f = scatter_sum((w.double() * sd[dst]), src, graph.num_nodes)  # f_i
    two_m = scatter_sum(w.double(), graph.edge_batch, b).clamp_min(1e-12)  # 2m
    ks = scatter_sum(k * sd, graph.batch, b)  # Σ k_i s_i
    m2 = two_m.index_select(0, graph.batch)  # [N]
    ksn = ks.index_select(0, graph.batch)  # [N]
    d_agree = -4.0 * sd * f
    d_ks2 = -4.0 * ksn * k * sd + 4.0 * k * k
    return ((d_agree - d_ks2 / m2) / (2.0 * m2)).to(s.dtype)  # [N]


@HEURISTICS.register("modularity_local_search", aliases=["greedy_modularity"])
class ModularityLocalSearch(Heuristic):
    """Greedy 1-flip hill climbing with random restarts.

    From the trivial all-in-one-community split no *single* flip improves Q (moving
    one node out is always unfavorable), so we restart from random ±1 splits and
    keep the best local optimum per graph.
    """

    name = "modularity_local_search"
    supports = (ModularityEnv,)

    def __init__(self, restarts: int = 10, max_iters: int | None = None):
        self.restarts = int(restarts)
        self.max_iters = max_iters

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t0 = time.time()
        n, b = graph.num_nodes, graph.num_graphs
        cap = self.max_iters if self.max_iters is not None else n + 1
        best = torch.full((b,), float("-inf"), device=graph.device)
        for _ in range(self.restarts):
            s = torch.where(torch.rand(n, device=graph.device) < 0.5, 1.0, -1.0)  # random ±1
            for _ in range(int(cap)):
                dQ = modularity_delta(graph, s)
                action = segment_argmax(dQ, graph.batch, b, dQ > 1e-9)  # best improving flip / graph
                flip = action[action >= 0]
                if flip.numel() == 0:
                    break
                s[flip] = -s.index_select(0, flip)
            best = torch.maximum(best, ModularityEnv.objective(graph, {"selected": s < 0}))
        return HeuristicResult(objective=best.detach(), ret=best.detach(), seconds=time.time() - t0)


@HEURISTICS.register("modularity_spectral", aliases=["newman_spectral"])
class ModularitySpectral(Heuristic):
    """Newman leading-eigenvector split of the modularity matrix (dense; small n)."""

    name = "modularity_spectral"
    supports = (ModularityEnv,)

    def __init__(self, max_nodes: int = 600):
        self.max_nodes = int(max_nodes)

    def applicable(self, env, graphs=None) -> bool:
        if not isinstance(env, self.supports):
            return False
        return graphs is None or int(graphs.graph_num_nodes.max()) <= self.max_nodes

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t0 = time.time()
        dev = graph.device
        selected = torch.zeros(graph.num_nodes, dtype=torch.bool, device=dev)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=dev)
        for g in range(graph.num_graphs):
            lo, hi = int(graph.ptr[g]), int(graph.ptr[g + 1])
            n = hi - lo
            if n < 2:
                continue
            em = graph.edge_batch == g
            A = torch.zeros(n, n, device=dev)
            A[(src[em] - lo), (dst[em] - lo)] = w[em]
            k = A.sum(1)
            two_m = k.sum().clamp_min(1e-12)
            B = A - torch.outer(k, k) / two_m  # modularity matrix
            B = (B + B.t()) / 2.0  # symmetric for eigh
            vec = torch.linalg.eigh(B).eigenvectors[:, -1]  # leading eigenvector
            selected[lo:hi] = vec < 0  # sign split (community -1)
        obj = ModularityEnv.objective(graph, {"selected": selected})
        return HeuristicResult(objective=obj.detach(), ret=obj.detach(), seconds=time.time() - t0)
