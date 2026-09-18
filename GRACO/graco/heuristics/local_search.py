"""Deterministic local-search / approximation heuristics.

These operate on a full solution and evaluate it with the env's pure
``objective``, fully vectorized across the batch:

* :class:`GreedyLocalSearchMaxCut` — 1-flip hill climbing to a local optimum.
* :class:`TwoApproxVertexCover` — the classic pick-both-endpoints 2-approx MVC.
"""

from __future__ import annotations

import time

import torch

from graco.data.batch import BatchedGraph
from graco.envs.base import VectorizedEnv
from graco.envs.maxcut import MaxCutEnv, SpinGlassEnv
from graco.envs.mvc import MVCEnv
from graco.envs.qubo import QUBOEnv
from graco.heuristics.base import HEURISTICS, Heuristic, HeuristicResult
from graco.utils.scatter import scatter_min, scatter_sum
from graco.utils.segment_ops import segment_argmax


@HEURISTICS.register("local_search_maxcut")
class GreedyLocalSearchMaxCut(Heuristic):
    """Hill-climbing to a 1-flip local optimum for MaxCut / Spin-Glass.

    Repeatedly computes every node's flip gain in the **maximized** objective
    (energy or cut) via a scatter, flips the single best positive-gain node per
    graph, and stops when no graph has a positive-gain flip left (or after
    ``max_iters``).  The returned solution is therefore a 1-flip local optimum.
    """

    name = "local_search_maxcut"
    supports = (MaxCutEnv, SpinGlassEnv)

    def __init__(self, max_iters: int | None = None):
        self.max_iters = max_iters

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t_start = time.time()
        dev = graph.device
        n, b = graph.num_nodes, graph.num_graphs
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=dev)
        is_cut = getattr(env, "objective_kind", "energy") == "cut"

        s = torch.ones(n, device=dev)  # all +1 (nothing flipped)
        cap = self.max_iters if self.max_iters is not None else n + 1
        for _ in range(int(cap)):
            h = scatter_sum(w * s.index_select(0, dst), src, n)  # local field
            gain = (s * h) if is_cut else (-2.0 * s * h)         # flip gain [N]
            # best strictly-positive-gain node per graph (masked argmax)
            mask = gain > 1e-9
            action = segment_argmax(gain, graph.batch, b, mask)  # [B] global id or -1
            flip = action[action >= 0]
            if flip.numel() == 0:
                break
            s[flip] = -s.index_select(0, flip)

        sel = (s < 0)
        obj = env.objective(graph, {"selected": sel})
        return HeuristicResult(objective=obj.detach(), ret=obj.detach(),
                               seconds=time.time() - t_start)


@HEURISTICS.register("qubo_local_search", aliases=["greedy_qubo"])
class QUBOLocalSearch(Heuristic):
    """1-flip steepest-descent local search for QUBO (``min x^T Q x``).

    From ``x = 0``, repeatedly flip the single bit whose flip most **decreases**
    the energy (``ΔE_i = (1−2x_i)(b_i + Σ_j c_ij x_j)``), stopping at a 1-flip
    local optimum.  Vectorized across the batch; the per-node bias comes from
    ``node_attr['q_diag']`` and couplings from the edge weights.
    """

    name = "qubo_local_search"
    supports = (QUBOEnv,)

    def __init__(self, max_iters: int | None = None):
        self.max_iters = max_iters

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t_start = time.time()
        dev = graph.device
        n, b = graph.num_nodes, graph.num_graphs
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=dev)
        bias = graph.node_attr.get("q_diag")
        bias = bias.to(dev) if bias is not None else torch.zeros(n, device=dev)

        x = torch.zeros(n, device=dev)  # start all-zero
        cap = self.max_iters if self.max_iters is not None else n + 1
        for _ in range(int(cap)):
            field = bias + scatter_sum(w * x.index_select(0, dst), src, n)  # f_i
            dE = (1.0 - 2.0 * x) * field  # energy change if flipped [N]
            mask = dE < -1e-9  # only improving (energy-decreasing) flips
            action = segment_argmax(-dE, graph.batch, b, mask)  # most-negative ΔE per graph
            flip = action[action >= 0]
            if flip.numel() == 0:
                break
            x[flip] = 1.0 - x.index_select(0, flip)

        obj = env._energy(graph, x.bool())  # pure energy (env.objective is best-so-far state)
        return HeuristicResult(objective=obj.detach(), ret=obj.detach(),
                               seconds=time.time() - t_start)


@HEURISTICS.register("mvc_2approx")
class TwoApproxVertexCover(Heuristic):
    """Classic 2-approximation for Minimum Vertex Cover.

    While uncovered edges remain, pick one uncovered edge per graph and add
    **both** its endpoints to the cover.  Building the ``selected`` mask directly
    (bypassing ``env.step``) keeps it a few scatter ops per round; the number of
    rounds is bounded by the maximal-matching size.
    """

    name = "mvc_2approx"
    supports = (MVCEnv,)

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t_start = time.time()
        dev = graph.device
        n, e = graph.num_nodes, graph.num_edges
        src, dst = graph.edge_index[0], graph.edge_index[1]
        edge_batch = graph.edge_batch
        edge_ids = torch.arange(e, device=dev)

        selected = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(n + 1):
            unsel = ~selected
            uncovered = unsel[src] & unsel[dst]  # [E]
            if not bool(uncovered.any()):
                break
            # smallest uncovered edge id per graph (E = "none")
            cand = torch.where(uncovered, edge_ids, torch.full_like(edge_ids, e))
            pick = scatter_min(cand, edge_batch, graph.num_graphs, fill_value=float(e)).long()
            valid = pick < e
            chosen = pick[valid]
            selected[src.index_select(0, chosen)] = True
            selected[dst.index_select(0, chosen)] = True

        obj = env.objective(graph, {"selected": selected})
        return HeuristicResult(objective=obj.detach(), ret=obj.detach(),
                               seconds=time.time() - t_start)
