"""Simulated-annealing metaheuristics.

Unlike the constructive :class:`~graco.heuristics.base.ScoreHeuristic` family,
these operate on a **full solution** — a ``selected`` bool mask per graph — and
evaluate it with the env's pure ``objective``.  Each graph in the batch anneals
independently (its own temperature and its own current/best solution); moves use
incremental delta-energy so a sweep is a handful of scatter ops.

* :class:`SimulatedAnnealingMaxCut` — classic single-spin-flip Metropolis over
  ``s ∈ {+1,-1}^N`` (``selected = (s == -1)``), vectorized across the batch.
* :class:`SimulatedAnnealingMIS` — anneal over an independent set with add/remove
  moves and independence repair (per-graph loop; move maths are ``O(deg)``).
"""

from __future__ import annotations

import math
import random
import time

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import VectorizedEnv
from graco.envs.maxcut import MaxCutEnv, SpinGlassEnv
from graco.envs.mis import MISEnv
from graco.envs.mwis import MWISEnv, _node_weight
from graco.heuristics.base import HEURISTICS, Heuristic, HeuristicResult
from graco.utils.scatter import lengths_to_ptr, scatter_sum


@HEURISTICS.register("sa_maxcut")
class SimulatedAnnealingMaxCut(Heuristic):
    """Single-spin-flip simulated annealing for MaxCut / Spin-Glass.

    Every graph anneals its own spin vector with its own geometric temperature
    schedule.  A *sweep* visits the nodes of each graph in a fresh random order
    and applies Metropolis flips one node per graph at a time; the per-node
    delta of the **maximized** objective (Ising energy, or cut) is read off an
    incrementally-maintained local field ``h_i = Σ_{j∼i} w_ij s_j``.
    """

    name = "sa_maxcut"
    supports = (MaxCutEnv, SpinGlassEnv)

    def __init__(self, sweeps: int = 50, t0: float | None = None,
                 t_end: float | None = None, seed: int | None = None):
        self.sweeps = int(sweeps)
        self.t0 = t0
        self.t_end = t_end
        self.seed = seed

    @staticmethod
    def _local_field(w: Tensor, s: Tensor, src: Tensor, dst: Tensor, n: int) -> Tensor:
        """``h_i = Σ_{j∼i} w_ij s_j`` — sum of neighbour spins over out-edges."""
        return scatter_sum(w * s.index_select(0, dst), src, n)

    def _perm_matrix(self, graph: BatchedGraph, max_n: int, gen: torch.Generator) -> Tensor:
        """``[B, max_n]`` of global node ids in a per-graph random order (-1 = pad)."""
        n, b = graph.num_nodes, graph.num_graphs
        dev = graph.device
        r = torch.rand(n, device=dev, generator=gen)
        # sort primarily by graph id (integer gap), secondarily by the random key
        order = torch.argsort(graph.batch.to(r.dtype) + r)
        pos = torch.empty(n, dtype=torch.long, device=dev)
        pos[order] = torch.arange(n, device=dev)
        rank = pos - graph.ptr.index_select(0, graph.batch)  # 0..n_b-1 within graph
        mat = torch.full((b, max_n), -1, dtype=torch.long, device=dev)
        mat[graph.batch, rank] = torch.arange(n, device=dev)
        return mat

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t_start = time.time()
        dev = graph.device
        n, b = graph.num_nodes, graph.num_graphs
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = graph.edge_weight if graph.edge_weight is not None else torch.ones(graph.num_edges, device=dev)
        is_cut = getattr(env, "objective_kind", "energy") == "cut"

        gen = torch.Generator(device=dev)
        gen.manual_seed(self.seed if self.seed is not None else torch.seed() % (2**31))

        # current spins: all +1 (nothing flipped) and its local field
        s = torch.ones(n, device=dev)
        h = self._local_field(w, s, src, dst, n)

        # per-node delta of the MAXIMIZED objective if node i is flipped
        def deltas() -> Tensor:
            return (s * h) if is_cut else (-2.0 * s * h)

        # temperature schedule (per graph); auto-scale from the initial move sizes
        if self.t0 is None:
            d0 = deltas().abs()
            t0 = (scatter_sum(d0, graph.batch, b) / graph.graph_num_nodes.clamp_min(1).float())
            t0 = t0.clamp_min(1e-3)
        else:
            t0 = torch.full((b,), float(self.t0), device=dev)
        t_end = (t0 * 1e-2) if self.t_end is None else torch.full((b,), float(self.t_end), device=dev)
        t_end = t_end.clamp_min(1e-6)

        # best-so-far starts at the initial all-+1 solution
        def objective(sel: Tensor) -> Tensor:
            return env.objective(graph, {"selected": sel})

        best_sel = (s < 0)
        best_obj = objective(best_sel)

        max_n = int(graph.graph_num_nodes.max().item()) if n > 0 else 0
        denom = max(self.sweeps - 1, 1)
        ratio = t_end / t0
        for k in range(self.sweeps):
            temp = (t0 * ratio.pow(k / denom)).index_select(0, graph.batch)  # [N] broadcast
            perm = self._perm_matrix(graph, max_n, gen)
            for p in range(max_n):
                cand = perm[:, p]                     # [B] global node id or -1
                valid = cand >= 0
                idx = cand.clamp_min(0)
                d = (s.index_select(0, idx) * h.index_select(0, idx))
                d = d if is_cut else (-2.0 * d)       # delta of maximized objective [B]
                tc = temp.index_select(0, idx)
                accept_p = torch.exp(torch.clamp(d / tc, max=0.0))  # 1 when d>=0
                u = torch.rand(b, device=dev, generator=gen)
                accept = valid & (u < accept_p)
                flip = cand[accept]
                if flip.numel() == 0:
                    continue
                # incremental field update: neighbours see Δs = -2 s_old
                ds = torch.zeros(n, device=dev)
                ds[flip] = -2.0 * s.index_select(0, flip)
                h = h + scatter_sum(w * ds.index_select(0, src), dst, n)
                s[flip] = -s.index_select(0, flip)
            # end sweep: keep the best solution seen
            sel = (s < 0)
            obj = objective(sel)
            improve = obj > best_obj
            best_obj = torch.where(improve, obj, best_obj)
            best_sel = torch.where(improve.index_select(0, graph.batch), sel, best_sel)

        final = objective(best_sel)
        return HeuristicResult(objective=final.detach(), ret=final.detach(),
                               seconds=time.time() - t_start)


@HEURISTICS.register("sa_mis")
class SimulatedAnnealingMIS(Heuristic):
    """Simulated annealing over a (weighted) independent set.

    Moves propose adding or removing a single node; adding a node repairs
    independence by evicting its in-set neighbours, so the current solution is
    always a valid independent set.  Metropolis acceptance is on the change in
    the (weighted) set size.  Runs a per-graph Python loop — the move maths are
    only ``O(deg)`` — and reports the batched ``env.objective`` of the best set.
    """

    name = "sa_mis"
    supports = (MISEnv, MWISEnv)

    def __init__(self, sweeps: int = 50, t0: float | None = None,
                 t_end: float | None = None, seed: int | None = None):
        self.sweeps = int(sweeps)
        self.t0 = t0
        self.t_end = t_end
        self.seed = seed

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t_start = time.time()
        dev = graph.device
        n = graph.num_nodes
        src, dst = graph.edge_index[0], graph.edge_index[1]

        # node weights consistent with env.objective (unit for plain MIS)
        if isinstance(env, MWISEnv):
            w = _node_weight(graph, env.weight_mode)
        else:
            w = torch.ones(n, device=dev)
        w_list = w.tolist()

        # CSR adjacency (global ids)
        order = torch.argsort(src)
        sorted_dst = dst.index_select(0, order)
        counts = torch.bincount(src, minlength=n)
        nptr = lengths_to_ptr(counts)
        nptr_l = nptr.tolist()
        dst_l = sorted_dst.tolist()
        ptr_l = graph.ptr.tolist()

        best_selected = torch.zeros(n, dtype=torch.bool, device=dev)
        base_seed = self.seed if self.seed is not None else random.randrange(2**31)

        for b in range(graph.num_graphs):
            lo, hi = ptr_l[b], ptr_l[b + 1]
            n_b = hi - lo
            if n_b == 0:
                continue
            rng = random.Random(base_seed + b)
            in_set = [False] * n_b
            cur = 0.0
            best_val = 0.0
            best = list(in_set)

            # per-graph temperature scale (mean node weight ~ one move's size)
            if self.t0 is None:
                t0 = max(sum(w_list[lo:hi]) / n_b, 1e-3)
            else:
                t0 = float(self.t0)
            t_end = max(t0 * 1e-2 if self.t_end is None else float(self.t_end), 1e-6)
            log_ratio = math.log(t_end / t0)

            total = max(self.sweeps * n_b, 1)
            for t in range(total):
                temp = t0 * math.exp(log_ratio * t / total)
                u = rng.randrange(n_b)              # local node
                gnode = lo + u
                nbrs = dst_l[nptr_l[gnode]:nptr_l[gnode + 1]]
                if in_set[u]:                       # propose removal (always legal)
                    delta = -w_list[gnode]
                    if delta >= 0 or rng.random() < math.exp(delta / temp):
                        in_set[u] = False
                        cur += delta
                else:                               # propose add + evict conflicts
                    conf = [j - lo for j in nbrs if in_set[j - lo]]
                    delta = w_list[gnode] - sum(w_list[lo + j] for j in conf)
                    if delta >= 0 or rng.random() < math.exp(delta / temp):
                        for j in conf:
                            in_set[j] = False
                        in_set[u] = True
                        cur += delta
                if cur > best_val:
                    best_val = cur
                    best = list(in_set)

            if any(best):
                idx = torch.tensor([lo + u for u, v in enumerate(best) if v],
                                   dtype=torch.long, device=dev)
                best_selected[idx] = True

        obj = env.objective(graph, {"selected": best_selected})
        return HeuristicResult(objective=obj.detach(), ret=obj.detach(),
                               seconds=time.time() - t_start)
