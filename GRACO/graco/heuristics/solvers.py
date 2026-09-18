"""Exact / solver baselines for optimality-gap evaluation.

These are *reference* solvers, not learning methods:

* :class:`BruteForceMaxCut` — enumerates all ``2^n`` spin configurations on the
  GPU (vectorized) to get the exact MaxCut / Ising optimum. No dependencies;
  only applicable to small graphs (``n <= max_nodes``).
* :class:`ILPSolver` — an exact integer-program solver (PuLP → CBC by default,
  optional Gurobi backend) for MaxCut, MVC, MIS, Minimum Dominating Set and
  Set Cover; solves each graph to optimality (or a time limit).

Both opt out of large instances via :meth:`applicable`, so the benchmark only
runs them when the graphs are small enough to be worth an exact answer — giving
a true optimality gap for the learned policy and heuristics.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import torch

from graco.data.batch import BatchedGraph
from graco.envs.dominating_set import MinDominatingSetEnv
from graco.envs.maxcut import MaxCutEnv, SpinGlassEnv
from graco.envs.mis import MISEnv
from graco.envs.mvc import MVCEnv
from graco.envs.qubo import QUBOEnv
from graco.envs.setcover import SetCoverEnv
from graco.heuristics.base import HEURISTICS, Heuristic, HeuristicResult


def _undirected_edges(graph: BatchedGraph, g: int) -> Tuple[int, Dict[Tuple[int, int], float]]:
    """Per-graph node count and undirected edge dict ``{(u,v): w}`` with ``u<v``."""
    lo, hi = int(graph.ptr[g]), int(graph.ptr[g + 1])
    n = hi - lo
    em = graph.edge_batch == g
    ei = (graph.edge_index[:, em] - lo).cpu()
    w = graph.edge_weight[em].cpu() if graph.edge_weight is not None else None
    und: Dict[Tuple[int, int], float] = {}
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        und[key] = float(w[k]) if w is not None else 1.0
    return n, und


@HEURISTICS.register("brute_force_maxcut", aliases=["exact_maxcut"])
class BruteForceMaxCut(Heuristic):
    name = "brute_force_maxcut"
    supports = (MaxCutEnv, SpinGlassEnv)

    def __init__(self, max_nodes: int = 16):
        self.max_nodes = int(max_nodes)

    def applicable(self, env, graphs=None) -> bool:
        if not isinstance(env, self.supports):
            return False
        if graphs is None:
            return True
        return int(graphs.graph_num_nodes.max()) <= self.max_nodes

    @torch.no_grad()
    def solve(self, env, graph: BatchedGraph) -> HeuristicResult:
        t0 = time.time()
        dev = graph.device
        B = graph.num_graphs
        cut_mode = getattr(env, "objective_kind", "energy") == "cut"
        objs = torch.zeros(B, device=dev)
        for g in range(B):
            n, und = _undirected_edges(graph, g)
            if n == 0 or not und:
                continue
            configs = 1 << n
            idx = torch.arange(configs, device=dev)
            bits = (idx.unsqueeze(1) >> torch.arange(n, device=dev)) & 1  # [C, n]
            s = 1.0 - 2.0 * bits.float()  # spins in {+1,-1}
            us = torch.tensor([u for (u, _) in und], device=dev)
            vs = torch.tensor([v for (_, v) in und], device=dev)
            w = torch.tensor([und[k] for k in und], device=dev)
            agree = (s[:, us] * s[:, vs] * w).sum(1)  # H per config = sum w s_u s_v
            if cut_mode:
                w_total = w.sum()
                objs[g] = ((w_total - agree) / 2.0).max()  # cut = (W - H)/2
            else:
                objs[g] = agree.max()  # Ising energy
        return HeuristicResult(objective=objs, ret=objs, seconds=time.time() - t0)


@HEURISTICS.register("brute_force_qubo", aliases=["exact_qubo"])
class BruteForceQUBO(Heuristic):
    """Exact QUBO optimum by enumerating all ``2^n`` bit strings (small ``n``).

    ``min_x x^T Q x`` with ``Q``'s diagonal read from ``node_attr['q_diag']`` and
    the off-diagonal couplings from the (symmetrized) edge weights. Vectorized
    over configurations on the GPU; opts out of large instances.
    """

    name = "brute_force_qubo"
    supports = (QUBOEnv,)

    def __init__(self, max_nodes: int = 18):
        self.max_nodes = int(max_nodes)

    def applicable(self, env, graphs=None) -> bool:
        if not isinstance(env, self.supports):
            return False
        if graphs is None:
            return True
        return int(graphs.graph_num_nodes.max()) <= self.max_nodes

    @torch.no_grad()
    def solve(self, env, graph: BatchedGraph) -> HeuristicResult:
        t0 = time.time()
        dev = graph.device
        B = graph.num_graphs
        diag = graph.node_attr.get("q_diag")
        objs = torch.zeros(B, device=dev)
        for g in range(B):
            n, und = _undirected_edges(graph, g)
            lo = int(graph.ptr[g])
            b = diag[lo : lo + n].to(dev) if diag is not None else torch.zeros(n, device=dev)
            idx = torch.arange(1 << n, device=dev)
            bits = ((idx.unsqueeze(1) >> torch.arange(n, device=dev)) & 1).float()  # [C,n] in {0,1}
            energy = bits @ b  # linear term [C]
            if und:
                us = torch.tensor([u for (u, _) in und], device=dev)
                vs = torch.tensor([v for (_, v) in und], device=dev)
                w = torch.tensor([und[k] for k in und], device=dev)
                energy = energy + (bits[:, us] * bits[:, vs] * w).sum(1)  # quadratic term
            objs[g] = energy.min()  # QUBO is minimized
        return HeuristicResult(objective=objs, ret=objs, seconds=time.time() - t0)


@HEURISTICS.register("ilp", aliases=["ilp_exact", "exact_ilp"])
class ILPSolver(Heuristic):
    """Exact ILP baseline (PuLP/CBC, optional Gurobi) for several problems."""

    name = "ilp"
    supports = (MaxCutEnv, SpinGlassEnv, MVCEnv, MISEnv, MinDominatingSetEnv, SetCoverEnv)

    def __init__(self, max_nodes: int = 120, time_limit: float = 10.0, backend: str = "cbc"):
        self.max_nodes = int(max_nodes)
        self.time_limit = float(time_limit)
        self.backend = backend

    def applicable(self, env, graphs=None) -> bool:
        if not isinstance(env, self.supports):
            return False
        try:
            import pulp  # noqa: F401
        except Exception:
            return False
        if graphs is None:
            return True
        return int(graphs.graph_num_nodes.max()) <= self.max_nodes

    def _solver(self):
        import pulp

        if self.backend == "gurobi":
            try:
                return pulp.GUROBI(msg=0, timeLimit=self.time_limit)
            except Exception:
                pass
        return pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)

    @torch.no_grad()
    def solve(self, env, graph: BatchedGraph) -> HeuristicResult:
        import pulp

        t0 = time.time()
        B = graph.num_graphs
        objs = torch.zeros(B)
        solver = self._solver()
        for g in range(B):
            n, und = _undirected_edges(graph, g)
            objs[g] = self._solve_one(env, graph, g, n, und, pulp, solver)
        objs = objs.to(graph.device)
        return HeuristicResult(objective=objs, ret=objs, seconds=time.time() - t0)

    def _solve_one(self, env, graph, g, n, und, pulp, solver) -> float:
        edges = list(und.items())
        if isinstance(env, (MaxCutEnv, SpinGlassEnv)):
            prob = pulp.LpProblem("cut", pulp.LpMaximize)
            x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)}
            y = {}
            for e, ((u, v), w) in enumerate(edges):
                ye = pulp.LpVariable(f"y{e}", cat="Binary")
                y[e] = ye
                prob += ye <= x[u] + x[v]
                prob += ye <= 2 - x[u] - x[v]
                prob += ye >= x[u] - x[v]
                prob += ye >= x[v] - x[u]
            wsum = sum(w for (_, w) in edges)
            cut = pulp.lpSum(w * y[e] for e, ((_, _), w) in enumerate(edges))
            if getattr(env, "objective_kind", "energy") == "cut":
                prob += cut
                prob.solve(solver)
                return float(pulp.value(cut))
            prob += wsum - 2 * cut  # Ising energy H = W - 2*cut
            prob.solve(solver)
            return float(wsum - 2 * pulp.value(cut))
        if isinstance(env, MVCEnv):
            prob = pulp.LpProblem("mvc", pulp.LpMinimize)
            x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)}
            for (u, v), _ in edges:
                prob += x[u] + x[v] >= 1
            prob += pulp.lpSum(x.values())
            prob.solve(solver)
            return float(pulp.value(prob.objective))
        if isinstance(env, MISEnv):
            prob = pulp.LpProblem("mis", pulp.LpMaximize)
            x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)}
            for (u, v), _ in edges:
                prob += x[u] + x[v] <= 1
            weight = getattr(env, "_w", None)
            if weight is not None:  # MWIS
                lo = int(graph.ptr[g])
                prob += pulp.lpSum(float(weight[lo + i]) * x[i] for i in range(n))
            else:
                prob += pulp.lpSum(x.values())
            prob.solve(solver)
            return float(pulp.value(prob.objective))
        if isinstance(env, MinDominatingSetEnv):
            prob = pulp.LpProblem("mds", pulp.LpMinimize)
            x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)}
            nbr = {i: [i] for i in range(n)}
            for (u, v), _ in edges:
                nbr[u].append(v)
                nbr[v].append(u)
            for i in range(n):
                prob += pulp.lpSum(x[j] for j in nbr[i]) >= 1
            prob += pulp.lpSum(x.values())
            prob.solve(solver)
            return float(pulp.value(prob.objective))
        if isinstance(env, SetCoverEnv):
            lo = int(graph.ptr[g])
            nt = graph.node_type
            is_set = {i: (nt is not None and int(nt[lo + i]) == 1) for i in range(n)}
            cover = {i: [] for i in range(n) if not is_set[i]}  # element -> covering sets
            for (u, v), _ in edges:
                s_node, e_node = (u, v) if is_set[u] else (v, u)
                if is_set[s_node] and not is_set[e_node]:
                    cover.setdefault(e_node, []).append(s_node)
            prob = pulp.LpProblem("setcover", pulp.LpMinimize)
            x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n) if is_set[i]}
            for e_node, sets in cover.items():
                if sets:
                    prob += pulp.lpSum(x[s] for s in sets) >= 1
            prob += pulp.lpSum(x.values())
            prob.solve(solver)
            return float(pulp.value(prob.objective))
        return float("nan")
