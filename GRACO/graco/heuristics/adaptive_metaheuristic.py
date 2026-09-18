"""Adaptive meta-heuristic (AMH) dismantling.

Reproduces the **AMH** algorithm of Wu et al., *A reinforcement learning-enhanced
meta-heuristic framework for network dismantling* (J. Phys. Complex. 2025): a
metaheuristic that searches directly over the **removal order** rather than
scoring nodes one at a time.

* **Initial solution** — nodes in descending degree order.
* **Objective** — Schneider ``R`` (the area under the LCC-vs-fraction-removed
  curve), evaluated fast by *reverse* node addition with a union-find, so a whole
  removal curve costs ``O((N+E) α(N))`` instead of ``N`` recomputations.
* **Three neighbourhood operators** on the order — *Swap* (exchange two
  positions), *Shift* (move a node to another position), *Reverse* (reverse a
  random segment).
* **Adaptive operator selection** — each operator keeps success / total counts
  and is drawn with probability proportional to its (Laplace-smoothed) success
  rate; an improving move is accepted, otherwise reverted.

The optimized order is turned into a static per-node priority and replayed
through the env with the shared vectorized rollout (:class:`ScoreHeuristic`), so
the reported return / objective is exactly comparable to every learned agent.
Applies to plain (LCC) network-dismantling envs.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

from graco.envs.base import Observation, VectorizedEnv
from graco.envs.dismantling import CostAwareDismantlingEnv, DismantlingEnv
from graco.heuristics.base import HEURISTICS, ScoreHeuristic


def _adjacency(num_nodes: int, edge_index: np.ndarray) -> List[np.ndarray]:
    """Undirected adjacency lists from a (already both-directions) edge array."""
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src.tolist(), dst.tolist()):
        if s != d:
            adj[s].append(d)
    return [np.asarray(a, dtype=np.int64) for a in adj]


def _schneider_r(adj: List[np.ndarray], order: np.ndarray) -> float:
    """Accumulated LCC (ANC, lower = better) for a removal ``order``.

    Computed by adding nodes back in reverse removal order with a union-find and
    tracking the largest component after each addition; ``LCC`` after removing the
    first ``k`` nodes equals the largest component once the last ``N-k`` nodes are
    present.  The sum runs only over the steps the env actually scores — from the
    shatter prefix (the first point at which the residual graph has an edge)
    onward — so the proxy matches the env's episode return exactly (up to the
    constant ``1/N²`` scale), and every accepted move improves the reported metric.
    """
    n = len(order)
    parent = np.full(n, -1, dtype=np.int64)
    size = np.zeros(n, dtype=np.int64)
    present = np.zeros(n, dtype=bool)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    lcc = 0
    first_edge = None  # first reverse-add step at which two present nodes connect
    add_lcc = np.zeros(n + 1, dtype=np.int64)  # add_lcc[j] = LCC with first j (reverse) present
    for j, v in enumerate(order[::-1], start=1):
        present[v] = True
        parent[v] = v
        size[v] = 1
        cur = 1
        for u in adj[v]:
            if present[u]:
                ru, rv = find(int(u)), find(v)
                if ru != rv:
                    if size[ru] < size[rv]:
                        ru, rv = rv, ru
                    parent[rv] = ru
                    size[ru] += size[rv]
                cur = max(cur, size[find(v)])
        if first_edge is None and cur > 1:
            first_edge = j
        lcc = max(lcc, cur)
        add_lcc[j] = lcc
    if first_edge is None:
        return 0.0  # edgeless graph — nothing to dismantle
    j0 = first_edge - 1  # shatter prefix: last edgeless present-set (LCC floor 1)
    return float(add_lcc[j0:n].sum())


_OPS = ("swap", "shift", "reverse")


def _apply_op(order: np.ndarray, k: int, n: int, rng) -> np.ndarray:
    """Apply neighbourhood operator ``k`` to a removal ``order`` (returns a copy)."""
    cand = order.copy()
    if _OPS[k] == "swap":            # exchange two positions
        i, j = rng.integers(0, n, size=2)
        cand[i], cand[j] = cand[j], cand[i]
    elif _OPS[k] == "shift":         # move a node to another position
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        v = cand[i]
        cand = np.delete(cand, i)
        cand = np.insert(cand, j, v)
    else:                            # reverse a random segment
        i, j = sorted(rng.integers(0, n, size=2).tolist())
        cand[i:j + 1] = cand[i:j + 1][::-1]
    return cand


@HEURISTICS.register("amh", aliases=["adaptive_metaheuristic"])
class AdaptiveMetaHeuristic(ScoreHeuristic):
    """Adaptive meta-heuristic search over the node-removal order (Wu et al. 2025)."""

    name = "amh"

    def __init__(self, iters: int = 400, seed: int = 0):
        self.iters = int(iters)
        self.seed = int(seed)
        self._priority: Tensor | None = None

    def applicable(self, env: VectorizedEnv, graphs=None) -> bool:
        # plain LCC dismantling only (specialized subclasses redefine the objective)
        return type(env) in (DismantlingEnv, CostAwareDismantlingEnv)

    def _search_order(self, adj: List[np.ndarray], deg: np.ndarray, rng) -> np.ndarray:
        n = len(adj)
        order = np.argsort(-deg, kind="stable")  # initial: degree-descending
        if n <= 2:
            return order
        best_r = _schneider_r(adj, order)
        succ = np.ones(3)   # Laplace-smoothed success / total counts
        total = np.ones(3)
        for _ in range(self.iters):
            p = succ / total
            k = int(rng.choice(3, p=p / p.sum()))  # adaptive roulette by success rate
            cand = _apply_op(order, k, n, rng)
            r = _schneider_r(adj, cand)
            total[k] += 1
            if r < best_r:  # improving move accepted
                best_r, order, succ[k] = r, cand, succ[k] + 1
        return order

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph):
        # 1) search a good removal order per graph, build a static per-node priority
        priority = torch.zeros(graph.num_nodes, device=graph.device)
        ei = graph.edge_index.cpu().numpy()
        deg_all = graph.degree().cpu().numpy()
        ptr = graph.ptr.cpu().numpy()
        rng = np.random.default_rng(self.seed)
        for b in range(graph.num_graphs):
            lo, hi = int(ptr[b]), int(ptr[b + 1])
            n = hi - lo
            if n == 0:
                continue
            emask = (ei[0] >= lo) & (ei[0] < hi)
            local_ei = ei[:, emask] - lo
            adj = _adjacency(n, local_ei)
            order = self._search_order(adj, deg_all[lo:hi], rng)
            # earlier in the removal order => higher priority (removed first)
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            priority[lo:hi] = torch.as_tensor(n - rank, dtype=torch.float32, device=graph.device)
        self._priority = priority
        # 2) replay through the shared vectorized rollout for an exact, comparable metric
        return super().solve(env, graph)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        return self._priority  # static priority; segment_argmax masks invalid nodes each step


@HEURISTICS.register("mhrl", aliases=["metaheuristic_rl"])
class MetaHeuristicRL(AdaptiveMetaHeuristic):
    """MHRL — the reinforcement-learning operator-selection variant of AMH.

    Same removal-order search and neighbourhood operators as :class:`AdaptiveMetaHeuristic`,
    but instead of AMH's success-rate roulette the operator is chosen by **tabular
    Q-learning**: the state is the last operator applied (plus a distinct start
    state), the action is which operator to try next, and the reward is the
    relative decrease in Schneider ``R`` the move achieves (0 for a rejected move).
    An ε-greedy policy (ε decaying over iterations) selects operators; the Q-table
    is updated online.  This is the "select the optimal operation based on the
    current state" mechanism of Wu et al. (2025).
    """

    name = "mhrl"

    def __init__(self, iters: int = 400, seed: int = 0, lr: float = 0.5,
                 gamma: float = 0.9, eps_start: float = 0.9, eps_end: float = 0.05):
        super().__init__(iters=iters, seed=seed)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.eps_start = float(eps_start)
        self.eps_end = float(eps_end)

    def _search_order(self, adj: List[np.ndarray], deg: np.ndarray, rng) -> np.ndarray:
        n = len(adj)
        order = np.argsort(-deg, kind="stable")
        if n <= 2:
            return order
        best_r = _schneider_r(adj, order)
        q = np.zeros((4, 3))  # states: 0..2 = last op, 3 = start; actions: 3 operators
        s = 3
        for t in range(self.iters):
            eps = self.eps_end + (self.eps_start - self.eps_end) * (1.0 - t / max(1, self.iters))
            k = int(rng.integers(0, 3)) if rng.random() < eps else int(np.argmax(q[s]))
            cand = _apply_op(order, k, n, rng)
            r = _schneider_r(adj, cand)
            improved = r < best_r
            reward = (best_r - r) / max(best_r, 1e-8) if improved else 0.0
            if improved:
                best_r, order = r, cand
            s_next = k  # next state = the operator just tried
            q[s, k] += self.lr * (reward + self.gamma * q[s_next].max() - q[s, k])
            s = s_next
        return order

