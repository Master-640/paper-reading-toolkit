"""Heuristic / metaheuristic baselines and their common interface.

Every heuristic implements ``solve(env, graph) -> HeuristicResult`` and declares
which problems it supports.  Two big families:

* **Constructive** (:class:`ScoreHeuristic`) — greedily pick the highest-scoring
  legal node each step and step the env until done (degree, adaptive-degree,
  greedy-gain, min-degree, betweenness, collective-influence, ...). These reuse
  the framework's vectorized rollout, so they run on a whole batch at once.
* **Metaheuristic** (:class:`Heuristic` directly) — simulated annealing, local
  search, tabu, etc. — which operate on a full solution and evaluate it with the
  env's pure ``objective``.

Heuristics register on :data:`HEURISTICS`; the benchmark runner asks each one
``applicable(env)`` and runs every one that fits.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import Observation, VectorizedEnv
from graco.registries import Registry
from graco.utils.segment_ops import segment_argmax

#: registry of all heuristic solvers
HEURISTICS = Registry("heuristic")


@dataclass
class HeuristicResult:
    objective: Tensor  # [B] final objective value per graph
    ret: Tensor  # [B] episode return (sum of step rewards; e.g. -ANC for dismantling)
    seconds: float  # wall-clock for the whole batch


class Heuristic(ABC):
    """Base heuristic. ``supports=None`` means it applies to any node-selection env."""

    name: str = "heuristic"
    supports: Optional[Tuple[type, ...]] = None

    def applicable(self, env: VectorizedEnv, graphs=None) -> bool:
        """Whether this heuristic can run on ``env`` (optionally given the batch).

        ``graphs`` lets size-limited solvers (brute-force / ILP) opt out of large
        instances; most heuristics ignore it.
        """
        return self.supports is None or isinstance(env, self.supports)

    @abstractmethod
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult: ...


class ScoreHeuristic(Heuristic):
    """Constructive heuristic: argmax(score) over valid nodes each step.

    Subclasses implement :meth:`score` (a per-node score ``[N]``); the base runs
    the vectorized rollout over the whole batch and reports the final objective
    and the accumulated return.
    """

    @abstractmethod
    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor: ...

    @torch.no_grad()
    def solve(self, env: VectorizedEnv, graph: BatchedGraph) -> HeuristicResult:
        t0 = time.time()
        obs = env.reset(graph.clone())
        ret = torch.zeros(env.num_envs, device=env.device)
        max_steps = int(graph.graph_num_nodes.max().item()) + 2
        steps = 0
        while not obs.done.all():
            scores = self.score(env, obs)
            action = segment_argmax(scores, obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
            step = env.step(action)
            ret += step.reward
            obs = step.obs
            steps += 1
            if steps > max_steps:
                break
        obj = env.objective(env.graph, env.state)
        return HeuristicResult(objective=obj.detach(), ret=ret.detach(), seconds=time.time() - t0)
