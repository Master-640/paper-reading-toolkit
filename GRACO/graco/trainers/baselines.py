"""Heuristic baselines and a shared greedy-rollout evaluator.

Baselines score nodes with a simple rule (random / static degree / adaptive
degree) and run the *same* vectorized env loop as the learned policy, so their
objective values are directly comparable.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import Tensor

from graco.envs.base import Observation, VectorizedEnv
from graco.utils.scatter import scatter_sum

ScoreFn = Callable[[Observation], Tensor]  # obs -> per-node scores [N]


def random_score(obs: Observation) -> Tensor:
    return torch.rand(obs.num_nodes, device=obs.graph.device)


def degree_score(obs: Observation) -> Tensor:
    return obs.graph.degree()


def adaptive_degree_score(obs: Observation) -> Tensor:
    """Degree over the still-active subgraph (uses valid endpoints)."""
    g = obs.graph
    valid = obs.action_mask
    src, dst = g.edge_index[0], g.edge_index[1]
    live = (valid[src] & valid[dst]).float()
    return scatter_sum(live, src, g.num_nodes)


BASELINES: Dict[str, ScoreFn] = {
    "random": random_score,
    "degree": degree_score,
    "adaptive_degree": adaptive_degree_score,
}


def baseline_score_fn(name: str, env: VectorizedEnv | None = None) -> ScoreFn:
    """Resolve a baseline name to a per-node score function.

    Native names (``random`` / ``degree`` / ``adaptive_degree``) are returned
    directly.  Any other name falls back to a registered
    :class:`~graco.heuristics.base.ScoreHeuristic` (e.g. ``corehd``,
    ``collective_influence``, ``betweenness``), so the whole constructive-heuristic
    zoo is usable as an eval baseline with no duplication.
    """
    if name in BASELINES:
        return BASELINES[name]
    from graco.heuristics.base import HEURISTICS, ScoreHeuristic  # lazy: avoid import cycle

    try:
        heuristic = HEURISTICS.get(name)()
    except Exception as exc:  # unknown name
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {sorted(BASELINES)} + registered ScoreHeuristics"
        ) from exc
    if not isinstance(heuristic, ScoreHeuristic):
        raise KeyError(f"Baseline '{name}' is not a per-node ScoreHeuristic")
    return lambda obs: heuristic.score(env, obs)
