"""Inference-time decoding strategies (F3).

A trained policy can be *decoded* several ways to trade compute for quality:

* ``greedy`` — one rollout taking the argmax action each step (the default).
* ``sample`` — draw ``samples`` stochastic rollouts (Boltzmann over the policy's
  per-node scores at ``temperature``) and keep the best per graph.
* ``multistart`` — ``samples`` greedy rollouts on node-permuted copies of each
  graph (augmentation multi-start, à la POMO inference) and keep the best.

All strategies reuse GRACO's vectorized step and disjoint-union batching: the
``samples`` copies of every graph are one big batch, rolled out in lock-step,
then reduced per source graph. Works with any algorithm exposing ``score(obs)``
(DQN Q-values or policy logits).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from graco.data.augment import repeat_graphs
from graco.data.batch import BatchedGraph
from graco.envs.base import VectorizedEnv
from graco.utils.segment_ops import segment_argmax, segment_gumbel_sample


@torch.no_grad()
def _rollout(env: VectorizedEnv, graph: BatchedGraph, act_fn) -> Tuple[Tensor, Tensor]:
    obs = env.reset(graph)
    ret = torch.zeros(env.num_envs, device=env.device)
    max_steps = int(graph.graph_num_nodes.max().item()) + 2
    steps = 0
    while not obs.done.all():
        action = act_fn(obs)
        step = env.step(action)
        ret += step.reward
        obs = step.obs
        steps += 1
        if steps > max_steps:
            break
    return ret, env.objective(env.graph, env.state)


@torch.no_grad()
def decode(
    env: VectorizedEnv,
    graph: BatchedGraph,
    algo,
    strategy: str = "greedy",
    samples: int = 8,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    """Decode ``algo`` on ``graph``; return best ``(objective[B], return[B])`` per graph."""
    B = graph.num_graphs

    if strategy == "greedy":
        k, rep = 1, graph
        def act_fn(obs):
            return segment_argmax(algo.score(obs), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
    elif strategy == "sample":
        k = int(samples)
        rep, _ = repeat_graphs(graph, k, permute=False, generator=generator)
        def act_fn(obs):
            sc = algo.score(obs) / max(1e-6, temperature)
            return segment_gumbel_sample(sc, obs.graph.batch, obs.graph.num_graphs, obs.action_mask, generator)
    elif strategy in ("multistart", "augment"):
        k = int(samples)
        rep, _ = repeat_graphs(graph, k, permute=True, generator=generator)
        def act_fn(obs):
            return segment_argmax(algo.score(obs), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
    else:
        raise ValueError(f"Unknown decode strategy {strategy!r} (greedy|sample|multistart)")

    ret, obj = _rollout(env, rep, act_fn)  # [B*k]
    # copies of each source graph are contiguous (repeat_graphs order) -> [B, k]
    obj2, ret2 = obj.view(B, k), ret.view(B, k)
    metric = getattr(env, "eval_metric", "objective")
    primary = obj2 if metric == "objective" else ret2
    win = primary.argmax(1) if getattr(env, "maximize", True) else primary.argmin(1)
    idx = win.unsqueeze(1)
    return obj2.gather(1, idx).squeeze(1), ret2.gather(1, idx).squeeze(1)
