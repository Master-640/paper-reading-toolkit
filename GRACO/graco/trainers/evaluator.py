"""Evaluation: roll out the greedy policy on a fixed validation set and compare
against heuristic baselines."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import Observation, VectorizedEnv
from graco.trainers.baselines import baseline_score_fn
from graco.utils.segment_ops import segment_argmax


@torch.no_grad()
def _rollout(
    env: VectorizedEnv, graph: BatchedGraph, act_fn: Callable[[Observation], Tensor]
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
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
    extra = env.eval_extra_metrics(env.graph, env.state)
    return ret, env.objective(env.graph, env.state), extra


class Evaluator:
    def __init__(
        self,
        env: VectorizedEnv,
        val_graphs: List[BatchedGraph],
        baselines: Optional[List[str]] = None,
    ):
        self.env = env
        self.val_graphs = val_graphs
        self.baselines = baselines or []
        self.metric = getattr(env, "eval_metric", "objective")

    def _run(self, act_fn) -> Dict[str, Tensor]:
        rets, objs = [], []
        extras: Dict[str, List[Tensor]] = {}
        for g in self.val_graphs:
            r, o, ex = _rollout(self.env, g.clone(), act_fn)
            rets.append(r)
            objs.append(o)
            for k, v in ex.items():
                extras.setdefault(k, []).append(v)
        out = {"return": torch.cat(rets), "objective": torch.cat(objs)}
        for k, vs in extras.items():
            out[k] = torch.cat(vs)
        return out

    @torch.no_grad()
    def evaluate(self, algo) -> Dict[str, float]:
        policy = self._run(lambda obs: algo.act(obs, explore=False))
        out = {
            "eval/return": float(policy["return"].mean()),
            "eval/objective": float(policy["objective"].mean()),
            "eval/objective_std": float(policy["objective"].std()),
        }
        # generic per-graph diagnostics surfaced by the env (e.g. a shadow estimator)
        for k, v in policy.items():
            if k not in ("return", "objective"):
                out[f"eval/{k}"] = float(v.mean())
        primary = policy[self.metric].mean()
        for name in self.baselines:
            fn = baseline_score_fn(name, self.env)
            b = self._run(
                lambda obs, fn=fn: segment_argmax(
                    fn(obs), obs.graph.batch, obs.graph.num_graphs, obs.action_mask
                )
            )
            out[f"eval/baseline_{name}_objective"] = float(b["objective"].mean())
            out[f"eval/baseline_{name}_return"] = float(b["return"].mean())
            b_primary = float(b[self.metric].mean())
            denom = b_primary if abs(b_primary) > 1e-8 else 1e-8
            out[f"eval/ratio_vs_{name}"] = float(primary) / denom
        return out
