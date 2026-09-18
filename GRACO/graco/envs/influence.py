"""Influence Maximization environment (Independent Cascade model).

Pick a budget of ``k`` seed nodes to maximize expected spread under the IC model
(each active node activates each inactive neighbour with probability ``p``).  The
per-step reward is the **marginal** expected spread of the newly added seed,
estimated by a fully-vectorized batched Monte-Carlo simulation on GPU.  The
episode return therefore equals the total expected influence of the chosen seed
set.  This is the heaviest environment (its reward runs an IC simulation each
step); reduce ``mc_samples`` for speed.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


@torch.no_grad()
def expected_influence(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    seeds: Tensor,
    p: float,
    mc_samples: int,
    max_iter: int,
) -> Tensor:
    """Expected #activated nodes per graph ``[B]`` under IC from ``seeds`` (bool [N])."""
    device = edge_index.device
    if edge_index.numel() == 0 or seeds.sum() == 0:
        return scatter_sum(seeds.float(), batch, num_graphs)
    src, dst = edge_index[0], edge_index[1]
    e = edge_index.shape[1]
    active = seeds.unsqueeze(1).repeat(1, mc_samples)  # [N, mc]
    for _ in range(int(max_iter)):
        fire = torch.rand(e, mc_samples, device=device) < p  # per-edge, per-sim
        msg = active.index_select(0, src) & fire  # [E, mc]
        incoming = scatter_sum(msg.float(), dst, num_nodes)  # [N, mc]
        newly = (incoming > 0) & (~active)
        if not bool(newly.any()):
            break
        active = active | newly
    per_graph = scatter_sum(active.float(), batch, num_graphs)  # [B, mc]
    return per_graph.mean(dim=1)


@ENVS.register("influence_max", aliases=["influence", "im"])
class InfluenceMaxEnv(VectorizedEnv):
    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = True
    eval_metric = "return"  # total expected influence = sum of marginal rewards
    state_spec = {"selected": "node"}

    def __init__(
        self,
        reward_scale: float = 1.0,
        max_steps_frac: float = 1.0,
        budget: Optional[int] = None,
        budget_frac: float = 0.1,
        activation_prob: float = 0.1,
        mc_samples: int = 16,
        mc_iters: Optional[int] = None,
    ):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        self.budget = budget
        self.budget_frac = budget_frac
        self.p = activation_prob
        self.mc_samples = mc_samples
        self.mc_iters = mc_iters

    def _init_state(self, graph: BatchedGraph) -> State:
        n = graph.graph_num_nodes
        if self.budget is not None:
            self._budget = torch.full_like(n, int(self.budget))
        else:
            self._budget = (n.float() * self.budget_frac).ceil().long().clamp_min(1)
        self._prev_infl = torch.zeros(graph.num_graphs, device=graph.device)
        self._iters = int(self.mc_iters or int(n.max()))
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        self._mark_selected(action, active)
        infl = expected_influence(
            g.edge_index, g.num_nodes, g.batch, g.num_graphs,
            self.state["selected"], self.p, self.mc_samples, self._iters,
        )
        reward = infl - self._prev_infl
        self._prev_infl = torch.where(active, infl, self._prev_infl)
        return reward

    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"].float().unsqueeze(-1)
        deg = graph.degree()
        deg_max = graph.broadcast_to_nodes(graph.pool(deg.unsqueeze(-1), reduce="max")).clamp_min(1)
        return torch.cat([sel, deg.unsqueeze(-1) / deg_max, torch.ones_like(sel)], dim=-1)

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        return ~state["selected"]

    def is_done(self, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        n_sel = scatter_sum(state["selected"].long(), graph.batch, graph.num_graphs)
        return n_sel >= self._budget

    @staticmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        # seed-set size; the meaningful influence value is captured by the return
        return scatter_sum(state["selected"].float(), graph.batch, graph.num_graphs)
