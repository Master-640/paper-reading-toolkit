"""Sentinel-node-selection environment (see why.md).

An MDP for choosing a small sentinel set S whose readout estimates the global
steady-state mean across M dynamical conditions:

* **state** = ``selected`` mask (the sentinel set so far).
* **action** = add one un-selected node to S.
* **estimator** ``ŷ(S)``: either an **injected learned predictor** (the GLV /
  generalization setting — reads the sentinels' trajectories, extrapolates across
  interaction strengths) or, when none is injected, the analytic **sentinel mean**
  of the per-condition steady activity (transductive baseline, backward-compatible
  with the legacy tanh-``map`` config).
* **error** ``E(S) = mean_m |ŷ_m(S) − y_m| / max(|y_m|, floor)`` over the
  **converged** conditions (``graph_attr['converged']`` when present).
* **reward** = ``E(S_before) − E(S_after) − λ`` (per-pick error improvement minus
  selection cost).  Because ``E`` is a pure function of ``selected`` (the predictor
  is stateless given the graph + mask), the per-episode return telescopes to
  ``E(∅) − E(S_final) − Kλ`` and replay reconstruction is exact — *provided the
  reward predictor is frozen within an episode* (the sentinel_dqn algo does this).
* **terminal** = budget ``K`` reached.  Objective (minimized) = ``E(S)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


@ENVS.register("sentinel", aliases=["sentinel_selection"])
class SentinelSelectionEnv(VectorizedEnv):
    edge_feature_dim = 0
    maximize = False  # minimize estimation error E(S)
    eval_metric = "objective"
    state_spec = {"selected": "node"}
    eps = 1e-6

    def __init__(
        self,
        num_conditions: int = 5,
        budget: int = 10,
        selection_cost: float = 0.0,
        err_floor: float = 0.1,
        reward_scale: float = 1.0,
        max_steps_frac: float = 1.0,
    ):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        self.M = int(num_conditions)
        self.node_feature_dim = self.M + 2  # [a_norm(M), selected, degree]
        self.budget = int(budget)
        self.selection_cost = float(selection_cost)
        self.err_floor = float(err_floor)
        self.predictor = None  # injected by the sentinel_dqn algo (else analytic mean)

    def set_predictor(self, predictor) -> None:
        """Inject the learned readout used to score ``E(S)`` (None => analytic mean)."""
        self.predictor = predictor

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _estimate_analytic(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Transductive baseline: mean of the selected steady values ``[B, M]``."""
        a = graph.node_attr["dyn_mean"]  # [N, M]
        sel = selected.to(a.dtype)
        b = graph.num_graphs
        cnt = scatter_sum(sel, graph.batch, b).clamp_min(1.0).unsqueeze(-1)  # [B,1]
        return scatter_sum(a * sel.unsqueeze(-1), graph.batch, b) / cnt  # [B,M] (empty -> 0)

    def _estimate(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """``ŷ(S)`` per graph/condition ``[B, M]`` (learned predictor if injected)."""
        if self.predictor is not None:
            self.predictor.eval()
            with torch.no_grad():
                return self.predictor(graph, selected)
        return self._estimate_analytic(graph, selected)

    def _rel_error(self, yhat: Tensor, graph: BatchedGraph) -> Tensor:
        """Scale-aware relative error over the converged conditions ``[B]``."""
        y = graph.graph_attr["y"]  # [B, M]
        err = (yhat - y).abs() / y.abs().clamp_min(self.err_floor)  # [B, M]
        conv = graph.graph_attr.get("converged")
        if conv is not None:
            cf = conv.to(err.dtype)
            cnt = cf.sum(dim=1).clamp_min(1.0)  # empty-converged graph -> safe 0 error
            return (err * cf).sum(dim=1) / cnt  # [B]
        return err.mean(dim=1)  # [B]

    def _error(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Error of the active estimator (learned predictor when injected) ``[B]``."""
        return self._rel_error(self._estimate(graph, selected), graph)

    def _error_analytic(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Error of the transductive sentinel-mean, regardless of the injected predictor."""
        return self._rel_error(self._estimate_analytic(graph, selected), graph)

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        e_before = self._error(self.graph, self.state["selected"])
        self._mark_selected(action, active)
        e_after = self._error(self.graph, self.state["selected"])
        return (e_before - e_after) - self.selection_cost

    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        a = graph.node_attr["dyn_mean"]  # [N, M]
        # per-graph/per-condition standardization keeps the policy encoder in-range
        # across regimes (raw magnitudes are OOD under new coupling).
        mean = graph.broadcast_to_nodes(graph.pool(a, reduce="mean"))
        var = graph.broadcast_to_nodes(graph.pool((a - mean) ** 2, reduce="mean"))
        a_norm = (a - mean) / (var + 1e-6).sqrt()
        sel = state["selected"].to(a.dtype).unsqueeze(-1)
        deg = graph.degree()
        deg_max = graph.broadcast_to_nodes(graph.pool(deg.unsqueeze(-1), reduce="max")).clamp_min(1)
        return torch.cat([a_norm, sel, deg.unsqueeze(-1) / deg_max], dim=-1)  # [N, M+2]

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        return ~state["selected"]

    def is_done(self, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        n_sel = scatter_sum(state["selected"].long(), graph.batch, graph.num_graphs)
        no_valid = n_sel >= graph.graph_num_nodes
        return (n_sel >= self.budget) | no_valid

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:
        return self._error(graph, state["selected"])

    def eval_extra_metrics(self, graph: BatchedGraph, state: State):
        """Shadow diagnostic: the transductive sentinel-mean error on the SAME set.

        Reported at eval as ``eval/objective_analytic``; comparing it against
        ``eval/objective`` (the learned predictor) on the disjoint coupling range
        shows whether the learned readout extrapolates where the mean cannot.
        """
        return {"objective_analytic": self._error_analytic(graph, state["selected"])}
