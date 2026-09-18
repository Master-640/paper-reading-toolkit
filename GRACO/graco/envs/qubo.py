"""QUBO — Quadratic Unconstrained Binary Optimization.

Solve ``min_x  x^T Q x`` over ``x ∈ {0,1}^n``.  QUBO is the canonical binary
optimization problem: MaxCut, MVC, MIS, set-partitioning, Ising ground states,
… all reduce to it.  It maps onto a graph exactly:

* the **diagonal** ``Q_ii`` is a per-node **bias / external field** ``b_i``
  (carried in ``graph.node_attr['q_diag']`` — survives replay), and
* the **off-diagonal** ``Q_ij + Q_ji`` are the pairwise **couplings**, stored as
  the (symmetrized, both-directions) edge weights.

so ``x^T Q x = Σ_i b_i x_i + Σ_{i<j} c_ij x_i x_j``.  QUBO is thus an Ising model
*with a field* — which the plain spin-glass env lacks.

This is an **ECO-DQN-style flip-improvement** MDP (the standard DRL formulation
for QUBO/Ising): start from ``x = 0`` and repeatedly **flip** a bit over a fixed
``≈ horizon_frac · n`` horizon (revisitable / non-monotone); the reward is the
best-so-far *decrease* of the energy, normalised by the instance's weight scale.
The observation is a pure function of ``(graph, selected)`` — current bit, the
immediate flip gain ``ΔE``, the node bias, and degree — so replay stays exact;
the trajectory-dependent best-so-far lives only in the reward and ``objective``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum

_EPS = 1e-8


def _bias(graph: BatchedGraph) -> Tensor:
    b = graph.node_attr.get("q_diag")
    return b if b is not None else torch.zeros(graph.num_nodes, device=graph.device)


def _weights(graph: BatchedGraph) -> Tensor:
    return (
        graph.edge_weight
        if graph.edge_weight is not None
        else torch.ones(graph.num_edges, device=graph.device)
    )


def _local_field(graph: BatchedGraph, selected: Tensor) -> Tensor:
    """``f_i = b_i + Σ_{j~i} c_ij x_j`` — the marginal of flipping node ``i`` [N]."""
    x = selected.to(torch.float32)
    src, dst = graph.edge_index[0], graph.edge_index[1]
    return _bias(graph) + scatter_sum(_weights(graph) * x[dst], src, graph.num_nodes)


@ENVS.register("qubo", aliases=["eco_qubo"])
class QUBOEnv(VectorizedEnv):
    node_feature_dim = 4  # [x, flip_gain, bias, degree] (all normalized)
    edge_feature_dim = 0
    maximize = False  # minimize x^T Q x
    eval_metric = "objective"
    state_spec = {"selected": "node"}

    #: horizon = ceil(horizon_frac * n) flips per graph
    horizon_frac: float = 2.0

    def __init__(
        self,
        horizon_frac: float | None = None,
        reward_scale: float = 1.0,
        max_steps_frac: float = 1.0,
        **kwargs,
    ):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        if kwargs:
            import warnings

            warnings.warn(f"{type(self).__name__} ignoring config keys: {sorted(kwargs)}")
        if horizon_frac is not None:
            type(self).horizon_frac = float(horizon_frac)
        self._best: Tensor | None = None  # [B] best (lowest) energy seen
        self._scale: Tensor | None = None  # [B] per-graph energy scale for the reward

    # ------------------------------------------------------------- energy
    @classmethod
    def _energy(cls, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """``E(x) = Σ_i b_i x_i + Σ_{i<j} c_ij x_i x_j`` per graph ``[B]``."""
        x = selected.to(torch.float32)
        b = graph.num_graphs
        src, dst = graph.edge_index[0], graph.edge_index[1]
        lin = scatter_sum((_bias(graph) * x).double(), graph.batch, b)
        quad = 0.5 * scatter_sum((_weights(graph) * x[src] * x[dst]).double(), graph.edge_batch, b)
        return (lin + quad).to(torch.float32)

    # ------------------------------------------------- stateful bits (flip MDP)
    def _init_state(self, graph: BatchedGraph) -> State:
        selected = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)
        self._best = self._energy(graph, selected).detach()  # E(0) = 0
        b = graph.num_graphs
        scale = scatter_sum(_bias(graph).abs().double(), graph.batch, b) + 0.5 * scatter_sum(
            _weights(graph).abs().double(), graph.edge_batch, b
        )
        self._scale = scale.to(torch.float32).clamp_min(_EPS)
        return {"selected": selected}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        chosen = action[active]
        self.state["selected"][chosen] ^= True  # revisitable flip (XOR)
        e = self._energy(self.graph, self.state["selected"])
        new_best = torch.minimum(self._best, e)  # minimization
        reward = (self._best - new_best).clamp_min(0.0) / self._scale  # best-so-far decrease
        self._best = torch.where(active, new_best, self._best)
        return reward

    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        """Best (lowest) energy visited on the live instance (trajectory-dependent)."""
        if self._best is None:
            return self._energy(graph, state["selected"])
        return self._best

    # ------------------------------------------------------ pure observation fns
    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        x = state["selected"].to(torch.float32).unsqueeze(-1)  # [N,1]
        f = _local_field(graph, state["selected"])
        gain = ((1.0 - 2.0 * state["selected"].to(torch.float32)) * f).unsqueeze(-1)  # ΔE if flipped [N,1]
        gain_max = graph.broadcast_to_nodes(graph.pool(gain.abs(), reduce="max")).clamp_min(_EPS)
        b = _bias(graph).unsqueeze(-1)  # [N,1]
        b_max = graph.broadcast_to_nodes(graph.pool(b.abs(), reduce="max")).clamp_min(_EPS)
        deg = graph.degree().unsqueeze(-1)  # [N,1]
        deg_max = graph.broadcast_to_nodes(graph.pool(deg, reduce="max")).clamp_min(1)
        return torch.cat([x, gain / gain_max, b / b_max, deg / deg_max], dim=-1)  # [N, 4]

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        return torch.ones(graph.num_nodes, dtype=torch.bool, device=graph.device)

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        horizon = (graph.graph_num_nodes.float() * cls.horizon_frac).ceil().long()
        return step_count >= horizon
