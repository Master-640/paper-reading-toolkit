"""Base class for fully-vectorized combinatorial-optimization environments.

Every environment is a *batched* construction/dismantling MDP: it holds ``B``
graphs at once (one :class:`~graco.data.batch.BatchedGraph`) and, on each
step, selects **one node per graph** — flipping a spin (MaxCut/Ising), adding a
node to a cover (MVC), removing a node (dismantling)...  All ``B`` transitions
happen simultaneously through scatter ops, so a step is a few GPU kernels
regardless of batch size.

Design contract
---------------
The dynamic state of the batch lives in ``self.state``: a dict of tensors, each
tagged in ``state_spec`` as ``"node"`` (shape ``[N]``), ``"graph"`` (``[B]``) or
``"edge"`` (``[E]``).  Crucially, **observations are pure functions of
(graph, state)** — the static methods :meth:`node_features`, :meth:`edge_features`,
:meth:`valid_mask`, :meth:`is_done`, :meth:`objective`.  This purity is what lets
the replay buffer store a tiny per-graph state snapshot and faithfully
reconstruct the full observation (features + action mask) at learning time,
exactly like DIRAC/FINDER's ``PrepareBatchGraph`` rebuilds batches from stored
transitions — but vectorized and on-GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph

State = Dict[str, Tensor]


@dataclass
class Observation:
    """What a policy sees: a graph with dynamic node/edge features + a mask."""

    graph: BatchedGraph
    action_mask: Tensor  # [N] bool, True where the node is a legal action
    done: Tensor  # [B] bool
    step_count: Tensor  # [B] long

    @property
    def num_graphs(self) -> int:
        return self.graph.num_graphs

    @property
    def num_nodes(self) -> int:
        return self.graph.num_nodes


@dataclass
class StepResult:
    obs: Observation
    reward: Tensor  # [B]
    done: Tensor  # [B]
    info: Dict[str, Any] = field(default_factory=dict)


class VectorizedEnv(ABC):
    """Abstract batched environment. See module docstring for the contract."""

    #: number of dynamic node features exposed by :meth:`node_features`
    node_feature_dim: int = 1
    #: number of dynamic edge features exposed by :meth:`edge_features` (0 = none)
    edge_feature_dim: int = 0
    #: larger objective is better? (evaluation reporting only)
    maximize: bool = True
    #: which metric the evaluator ranks on: "objective" or "return" (episode sum
    #: of rewards). Dismantling uses "return" (= -ANC) since terminal objective
    #: is trivially the isolated-node count.
    eval_metric: str = "objective"
    #: dynamic-state layout: name -> "node" | "graph" | "edge"
    state_spec: Dict[str, str] = {"selected": "node"}

    def __init__(self, reward_scale: float = 1.0, max_steps_frac: float = 1.0):
        self.reward_scale = float(reward_scale)
        self.max_steps_frac = float(max_steps_frac)
        self.graph: Optional[BatchedGraph] = None
        self.state: State = {}
        self.step_count: Optional[Tensor] = None  # [B] long
        self.done: Optional[Tensor] = None  # [B] bool

    # --------------------------------------------------------------- properties
    @property
    def num_envs(self) -> int:
        return self.graph.num_graphs if self.graph is not None else 0

    @property
    def device(self) -> torch.device:
        return self.graph.device

    @property
    def selected(self) -> Tensor:
        return self.state["selected"]

    # -------------------------------------------------------------------- reset
    def reset(self, graph: BatchedGraph) -> Observation:
        self.graph = graph
        b = graph.num_graphs
        self.state = self._init_state(graph)
        assert "selected" in self.state, "state must contain a 'selected' node mask"
        self.step_count = torch.zeros(b, dtype=torch.long, device=graph.device)
        self.done = self.is_done(graph, self.state, self.step_count).clone()
        return self.observe()

    # --------------------------------------------------------------------- step
    def step(self, action: Tensor) -> StepResult:
        """Apply one action per graph (``action`` = global node ids ``[B]``).

        ``action[i] < 0`` or an already-``done`` graph is a no-op (reward 0).
        """
        assert self.graph is not None, "call reset() first"
        action = action.to(self.device).long()
        active = (~self.done) & (action >= 0)

        reward = torch.zeros(self.num_envs, device=self.device)
        if active.any():
            r = self._step_core(action, active)
            reward = torch.where(active, r, reward)

        self.step_count = self.step_count + active.long()
        self.done = self.done | self.is_done(self.graph, self.state, self.step_count)
        reward = reward * self.reward_scale
        return StepResult(obs=self.observe(), reward=reward, done=self.done.clone(), info={})

    # ------------------------------------------------------------------ observe
    def observe(self) -> Observation:
        return Observation(
            graph=self.obs_graph(self.graph, self.state),
            action_mask=self.valid_mask(self.graph, self.state),
            done=self.done.clone(),
            step_count=self.step_count.clone(),
        )

    @classmethod
    def obs_graph(cls, graph: BatchedGraph, state: State) -> BatchedGraph:
        """Attach dynamic node/edge features to the (static) topology.

        Shares the source graph's topology ``_cache`` (degree / edge_batch / sparse
        adjacency) so these are computed **once** and reused across every rollout
        step instead of being rebuilt each step — exact and correctness-preserving
        (the cache holds topology-only quantities).
        """
        edge_feats = cls.edge_features(graph, state)
        return BatchedGraph(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            batch=graph.batch,
            ptr=graph.ptr,
            edge_weight=graph.edge_weight,
            edge_attr=edge_feats if edge_feats is not None else graph.edge_attr,
            x=cls.node_features(graph, state),
            node_type=graph.node_type,
            edge_type=graph.edge_type,
            node_attr=graph.node_attr,
            graph_attr=graph.graph_attr,
            meta=graph.meta,
            _cache=graph._cache,  # share topology-only cache (degree/adj/edge_batch)
        )

    # --------------------------------------------------------- shared helpers
    def _mark_selected(self, action: Tensor, active: Tensor) -> Tensor:
        chosen = action[active]
        self.state["selected"][chosen] = True
        return chosen

    def budget_reached(self, graph: BatchedGraph, step_count: Tensor) -> Tensor:
        if self.max_steps_frac >= 1.0:
            return torch.zeros_like(step_count, dtype=torch.bool)
        budget = (graph.graph_num_nodes.float() * self.max_steps_frac).ceil().long()
        return step_count >= budget

    # ------------------------------------------------- subclass: stateful bits
    @abstractmethod
    def _init_state(self, graph: BatchedGraph) -> State:
        """Return the initial dynamic-state dict (must contain 'selected')."""

    @abstractmethod
    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        """Mutate ``self.state`` for ``active`` graphs; return reward ``[B]``."""

    # ------------------------------------------- subclass: pure observation fns
    @staticmethod
    @abstractmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        """Dynamic node features ``[N, node_feature_dim]`` from (graph, state)."""

    @staticmethod
    def edge_features(graph: BatchedGraph, state: State) -> Optional[Tensor]:
        return None

    @staticmethod
    @abstractmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        """Legal-action mask ``[N]`` (bool) from (graph, state)."""

    @staticmethod
    @abstractmethod
    def is_done(graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        """Termination flags ``[B]`` (bool) from (graph, state, step_count)."""

    @staticmethod
    @abstractmethod
    def objective(graph: BatchedGraph, state: State) -> Tensor:
        """Objective value per graph ``[B]`` (for evaluation/metrics)."""

    def eval_extra_metrics(self, graph: BatchedGraph, state: State) -> Dict[str, Tensor]:
        """Optional per-graph diagnostic metrics ``[B]`` reported at eval time.

        Default: none.  Override to surface extra comparisons (e.g. a shadow
        estimator) as ``eval/<key>``; the trainer/evaluator average them over the
        validation set.  Purely additive — envs that don't override are unaffected.
        """
        return {}
