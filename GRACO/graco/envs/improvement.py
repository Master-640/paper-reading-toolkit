"""ECO-DQN-style *flip-improvement* environments (MaxCut / Spin-Glass).

Where the rest of GRACO's spin envs (:mod:`graco.envs.maxcut`) are *monotone
construction* MDPs — each node is flipped at most once and the episode ends
one flip short of the whole graph — this module implements the complementary
**improvement** paradigm of ECO-DQN (Barrett et al., AAAI 2020):

* Start from a *full* configuration (all spins ``+1``) and repeatedly **flip**
  nodes over a fixed horizon (``≈ horizon_frac · n`` steps).
* Flips are **revisitable / non-monotone**: any node may be flipped on any
  step, and flipping it again undoes the previous flip (XOR on the ``selected``
  mask, so ``spin = 1 − 2·selected`` as usual).
* Reward is the classic ECO-DQN *best-so-far* shaping: a step is only rewarded
  when it discovers a configuration better than any seen so far on the instance
  (``max(0, new_best − old_best) / |E|``). This is trajectory-dependent, but it
  is *stored* in replay (never recomputed from the state), so the purity
  contract for observations still holds.

The observation is a pure function of ``(graph, selected)`` exactly like every
other GRACO env: ECO-DQN's replay-safe per-node features are the current spin,
the *immediate flip gain* (``ΔH`` if this node were flipped) normalized per
graph, and the normalized degree. The trajectory-dependent bits (best-so-far)
live only in the reward and in :meth:`objective` (an instance method read off
the live env at eval time), never in ``node_features``.

The per-flip ``ΔH`` mechanics reuse the vectorized formula from
:class:`~graco.envs.maxcut._IsingEnv`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.envs.maxcut import _IsingEnv, _spin
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum

_EPS = 1e-8


class _FlipImprovementEnv(VectorizedEnv):
    """Shared ECO-DQN flip mechanics; subclasses set ``objective_kind``."""

    node_feature_dim = 3
    edge_feature_dim = 0
    maximize = True
    eval_metric = "objective"
    state_spec = {"selected": "node"}

    #: horizon = ceil(horizon_frac * n) flips per graph (ECO-DQN uses ~2n)
    horizon_frac: float = 2.0
    #: "energy" (Ising) or "cut" (MaxCut) — what the objective reports
    objective_kind: str = "cut"

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
            # is_done reads cls.horizon_frac; honor the configured value.
            type(self).horizon_frac = float(horizon_frac)
        self._best: Tensor | None = None  # [B] best objective seen so far

    # -------------------------------------------------- objective (shared calc)
    @classmethod
    def _config_objective(cls, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Objective of the *current full config*, reusing _IsingEnv's formula.

        ``_IsingEnv.objective`` is a classmethod keyed on ``cls.objective_kind``;
        we dispatch it with *our* class so 'cut' vs 'energy' is respected.
        """
        return _IsingEnv.objective.__func__(cls, graph, {"selected": selected})

    # ------------------------------------------------- subclass: stateful bits
    def _init_state(self, graph: BatchedGraph) -> State:
        selected = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)
        # best-so-far initialized to the starting (all-+1) configuration
        self._best = self._config_objective(graph, selected).detach()
        return {"selected": selected}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        chosen = action[active]
        # revisitable flip: XOR the spin (flip again == un-flip)
        self.state["selected"][chosen] ^= True

        obj = self._config_objective(g, self.state["selected"])  # [B] current config
        new_best = torch.maximum(self._best, obj)
        norm = g.graph_num_undirected_edges.clamp_min(1).float()
        reward = (new_best - self._best).clamp_min(0.0) / norm  # ECO-DQN shaping
        # commit the running best only for graphs that actually acted
        self._best = torch.where(active, new_best, self._best)
        return reward

    # ------------------------------------------------ objective (eval, instance)
    def objective(self, graph: BatchedGraph, state: State) -> Tensor:  # type: ignore[override]
        """Best objective seen so far on the live instance (ECO-DQN eval target).

        Instance method by design: eval cares about the best config *visited*
        over the episode, which is trajectory-dependent. Only ever called on the
        live env (trainer/evaluator/baselines), never in replay.
        """
        if self._best is None:
            return self._config_objective(graph, state["selected"])
        return self._best

    # ------------------------------------------------------ pure observation fns
    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        """Replay-safe ECO-DQN features: [spin, flip_gain_norm, degree_norm]."""
        selected = state["selected"]
        spin = _spin(selected)  # [N], +1 / -1
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = (
            graph.edge_weight
            if graph.edge_weight is not None
            else torch.ones(graph.num_edges, device=graph.device)
        )
        # local field h_i = Σ_{j~i} w_ij s_j  ->  ΔH from flipping i is -2 s_i h_i
        field = scatter_sum(w * spin[dst], src, graph.num_nodes)  # [N]
        gain = -2.0 * spin * field  # [N] immediate flip gain (energy delta)
        gain_max = graph.broadcast_to_nodes(
            graph.pool(gain.abs().unsqueeze(-1), reduce="max")
        ).clamp_min(_EPS)
        gain_norm = gain.unsqueeze(-1) / gain_max  # [N,1]

        deg = graph.degree()
        deg_max = graph.broadcast_to_nodes(
            graph.pool(deg.unsqueeze(-1), reduce="max")
        ).clamp_min(1)
        deg_norm = deg.unsqueeze(-1) / deg_max  # [N,1]

        return torch.cat([spin.unsqueeze(-1), gain_norm, deg_norm], dim=-1)  # [N,3]

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        # every node is flippable on every step (revisitable, non-monotone)
        return torch.ones(graph.num_nodes, dtype=torch.bool, device=graph.device)

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        horizon = (graph.graph_num_nodes.float() * cls.horizon_frac).ceil().long()
        return step_count >= horizon


@ENVS.register("eco_maxcut", aliases=["improvement_maxcut"])
class FlipImprovementEnv(_FlipImprovementEnv):
    """ECO-DQN flip-improvement environment for (weighted) MaxCut."""

    objective_kind = "cut"


@ENVS.register("eco_spinglass", aliases=["eco_ising", "improvement_spinglass"])
class SpinGlassFlipImprovementEnv(_FlipImprovementEnv):
    """ECO-DQN flip-improvement environment for the Ising spin-glass energy."""

    objective_kind = "energy"
