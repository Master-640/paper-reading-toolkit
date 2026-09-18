"""MaxCut and Spin-Glass (Ising) environments.

Both are the DIRAC spin-flip MDP: start every spin at ``+1``; each step flips
one un-flipped node's spin; the episode ends one flip short of the whole graph
(``n_selected + 1 >= n``, the reference off-by-one).  The two problems differ
only in what they optimize:

* **Spin-Glass / Ising** — maximize energy ``H = Σ_{(u,v)} w_uv s_u s_v``.
  Per-flip reward ``ΔH / |E|`` (DIRAC exactly).
* **MaxCut** — maximize the cut ``Σ_{(u,v)} w_uv (1 − s_u s_v)/2`` (partition =
  flipped vs. un-flipped). Reward is ``−ΔH / (2|E|)`` (Δcut = −ΔH/2).

The delta is computed with the reference's flip-first rule: flip the acted
node's spin, then sum ``2·s_new[a]·s[nbr]·w`` over its out-edges — vectorized
across the batch since each graph flips exactly one node.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


def _spin(selected: Tensor) -> Tensor:
    """Spin from the flipped mask: +1 if never flipped, -1 if flipped."""
    return torch.where(selected, -1.0, 1.0)


class _IsingEnv(VectorizedEnv):
    """Shared spin-flip mechanics for MaxCut / Spin-Glass."""

    node_feature_dim = 3
    edge_feature_dim = 4
    maximize = True
    state_spec = {"selected": "node"}

    #: coefficient applied to ΔH/|E| to get the reward (+1 energy, -0.5 cut)
    reward_coeff: float = 1.0
    #: "energy" or "cut" — what :meth:`objective` reports
    objective_kind: str = "energy"

    def _init_state(self, graph: BatchedGraph) -> State:
        return {"selected": torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.device)}

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        chosen = self._mark_selected(action, active)  # sets selected True
        spin = _spin(self.state["selected"])  # [N], flipped nodes now -1
        flipped = torch.zeros(g.num_nodes, dtype=torch.bool, device=g.device)
        flipped[chosen] = True

        src, dst = g.edge_index[0], g.edge_index[1]
        w = g.edge_weight if g.edge_weight is not None else torch.ones(g.num_edges, device=g.device)
        m = flipped[src]  # only out-edges of nodes flipped this step
        val = 2.0 * spin[src] * spin[dst] * w  # s_new at src, s_old at neighbor
        dH = scatter_sum((val * m).double(), g.edge_batch, g.num_graphs).to(torch.float32)
        norm = g.graph_num_undirected_edges.clamp_min(1).float()
        return self.reward_coeff * dH / norm

    # ------------------------------------------------------ pure observation fns
    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        spin = _spin(state["selected"]).unsqueeze(-1)  # [N,1]
        deg = graph.degree()
        if "maxcut_deg_max" not in graph._cache:  # pure topology — cache across the episode
            graph._cache["maxcut_deg_max"] = graph.broadcast_to_nodes(
                graph.pool(deg.unsqueeze(-1), reduce="max")
            ).clamp_min(1)
        deg_max = graph._cache["maxcut_deg_max"]
        deg_norm = (deg.unsqueeze(-1) / deg_max)
        ones = torch.ones_like(spin)
        return torch.cat([spin, deg_norm, ones], dim=-1)  # [N,3]

    @staticmethod
    def edge_features(graph: BatchedGraph, state: State) -> Tensor:
        sel = state["selected"]
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = (
            graph.edge_weight
            if graph.edge_weight is not None
            else torch.ones(graph.num_edges, device=graph.device)
        ).unsqueeze(-1)
        sel_src = sel[src].float().unsqueeze(-1)
        cut_ind = (sel[src] ^ sel[dst]).float().unsqueeze(-1)  # exactly-one-flipped
        ones = torch.ones_like(sel_src)
        return torch.cat([w, sel_src, cut_ind, ones], dim=-1)  # [E,4]

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        return ~state["selected"]  # any un-flipped node

    @staticmethod
    def is_done(graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        n_sel = scatter_sum(state["selected"].long(), graph.batch, graph.num_graphs)
        return (n_sel + 1) >= graph.graph_num_nodes

    @classmethod
    def objective(cls, graph: BatchedGraph, state: State) -> Tensor:
        spin = _spin(state["selected"])
        src, dst = graph.edge_index[0], graph.edge_index[1]
        w = (
            graph.edge_weight
            if graph.edge_weight is not None
            else torch.ones(graph.num_edges, device=graph.device)
        )
        agree = scatter_sum((w * spin[src] * spin[dst]).double(), graph.edge_batch, graph.num_graphs)
        energy = (agree / 2.0).to(torch.float32)  # H = Σ_{undirected} w s_u s_v
        if cls.objective_kind == "cut":
            w_total = scatter_sum(w.double(), graph.edge_batch, graph.num_graphs).to(torch.float32) / 2.0
            return (w_total - energy) / 2.0  # cut = (W - H)/2
        return energy


@ENVS.register("spinglass", aliases=["ising", "spin_glass"])
class SpinGlassEnv(_IsingEnv):
    """Maximize the Ising energy ``H = Σ w_ij s_i s_j`` (DIRAC)."""

    reward_coeff = 1.0
    objective_kind = "energy"


@ENVS.register("maxcut")
class MaxCutEnv(_IsingEnv):
    """Maximize the (weighted) cut of a spin partition."""

    reward_coeff = -0.5  # Δcut = -ΔH/2
    objective_kind = "cut"
