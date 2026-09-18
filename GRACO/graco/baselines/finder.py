"""Faithful FINDER baseline (Fan et al., Nature Machine Intelligence 2020).

FINDER = GraphSAGE embedding + n-step DQN for network dismantling
(reward = -LCC/n²).  The dismantling reward, terminal condition and batched
connected-components come straight from the framework; the FINDER-specific
pieces provided here are:

* :class:`FinderDismantlingEnv` — constant ``ones[n, 2]`` node inputs plus a
  graph-level ``aux_feat`` (dim 4: removed-fraction, covered-edge-fraction,
  2-hop density, bias) recomputed purely from state so it survives replay.
* :class:`FinderQHead` — the bilinear cross-product Q head
  ``embed_s_a = action_embed · <y_potential, w_cross>`` with the aux features
  concatenated before the final layer.
* :class:`FinderDQN` — DQN wired to the above with FINDER defaults (n_step=5, no
  double-DQN, no bootstrap clip) and the graph **reconstruction loss**
  ``α · Σ_edge ||H_u - H_v||² / |E|`` added via the ``_aux_loss`` hook.

Run it with ``graco train -c finder``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graco.algos.dqn import DQNAlgorithm
from graco.data.batch import BatchedGraph
from graco.envs.base import State
from graco.envs.dismantling import DismantlingEnv
from graco.registries import ALGOS, ENVS, HEADS
from graco.utils.scatter import scatter_sum


@ENVS.register("finder_dismantling", aliases=["finder_env"])
class FinderDismantlingEnv(DismantlingEnv):
    """Dismantling with FINDER's constant node inputs + graph aux features."""

    node_feature_dim = 2  # constant ones[n, 2]
    edge_feature_dim = 0

    @staticmethod
    def node_features(graph: BatchedGraph, state: State) -> Tensor:
        return torch.ones(graph.num_nodes, 2, device=graph.device)

    @staticmethod
    def _aux(graph: BatchedGraph, state: State) -> Tensor:
        """Graph-level readout features [B, 4] (FINDER aux_feat)."""
        removed = state["selected"]
        active = ~removed
        b, n = graph.num_graphs, graph.graph_num_nodes.float().clamp_min(1)
        n_removed = scatter_sum(removed.float(), graph.batch, b)
        src, dst = graph.edge_index[0], graph.edge_index[1]
        covered_edge = (removed[src] | removed[dst]).float()  # directed; ratio is scale-free
        n_edges = graph.graph_num_edges.float().clamp_min(1)
        covered_frac = scatter_sum(covered_edge, graph.edge_batch, b) / n_edges
        # 2-hop density among still-active nodes: sum active_deg*(active_deg-1)/2 / n^2
        active_edge = (active[src] & active[dst]).float()
        active_deg = scatter_sum(active_edge, src, graph.num_nodes)
        twohop = scatter_sum(active_deg * (active_deg - 1.0) / 2.0, graph.batch, b) / (n * n)
        ones = torch.ones(b, device=graph.device)
        return torch.stack([n_removed / n, covered_frac, twohop, ones], dim=1)

    @classmethod
    def obs_graph(cls, graph: BatchedGraph, state: State) -> BatchedGraph:
        bg = super().obs_graph(graph, state)
        bg.meta = {**bg.meta, "aux_feat": cls._aux(graph, state)}
        return bg


@HEADS.register("finder_q")
class FinderQHead(nn.Module):
    """FINDER bilinear cross-product Q head with graph aux features."""

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, aux_dim: int = 4, **kw):
        super().__init__()
        hidden_dim = hidden_dim or max(8, embed_dim // 2)
        self.aux_dim = aux_dim
        self.w_cross = nn.Linear(embed_dim, 1, bias=False)  # <y_potential, w_cross>
        self.h1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.h2 = nn.Linear(hidden_dim + aux_dim, 1, bias=False)

    def forward(self, h: Tensor, graph: BatchedGraph, aux: Optional[Tensor] = None) -> Tensor:
        # y_potential: sum-pool over ACTIVE nodes only (degree>0 on the residual
        # subgraph), then L2-normalize — matches FINDER's unit-vector graph embedding
        # and keeps the bilinear gate scale-stable across the episode.
        active = (graph.degree() > 0).float().unsqueeze(-1)
        y = scatter_sum(h * active, graph.batch, graph.num_graphs)  # [B, H]
        y = F.normalize(y, p=2.0, dim=-1, eps=1e-12)
        scale = self.w_cross(y)  # [B, 1]
        embed_sa = h * graph.broadcast_to_nodes(scale)  # [N, H] bilinear cross-product
        hidden = F.relu(self.h1(embed_sa))  # [N, hidden]
        if self.aux_dim:
            if aux is None:
                aux = h.new_zeros(graph.num_graphs, self.aux_dim)
            hidden = torch.cat([hidden, graph.broadcast_to_nodes(aux)], dim=-1)
        return self.h2(hidden).squeeze(-1)


@ALGOS.register("finder")
class FinderDQN(DQNAlgorithm):
    """FINDER: dismantling DQN with the bilinear head + reconstruction loss."""

    def __init__(self, env, model: Optional[dict] = None, recon_alpha: float = 1e-4, **kw):
        model = dict(model or {})
        model.setdefault("encoder", {"type": "graphsage", "hidden_dim": 64, "num_layers": 3})
        head = dict(model.get("head") or {})
        head.setdefault("type", "finder_q")
        head.setdefault("aux_dim", 4)
        model["head"] = head
        kw.setdefault("n_step", 5)
        kw.setdefault("double", False)
        kw.setdefault("bootstrap_clip_min", None)
        self.recon_alpha = float(recon_alpha)
        super().__init__(env, model=model, **kw)

    def _aux_loss(self, obs_t: BatchedGraph, mask_t: Tensor) -> Tensor:
        if self.recon_alpha <= 0:
            return torch.zeros((), device=self.device)
        h = self.policy.encoder(obs_t)  # [N, H]
        src, dst = obs_t.edge_index[0], obs_t.edge_index[1]
        diff = (h.index_select(0, src) - h.index_select(0, dst)).pow(2).sum(-1)
        edge_num = max(1, obs_t.num_edges)
        return self.recon_alpha * diff.sum() / edge_num
