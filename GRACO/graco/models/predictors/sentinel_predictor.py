"""Learned sentinel predictor (see why.md, redesign).

Replaces the *transductive* "sentinel mean" estimator — which only holds inside
the trained interaction-strength range — with a network that reads **only the
selected sentinels' time series** and predicts the global steady-state mean, so
it can generalize to brand-new coupling values on the same system.

Pipeline, per condition ``m``:

1. **Regime inference (temporal Transformer).** Each selected sentinel's series
   ``traj[i, m, :]`` is *shape-normalised* (subtract its own level, divide by its
   own robust scale) and passed through a Transformer over time; the trajectory
   *shape* (rise time, overshoot, plateau) encodes the dynamical regime, which is
   what lets the model extrapolate rather than memorise a value.  The removed
   ``level`` is fed back through a separate linear (the global mean is a function
   of absolute magnitude), and the sentinel's log-degree (a deployment-known
   topology summary) is added as context.
2. **Permutation-invariant set pooling (PMA).** A single learned query attends
   over the selected tokens *within each graph* via segment-softmax, so the
   output is invariant to sentinel ordering and handles any ``|S|``.  The empty
   set maps to a learned embedding (this defines ``E(∅))``.
3. **Head.** An MLP maps the pooled vector to a scalar ``ŷ[b, m]``.

There is **no condition-slot embedding** (the slot index is re-randomised every
sample, so it would only inject a train-domain prior) and **no coupling input**
(that would defeat the point) — the regime is inferred from the series alone.

**Residual mode** (default): rather than predicting the absolute global mean from
scratch (which forces the network to extrapolate the magnitude and can do *worse*
than a plain average when the sentinels are unbiased), the network predicts a
*correction* to the analytic sentinel-mean anchor:

    ŷ(S) = mean_{i∈S} steady_i  +  g_φ(sentinel trajectories)

The correction head is zero-initialised, so training starts exactly at the
transductive baseline and only has to learn the coupling-dependent bias — the
predictor is never worse than the mean when there is no bias to correct, and
corrects it when there is (e.g. the hub regime).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.utils.scatter import scatter_softmax, scatter_sum


class SentinelPredictor(nn.Module):
    def __init__(
        self,
        num_conditions: int,
        d: int = 64,
        n_heads: int = 4,
        tf_layers: int = 2,
        max_timesteps: int = 256,
        dropout: float = 0.0,
        use_topo: bool = True,
        residual: bool = True,
        scale_eps: float = 1e-3,
        level_floor: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.M = int(num_conditions)
        self.d = int(d)
        self.use_topo = bool(use_topo)
        self.residual = bool(residual)
        self.scale_eps = float(scale_eps)
        self.level_floor = float(level_floor)

        self.val_embed = nn.Linear(1, d)  # per-timestep value -> token
        self.pos = nn.Parameter(torch.zeros(max_timesteps, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=2 * d, batch_first=True, dropout=dropout
        )
        self.time_tf = nn.TransformerEncoder(layer, tf_layers)
        self.level_embed = nn.Linear(1, d)  # absolute magnitude (needed for the mean)
        self.topo_embed = nn.Linear(1, d) if self.use_topo else None

        # permutation-invariant pooling over the selected set (PMA)
        self.q = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.q, std=0.02)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.empty = nn.Parameter(torch.zeros(d))  # E(empty set) representation
        nn.init.normal_(self.empty, std=0.02)

        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        if self.residual:
            # start at the analytic sentinel-mean anchor (correction ≈ 0), so the
            # learned readout is never worse than the transductive baseline and only
            # has to learn the coupling-dependent bias.
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    # ---------------------------------------------------------------- encoding
    def _encode_series(self, shape: Tensor) -> Tensor:
        """``shape`` [P, M, T] (level-removed) -> tokens [P, M, d]."""
        p, m, t = shape.shape
        v = self.val_embed(shape.reshape(p * m, t, 1)) + self.pos[:t].unsqueeze(0)
        z = self.time_tf(v).mean(dim=1)  # pool over time -> [P*M, d]
        return z.reshape(p, m, self.d)

    def _node_tokens(self, graph, idx, anchor_sel=None, scale_sel=None):
        """Per-(selected node, condition) token [P, M, d].

        The trajectory *shape* (regime) is scale-invariant by construction.  The
        level feature is the node's **steady value** (``dyn_mean`` — same quantity
        as the anchor, no transient), and in residual mode it is fed as a
        **scale-invariant deviation from the sentinel-set anchor** ``(steady −
        anchor)/scale`` rather than an absolute magnitude — so nothing that drifts
        monotonically with coupling enters the correction head.
        """
        traj = graph.node_attr.get("traj")
        if traj is None:  # graceful fallback: steady mean as a length-1 series
            traj = graph.node_attr["dyn_mean"].unsqueeze(-1)
        traj = torch.nan_to_num(traj.float(), nan=0.0, posinf=0.0, neginf=0.0)
        sel = traj.index_select(0, idx)  # [P, M, T]
        shape = (sel - sel.mean(dim=-1, keepdim=True)) / sel.std(dim=-1, keepdim=True).clamp_min(self.scale_eps)
        z = self._encode_series(shape)  # [P, M, d] — scale-invariant regime signature
        steady = graph.node_attr["dyn_mean"].index_select(0, idx).float()  # [P, M] steady value
        if anchor_sel is not None:  # residual: scale-invariant deviation from the set mean
            level_feat = (steady - anchor_sel) / scale_sel
        else:  # absolute (non-residual) mode
            level_feat = steady
        z = z + self.level_embed(level_feat.unsqueeze(-1))
        if self.topo_embed is not None:
            deg = graph.degree().index_select(0, idx).clamp_min(0.0).float()  # [P]
            z = z + self.topo_embed(torch.log1p(deg).unsqueeze(-1)).unsqueeze(1)  # [P,1,d]
        return z

    # ---------------------------------------------------------------- forward
    def _anchor(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Analytic sentinel-mean of the observed steady values ``[B, M]`` (residual base)."""
        a = graph.node_attr["dyn_mean"]  # [N, M] — observable at the sentinels
        sel = selected.to(a.dtype)
        B = graph.num_graphs
        cnt = scatter_sum(sel, graph.batch, B).clamp_min(1.0).unsqueeze(-1)  # [B,1]
        return scatter_sum(a * sel.unsqueeze(-1), graph.batch, B) / cnt  # [B, M]

    def forward(self, graph: BatchedGraph, selected: Tensor) -> Tensor:
        """Return the predicted global steady mean per graph/condition ``[B, M]``.

        Residual mode: ``ŷ = anchor + scale · g_φ`` where ``anchor`` is the analytic
        sentinel-mean and ``scale = |anchor|`` (floored). Both the level input and
        the correction output are scale-relative, so the correction transfers across
        coupling regimes instead of being locked to the training magnitude.
        """
        B, M, d = graph.num_graphs, self.M, self.d
        if self.residual:
            anchor = self._anchor(graph, selected)  # [B, M]
            scale = anchor.abs().clamp_min(self.level_floor)  # [B, M] regime magnitude
        idx = selected.bool().nonzero(as_tuple=False).flatten()  # [P] selected node ids
        if idx.numel() == 0:  # every set empty -> learned empty embedding
            corr = self.head(self.empty.view(1, 1, d).expand(B, M, d)).squeeze(-1)
            return anchor + scale * corr if self.residual else corr

        bsel = graph.batch.index_select(0, idx)  # [P] owning graph of each selected node
        anchor_sel = anchor.index_select(0, bsel) if self.residual else None
        scale_sel = scale.index_select(0, bsel) if self.residual else None
        z = self._node_tokens(graph, idx, anchor_sel, scale_sel)  # [P, M, d]
        scores = (self.k_proj(z) * self.q.view(1, 1, d)).sum(-1) / (d ** 0.5)  # [P, M]
        alpha = scatter_softmax(scores, bsel, B)  # segment-softmax over S within each graph
        pooled = scatter_sum(alpha.unsqueeze(-1) * self.v_proj(z), bsel, B)  # [B, M, d]
        has_sel = scatter_sum(torch.ones_like(bsel, dtype=z.dtype), bsel, B) > 0  # [B]
        pooled = torch.where(has_sel.view(B, 1, 1), pooled, self.empty.view(1, 1, d))
        corr = self.head(pooled).squeeze(-1)  # [B, M]
        return anchor + scale * corr if self.residual else corr  # [B, M]
