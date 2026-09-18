"""Sentinel dynamics encoder (see why.md): trajectory Transformer + GraphSAGE.

Encodes each node's per-condition activity trajectory ``[N, M, T]`` with a
Transformer over time (shared across conditions, then pooled over conditions) to
a node representation, optionally concatenates the env's steady-mean features
(``graph.x``), and passes the result through an inner GraphSAGE stack for
inductive neighbourhood aggregation.  If no trajectory is present in
``graph.node_attr['traj']`` (e.g. the generator ran with ``store_traj=False``),
it falls back to the env node features — so the same config degrades gracefully
to the steady-mean baseline.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.models.base import GNNEncoder
from graco.registries import ENCODERS


@ENCODERS.register("sentinel_encoder", aliases=["dynamics_transformer"])
class SentinelEncoder(GNNEncoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_dim: int = 0,
        dyn_dim: int = 32,
        n_heads: int = 4,
        transformer_layers: int = 2,
        max_timesteps: int = 256,
        concat_features: bool = True,
        **kwargs,
    ):
        super().__init__(in_dim, hidden_dim, num_layers, edge_dim=edge_dim, **kwargs)
        self.dyn_dim = dyn_dim
        self.concat_features = concat_features
        self.val_embed = nn.Linear(1, dyn_dim)
        self.pos = nn.Parameter(torch.zeros(max_timesteps, dyn_dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dyn_dim, n_heads, dim_feedforward=2 * dyn_dim, batch_first=True, dropout=self.dropout
        )
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)

        inner_in = dyn_dim + (in_dim if concat_features else 0)
        self.inner = ENCODERS.build(
            {"type": "graphsage", "hidden_dim": hidden_dim, "num_layers": num_layers,
             "aggr": "sum", "norm": self.norm_kind},
            in_dim=inner_in, edge_dim=0,
        )
        self.out_dim = self.inner.out_dim

    def _encode_traj(self, traj: Tensor) -> Tensor:
        n, m, t = traj.shape
        v = self.val_embed(traj.reshape(n * m, t, 1))  # [N*M, T, d]
        v = v + self.pos[:t].unsqueeze(0)
        z = self.transformer(v).mean(dim=1)  # pool time -> [N*M, d]
        return z.reshape(n, m, self.dyn_dim).mean(dim=1)  # pool conditions -> [N, d]

    def forward(self, graph: BatchedGraph) -> Tensor:
        traj = graph.node_attr.get("traj")
        if traj is not None:
            h = self._encode_traj(traj.float())
            if self.concat_features and graph.x is not None:
                h = torch.cat([h, graph.x.float()], dim=-1)
        else:  # graceful fallback to env features
            h = self.input_x(graph)
            if self.concat_features:  # keep the inner-encoder input width consistent
                h = torch.cat([h.new_zeros(h.shape[0], self.dyn_dim), h], dim=-1)
        aug = dataclasses.replace(graph, x=h, _cache={})
        return self.inner(aug)
