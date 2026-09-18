"""Register a custom GNN encoder and use it — no framework fork needed.

Run:  python examples/custom_encoder.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from graco.models.base import GNNEncoder
from graco.registries import ENCODERS
from graco.utils.scatter import scatter_mean


@ENCODERS.register("mean_mlp")
class MeanMLPEncoder(GNNEncoder):
    """A tiny mean-aggregation encoder to demonstrate the extension point."""

    def __init__(self, in_dim, hidden_dim=64, num_layers=3, edge_dim=0, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, edge_dim=edge_dim, **kw)
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers))

    def forward(self, graph):
        src, dst = graph.edge_index[0], graph.edge_index[1]
        h = self.act(self.inp(self.input_x(graph)))
        for lin in self.layers:
            agg = scatter_mean(h[src], dst, graph.num_nodes)
            h = self.act(lin(torch.cat([h, agg], dim=-1)))
        return h


if __name__ == "__main__":
    from graco.trainers import Trainer
    from graco.utils.config import load_config

    cfg = load_config(
        "debug",
        overrides=["algo.model.encoder.type=mean_mlp", "algo.model.encoder.hidden_dim=32"],
    )
    Trainer(cfg).train()
    print("Trained with a custom-registered encoder:", cfg.algo.model.encoder.type)
