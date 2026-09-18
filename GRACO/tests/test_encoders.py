import pytest
import torch

import graco.models  # noqa: F401  (register encoders + heads)
from graco.data.batch import BatchedGraph
from graco.registries import ENCODERS

ENCODER_NAMES = [
    "gcn", "graphsage", "gin", "gine", "gat", "gatv2", "s2v",
    "gatedgcn", "pna", "mpnn", "nnconv", "graph_transformer", "edgeconv",
    "gcnii", "appnp", "sgc", "chebnet", "arma", "tagcn",
    "han", "typed_mpnn", "multiplex_gcn", "hypergcn", "hnhn", "rgcn",
]


def _batch(edge_dim=0):
    ei1 = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    ei2 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    bg = BatchedGraph.from_graph_list(
        [ei1, ei2], [3, 4], [torch.ones(6), torch.ones(6)]
    )
    bg.x = torch.randn(bg.num_nodes, 4)
    if edge_dim:
        bg.edge_attr = torch.randn(bg.num_edges, edge_dim)
    return bg


@pytest.mark.parametrize("name", ENCODER_NAMES)
def test_encoder_forward_shape_and_grad(name):
    bg = _batch(edge_dim=3)
    enc = ENCODERS.build({"type": name, "hidden_dim": 16, "num_layers": 2}, in_dim=4, edge_dim=3)
    out = enc(bg)
    assert out.shape == (bg.num_nodes, enc.out_dim)
    assert torch.isfinite(out).all()
    # gradient flows to encoder params under a non-invariant loss
    loss = (out * torch.randn_like(out)).sum()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)


@pytest.mark.parametrize("name", ["gcn", "gat", "s2v", "gin"])
def test_encoder_without_edge_features(name):
    bg = _batch(edge_dim=0)
    enc = ENCODERS.build({"type": name, "hidden_dim": 16, "num_layers": 2}, in_dim=4, edge_dim=0)
    out = enc(bg)
    assert out.shape == (bg.num_nodes, enc.out_dim)
    assert torch.isfinite(out).all()


def test_graphsage_sum_encodes_degree():
    # regression for the FINDER audit: with constant node inputs, SUM aggregation
    # must let the GNN distinguish nodes of different degree (MEAN collapses them,
    # which made the dismantling agent degree-blind). Star graph: hub vs leaf.
    n = 6
    ei = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
    bg = BatchedGraph.from_graph_list([ei], [n])
    bg.x = torch.ones(n, 2)
    enc = ENCODERS.build(
        {"type": "graphsage", "aggr": "sum", "hidden_dim": 8, "num_layers": 2,
         "norm": "none", "residual": False},
        in_dim=2, edge_dim=0,
    )
    out = enc(bg)
    assert not torch.allclose(out[0], out[1], atol=1e-4)  # hub != leaf


def test_isolated_node_safe():
    # graph with an isolated node (node 2 has no edges)
    ei = torch.tensor([[0, 1], [1, 0]])
    bg = BatchedGraph.from_graph_list([ei], [3])
    bg.x = torch.randn(3, 4)
    for name in ["gcn", "graphsage", "s2v", "gat"]:
        enc = ENCODERS.build({"type": name, "hidden_dim": 8, "num_layers": 2}, in_dim=4, edge_dim=0)
        out = enc(bg)
        assert torch.isfinite(out).all()


def test_pe_wrapper_encoder():
    bg = _batch(edge_dim=0)
    enc = ENCODERS.build(
        {"type": "pe", "pe_types": ["degree", "neighbor_degree", "two_hop"], "lap_k": 2,
         "encoder": {"type": "gat", "num_heads": 2}, "hidden_dim": 16, "num_layers": 2},
        in_dim=4, edge_dim=0,
    )
    out = enc(bg)
    assert out.shape == (bg.num_nodes, enc.out_dim)
    assert torch.isfinite(out).all()
    assert enc.pe_dim == 1 + 2 + 1 + 2  # degree + neighbor(2) + two_hop + lap_k
