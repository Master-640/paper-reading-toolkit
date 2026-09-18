import pytest
import torch

import graco.generators  # noqa: F401
from graco.registries import GENERATORS

SINGLE = ["er", "ba", "ws", "complete", "lattice", "regular", "rgg", "sbm", "tree",
          "powerlaw_cluster", "caveman"]


@pytest.mark.parametrize("name", SINGLE)
def test_generator_builds_valid_batch(name):
    gen = GENERATORS.build({"type": name, "num_nodes": [16, 24]})
    bg = gen.sample(4, device="cpu")
    assert bg.num_graphs == 4
    assert bg.num_nodes > 0
    # edges never cross graphs
    if bg.num_edges:
        assert bool((bg.batch[bg.edge_index[0]] == bg.batch[bg.edge_index[1]]).all())
        assert int(bg.edge_index.max()) < bg.num_nodes


def test_weighted_generator():
    gen = GENERATORS.build(
        {"type": "er", "num_nodes": 20, "p": 0.3, "weighted": True, "weight_dist": "normal"}
    )
    bg = gen.sample(2, device="cpu")
    assert bg.edge_weight is not None
    assert bg.edge_weight.std() > 0  # not all equal


def test_heterogeneous_has_types():
    gen = GENERATORS.build({"type": "hetero", "num_nodes": 20})
    bg = gen.sample(2, device="cpu")
    assert bg.node_type is not None and bg.edge_type is not None
    assert "num_node_types" in bg.meta and "num_edge_types" in bg.meta


def test_multiplex_has_layers():
    gen = GENERATORS.build({"type": "multiplex", "num_nodes": 20})
    bg = gen.sample(2, device="cpu")
    assert bg.edge_type is not None
    assert "num_layers" in bg.meta


def test_hypergraph_has_incidence():
    gen = GENERATORS.build({"type": "hypergraph", "num_nodes": 20})
    bg = gen.sample(2, device="cpu")
    assert "inc_node" in bg.meta and "inc_hyperedge" in bg.meta
    assert bg.meta["inc_node"].numel() == bg.meta["inc_hyperedge"].numel()
