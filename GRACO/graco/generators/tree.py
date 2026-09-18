"""Uniform random labelled trees."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import EMPTY, nx_to_edges, rng_seed
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("tree")
class Tree(GraphGenerator):
    """A uniformly random labelled tree on ``n`` nodes (``n-1`` edges)."""

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 2:
            return EMPTY()
        seed = rng_seed(rng)
        if hasattr(nx, "random_labeled_tree"):
            g = nx.random_labeled_tree(n, seed=seed)
        else:  # older networkx
            g = nx.random_tree(n, seed=seed)
        return nx_to_edges(g)
