"""Random geometric graphs (RGG)."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import EMPTY, nx_to_edges, rng_seed
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("rgg", aliases=["random_geometric"])
class RandomGeometric(GraphGenerator):
    """Nodes placed uniformly in the unit square; edges within ``radius``."""

    def __init__(self, radius: float = 0.2, **base_kwargs):
        super().__init__(**base_kwargs)
        self.radius = float(radius)

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 2:
            return EMPTY()
        g = nx.random_geometric_graph(n, self.radius, seed=rng_seed(rng))
        return nx_to_edges(g)
