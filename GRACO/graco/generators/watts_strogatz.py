"""Watts-Strogatz small-world graphs."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import EMPTY, nx_to_edges, rng_seed
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("watts_strogatz", aliases=["ws"])
class WattsStrogatz(GraphGenerator):
    """Ring lattice with ``k`` neighbours, each edge rewired with prob ``p``.

    Prefers the connected variant, falling back to the plain generator when a
    connected instance cannot be produced.
    """

    def __init__(self, k: int = 6, p: float = 0.1, **base_kwargs):
        super().__init__(**base_kwargs)
        self.k = int(k)
        self.p = float(p)

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 3:
            return EMPTY()
        k = max(2, min(self.k, n - 1))
        seed = rng_seed(rng)
        try:
            g = nx.connected_watts_strogatz_graph(n, k, self.p, seed=seed)
        except nx.NetworkXError:
            g = nx.watts_strogatz_graph(n, k, self.p, seed=seed)
        return nx_to_edges(g)
