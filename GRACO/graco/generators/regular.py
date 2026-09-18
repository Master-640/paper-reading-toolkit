"""Random d-regular graphs."""

from __future__ import annotations

import networkx as nx

from graco.generators._common import EMPTY, VariableSizeGenerator, nx_to_edges, rng_seed
from graco.registries import GENERATORS


@GENERATORS.register("regular")
class Regular(VariableSizeGenerator):
    """A uniformly random ``d``-regular graph.

    ``networkx`` requires ``n * d`` even; when the request is odd the node
    count is bumped by one (reflected in the realized size).
    """

    def __init__(self, d: int = 3, **base_kwargs):
        super().__init__(**base_kwargs)
        self.d = int(d)

    def _edges_n(self, n: int, rng):
        n = int(n)
        if n < 2 or self.d < 1:
            return EMPTY(), max(n, 0)
        d = min(self.d, n - 1)
        n_eff = n + 1 if (n * d) % 2 else n  # ensure n*d even by bumping n
        d = min(d, n_eff - 1)
        g = nx.random_regular_graph(d, n_eff, seed=rng_seed(rng))
        return nx_to_edges(g), n_eff
