"""Complete graphs K_n."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import EMPTY, nx_to_edges
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("complete")
class Complete(GraphGenerator):
    """The complete graph on ``n`` nodes (every pair connected)."""

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 2:
            return EMPTY()
        return nx_to_edges(nx.complete_graph(n))
