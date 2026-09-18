"""Erdos-Renyi G(n, p) random graphs."""

from __future__ import annotations

from torch import Tensor

from graco.generators._common import er_edges
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("erdos_renyi", aliases=["er"])
class ErdosRenyi(GraphGenerator):
    """G(n, p): each of the ``n(n-1)/2`` node pairs is an edge with prob ``p``.

    Uses a fast tensorized sampler over the upper triangle rather than
    networkx.
    """

    def __init__(self, p: float = 0.15, **base_kwargs):
        super().__init__(**base_kwargs)
        self.p = float(p)

    def _edges(self, n: int, rng) -> Tensor:
        return er_edges(int(n), self.p, rng)
