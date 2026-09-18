"""Barabasi-Albert preferential-attachment graphs and the powerlaw-cluster variant."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import EMPTY, ba_edges, nx_to_edges, rng_seed
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("barabasi_albert", aliases=["ba"])
class BarabasiAlbert(GraphGenerator):
    """Barabasi-Albert graph: each new node attaches ``m`` preferential edges."""

    def __init__(self, m: int = 4, **base_kwargs):
        super().__init__(**base_kwargs)
        self.m = int(m)

    def _edges(self, n: int, rng) -> Tensor:
        return ba_edges(int(n), self.m, rng)


@GENERATORS.register("powerlaw_cluster", aliases=["plc"])
class PowerlawCluster(GraphGenerator):
    """Holme-Kim powerlaw-cluster graph: BA growth with triad-closure prob ``p``."""

    def __init__(self, m: int = 4, p: float = 0.1, **base_kwargs):
        super().__init__(**base_kwargs)
        self.m = int(m)
        self.p = float(p)

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 2:
            return EMPTY()
        m = max(1, min(self.m, n - 1))
        g = nx.powerlaw_cluster_graph(n, m, self.p, seed=rng_seed(rng))
        return nx_to_edges(g)
