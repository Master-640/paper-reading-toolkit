"""Miscellaneous topologies: caveman communities and powerlaw trees."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import (
    EMPTY,
    VariableSizeGenerator,
    nx_to_edges,
    rng_seed,
)
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("caveman", aliases=["relaxed_caveman"])
class RelaxedCaveman(VariableSizeGenerator):
    """``l`` cliques of size ``clique_size`` with edges rewired at prob ``p``.

    ``l`` is chosen so ``l * clique_size`` approximates the requested ``n``;
    the realized node count is exactly ``l * clique_size``.
    """

    def __init__(self, clique_size: int = 5, p: float = 0.1, **base_kwargs):
        super().__init__(**base_kwargs)
        self.clique_size = max(2, int(clique_size))
        self.p = float(p)

    def _edges_n(self, n: int, rng):
        k = self.clique_size
        n = int(n)
        if n < k:
            return EMPTY(), max(n, 0)
        cliques = max(2, round(n / k))
        g = nx.relaxed_caveman_graph(cliques, k, self.p, seed=rng_seed(rng))
        return nx_to_edges(g), cliques * k


@GENERATORS.register("powerlaw_tree")
class PowerlawTree(GraphGenerator):
    """A random tree whose degree sequence follows a powerlaw with exponent ``gamma``."""

    def __init__(self, gamma: float = 3.0, tries: int = 1000, **base_kwargs):
        super().__init__(**base_kwargs)
        self.gamma = float(gamma)
        self.tries = int(tries)

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 2:
            return EMPTY()
        seed = rng_seed(rng)
        try:
            g = nx.random_powerlaw_tree(n, gamma=self.gamma, seed=seed, tries=self.tries)
        except nx.NetworkXError:  # degree sequence not realizable -> plain tree
            g = (
                nx.random_labeled_tree(n, seed=seed)
                if hasattr(nx, "random_labeled_tree")
                else nx.random_tree(n, seed=seed)
            )
        return nx_to_edges(g)
