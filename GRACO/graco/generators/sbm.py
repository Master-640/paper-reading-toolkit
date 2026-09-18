"""Stochastic block model (SBM) community graphs."""

from __future__ import annotations

import networkx as nx
from torch import Tensor

from graco.generators._common import EMPTY, nx_to_edges, rng_seed
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("sbm", aliases=["stochastic_block_model"])
class StochasticBlockModel(GraphGenerator):
    """``n`` nodes split into ``num_blocks`` roughly equal communities.

    Intra-block edge prob ``p_in``, inter-block prob ``p_out``.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        p_in: float = 0.3,
        p_out: float = 0.02,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.num_blocks = int(num_blocks)
        self.p_in = float(p_in)
        self.p_out = float(p_out)

    def _edges(self, n: int, rng) -> Tensor:
        n = int(n)
        if n < 2:
            return EMPTY()
        b = max(1, min(self.num_blocks, n))
        base, rem = divmod(n, b)
        sizes = [base + (1 if i < rem else 0) for i in range(b)]
        probs = [[self.p_in if i == j else self.p_out for j in range(b)] for i in range(b)]
        g = nx.stochastic_block_model(sizes, probs, seed=rng_seed(rng))
        return nx_to_edges(g)
