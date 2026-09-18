"""Random hypergraphs with clique-expanded topology and incidence metadata."""

from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators._common import EMPTY, build_batch
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("hypergraph")
class Hypergraph(GraphGenerator):
    """``num_hyperedges`` random hyperedges, each covering ``hyperedge_size`` nodes.

    The ordinary :class:`BatchedGraph` topology is the *clique expansion*
    (all node pairs inside a hyperedge are connected) so standard encoders
    still work.  The bipartite node<->hyperedge incidence is stored in
    ``bg.meta`` with per-graph offsets applied so it batches correctly:

    * ``inc_node``               long ``[nnz]`` global node ids
    * ``inc_hyperedge``          long ``[nnz]`` global hyperedge ids
    * ``num_hyperedges_per_graph`` long ``[B]``
    * ``num_hyperedges``         int (batch total)
    """

    def __init__(self, num_hyperedges: int = 10, hyperedge_size: int = 3, **base_kwargs):
        super().__init__(**base_kwargs)
        self.num_hyperedges = int(num_hyperedges)
        self.hyperedge_size = int(hyperedge_size)

    def _one(self, n: int, rng):
        """Build one graph: (undirected clique edges, inc_node, inc_he, H)."""
        if n < 1 or self.num_hyperedges < 1:
            return EMPTY(), [], [], 0
        inc_node: List[int] = []
        inc_he: List[int] = []
        edge_set = set()
        for h in range(self.num_hyperedges):
            size = min(self.hyperedge_size, n)
            if size < 1:
                continue
            members = sorted(int(x) for x in rng.choice(n, size=size, replace=False))
            for u in members:
                inc_node.append(u)
                inc_he.append(h)
            for a, b in combinations(members, 2):
                edge_set.add((a, b))
        if edge_set:
            und = torch.tensor(sorted(edge_set), dtype=torch.long).t().contiguous()
        else:
            und = EMPTY()
        return und, inc_node, inc_he, self.num_hyperedges

    def _edges(self, n: int, rng) -> Tensor:
        return self._one(int(n), rng)[0]

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes = self._sizes(batch_size, generator)

        und_list: List[Tensor] = []
        inc_node_all: List[Tensor] = []
        inc_he_all: List[Tensor] = []
        h_per_graph: List[int] = []
        node_off = 0
        he_off = 0
        for n in sizes:
            n = int(n)
            und, inc_node, inc_he, h = self._one(n, rng)
            und_list.append(und)
            inc_node_all.append(torch.tensor(inc_node, dtype=torch.long) + node_off)
            inc_he_all.append(torch.tensor(inc_he, dtype=torch.long) + he_off)
            h_per_graph.append(h)
            node_off += n
            he_off += h

        bg = build_batch(self, und_list, sizes, device=device, generator=generator)
        bg.meta["inc_node"] = torch.cat(inc_node_all).to(bg.device)
        bg.meta["inc_hyperedge"] = torch.cat(inc_he_all).to(bg.device)
        bg.meta["num_hyperedges_per_graph"] = torch.tensor(
            h_per_graph, dtype=torch.long, device=bg.device
        )
        bg.meta["num_hyperedges"] = int(he_off)
        return bg
