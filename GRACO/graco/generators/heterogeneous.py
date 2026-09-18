"""Heterogeneous graphs: a base topology decorated with node and edge types."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators._common import ba_edges, build_batch, er_edges
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("heterogeneous", aliases=["hetero"])
class Heterogeneous(GraphGenerator):
    """A base graph (ER or BA) with random per-node and per-edge type labels.

    Types are drawn uniformly in ``[0, num_node_types)`` / ``[0, num_edge_types)``.
    Edge types are assigned per undirected edge (shared by both directions).
    ``bg.meta`` records ``num_node_types`` and ``num_edge_types``.
    """

    def __init__(
        self,
        num_node_types: int = 3,
        num_edge_types: int = 3,
        base: str = "er",
        p: float = 0.15,
        m: int = 4,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.num_node_types = max(1, int(num_node_types))
        self.num_edge_types = max(1, int(num_edge_types))
        self.base = str(base).lower()
        self.p = float(p)
        self.m = int(m)

    def _base_und(self, n: int, rng) -> Tensor:
        if self.base in ("ba", "barabasi_albert"):
            return ba_edges(n, self.m, rng)
        return er_edges(n, self.p, rng)

    def _edges(self, n: int, rng) -> Tensor:
        return self._base_und(int(n), rng)

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes = self._sizes(batch_size, generator)

        und_list, nt_list, et_list = [], [], []
        for n in sizes:
            n = int(n)
            und = self._base_und(n, rng)
            und_list.append(und)
            nt_list.append(
                torch.as_tensor(rng.integers(0, self.num_node_types, size=n), dtype=torch.long)
            )
            et_list.append(
                torch.as_tensor(
                    rng.integers(0, self.num_edge_types, size=und.shape[1]), dtype=torch.long
                )
            )

        bg = build_batch(
            self,
            und_list,
            sizes,
            device=device,
            generator=generator,
            node_type_list=nt_list,
            edge_type_und_list=et_list,
        )
        bg.meta["num_node_types"] = self.num_node_types
        bg.meta["num_edge_types"] = self.num_edge_types
        return bg
