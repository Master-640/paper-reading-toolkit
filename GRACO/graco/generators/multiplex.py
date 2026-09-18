"""Multiplex / multilayer graphs: several layers over a shared node set."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators._common import EMPTY, ba_edges, build_batch, er_edges
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("multiplex", aliases=["multilayer"])
class Multiplex(GraphGenerator):
    """``num_layers`` graphs on the SAME ``n`` nodes, flattened to one graph.

    Every edge carries its layer id in ``edge_type`` (both directions share the
    id); ``bg.meta['num_layers']`` records the layer count.
    """

    def __init__(
        self,
        num_layers: int = 2,
        base: str = "er",
        p: float = 0.15,
        m: int = 4,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.num_layers = max(1, int(num_layers))
        self.base = str(base).lower()
        self.p = float(p)
        self.m = int(m)

    def _layer_und(self, n: int, rng) -> Tensor:
        if self.base in ("ba", "barabasi_albert"):
            return ba_edges(n, self.m, rng)
        return er_edges(n, self.p, rng)

    def _stack(self, n: int, rng):
        """Return combined undirected edges and per-edge layer ids for one graph."""
        unds, types = [], []
        for layer in range(self.num_layers):
            u = self._layer_und(n, rng)
            if u.shape[1]:
                unds.append(u)
                types.append(torch.full((u.shape[1],), layer, dtype=torch.long))
        if not unds:
            return EMPTY(), torch.zeros(0, dtype=torch.long)
        return torch.cat(unds, dim=1), torch.cat(types)

    def _edges(self, n: int, rng) -> Tensor:
        return self._stack(int(n), rng)[0]

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes = self._sizes(batch_size, generator)

        und_list, et_list = [], []
        for n in sizes:
            und, et = self._stack(int(n), rng)
            und_list.append(und)
            et_list.append(et)

        bg = build_batch(
            self,
            und_list,
            sizes,
            device=device,
            generator=generator,
            edge_type_und_list=et_list,
        )
        bg.meta["num_layers"] = self.num_layers
        return bg
