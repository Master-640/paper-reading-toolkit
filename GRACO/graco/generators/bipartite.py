"""Bipartite set-cover instance generator.

Produces a bipartite graph: ``ne`` *element* nodes (``node_type=0``) and ``ns``
*set* nodes (``node_type=1``); each set node is connected to the elements it
covers.  Feasibility is guaranteed (every element is covered by >=1 set).  Used
by the :class:`~graco.envs.setcover.SetCoverEnv`.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("set_cover", aliases=["bipartite_cover"])
class SetCover(GraphGenerator):
    def __init__(self, num_elements=(20, 30), num_sets=None, cover_size: int = 4, **base_kwargs):
        # `num_elements` plays the role of the base `num_nodes` range; drop any
        # inherited `num_nodes`/`m` so config `base:` merges don't collide.
        base_kwargs.pop("num_nodes", None)
        base_kwargs.pop("m", None)
        super().__init__(num_nodes=num_elements, **base_kwargs)
        self.num_sets = num_sets
        self.cover_size = int(cover_size)

    def _edges(self, n: int, rng) -> Tensor:  # not used; sample() is overridden
        return torch.zeros(2, 0, dtype=torch.long)

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        elem_sizes = self._sizes(batch_size, generator)

        ei_list: List[Tensor] = []
        sizes: List[int] = []
        nt_list: List[Tensor] = []
        for ne in elem_sizes:
            ne = int(ne)
            ns = int(self.num_sets or max(2, ne // 2))
            us, vs = [], []
            covered = set()
            for s in range(ns):
                k = min(self.cover_size, ne)
                if k < 1:
                    continue
                members = rng.choice(ne, size=k, replace=False)
                for e in members:
                    us.append(ne + s)
                    vs.append(int(e))
                    covered.add(int(e))
            for e in range(ne):  # feasibility: cover any missed element
                if e not in covered:
                    s = int(rng.integers(ns))
                    us.append(ne + s)
                    vs.append(e)
            if us:
                u = torch.tensor(us, dtype=torch.long)
                v = torch.tensor(vs, dtype=torch.long)
                ei = torch.stack([torch.cat([u, v]), torch.cat([v, u])])
            else:
                ei = torch.zeros(2, 0, dtype=torch.long)
            total = ne + ns
            nt = torch.zeros(total, dtype=torch.long)
            nt[ne:] = 1  # sets
            ei_list.append(ei)
            sizes.append(total)
            nt_list.append(nt)

        bg = BatchedGraph.from_graph_list(ei_list, sizes, node_type_list=nt_list, device=device)
        bg.meta["num_node_types"] = 2
        return bg
