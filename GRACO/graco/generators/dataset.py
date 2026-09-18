"""Real-graph dataset generator.

Loads real-world graphs / COP instances from disk (any format understood by
:mod:`graco.data.datasets`) and serves them through the standard
:class:`~graco.generators.base.GraphGenerator` interface, so **training and
one-click benchmarking run on real data with zero extra plumbing**::

    generator: {type: dataset, path: data/snap/*.txt, mode: random}   # train on real graphs
    generator: {type: dataset, path: data/gset,       mode: all}      # eval on the whole set

``mode``: ``random`` samples ``batch_size`` graphs (with replacement) — for
training; ``all`` returns the entire pool as one batch — for evaluation.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.data.datasets import load_graphs
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("dataset", aliases=["files", "real"])
class FileDataset(GraphGenerator):
    def __init__(
        self,
        path: str,
        fmt: str = "auto",
        weighted: Optional[bool] = None,
        max_nodes: Optional[int] = None,
        mode: str = "random",
        **base_kwargs,
    ):
        base_kwargs.pop("num_nodes", None)  # not used; avoids config-merge clash
        super().__init__(**base_kwargs)
        self.path = path
        self.fmt = fmt
        self.file_weighted = weighted
        self.max_nodes = max_nodes
        assert mode in ("random", "all")
        self.mode = mode
        self._graphs: Optional[List] = None

    def _pool(self) -> List:
        if self._graphs is None:
            self._graphs = load_graphs(self.path, self.fmt, self.file_weighted, self.max_nodes)
        return self._graphs

    def _edges(self, n: int, rng) -> Tensor:  # not used; sample() overridden
        raise NotImplementedError

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        pool = self._pool()
        if self.mode == "all":
            chosen = pool
        else:
            if rng is None:
                rng = np.random.default_rng(self._np_seed)
            idx = rng.integers(0, len(pool), size=int(batch_size))
            chosen = [pool[int(i)] for i in idx]
        return BatchedGraph.from_networkx(chosen, device=device)
