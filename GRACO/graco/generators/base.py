"""Graph generator base class.

Generators produce a :class:`~graco.data.batch.BatchedGraph` of ``batch_size``
random instances.  Subclasses only emit an **undirected** edge list per graph
(each edge once); the base symmetrizes to the directed both-directions layout,
samples/attaches edge weights, and assembles the batch — so weighting,
directionality and batching are handled uniformly for every topology.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph


def sample_weights(
    num_edges: int,
    dist: Optional[str],
    params: Optional[dict],
    device,
    generator: Optional[torch.Generator] = None,
) -> Optional[Tensor]:
    """Sample ``num_edges`` undirected edge weights from a named distribution.

    Supported ``dist``: ``None``/``unweighted`` (ones), ``normal`` (mean,std),
    ``uniform`` (low,high), ``negative_uniform`` (−U(0,1)), ``bimodal`` (±1),
    ``lognormal``, ``exponential``, ``poisson``.
    """
    params = params or {}
    if dist in (None, "none", "unweighted", "unit"):
        return None
    g = generator
    if dist == "normal":
        w = torch.randn(num_edges, device=device, generator=g)
        w = w * params.get("std", 1.0) + params.get("mean", 0.0)
    elif dist == "uniform":
        low, high = params.get("low", 0.0), params.get("high", 1.0)
        w = torch.rand(num_edges, device=device, generator=g) * (high - low) + low
    elif dist == "negative_uniform":
        w = -torch.rand(num_edges, device=device, generator=g)
    elif dist == "bimodal":
        w = torch.where(
            torch.rand(num_edges, device=device, generator=g) < 0.5,
            torch.full((num_edges,), -1.0, device=device),
            torch.full((num_edges,), 1.0, device=device),
        )
    elif dist == "lognormal":
        w = torch.randn(num_edges, device=device, generator=g)
        w = (w * params.get("std", 1.0) + params.get("mean", 0.0)).exp()
    elif dist == "exponential":
        u = torch.rand(num_edges, device=device, generator=g).clamp_min(1e-12)
        w = -u.log() / params.get("rate", 1.0)
    elif dist == "poisson":
        rate = params.get("rate", 1.0)
        w = torch.poisson(torch.full((num_edges,), float(rate), device=device), generator=g)
    else:
        raise ValueError(f"Unknown weight distribution '{dist}'")
    return w.float()


class GraphGenerator(ABC):
    def __init__(
        self,
        num_nodes: Sequence[int] | int = (30, 50),
        weighted: bool = False,
        weight_dist: Optional[str] = None,
        weight_params: Optional[dict] = None,
        directed: bool = False,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if isinstance(num_nodes, int):
            num_nodes = (num_nodes, num_nodes)
        self.num_nodes_range: Tuple[int, int] = (int(num_nodes[0]), int(num_nodes[-1]))
        self.weighted = weighted
        self.weight_dist = weight_dist or ("normal" if weighted else None)
        self.weight_params = weight_params or {}
        self.directed = directed
        self._np_seed = seed
        self.extra = kwargs

    def _sizes(self, batch_size: int, generator: Optional[torch.Generator]) -> List[int]:
        lo, hi = self.num_nodes_range
        if lo == hi:
            return [lo] * batch_size
        return (
            torch.randint(lo, hi + 1, (batch_size,), generator=generator).tolist()
        )

    @abstractmethod
    def _edges(self, n: int, rng) -> Tensor:
        """Return an UNDIRECTED edge list ``[2, e]`` (each edge once) for one graph.

        ``rng`` is a ``numpy.random.Generator`` for topology randomness.
        """

    def sample(
        self,
        batch_size: int,
        device="cpu",
        generator: Optional[torch.Generator] = None,
        rng=None,
    ) -> BatchedGraph:
        import numpy as np

        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes = self._sizes(batch_size, generator)

        ei_list: List[Tensor] = []
        ew_list: List[Optional[Tensor]] = []
        for n in sizes:
            und = self._edges(int(n), rng).to(torch.long)  # [2, e]
            e = und.shape[1]
            if e == 0:
                ei_list.append(torch.zeros(2, 0, dtype=torch.long))
                ew_list.append(None if not self.weighted else torch.zeros(0))
                continue
            # symmetrize -> directed both directions, aligned weights
            src = torch.cat([und[0], und[1]])
            dst = torch.cat([und[1], und[0]])
            ei_list.append(torch.stack([src, dst]))
            if self.weighted:
                w = sample_weights(e, self.weight_dist, self.weight_params, "cpu", generator)
                if w is None:
                    w = torch.ones(e)
                ew_list.append(torch.cat([w, w]))
            else:
                ew_list.append(None)

        return BatchedGraph.from_graph_list(
            ei_list, sizes, ew_list if self.weighted else None, device=device
        )
