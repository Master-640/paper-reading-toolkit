"""d-dimensional grid / lattice graphs (optionally periodic torus)."""

from __future__ import annotations

import networkx as nx

from graco.generators._common import VariableSizeGenerator, nx_to_edges
from graco.registries import GENERATORS


@GENERATORS.register("lattice", aliases=["grid"])
class Lattice(VariableSizeGenerator):
    """A ``dim``-dimensional grid of side ``round(n ** (1/dim))``.

    The realized node count is ``side ** dim`` and may differ from the
    requested ``n`` (handled via :class:`VariableSizeGenerator`).  ``periodic``
    wraps the grid into a torus.
    """

    def __init__(self, dim: int = 2, periodic: bool = True, **base_kwargs):
        super().__init__(**base_kwargs)
        self.dim = int(dim)
        self.periodic = bool(periodic)

    def _edges_n(self, n: int, rng):
        side = max(2, round(int(n) ** (1.0 / self.dim)))
        g = nx.grid_graph([side] * self.dim, periodic=self.periodic)
        return nx_to_edges(g), side**self.dim
