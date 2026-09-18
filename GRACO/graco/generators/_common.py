"""Shared helpers for concrete graph generators.

These utilities keep the individual topology files small and consistent: an
integer-seed derivation from a ``numpy`` generator, networkx-to-edge-tensor
conversion, undirected->directed symmetrization, and a batch assembler that
mirrors :meth:`GraphGenerator.sample` but also supports variable node counts
and per-edge / per-node type tensors.  Nothing here is registered.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators.base import GraphGenerator, sample_weights

EMPTY = lambda: torch.zeros(2, 0, dtype=torch.long)  # noqa: E731


def rng_seed(rng: np.random.Generator) -> int:
    """Derive a plain ``int`` seed from a numpy Generator.

    networkx generators accept an ``int`` seed for both ``py_random_state`` and
    ``np_random_state`` decorated functions, so this works universally while
    keeping the stream deterministic w.r.t. ``rng``.
    """
    return int(rng.integers(0, 2**31 - 1))


def nx_to_edges(g) -> Tensor:
    """Convert a networkx graph to an UNDIRECTED edge list ``[2, e]``.

    Node labels are relabelled to integers ``0..n-1`` (needed for grid/tuple
    labels); empty graphs return ``[2, 0]``.
    """
    import networkx as nx

    g = nx.convert_node_labels_to_integers(g)
    if g.number_of_edges() == 0:
        return EMPTY()
    return torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()


def symmetrize(und: Tensor) -> Tensor:
    """Undirected ``[2, e]`` -> directed both-directions ``[2, 2e]``."""
    if und.numel() == 0:
        return EMPTY()
    src = torch.cat([und[0], und[1]])
    dst = torch.cat([und[1], und[0]])
    return torch.stack([src, dst])


# ----------------------------------------------------------------- topologies
def er_edges(n: int, p: float, rng: np.random.Generator) -> Tensor:
    """Fast tensorized Erdos-Renyi G(n, p) undirected edge list."""
    if n < 2:
        return EMPTY()
    iu = torch.triu_indices(n, n, offset=1)  # [2, n(n-1)/2]
    mask = torch.from_numpy(rng.random(iu.shape[1]) < p)
    return iu[:, mask]


def ba_edges(n: int, m: int, rng: np.random.Generator) -> Tensor:
    """Barabasi-Albert undirected edge list (networkx)."""
    import networkx as nx

    if n < 2:
        return EMPTY()
    m = max(1, min(int(m), n - 1))
    g = nx.barabasi_albert_graph(n, m, seed=rng_seed(rng))
    return nx_to_edges(g)


# ------------------------------------------------------------------ assembler
def build_batch(
    gen: GraphGenerator,
    und_list: Sequence[Tensor],
    sizes: Sequence[int],
    device="cpu",
    generator: Optional[torch.Generator] = None,
    *,
    node_type_list: Optional[Sequence[Tensor]] = None,
    edge_type_und_list: Optional[Sequence[Tensor]] = None,
) -> BatchedGraph:
    """Symmetrize per-graph undirected edges, sample weights, and batch.

    Mirrors :meth:`GraphGenerator.sample` but accepts explicit (possibly
    variable) node ``sizes`` and optional per-node / per-undirected-edge type
    tensors, which are attached to the resulting :class:`BatchedGraph`.
    ``edge_type_und_list`` entries are per-undirected-edge and duplicated to
    match the two stored directions.
    """
    ei_list: List[Tensor] = []
    ew_list: List[Optional[Tensor]] = []
    et_list: List[Tensor] = []
    for i, und in enumerate(und_list):
        und = und.to(torch.long)
        e = und.shape[1]
        ei_list.append(symmetrize(und))
        if gen.weighted:
            w = sample_weights(e, gen.weight_dist, gen.weight_params, "cpu", generator)
            if w is None:
                w = torch.ones(e)
            ew_list.append(torch.cat([w, w]))
        else:
            ew_list.append(None)
        if edge_type_und_list is not None:
            et = edge_type_und_list[i].to(torch.long)
            et_list.append(torch.cat([et, et]))

    return BatchedGraph.from_graph_list(
        ei_list,
        [int(s) for s in sizes],
        edge_weight_list=ew_list if gen.weighted else None,
        node_type_list=list(node_type_list) if node_type_list is not None else None,
        edge_type_list=et_list if edge_type_und_list is not None else None,
        device=device,
    )


class VariableSizeGenerator(GraphGenerator):
    """Base for generators whose realized node count differs from the request.

    Subclasses implement :meth:`_edges_n` returning ``(undirected_edges,
    actual_num_nodes)``; ``sample`` is overridden to feed the *actual* counts
    into batching so node offsets stay consistent.
    """

    def _edges_n(self, n: int, rng: np.random.Generator):  # -> (Tensor, int)
        raise NotImplementedError

    def _edges(self, n: int, rng) -> Tensor:  # ABC satisfaction / direct use
        return self._edges_n(int(n), rng)[0]

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        sizes = self._sizes(batch_size, generator)
        und_list: List[Tensor] = []
        actual: List[int] = []
        for n in sizes:
            und, na = self._edges_n(int(n), rng)
            und_list.append(und)
            actual.append(int(na))
        return build_batch(self, und_list, actual, device=device, generator=generator)
