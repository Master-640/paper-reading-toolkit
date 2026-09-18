"""Graph instance generators.

Importing this package registers every generator on the ``GENERATORS`` registry:
random graph families (ER, BA, WS, powerlaw-cluster, complete, lattice, regular,
RGG, SBM, tree, caveman), weighted/unweighted variants, and multi-type graphs
(heterogeneous, multiplex/multilayer, hypergraph).
"""

from graco.generators import (  # noqa: F401
    barabasi_albert,
    bipartite,
    combat,
    complete,
    dataset,
    erdos_renyi,
    heterogeneous,
    hypergraph,
    lattice,
    line_graph,
    misc,
    multiplex,
    qubo,
    regular,
    rgg,
    sbm,
    sentinel,
    tree,
    watts_strogatz,
)
from graco.generators.base import GraphGenerator, sample_weights  # noqa: F401

__all__ = ["GraphGenerator", "sample_weights"]
