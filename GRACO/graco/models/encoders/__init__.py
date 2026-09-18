"""GNN encoders.

Importing this package registers every encoder on the ``ENCODERS`` registry.
Message-passing: GCN, GraphSAGE, GIN/GINE, GAT/GATv2, structure2vec (S2V/DIRAC),
GatedGCN, PNA, MPNN/NNConv, GraphTransformer, EdgeConv.
Spectral / deep: GCNII, APPNP, SGC, ChebNet, ARMA, TAGCN.
Typed / multi: RGCN, HGT, HAN, typed-MPNN (heterogeneous); MultiplexGCN
(multilayer); HGNN, HyperGCN, HNHN (hypergraph).
Plus a composable positional/structural-encoding wrapper (PE).
"""

from graco.models import pe  # noqa: F401  (registers the "pe" wrapper encoder)
from graco.models.encoders import (  # noqa: F401
    appnp,
    arma,
    chebnet,
    edgeconv,
    gat,
    gatedgcn,
    gcn,
    gcnii,
    gin,
    graph_transformer,
    graphsage,
    han,
    hetero,
    hnhn,
    hypergcn,
    hypergraph,
    mind,
    mpnn,
    multiplex,
    pna,
    s2v,
    sentinel_encoder,
    sgc,
    tagcn,
    typed_mpnn,
    vns,
)
