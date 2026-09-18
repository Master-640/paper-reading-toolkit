"""Line-graph transform generator (edge-centric dismantling).

Wraps any base generator and returns the **line graph** ``L(G)`` of each sampled
graph: every edge of ``G`` becomes a node of ``L(G)``, and two such nodes are
connected iff the original edges share an endpoint.  Running an ordinary
node-selection policy on ``L(G)`` is therefore *edge* selection on ``G`` — the
substrate for the critical-edge / edge-dismantling papers (FIGHTER, IKEoN, SHEAR),
so they reuse the entire dismantling + FINDER/DQN stack unchanged.

The transform is a data-generation step (not in the RL hot loop), so it is built
per graph with ``networkx.line_graph`` for exactness, then batched by GRACO's
disjoint-union engine.  ``graph_attr['orig_edge']`` keeps, per line-graph node,
the original edge endpoints ``[M,2]`` (for cross-graph rewards that measure
connectivity on the *original* graph).
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.generators.base import GraphGenerator
from graco.registries import GENERATORS


@GENERATORS.register("line_graph", aliases=["linegraph"])
class LineGraphGenerator(GraphGenerator):
    def __init__(self, base: Dict[str, Any] | None = None, num_nodes=(1, 1), seed=None, **base_kwargs):
        super().__init__(num_nodes=num_nodes, seed=seed, **base_kwargs)
        self.base_cfg = dict(base or {"type": "barabasi_albert", "num_nodes": [30, 50], "m": 4})
        self._base = None

    def _edges(self, n: int, rng) -> Tensor:  # ABC satisfaction; sample() overridden
        return torch.zeros(2, 0, dtype=torch.long)

    @staticmethod
    def _line_graph_edges(und: Tensor, n: int):
        """Undirected edges of ``L(G)`` given ``G``'s undirected edges ``und`` [2,e]."""
        m = und.shape[1]
        if m == 0:
            return torch.zeros(2, 0, dtype=torch.long), m
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from([(int(u), int(v)) for u, v in und.t().tolist()])
        L = nx.line_graph(G)
        idx = {e: i for i, e in enumerate(L.nodes())}  # each L-node is an edge tuple of G
        le = [(idx[a], idx[b]) for a, b in L.edges()]
        if le:
            und_l = torch.tensor(le, dtype=torch.long).t().contiguous()
        else:
            und_l = torch.zeros(2, 0, dtype=torch.long)
        return und_l, len(idx)

    def sample(self, batch_size, device="cpu", generator=None, rng=None) -> BatchedGraph:
        if rng is None:
            rng = np.random.default_rng(self._np_seed)
        if self._base is None:
            self._base = GENERATORS.build(dict(self.base_cfg))

        ei_list: List[Tensor] = []
        sizes: List[int] = []
        orig_edges: List[Tensor] = []  # global original endpoints per L-node
        orig_batch: List[Tensor] = []  # graph id per original node
        n_off = 0  # running original-node offset (for global ids)
        for gi in range(batch_size):
            g = self._base.sample(1, device="cpu", rng=rng)  # one base graph
            ei = g.edge_index
            und = ei[:, ei[0] < ei[1]] if ei.numel() else ei  # unique undirected edges
            und_l, mnodes = self._line_graph_edges(und, g.num_nodes)
            # directed both-ways for GRACO's undirected convention
            if und_l.numel():
                src = torch.cat([und_l[0], und_l[1]])
                dst = torch.cat([und_l[1], und_l[0]])
                ei_list.append(torch.stack([src, dst]))
            else:
                ei_list.append(torch.zeros(2, 0, dtype=torch.long))
            sizes.append(max(mnodes, 1))
            oe = (und.t().contiguous() if und.numel() else torch.zeros(0, 2, dtype=torch.long))
            orig_edges.append(oe + n_off)  # global original node ids
            orig_batch.append(torch.full((g.num_nodes,), gi, dtype=torch.long))
            n_off += g.num_nodes

        bg = BatchedGraph.from_graph_list(ei_list, sizes, device=device)
        # per-line-graph-node original edge endpoints (global), + original-node batch,
        # so an edge-dismantling env can measure connectivity on the ORIGINAL graph.
        bg.node_attr["orig_edge"] = torch.cat(
            [oe if oe.shape[0] == s else torch.zeros(s, 2, dtype=torch.long)
             for oe, s in zip(orig_edges, sizes)]
        ).to(device)
        bg.meta["line_graph"] = True
        bg.meta["orig_num_nodes"] = int(n_off)
        bg.meta["orig_batch"] = torch.cat(orig_batch).to(device)
        return bg
