"""Vectorized batched-graph container.

A *batch* of graphs is stored as a single disjoint-union graph on one device.
Every environment steps **all** graphs in the batch simultaneously, and every
GNN layer is a handful of scatter ops over these flat tensors — this is what
makes GRACO fast on GPU.

Tensor schema
-------------
``edge_index``    LongTensor  ``[2, E]``   directed edges; ``[0]``=source, ``[1]``=target.
                                            Undirected graphs store both directions.
``edge_weight``   FloatTensor ``[E]``      edge weights (defaults to ones).
``edge_attr``     FloatTensor ``[E, Fe]``  optional edge features.
``batch``         LongTensor  ``[N]``      graph id in ``[0, B)`` for each node.
``ptr``           LongTensor  ``[B+1]``    CSR node offsets (``ptr[i]:ptr[i+1]``).
``x``             FloatTensor ``[N, Fx]``  optional static node features.
``node_type``     LongTensor  ``[N]``      optional (heterogeneous graphs).
``edge_type``     LongTensor  ``[E]``      optional (hetero / multiplex layer id).

``N`` = total nodes across the batch, ``E`` = total directed edges, ``B`` = #graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import torch
from torch import Tensor

from graco.utils.scatter import (
    lengths_to_ptr,
    scatter_max,
    scatter_mean,
    scatter_sum,
    segment_lengths_to_ids,
)


@dataclass
class BatchedGraph:
    edge_index: Tensor  # [2, E] long
    num_nodes: int
    batch: Tensor  # [N] long
    ptr: Tensor  # [B+1] long
    edge_weight: Optional[Tensor] = None  # [E] float
    edge_attr: Optional[Tensor] = None  # [E, Fe] float
    x: Optional[Tensor] = None  # [N, Fx] float
    node_type: Optional[Tensor] = None  # [N] long
    edge_type: Optional[Tensor] = None  # [E] long
    #: static per-node data carried through obs + replay (e.g. dynamics trajectories
    #: [N, ...] for the sentinel problem). Each value has leading dim N.
    node_attr: Dict[str, Tensor] = field(default_factory=dict)
    #: static per-graph data (e.g. global targets y [B, ...]). Leading dim B.
    graph_attr: Dict[str, Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[str, Tensor] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ basics
    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_graphs(self) -> int:
        return int(self.ptr.numel() - 1)

    @property
    def device(self) -> torch.device:
        return self.edge_index.device

    @property
    def src(self) -> Tensor:
        return self.edge_index[0]

    @property
    def dst(self) -> Tensor:
        return self.edge_index[1]

    @property
    def edge_batch(self) -> Tensor:
        """Graph id for each edge ``[E]``."""
        if "edge_batch" not in self._cache:
            self._cache["edge_batch"] = self.batch.index_select(0, self.edge_index[0])
        return self._cache["edge_batch"]

    @property
    def graph_num_nodes(self) -> Tensor:
        """Per-graph node count ``[B]``."""
        return self.ptr[1:] - self.ptr[:-1]

    @property
    def graph_num_edges(self) -> Tensor:
        """Per-graph edge count ``[B]`` (directed edges)."""
        if "graph_num_edges" not in self._cache:
            ones = torch.ones(self.num_edges, device=self.device, dtype=torch.long)
            self._cache["graph_num_edges"] = scatter_sum(ones, self.edge_batch, self.num_graphs)
        return self._cache["graph_num_edges"]

    @property
    def graph_num_undirected_edges(self) -> Tensor:
        """Per-graph undirected edge count ``[B]``.

        GRACO stores undirected graphs as two directed edges, so this is the
        directed count halved.  Used for MaxCut norms and MVC terminals, where
        the reference C++ counts each undirected edge once.
        """
        return self.graph_num_edges // 2

    def degree(self, weighted: bool = False) -> Tensor:
        """Out-degree (== in-degree for undirected) per node ``[N]``."""
        key = "wdegree" if weighted else "degree"
        if key not in self._cache:
            if weighted and self.edge_weight is not None:
                val = self.edge_weight
            else:
                val = torch.ones(self.num_edges, device=self.device)
            self._cache[key] = scatter_sum(val, self.edge_index[1], self.num_nodes)
        return self._cache[key]

    def adj_norm(self, aggr: str = "mean") -> Tensor:
        """Sparse aggregation operator ``Â`` (``[N, N]``) for SpMM message passing.

        ``Â @ x`` equals the scatter aggregation ``AGG_{j->i} x_j`` — ``Â[dst, src]``
        is ``1`` for ``sum`` or ``1/indeg[dst]`` for ``mean``.  Cached (topology-only)
        so it is built once and reused across every rollout step / GNN layer.
        """
        key = f"adj_{aggr}"
        if key not in self._cache:
            src, dst = self.edge_index[0], self.edge_index[1]
            vals = torch.ones(self.num_edges, device=self.device)
            if aggr == "mean":
                vals = vals / self.degree().clamp_min(1.0).index_select(0, dst)
            idx = torch.stack([dst, src])  # row=dst (aggregation target), col=src
            self._cache[key] = torch.sparse_coo_tensor(
                idx, vals, (self.num_nodes, self.num_nodes)
            ).coalesce()
        return self._cache[key]

    # ---------------------------------------------------------------- pooling
    def pool(self, x: Tensor, reduce: str = "mean") -> Tensor:
        """Node features ``[N, F]`` -> per-graph ``[B, F]``."""
        if reduce == "mean":
            return scatter_mean(x, self.batch, self.num_graphs)
        if reduce in ("sum", "add"):
            return scatter_sum(x, self.batch, self.num_graphs)
        if reduce in ("max", "amax"):
            return scatter_max(x, self.batch, self.num_graphs)
        raise ValueError(f"Unknown pool reduce='{reduce}'")

    def broadcast_to_nodes(self, graph_feat: Tensor) -> Tensor:
        """Per-graph features ``[B, F]`` -> per-node ``[N, F]`` (gather by batch)."""
        return graph_feat.index_select(0, self.batch)

    # --------------------------------------------------------------- movement
    def to(self, device: Any) -> "BatchedGraph":
        def mv(t: Optional[Tensor]) -> Optional[Tensor]:
            return t.to(device) if isinstance(t, Tensor) else t

        return BatchedGraph(
            edge_index=mv(self.edge_index),
            num_nodes=self.num_nodes,
            batch=mv(self.batch),
            ptr=mv(self.ptr),
            edge_weight=mv(self.edge_weight),
            edge_attr=mv(self.edge_attr),
            x=mv(self.x),
            node_type=mv(self.node_type),
            edge_type=mv(self.edge_type),
            node_attr={k: mv(v) for k, v in self.node_attr.items()},
            graph_attr={k: mv(v) for k, v in self.graph_attr.items()},
            meta=dict(self.meta),
        )

    def clone(self) -> "BatchedGraph":
        def cl(t: Optional[Tensor]) -> Optional[Tensor]:
            return t.clone() if isinstance(t, Tensor) else t

        return BatchedGraph(
            edge_index=cl(self.edge_index),
            num_nodes=self.num_nodes,
            batch=cl(self.batch),
            ptr=cl(self.ptr),
            edge_weight=cl(self.edge_weight),
            edge_attr=cl(self.edge_attr),
            x=cl(self.x),
            node_type=cl(self.node_type),
            edge_type=cl(self.edge_type),
            node_attr={k: cl(v) for k, v in self.node_attr.items()},
            graph_attr={k: cl(v) for k, v in self.graph_attr.items()},
            meta=dict(self.meta),
        )

    # ---------------------------------------------------------- construction
    @staticmethod
    def from_graph_list(
        edge_index_list: Sequence[Tensor],
        num_nodes_list: Sequence[int],
        edge_weight_list: Optional[Sequence[Optional[Tensor]]] = None,
        edge_attr_list: Optional[Sequence[Optional[Tensor]]] = None,
        x_list: Optional[Sequence[Optional[Tensor]]] = None,
        node_type_list: Optional[Sequence[Optional[Tensor]]] = None,
        edge_type_list: Optional[Sequence[Optional[Tensor]]] = None,
        device: Any = "cpu",
    ) -> "BatchedGraph":
        """Assemble a batch from per-graph tensors, offsetting node indices."""
        device = torch.device(device)
        counts = torch.tensor([int(n) for n in num_nodes_list], device=device)
        ptr = lengths_to_ptr(counts)
        batch = segment_lengths_to_ids(counts)
        offsets = ptr[:-1]

        eis, ews, eas, xs, nts, ets = [], [], [], [], [], []
        for i, ei in enumerate(edge_index_list):
            ei = ei.to(device=device, dtype=torch.long)
            eis.append(ei + offsets[i])
            e = ei.shape[1]
            if edge_weight_list is not None and edge_weight_list[i] is not None:
                ews.append(edge_weight_list[i].to(device))
            else:
                ews.append(torch.ones(e, device=device))
            if edge_attr_list is not None and edge_attr_list[i] is not None:
                eas.append(edge_attr_list[i].to(device))
            if x_list is not None and x_list[i] is not None:
                xs.append(x_list[i].to(device))
            if node_type_list is not None and node_type_list[i] is not None:
                nts.append(node_type_list[i].to(device))
            if edge_type_list is not None and edge_type_list[i] is not None:
                ets.append(edge_type_list[i].to(device))

        edge_index = (
            torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long, device=device)
        )
        edge_weight = torch.cat(ews) if ews else None
        edge_attr = torch.cat(eas, dim=0) if len(eas) == len(edge_index_list) and eas else None
        x = torch.cat(xs, dim=0) if len(xs) == len(edge_index_list) and xs else None
        node_type = torch.cat(nts) if len(nts) == len(edge_index_list) and nts else None
        edge_type = torch.cat(ets) if len(ets) == len(edge_index_list) and ets else None

        return BatchedGraph(
            edge_index=edge_index,
            num_nodes=int(counts.sum().item()),
            batch=batch,
            ptr=ptr,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            x=x,
            node_type=node_type,
            edge_type=edge_type,
        )

    @staticmethod
    def from_networkx(graphs, weight_key: str = "weight", device: Any = "cpu") -> "BatchedGraph":
        """Build a batch from a list of ``networkx`` graphs."""
        import networkx as nx

        if isinstance(graphs, nx.Graph):
            graphs = [graphs]
        ei_list, nn_list, ew_list = [], [], []
        for g in graphs:
            g = nx.convert_node_labels_to_integers(g)
            n = g.number_of_nodes()
            if g.number_of_edges() == 0:
                ei = torch.zeros(2, 0, dtype=torch.long)
                ew = torch.zeros(0)
            else:
                us, vs, ws = [], [], []
                for u, v, data in g.edges(data=True):
                    w = float(data.get(weight_key, 1.0))
                    us += [u, v]  # symmetrize (undirected)
                    vs += [v, u]
                    ws += [w, w]
                ei = torch.tensor([us, vs], dtype=torch.long)
                ew = torch.tensor(ws, dtype=torch.float)
            ei_list.append(ei)
            nn_list.append(n)
            ew_list.append(ew)
        return BatchedGraph.from_graph_list(ei_list, nn_list, ew_list, device=device)
