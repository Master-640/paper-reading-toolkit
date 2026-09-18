"""Experience buffers and the transition storage contract.

Key idea (mirrors DIRAC/FINDER): a transition does **not** store dense feature
tensors.  It stores a reference to the (episode-shared) single-graph topology
plus a compact dynamic-state snapshot (a dict of small tensors, typically just
the ``selected`` mask).  At learning time we collate K transitions into one
:class:`~graco.data.batch.BatchedGraph` and let the env's *pure* observation
functions rebuild features/masks.

The stored :class:`SingleGraph` also carries graph *type* information —
``node_type`` / ``edge_type`` (heterogeneous, multiplex layer id) and hypergraph
incidence (``inc_node`` / ``inc_hyperedge``) — so that typed encoders
(RGCN/HGT/HAN/multiplex/HGNN/HyperGCN) train correctly on hetero, multilayer and
hypergraph instances, not just on the plain topology.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph

State = Dict[str, Tensor]


@dataclass
class SingleGraph:
    """One graph's topology + type metadata (CPU tensors), shared by its transitions."""

    edge_index: Tensor  # [2, e] local ids
    num_nodes: int
    edge_weight: Optional[Tensor] = None  # [e]
    node_type: Optional[Tensor] = None  # [n] long
    edge_type: Optional[Tensor] = None  # [e] long
    node_pos: Optional[Tensor] = None  # [n, d] float (lattice / geometric coords)
    inc_node: Optional[Tensor] = None  # hypergraph incidence: local node ids
    inc_hyperedge: Optional[Tensor] = None  # hypergraph incidence: local hyperedge ids
    num_hyperedges: int = 0
    node_attr: Dict[str, Tensor] = field(default_factory=dict)  # per-node static data [n, ...]
    graph_attr: Dict[str, Tensor] = field(default_factory=dict)  # per-graph static data [...]
    meta_scalars: Dict[str, int] = field(default_factory=dict)


@dataclass
class Transition:
    graph: SingleGraph
    state: State
    action: int  # local node id chosen at s_t
    reward: float  # n-step return
    next_state: State
    terminal: bool


@dataclass
class SampledBatch:
    graph: BatchedGraph
    state: State
    next_state: State
    action: Tensor  # [K] GLOBAL node ids
    reward: Tensor  # [K]
    terminal: Tensor  # [K] bool
    indices: Optional[Tensor] = None
    is_weights: Optional[Tensor] = None


_SCALAR_META_KEYS = ("num_node_types", "num_edge_types", "num_layers")


def to_single_graphs(bg: BatchedGraph) -> List[SingleGraph]:
    """Split a batched graph into per-graph :class:`SingleGraph` (CPU), types kept."""
    bg = bg.to("cpu")
    eb = bg.edge_batch
    # hypergraph incidence (optional)
    inc_node = bg.meta.get("inc_node") if bg.meta else None
    inc_he = bg.meta.get("inc_hyperedge") if bg.meta else None
    he_per = bg.meta.get("num_hyperedges_per_graph") if bg.meta else None
    if inc_node is not None:
        inc_node = inc_node.cpu()
        inc_he = inc_he.cpu()
        he_per = [int(x) for x in he_per.cpu().tolist()] if he_per is not None else None
        he_ptr = [0]
        for h in (he_per or []):
            he_ptr.append(he_ptr[-1] + h)
    scalars = {k: int(bg.meta[k]) for k in _SCALAR_META_KEYS if bg.meta and k in bg.meta}
    node_pos = bg.meta.get("node_pos") if bg.meta else None
    if node_pos is not None:
        node_pos = node_pos.cpu()

    graphs: List[SingleGraph] = []
    for g in range(bg.num_graphs):
        lo = int(bg.ptr[g])
        n = int(bg.ptr[g + 1] - lo)
        em = eb == g
        ei = bg.edge_index[:, em] - lo
        ew = bg.edge_weight[em] if bg.edge_weight is not None else None
        nt = bg.node_type[lo : lo + n] if bg.node_type is not None else None
        et = bg.edge_type[em] if bg.edge_type is not None else None
        npos = node_pos[lo : lo + n] if node_pos is not None else None
        na = {k: v[lo : lo + n].cpu() for k, v in bg.node_attr.items()}
        ga = {k: v[g].cpu() for k, v in bg.graph_attr.items()}
        in_node = in_he = None
        n_he = 0
        if inc_node is not None and he_per is not None:
            m = (inc_node >= lo) & (inc_node < lo + n)
            in_node = inc_node[m] - lo
            in_he = inc_he[m] - he_ptr[g]
            n_he = int(he_per[g])
        graphs.append(
            SingleGraph(
                edge_index=ei,
                num_nodes=n,
                edge_weight=ew,
                node_type=nt,
                edge_type=et,
                node_pos=npos,
                inc_node=in_node,
                inc_hyperedge=in_he,
                num_hyperedges=n_he,
                node_attr=na,
                graph_attr=ga,
                meta_scalars=dict(scalars),
            )
        )
    return graphs


def collate_single_graphs(
    graphs: List[SingleGraph], device
) -> Tuple[BatchedGraph, Tensor]:
    """Collate K :class:`SingleGraph` into one BatchedGraph (topology + types).

    Returns ``(batched_graph, node_offsets[K])`` so callers can map per-graph
    local action ids to global ids.
    """
    ei_list, ew_list, nn_list, nt_list, et_list = [], [], [], [], []
    have_w = all(g.edge_weight is not None for g in graphs)
    have_nt = all(g.node_type is not None for g in graphs)
    have_et = all(g.edge_type is not None for g in graphs)
    have_inc = all(g.inc_node is not None for g in graphs)
    for g in graphs:
        ei_list.append(g.edge_index)
        nn_list.append(g.num_nodes)
        ew_list.append(g.edge_weight)
        nt_list.append(g.node_type)
        et_list.append(g.edge_type)
    bg = BatchedGraph.from_graph_list(
        ei_list,
        nn_list,
        ew_list if have_w else None,
        node_type_list=nt_list if have_nt else None,
        edge_type_list=et_list if have_et else None,
        device=device,
    )
    # scalar meta (num_node_types / num_edge_types / num_layers): take the max seen
    for k in _SCALAR_META_KEYS:
        vals = [g.meta_scalars.get(k) for g in graphs if k in g.meta_scalars]
        if vals:
            bg.meta[k] = max(vals)
    # node coordinates (lattice / geometric)
    if all(g.node_pos is not None for g in graphs):
        bg.meta["node_pos"] = torch.cat([g.node_pos.to(device) for g in graphs], dim=0)
    # generic static per-node / per-graph data (e.g. sentinel trajectories/targets)
    if graphs and graphs[0].node_attr:
        for k in graphs[0].node_attr:
            bg.node_attr[k] = torch.cat([g.node_attr[k].to(device) for g in graphs], dim=0)
    if graphs and graphs[0].graph_attr:
        for k in graphs[0].graph_attr:
            bg.graph_attr[k] = torch.stack([g.graph_attr[k].to(device) for g in graphs], dim=0)
    # hypergraph incidence: re-offset node ids and hyperedge ids
    if have_inc:
        offsets = bg.ptr[:-1]
        in_nodes, in_hes = [], []
        he_off = 0
        for i, g in enumerate(graphs):
            in_nodes.append(g.inc_node.to(device) + int(offsets[i]))
            in_hes.append(g.inc_hyperedge.to(device) + he_off)
            he_off += int(g.num_hyperedges)
        bg.meta["inc_node"] = torch.cat(in_nodes) if in_nodes else torch.zeros(0, dtype=torch.long, device=device)
        bg.meta["inc_hyperedge"] = torch.cat(in_hes) if in_hes else torch.zeros(0, dtype=torch.long, device=device)
        bg.meta["num_hyperedges"] = he_off
        bg.meta["num_hyperedges_per_graph"] = torch.tensor(
            [int(g.num_hyperedges) for g in graphs], dtype=torch.long, device=device
        )
    return bg, bg.ptr[:-1]


def collate(
    transitions: List[Transition],
    state_spec: Dict[str, str],
    device,
) -> Tuple[BatchedGraph, State, State, Tensor, Tensor, Tensor]:
    """Collate transitions into (batched_graph, state, next_state, action, reward, terminal)."""
    graph, offsets = collate_single_graphs([t.graph for t in transitions], device)

    def collate_state(key_from) -> State:
        out: State = {}
        for name, scope in state_spec.items():
            vals = [key_from(t)[name] for t in transitions]
            if scope == "graph":
                out[name] = torch.stack([v.reshape(()) for v in vals]).to(device)
            else:
                out[name] = torch.cat([v.reshape(-1) for v in vals]).to(device)
        return out

    state = collate_state(lambda t: t.state)
    next_state = collate_state(lambda t: t.next_state)
    action_local = torch.tensor([t.action for t in transitions], device=device)
    action = action_local + offsets
    reward = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
    terminal = torch.tensor([t.terminal for t in transitions], dtype=torch.bool, device=device)
    return graph, state, next_state, action, reward, terminal


class Buffer(ABC):
    """Abstract replay/rollout buffer."""

    state_spec: Dict[str, str] = {"selected": "node"}

    @abstractmethod
    def add(self, transition: Transition) -> None: ...

    @abstractmethod
    def sample(self, batch_size: int, device) -> SampledBatch: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def update_priorities(self, indices: Tensor, td_errors: Tensor) -> None:
        """No-op unless the buffer is prioritized."""
