"""Vectorized graph algorithms used by environments.

The centrepiece is a **batched connected-components** routine that runs entirely
on-device via label propagation (iterated ``scatter_max``).  Network-dismantling
rewards depend on the size of the largest connected component (LCC) after node
removals; recomputing that for a whole batch of graphs on GPU — rather than a
per-graph C++ union-find — is what keeps the vectorized env fast.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from graco.utils.scatter import scatter_max, scatter_sum


@torch.no_grad()
def connected_component_labels(
    edge_index: Tensor,
    num_nodes: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Return a component label ``[N]`` for every node via label propagation.

    Labels are the smallest-reachable-index convention would need scatter_min;
    we use ``scatter_max`` so a component's label is the largest global node id
    in it.  Only edges whose *both* endpoints are active propagate; inactive
    (removed) nodes keep their own id and are excluded downstream.

    Converges in O(diameter) iterations; we detect a fixed point and stop early.
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]
    labels = torch.arange(num_nodes, device=device)
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    if edge_index.numel() == 0:
        return labels

    # keep only edges with both endpoints active
    edge_ok = active_mask[src] & active_mask[dst]
    src_a = src[edge_ok]
    dst_a = dst[edge_ok]
    if src_a.numel() == 0:
        return labels

    cap = max_iter if max_iter is not None else num_nodes
    # Label propagation is monotone (labels only grow via ``maximum``) and idempotent
    # at the fixed point, so we only pay the GPU->CPU sync of the convergence check
    # every few iterations; a handful of extra idempotent rounds cannot change the
    # result but avoid serializing the stream on every step.
    check_every = 4
    for i in range(int(cap)):
        # message = source label along active edges; max-aggregate into dst
        cand = scatter_max(labels.index_select(0, src_a), dst_a, num_nodes, fill_value=-1)
        new_labels = torch.maximum(labels, cand)
        if (i % check_every == check_every - 1) and torch.equal(new_labels, labels):
            break
        labels = new_labels
    return labels


@torch.no_grad()
def largest_component_size(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Size of the largest connected component per graph ``[B]`` (active nodes only)."""
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    labels = connected_component_labels(edge_index, num_nodes, active_mask, max_iter)
    active_f = active_mask.to(torch.float32)
    # component size = #active nodes sharing a label
    comp_size = scatter_sum(active_f, labels, num_nodes)  # indexed by label id
    size_of_node = comp_size.index_select(0, labels) * active_f  # inactive -> 0
    lcc = scatter_max(size_of_node, batch, num_graphs, fill_value=0.0)
    # match the reference C++ `maxRankCount` floor of 1 for any graph that still
    # has at least one active node (isolated singletons count as size 1).
    has_active = scatter_sum(active_f, batch, num_graphs) > 0
    return torch.where(has_active, lcc.clamp_min(1.0), lcc)


@torch.no_grad()
def pairwise_connectivity(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Sum over components of ``C(C-1)/2`` per graph ``[B]`` (pairwise connectivity).

    This is the objective used by some dismantling formulations (minimize the
    number of still-connected node pairs).
    """
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    labels = connected_component_labels(edge_index, num_nodes, active_mask, max_iter)
    active_f = active_mask.to(torch.float32)
    comp_size = scatter_sum(active_f, labels, num_nodes)  # per-label size
    pairs = comp_size * (comp_size - 1.0) / 2.0
    # attribute each component's pair-count to its label's graph (label id is a
    # real node id, so batch[label] is well-defined)
    label_batch = batch  # labels index nodes; comp_size is indexed by node-id-as-label
    return scatter_sum(pairs * (comp_size > 0), label_batch, num_graphs)


@torch.no_grad()
def mutually_connected_component_size(
    edge_index: Tensor,
    edge_type: Tensor,
    num_layers: int,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    max_iter: Optional[int] = None,
) -> Tensor:
    """Size of the giant **mutually connected component** per graph ``[B]`` (multiplex).

    Two nodes are mutually connected iff they lie in the same connected component in
    **every** layer, along paths that stay inside the mutual set — the connectivity
    that multiplex-dismantling (MINER) targets.  Computed by the standard iterative
    refinement: repeatedly recompute each layer's components restricted to the current
    mutual groups and intersect, until the partition is stable.
    """
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    if edge_type is None or num_layers <= 1:
        return largest_component_size(edge_index, num_nodes, batch, num_graphs, active_mask, max_iter)
    src, dst = edge_index[0], edge_index[1]
    group = batch.clone().to(torch.long)  # start: all active nodes of a graph together
    cap = max_iter if max_iter is not None else 2 * num_layers + 5
    for _ in range(int(cap)):
        new_group = group
        for a in range(num_layers):
            em = (edge_type == a) & active_mask[src] & active_mask[dst] & (new_group[src] == new_group[dst])
            lab = connected_component_labels(edge_index[:, em], num_nodes, active_mask)
            _, new_group = torch.unique(new_group * num_nodes + lab, return_inverse=True)
        if torch.equal(new_group, group):
            break
        group = new_group
    af = active_mask.to(torch.float32)
    size = scatter_sum(af, group, int(group.max().item()) + 1)  # active count per mutual group
    size_of_node = size.index_select(0, group) * af  # inactive -> 0
    gmcc = scatter_max(size_of_node, batch, num_graphs, fill_value=0.0)
    has_active = scatter_sum(af, batch, num_graphs) > 0
    return torch.where(has_active, gmcc.clamp_min(1.0), gmcc)


@torch.no_grad()
def combat_operational_capability(
    edge_index: Tensor,
    node_type: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    sensor_type: int = 0,
    decision_type: int = 1,
    influence_type: int = 2,
) -> Tensor:
    """Operational capability of a heterogeneous **combat network** per graph ``[B]``.

    Combat networks (SHATTER / HDGED) model three functional roles — Sensor (S),
    Decision (D) and Influence (I) — and their capability is the number of
    **combat chains** ``S → D → I`` (a target is sensed by S, ordered by D, struck
    by I).  Disintegration removes nodes to drive this to zero; the accumulated
    normalized value is the ANOC objective (analogue of ANC for plain graphs).

    Chains are directed, but GRACO stores edges symmetrized, so each arc direction
    is recovered from the endpoint **node types**: an ``S→D`` arc is a stored edge
    with ``type[src]==S`` and ``type[dst]==D``; a ``D→I`` arc has ``type[src]==D``
    and ``type[dst]==I``.  For every active decision node ``d`` the number of chains
    through it is ``(#active S feeding d) · (#active I fed by d)``; summed per graph.
    """
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    af = active_mask.to(torch.float32)
    ts, td = node_type.index_select(0, src), node_type.index_select(0, dst)
    # S→D arcs: count active sensors feeding each (active) decision node
    sd = (ts == sensor_type) & (td == decision_type) & active_mask[src] & active_mask[dst]
    in_s = scatter_sum(af.index_select(0, src) * sd.to(torch.float32), dst, num_nodes)
    # D→I arcs: count active influencers reachable from each (active) decision node
    di = (ts == decision_type) & (td == influence_type) & active_mask[src] & active_mask[dst]
    out_i = scatter_sum(af.index_select(0, dst) * di.to(torch.float32), src, num_nodes)
    is_dec = (node_type == decision_type).to(torch.float32) * af
    chains = is_dec * in_s * out_i  # combat chains routed through each decision node
    return scatter_sum(chains, batch, num_graphs)


@torch.no_grad()
def sis_infection_risk(
    edge_index: Tensor,
    num_nodes: int,
    batch: Tensor,
    num_graphs: int,
    active_mask: Optional[Tensor] = None,
    beta: float = 0.3,
    iters: int = 20,
) -> Tensor:
    """Accumulated SIS infection risk per graph ``[B]`` (individual-based mean-field).

    Steady-state per-node infection probability from the fixed point
    ``p_i = f_i / (1 + f_i)`` with ``f_i = β Σ_{j~i, active} p_j`` (only active edges),
    summed per graph.  This is the objective DeepELE minimizes by removing nodes; it
    is 0 for isolated/removed nodes and monotone in residual connectivity.
    """
    if active_mask is None:
        active_mask = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    ae = (active_mask[src] & active_mask[dst]).to(torch.float32)
    af = active_mask.to(torch.float32)
    p = af * 0.5
    for _ in range(int(iters)):
        f = beta * scatter_sum(p.index_select(0, dst) * ae, src, num_nodes)
        p = af * (f / (1.0 + f))
    return scatter_sum(p, batch, num_graphs)


@torch.no_grad()
def hypergraph_cascade_survivors(
    inc_node: Tensor,
    inc_hyperedge: Tensor,
    num_nodes: int,
    num_hyperedges: int,
    load: Tensor,
    capacity: Tensor,
    removed: Tensor,
    max_rounds: int = 20,
    eps: float = 1e-9,
) -> Tensor:
    """Nodes that stay **functional** after a load-redistribution cascade ``[N]`` (bool).

    Reproduces the two-step hypergraph cascade of Jiang et al. (Physica A 2025):
    when a node fails, its load is redistributed (1) to the valid hyperedges it
    belongs to (those still containing functional nodes) in proportion to their
    remaining capacity, then (2) from each hyperedge to its functional member
    nodes, again in proportion to remaining capacity.  A functional node whose
    load then exceeds its capacity fails, and the process repeats until no new
    failures occur (steady state).  ``inc_node`` / ``inc_hyperedge`` are the
    bipartite incidence (global node / hyperedge ids per nonzero); ``removed`` are
    the agent-removed nodes that seed the cascade.
    """
    load = load.clone()
    alive = ~removed                      # currently functional
    shedders = removed.clone()            # failed nodes whose load must redistribute
    for _ in range(int(max_rounds)):
        if not bool(shedders.any()):
            break
        rem = (capacity - load).clamp_min(0.0) * alive.to(load.dtype)  # headroom per functional node
        r_e = scatter_sum(rem.index_select(0, inc_node), inc_hyperedge, num_hyperedges)  # [M]
        he_valid = r_e > eps
        # step 1 — shedding node -> its valid incident hyperedges, ∝ hyperedge headroom
        e1 = shedders.index_select(0, inc_node) & he_valid.index_select(0, inc_hyperedge)
        denom = scatter_sum(r_e.index_select(0, inc_hyperedge) * e1.to(load.dtype), inc_node, num_nodes)
        share = (load.index_select(0, inc_node) * r_e.index_select(0, inc_hyperedge)
                 / denom.index_select(0, inc_node).clamp_min(eps)) * e1.to(load.dtype)
        he_gain = scatter_sum(share, inc_hyperedge, num_hyperedges)  # [M] load arriving at hyperedges
        # step 2 — hyperedge -> its functional member nodes, ∝ node headroom
        e2 = alive.index_select(0, inc_node) & he_valid.index_select(0, inc_hyperedge)
        node_gain_e = (he_gain.index_select(0, inc_hyperedge) * rem.index_select(0, inc_node)
                       / r_e.index_select(0, inc_hyperedge).clamp_min(eps)) * e2.to(load.dtype)
        node_gain = scatter_sum(node_gain_e, inc_node, num_nodes)
        load = load + node_gain * alive.to(load.dtype)
        load = load.masked_fill(shedders, 0.0)  # shed nodes are gone
        alive = alive & ~shedders
        shedders = alive & (load > capacity)    # newly overloaded -> next round's shedders
        alive = alive & ~shedders
    return alive
