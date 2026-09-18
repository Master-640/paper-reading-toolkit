"""Constructive (greedy) heuristics for the GRACO environments.

Each heuristic is a :class:`~graco.heuristics.base.ScoreHeuristic`: it implements
:meth:`score` returning a per-node score ``[N]`` (higher = picked first), and the
base class runs the vectorized batched rollout, masking invalid nodes and taking
a per-graph argmax each step.  Scores are computed purely from the current state
(``env.state['selected']``) and the (static) topology (``obs.graph``); everything
is vectorized over the whole batch with scatter ops — no python node loops
(except the exact-betweenness fallback, which is explicitly a per-graph call).

Every heuristic registers on :data:`~graco.heuristics.base.HEURISTICS` under a
short name.  ``supports=None`` means the rule applies to any node-selection env;
otherwise it lists the concrete env classes (subclasses match via ``isinstance``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.envs.base import Observation, VectorizedEnv
from graco.envs.dismantling import DismantlingEnv
from graco.envs.dominating_set import MinDominatingSetEnv
from graco.envs.influence import InfluenceMaxEnv
from graco.envs.maxclique import MaxCliqueEnv
from graco.envs.maxcut import MaxCutEnv, SpinGlassEnv
from graco.envs.mis import MISEnv
from graco.envs.mvc import MVCEnv
from graco.envs.mwis import MWISEnv
from graco.envs.setcover import SetCoverEnv
from graco.heuristics.base import HEURISTICS, ScoreHeuristic
from graco.utils.scatter import scatter_sum


def _edge_weight(graph) -> Tensor:
    """Edge weights ``[E]`` (ones when the graph is unweighted)."""
    if graph.edge_weight is not None:
        return graph.edge_weight
    return torch.ones(graph.num_edges, device=graph.device)


# --------------------------------------------------------------------- generic


@HEURISTICS.register("random")
class RandomHeuristic(ScoreHeuristic):
    """Pick a uniformly random valid node each step."""

    name = "random"
    supports = None

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        return torch.rand(obs.num_nodes, device=obs.graph.device)


@HEURISTICS.register("degree")
class DegreeHeuristic(ScoreHeuristic):
    """Static highest-degree first (HDA on the original graph)."""

    name = "degree"
    supports = None

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        return obs.graph.degree()


@HEURISTICS.register("adaptive_degree")
class AdaptiveDegreeHeuristic(ScoreHeuristic):
    """Highest-degree first on the still-active subgraph (adaptive HDA).

    An edge counts only when *both* endpoints are still valid actions, so the
    score re-ranks nodes as the graph is consumed.
    """

    name = "adaptive_degree"
    supports = None

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        valid = obs.action_mask
        src, dst = g.edge_index[0], g.edge_index[1]
        live = (valid[src] & valid[dst]).float()
        return scatter_sum(live, src, g.num_nodes)


# ------------------------------------------------------------ maxcut / ising


@HEURISTICS.register("greedy_gain")
class GreedyGainHeuristic(ScoreHeuristic):
    """Greedy spin-flip: pick the node with the largest immediate objective gain.

    With spins ``s = 1 - 2*selected`` (``+1`` un-flipped, ``-1`` flipped), the
    Ising energy change from flipping node ``a`` is
    ``ΔH_a = -2 * s_a * Σ_{v~a} w_av * s_v``.  Spin-Glass maximizes ``H`` so it
    picks ``max ΔH``; MaxCut maximizes the cut ``= (W - H)/2`` so its gain is
    ``-ΔH/2`` and it picks ``max(-ΔH)``.
    """

    name = "greedy_gain"
    supports = (MaxCutEnv, SpinGlassEnv)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        selected = env.state["selected"]
        spin = torch.where(selected, -1.0, 1.0)  # [N]
        src, dst = g.edge_index[0], g.edge_index[1]
        w = _edge_weight(g)
        # Σ_{v~a} w_av * s_v, accumulated at the source node a.
        nbr_field = scatter_sum(w * spin[dst], src, g.num_nodes)
        delta_h = -2.0 * spin * nbr_field  # ΔH for flipping each node
        # +ΔH maximizes energy (spin-glass); -ΔH maximizes the cut.
        sign = -1.0 if getattr(env, "objective_kind", "energy") == "cut" else 1.0
        return sign * delta_h


# ----------------------------------------------------------------------- mvc


@HEURISTICS.register("mvc_greedy")
class MaxCoverGreedy(ScoreHeuristic):
    """MVC greedy: pick the node covering the most still-uncovered edges.

    An edge is uncovered iff neither endpoint is selected; the score is that
    uncovered-incident-edge count per node.
    """

    name = "mvc_greedy"
    supports = (MVCEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        unsel = ~env.state["selected"]
        src, dst = g.edge_index[0], g.edge_index[1]
        uncovered = (unsel[src] & unsel[dst]).float()
        return scatter_sum(uncovered, src, g.num_nodes)


# ----------------------------------------------------------------- mis / mwis


@HEURISTICS.register("min_degree")
class MinDegreeGreedy(ScoreHeuristic):
    """Classic MIS greedy: add the valid node of smallest current degree.

    "Current degree" counts edges to other still-valid nodes; the score is its
    negation so the per-graph argmax picks the minimum-degree candidate.
    """

    name = "min_degree"
    supports = (MISEnv, MWISEnv)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        valid = obs.action_mask
        src, dst = g.edge_index[0], g.edge_index[1]
        live = (valid[src] & valid[dst]).float()
        valid_degree = scatter_sum(live, src, g.num_nodes)
        return -valid_degree


@HEURISTICS.register("weight_degree")
class WeightDegreeGreedy(ScoreHeuristic):
    """MWIS greedy: pick the valid node maximizing ``weight / (1 + current degree)``.

    The node weight matches the env's default (``degree + 1``); the current
    degree counts edges to other still-valid nodes.
    """

    name = "weight_degree"
    supports = (MWISEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        valid = obs.action_mask
        src, dst = g.edge_index[0], g.edge_index[1]
        live = (valid[src] & valid[dst]).float()
        valid_degree = scatter_sum(live, src, g.num_nodes)
        weight = g.degree() + 1.0
        return weight / (1.0 + valid_degree)


# ----------------------------------------------------------------- maxclique


@HEURISTICS.register("clique_greedy")
class CliqueGreedy(ScoreHeuristic):
    """Max-Clique greedy: among candidates (valid nodes adjacent to the whole
    current clique), extend with the one having the most edges to other
    candidates."""

    name = "clique_greedy"
    supports = (MaxCliqueEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        cand = obs.action_mask
        src, dst = g.edge_index[0], g.edge_index[1]
        live = (cand[src] & cand[dst]).float()
        return scatter_sum(live, src, g.num_nodes)


# ----------------------------------------------------------------------- mds


@HEURISTICS.register("mds_greedy")
class DominateGreedy(ScoreHeuristic):
    """Min-Dominating-Set greedy: pick the node that newly dominates the most
    nodes — itself (if still undominated) plus its undominated neighbours."""

    name = "mds_greedy"
    supports = (MinDominatingSetEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        sel = env.state["selected"]
        src, dst = g.edge_index[0], g.edge_index[1]
        nbr_selected = scatter_sum(sel[src].float(), dst, g.num_nodes) > 0
        undominated = ~(sel | nbr_selected)  # [N]
        # neighbours of each node that are currently undominated
        undom_nbr = scatter_sum(undominated[dst].float(), src, g.num_nodes)
        return undominated.float() + undom_nbr


# ------------------------------------------------------------------- setcover


@HEURISTICS.register("setcover_greedy")
class SetCoverGreedy(ScoreHeuristic):
    """Classic greedy set cover: pick the set node covering the most still-
    uncovered elements."""

    name = "setcover_greedy"
    supports = (SetCoverEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        nt = g.node_type
        if nt is None:
            nt = torch.zeros(g.num_nodes, dtype=torch.long, device=g.device)
        sel = env.state["selected"]
        src, dst = g.edge_index[0], g.edge_index[1]
        cov_count = scatter_sum(sel[src].float(), dst, g.num_nodes)
        uncovered_elem = (nt == 0) & (cov_count == 0)  # elements with no selected set
        # for each set node (source), count its uncovered element neighbours
        return scatter_sum(uncovered_elem[dst].float(), src, g.num_nodes)


# ---------------------------------------------------------------- dismantling


@HEURISTICS.register("collective_influence")
class CollectiveInfluenceHeuristic(ScoreHeuristic):
    """Collective Influence (ℓ=1) on the still-active subgraph.

    ``CI(i) = (k_i - 1) * Σ_{j ∈ N(i)} (k_j - 1)`` where degrees ``k`` and the
    neighbourhood are taken over the currently-active (un-removed) subgraph.
    """

    name = "collective_influence"
    supports = (DismantlingEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        active = ~env.state["selected"]
        src, dst = g.edge_index[0], g.edge_index[1]
        active_edge = (active[src] & active[dst]).float()
        active_deg = scatter_sum(active_edge, src, g.num_nodes)  # k_i on active graph
        km1 = active_deg - 1.0
        # Σ_{j~i} (k_j - 1) over active edges, accumulated at the source i
        nbr_sum = scatter_sum(active_edge * km1[dst], src, g.num_nodes)
        return km1 * nbr_sum


@HEURISTICS.register("betweenness")
class BetweennessHeuristic(ScoreHeuristic):
    """Highest betweenness-centrality node on the active subgraph (dismantling).

    Betweenness is computed per graph with ``networkx`` on the extracted active
    subgraph.  Exact for small graphs; for graphs above ``exact_cap`` active
    nodes it switches to the standard ``k``-source sampling approximation.
    """

    name = "betweenness"
    supports = (DismantlingEnv,)

    #: max active nodes for exact betweenness; above this use k-sampling
    exact_cap: int = 400
    #: number of pivot sources for the sampled approximation
    approx_k: int = 128

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        import networkx as nx

        g = obs.graph
        n = g.num_nodes
        scores = torch.zeros(n, device=g.device)
        active = (~env.state["selected"]).cpu().numpy()
        ei = g.edge_index.cpu().numpy()
        src_np, dst_np = ei[0], ei[1]
        ptr = g.ptr.cpu().tolist()
        batch = g.edge_batch.cpu().numpy()
        out = torch.zeros(n, dtype=torch.float64)

        for gi in range(g.num_graphs):
            lo, hi = ptr[gi], ptr[gi + 1]
            active_nodes = [v for v in range(lo, hi) if active[v]]
            if len(active_nodes) <= 2:
                continue
            emask = (batch == gi)
            gg = nx.Graph()
            gg.add_nodes_from(active_nodes)
            eidx = emask.nonzero()[0]
            for e in eidx:
                u, v = int(src_np[e]), int(dst_np[e])
                if u < v and active[u] and active[v]:
                    gg.add_edge(u, v)
            if gg.number_of_nodes() <= 2:
                continue
            k = None
            if gg.number_of_nodes() > self.exact_cap:
                k = min(self.approx_k, gg.number_of_nodes())
            bc = nx.betweenness_centrality(gg, k=k, normalized=True, seed=0)
            for node, val in bc.items():
                out[node] = val

        return out.to(device=g.device, dtype=scores.dtype)


# ---------------------------------------------------------------- influence


@HEURISTICS.register("influence_degree")
class InfluenceDegreeHeuristic(ScoreHeuristic):
    """Influence-maximization greedy: high-degree seeding with a mild discount
    for nodes already adjacent to the current seed set (reduces overlap)."""

    name = "influence_degree"
    supports = (InfluenceMaxEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        deg = g.degree()
        seeds = env.state["selected"]
        src, dst = g.edge_index[0], g.edge_index[1]
        adj_to_seed = scatter_sum(seeds[src].float(), dst, g.num_nodes)  # #seed neighbours
        return deg - adj_to_seed


# ------------------------------------------------ high-impact dismantling rules


@HEURISTICS.register("corehd", aliases=["core_hd"])
class CoreHDHeuristic(ScoreHeuristic):
    """CoreHD dismantling (Zdeborová, Zhang & Zhou, *Sci. Rep.* 2016).

    Repeatedly remove the **highest-degree node in the 2-core** of the remaining
    graph (decycling); once the 2-core is empty the graph is a forest and it falls
    back to plain degree (the tree-breaking phase).  The 2-core of the active
    subgraph is found by vectorized iterative peeling (drop active nodes whose
    active degree < 2 until stable).
    """

    name = "corehd"
    supports = (DismantlingEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        n = g.num_nodes
        src, dst = g.edge_index[0], g.edge_index[1]
        active = ~env.state["selected"]
        deg = scatter_sum((active[src] & active[dst]).float(), src, n)  # active-subgraph degree
        core = active.clone()
        for _ in range(n + 1):  # 2-core peeling (bounded; breaks at fixpoint)
            cdeg = scatter_sum((core[src] & core[dst]).float(), src, n)
            drop = core & (cdeg < 2)
            if not bool(drop.any()):
                break
            core = core & ~drop
        big = deg.max().clamp_min(1.0) + 1.0
        return deg + big * core.float()  # 2-core first (by degree), then forest by degree


@HEURISTICS.register("collective_influence_l2", aliases=["ci2", "collective_influence_2"])
class CollectiveInfluenceL2Heuristic(ScoreHeuristic):
    """Collective Influence at radius ℓ=2 (Morone & Makse, *Nature* 2015).

    ``CI₂(i) = (k_i-1) · Σ_{j in the 2-hop neighbourhood} (k_j-1)`` on the active
    subgraph — a longer-range version of the ℓ=1 rule, obtained by propagating
    ``(k-1)`` two hops with scatter.
    """

    name = "collective_influence_l2"
    supports = (DismantlingEnv,)

    def score(self, env: VectorizedEnv, obs: Observation) -> Tensor:
        g = obs.graph
        n = g.num_nodes
        src, dst = g.edge_index[0], g.edge_index[1]
        ae = (~env.state["selected"])[src] & (~env.state["selected"])[dst]
        ae = ae.float()
        km1 = (scatter_sum(ae, src, n) - 1.0).clamp_min(0.0)
        h1 = scatter_sum(ae * km1[dst], src, n)  # Σ over 1-hop of (k-1)
        h2 = scatter_sum(ae * h1[dst], src, n)  # propagate a 2nd hop
        return km1 * h2
