"""k-label constructive-labeling environments: max-k-cut and graph coloring.

A whole new class of problems where each node gets one of ``k`` **labels/colors**
rather than a binary in/out decision.  The MDP is *constructive labeling*: state
is a per-node label ``label[i] ∈ {-1 (unassigned), 0..k-1}``; each step assigns a
color to one unassigned node; the episode ends when every node is labeled.  The
action is a joint ``(node, color)`` pair — the :class:`~graco.algos.label_dqn`
algorithm scores ``[N, k]`` and flattens to an ``N·k`` action space per graph.

* **Max-k-cut** (``maxkcut``): partition nodes into ``k`` groups maximizing the
  number of edges whose endpoints differ.  Reward per assignment = # newly-cut
  edges (telescopes to the final cut).  Generalizes MaxCut (``k=2``).
* **Graph coloring** (``graph_coloring``): assign ``k`` colors minimizing the
  number of monochromatic (conflicting) edges.  Reward = −(# new conflicts).
  Constructive (a node's color is fixed once chosen) — a learned greedy ordering;
  a revisitable *improvement* coloring (recolor to escape conflicts) is future work.

Observations are pure functions of ``(graph, label)`` — the per-node neighbour
**colour histogram** ``[N, k]``, the node's own one-hot colour ``[N, k]``, and an
assigned flag — so replay reconstruction is exact.  Since labeling is monotone
(each node colored once), the chosen colour is recoverable from ``next_state`` and
need not be stored out of band.  The binary node-selection path is untouched.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graco.data.batch import BatchedGraph
from graco.envs.base import State, VectorizedEnv
from graco.registries import ENVS
from graco.utils.scatter import scatter_sum


class LabelEnv(VectorizedEnv):
    """Base constructive k-labeling MDP (see module docstring)."""

    num_labels: int = 3  # k (class attribute; a single k per training run)
    edge_feature_dim = 0  # no natural edge features — neighbour colours live in node features
    eval_metric = "objective"
    state_spec = {"label": "node"}  # only `label` is replayed; `selected` mirror is runtime-only

    def __init__(self, k: int | None = None, reward_scale: float = 1.0, max_steps_frac: float = 1.0):
        super().__init__(reward_scale=reward_scale, max_steps_frac=max_steps_frac)
        if k is not None:
            type(self).num_labels = int(k)  # fix k for this env class / training run
        self.node_feature_dim = 2 * self.num_labels + 1  # [nbr-hist(k), own-onehot(k), assigned]

    def _init_state(self, graph: BatchedGraph) -> State:
        n = graph.num_nodes
        return {
            "label": torch.full((n,), -1, dtype=torch.long, device=graph.device),
            "selected": torch.zeros(n, dtype=torch.bool, device=graph.device),  # mirror (base assert)
        }

    def _delta_reward(self, node: Tensor, color: Tensor, active: Tensor) -> Tensor:
        raise NotImplementedError

    def _step_core(self, action: Tensor, active: Tensor) -> Tensor:
        k = self.num_labels
        node = torch.div(action, k, rounding_mode="floor")  # decode flat (node,color)
        color = action % k
        reward = self._delta_reward(node, color, active)  # computed BEFORE writing the label
        sel_n, sel_c = node[active], color[active]
        self.state["label"][sel_n] = sel_c
        self.state["selected"][sel_n] = True
        return reward

    @staticmethod
    def valid_mask(graph: BatchedGraph, state: State) -> Tensor:
        return state["label"] < 0  # any unassigned node (all k colours valid → no dead-ends)

    @classmethod
    def is_done(cls, graph: BatchedGraph, state: State, step_count: Tensor) -> Tensor:
        n_asg = scatter_sum((state["label"] >= 0).long(), graph.batch, graph.num_graphs)
        # all nodes labeled, or a hard cap at n steps (bounded; guards against no-op stalls)
        return (n_asg >= graph.graph_num_nodes) | (step_count >= graph.graph_num_nodes)

    @classmethod
    def node_features(cls, graph: BatchedGraph, state: State) -> Tensor:
        k = cls.num_labels
        label = state["label"]
        n = graph.num_nodes
        src, dst = graph.edge_index[0], graph.edge_index[1]
        # neighbour colour histogram [N,k] (pure function of label)
        hist = torch.zeros(n, k, device=label.device)
        nl = label[dst]
        m = nl >= 0
        if bool(m.any()):
            hist.index_put_((src[m], nl[m]), torch.ones(int(m.sum()), device=label.device), accumulate=True)
        deg = graph.degree().clamp_min(1.0).unsqueeze(-1)
        hist = hist / deg  # normalized colour fractions
        own = torch.zeros(n, k, device=label.device)
        a = label >= 0
        if bool(a.any()):
            own[a.nonzero(as_tuple=False).squeeze(-1), label[a]] = 1.0
        return torch.cat([hist, own, a.float().unsqueeze(-1)], dim=-1)  # [N, 2k+1]


@ENVS.register("maxkcut", aliases=["max_k_cut"])
class MaxKCutEnv(LabelEnv):
    """Max-k-cut: partition into k groups maximizing the number of bichromatic edges."""

    maximize = True

    def _delta_reward(self, node: Tensor, color: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        label = self.state["label"]
        chosen = torch.full((g.num_nodes,), -1, dtype=torch.long, device=g.device)
        chosen[node[active]] = color[active]  # colour assigned THIS step
        src, dst = g.edge_index[0], g.edge_index[1]
        m = chosen[src] >= 0  # out-edges of nodes coloured this step
        cut_new = m & (label[dst] >= 0) & (label[dst] != chosen[src])
        return scatter_sum(cut_new.float(), g.edge_batch, g.num_graphs)  # each cut edge counted once

    @classmethod
    def objective(cls, graph: BatchedGraph, state: State) -> Tensor:
        label = state["label"]
        s, d = graph.edge_index[0], graph.edge_index[1]
        cut = (label[s] >= 0) & (label[d] >= 0) & (label[s] != label[d])
        return 0.5 * scatter_sum(cut.float(), graph.edge_batch, graph.num_graphs)  # both directions /2


@ENVS.register("graph_coloring", aliases=["coloring", "kcoloring"])
class GraphColoringEnv(LabelEnv):
    """k-coloring: assign k colours minimizing the number of monochromatic edges."""

    maximize = False

    def _delta_reward(self, node: Tensor, color: Tensor, active: Tensor) -> Tensor:
        g = self.graph
        label = self.state["label"]
        chosen = torch.full((g.num_nodes,), -1, dtype=torch.long, device=g.device)
        chosen[node[active]] = color[active]
        src, dst = g.edge_index[0], g.edge_index[1]
        m = chosen[src] >= 0
        conf_new = m & (label[dst] >= 0) & (label[dst] == chosen[src])
        return -scatter_sum(conf_new.float(), g.edge_batch, g.num_graphs)  # conflicts added → negative

    @classmethod
    def objective(cls, graph: BatchedGraph, state: State) -> Tensor:
        label = state["label"]
        s, d = graph.edge_index[0], graph.edge_index[1]
        conf = (label[s] >= 0) & (label[d] >= 0) & (label[s] == label[d])
        return 0.5 * scatter_sum(conf.float(), graph.edge_batch, graph.num_graphs)
