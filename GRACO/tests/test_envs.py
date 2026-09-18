import networkx as nx
import torch

import graco.envs  # noqa: F401  (register)
from graco.data.batch import BatchedGraph
from graco.registries import ENVS
from graco.utils.segment_ops import segment_argmax


def _cut_value(edges, weights, spin):
    return sum(w * (1 - spin[u] * spin[v]) / 2 for (u, v), w in zip(edges, weights))


def test_maxcut_reward_equals_cut_delta():
    # single weighted triangle
    edges = [(0, 1), (1, 2), (0, 2)]
    weights = [1.0, 2.0, -1.0]
    ei = torch.tensor([[u for u, v in edges] + [v for u, v in edges],
                       [v for u, v in edges] + [u for u, v in edges]])
    ew = torch.tensor(weights + weights)
    bg = BatchedGraph.from_graph_list([ei], [3], [ew])
    env = ENVS.build({"type": "maxcut"})
    env.reset(bg.clone())

    spin = {0: 1, 1: 1, 2: 1}
    cut_before = _cut_value(edges, weights, spin)
    # flip node 1
    step = env.step(torch.tensor([1]))
    spin[1] = -1
    cut_after = _cut_value(edges, weights, spin)
    assert abs(float(step.reward[0]) - (cut_after - cut_before) / 3.0) < 1e-5


def test_maxcut_terminal_off_by_one():
    ei = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    bg = BatchedGraph.from_graph_list([ei], [3])
    env = ENVS.build({"type": "maxcut"})
    env.reset(bg.clone())
    env.step(torch.tensor([0]))
    step = env.step(torch.tensor([1]))
    assert bool(step.done[0])  # 2 flips on a 3-node graph -> terminal (n_sel+1>=n)


def test_mvc_covers_all_edges_and_is_cover():
    torch.manual_seed(0)
    g = nx.erdos_renyi_graph(20, 0.2, seed=1)
    bg = BatchedGraph.from_networkx(g)
    env = ENVS.build({"type": "mvc"})
    obs = env.reset(bg.clone())
    chosen = []
    steps = 0
    while not obs.done.all():
        a = segment_argmax(env.graph.degree(), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        chosen.append(int(a[0]))
        obs = env.step(a).obs
        steps += 1
        assert steps < 40
    cover = set(chosen)
    G = nx.convert_node_labels_to_integers(g)
    assert all((u in cover or v in cover) for u, v in G.edges())


def test_mis_is_independent():
    torch.manual_seed(0)
    g = nx.erdos_renyi_graph(20, 0.2, seed=2)
    bg = BatchedGraph.from_networkx(g)
    env = ENVS.build({"type": "mis"})
    obs = env.reset(bg.clone())
    chosen = []
    steps = 0
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        chosen.append(int(a[0]))
        obs = env.step(a).obs
        steps += 1
        assert steps < 40
    G = nx.convert_node_labels_to_integers(g)
    S = set(chosen)
    assert all(not G.has_edge(u, v) for u in S for v in S if u != v)


def test_dismantling_return_is_negative_anc():
    torch.manual_seed(0)
    g = nx.barabasi_albert_graph(30, 3, seed=3)
    bg = BatchedGraph.from_networkx(g)
    env = ENVS.build({"type": "dismantling"})
    obs = env.reset(bg.clone())
    total = 0.0
    steps = 0
    while not obs.done.all():
        a = segment_argmax(env.graph.degree(), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        step = env.step(a)
        total += float(step.reward[0])
        obs = step.obs
        steps += 1
        assert steps < 40
    assert total <= 0.0  # every LCC/n^2 term is negative
    assert bool(obs.done[0])


def test_all_envs_run_batched():
    torch.manual_seed(0)
    graphs = [nx.erdos_renyi_graph(n, 0.2, seed=i) for i, n in enumerate([10, 12, 14])]
    bg = BatchedGraph.from_networkx(graphs)
    for name in ["maxcut", "spinglass", "mvc", "mis", "dismantling", "dismantling_cost"]:
        env = ENVS.build({"type": name})
        obs = env.reset(bg.clone())
        steps = 0
        while not obs.done.all():
            a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
            obs = env.step(a).obs
            steps += 1
            assert steps < 60
