import networkx as nx
import torch

from graco.data.batch import BatchedGraph
from graco.utils import graph_algos, scatter, segment_ops


def _triangle_and_path():
    ei1 = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    ei2 = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    return BatchedGraph.from_graph_list([ei1, ei2], [3, 4])


def test_batching_shapes_and_batch_vector():
    bg = _triangle_and_path()
    assert bg.num_graphs == 2
    assert bg.num_nodes == 7
    assert bg.graph_num_nodes.tolist() == [3, 4]
    # no edge crosses graphs
    assert bool((bg.batch[bg.edge_index[0]] == bg.batch[bg.edge_index[1]]).all())
    assert bg.graph_num_undirected_edges.tolist() == [3, 3]


def test_degree_and_pooling():
    bg = _triangle_and_path()
    assert bg.degree().tolist() == [2, 2, 2, 1, 2, 2, 1]
    x = torch.ones(bg.num_nodes, 1)
    assert bg.pool(x, "sum").flatten().tolist() == [3.0, 4.0]


def test_scatter_ops():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    idx = torch.tensor([0, 0, 1, 1])
    assert scatter.scatter_sum(src, idx, 2).tolist() == [3.0, 7.0]
    assert scatter.scatter_mean(src, idx, 2).tolist() == [1.5, 3.5]
    assert scatter.scatter_max(src, idx, 2).tolist() == [2.0, 4.0]
    sm = scatter.scatter_softmax(src, idx, 2)
    assert abs(float(sm[:2].sum()) - 1.0) < 1e-5


def test_segment_argmax_and_mask():
    bg = _triangle_and_path()
    scores = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.7, 0.3, 0.4])
    a = segment_ops.segment_argmax(scores, bg.batch, bg.num_graphs)
    assert a.tolist() == [1, 4]  # global ids
    mask = torch.ones(7, dtype=torch.bool)
    mask[1] = False
    a2 = segment_ops.segment_argmax(scores, bg.batch, bg.num_graphs, mask)
    assert a2.tolist() == [2, 4]
    # all-masked graph -> -1
    mask2 = torch.zeros(7, dtype=torch.bool)
    a3 = segment_ops.segment_argmax(scores, bg.batch, bg.num_graphs, mask2)
    assert a3.tolist() == [-1, -1]


def test_connected_components_match_networkx():
    torch.manual_seed(0)
    graphs = [nx.erdos_renyi_graph(n, 0.15, seed=i) for i, n in enumerate([20, 30, 25])]
    bg = BatchedGraph.from_networkx(graphs)
    # remove a random subset of nodes per graph, compare LCC to networkx
    active = torch.ones(bg.num_nodes, dtype=torch.bool)
    removed = {0: [1, 5], 1: [], 2: [3, 4, 7, 8]}
    for g, rlist in removed.items():
        for r in rlist:
            active[int(bg.ptr[g]) + r] = False
    lcc = graph_algos.largest_component_size(
        bg.edge_index, bg.num_nodes, bg.batch, bg.num_graphs, active
    )
    for g, rlist in removed.items():
        G = nx.convert_node_labels_to_integers(graphs[g]).copy()
        G.remove_nodes_from(rlist)
        expected = max((len(c) for c in nx.connected_components(G)), default=0)
        expected = max(expected, 1) if G.number_of_nodes() > 0 else 0
        assert int(lcc[g]) == expected, (g, int(lcc[g]), expected)


def test_from_networkx_weights_symmetric():
    g = nx.Graph()
    g.add_edge(0, 1, weight=2.5)
    g.add_edge(1, 2, weight=-1.0)
    bg = BatchedGraph.from_networkx(g)
    assert bg.num_edges == 4  # 2 undirected -> 4 directed
    assert bg.edge_weight is not None
    # each undirected weight appears twice
    assert sorted(bg.edge_weight.tolist()) == sorted([2.5, 2.5, -1.0, -1.0])
