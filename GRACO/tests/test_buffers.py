import torch

from graco.buffers.base import SingleGraph, Transition
from graco.buffers.prioritized import PrioritizedNStepReplay
from graco.buffers.replay import UniformNStepReplay, make_nstep_transitions


def _dummy_graph(n=5):
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    return SingleGraph(edge_index=ei, num_nodes=n, edge_weight=torch.ones(4))


def test_nstep_transition_construction():
    g = _dummy_graph()
    trans = make_nstep_transitions(g, actions=[0, 1, 2], rewards=[1.0, 2.0, 3.0], n_step=2, gamma=1.0)
    assert len(trans) == 3
    # i=0: 2-step reward 1+2=3, non-terminal, next selects {0,1}
    assert abs(trans[0].reward - 3.0) < 1e-6
    assert trans[0].terminal is False
    assert trans[0].next_state["selected"].tolist() == [True, True, False, False, False]
    assert trans[0].state["selected"].tolist() == [False] * 5
    # i=1: window runs off end -> terminal, reward 2+3=5
    assert abs(trans[1].reward - 5.0) < 1e-6
    assert trans[1].terminal is True
    assert trans[2].terminal is True


def test_nstep_discount():
    g = _dummy_graph()
    trans = make_nstep_transitions(g, [0, 1, 2], [1.0, 2.0, 3.0], n_step=2, gamma=0.5)
    assert abs(trans[0].reward - (1.0 + 0.5 * 2.0)) < 1e-6


def test_uniform_replay_sample_shapes():
    g = _dummy_graph()
    buf = UniformNStepReplay(capacity=100)
    buf.add_many(make_nstep_transitions(g, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0], n_step=2, gamma=1.0))
    batch = buf.sample(8, device="cpu")
    assert batch.action.shape == (8,)
    assert batch.reward.shape == (8,)
    assert batch.graph.num_graphs == 8
    assert batch.state["selected"].shape[0] == batch.graph.num_nodes


def test_prioritized_replay_proportional():
    torch.manual_seed(0)
    g = _dummy_graph()
    buf = PrioritizedNStepReplay(capacity=8, alpha=1.0)
    trans = make_nstep_transitions(g, [0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0], n_step=1, gamma=1.0)
    buf.add_many(trans)  # all get max priority initially
    # give transition 0 a huge TD error, others tiny
    idx = torch.tensor([0, 1, 2, 3])
    buf.update_priorities(idx, torch.tensor([10.0, 0.001, 0.001, 0.001]))
    counts = torch.zeros(4)
    for _ in range(40):
        b = buf.sample(16, device="cpu")
        for i in b.indices.tolist():
            counts[i] += 1
    assert counts[0] > counts[1:].sum()  # highest-priority sampled most


def test_prioritized_tree_sum_consistent():
    g = _dummy_graph()
    buf = PrioritizedNStepReplay(capacity=16, alpha=0.6)
    buf.add_many(make_nstep_transitions(g, [0, 1, 2], [0.0, 0.0, 0.0], n_step=1, gamma=1.0))
    buf.update_priorities(torch.tensor([0, 1, 2]), torch.tensor([0.5, 0.3, 0.2]))
    leaves = buf.tree[buf.capacity : buf.capacity + 3].sum()
    assert abs(float(buf.tree[1]) - float(leaves)) < 1e-6  # root == sum of leaves
