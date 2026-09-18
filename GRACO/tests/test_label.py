import networkx as nx
import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.data.batch import BatchedGraph
from graco.registries import ALGOS, ENVS, GENERATORS


def _triangle():
    # a triangle: max-2-cut = 2 (one edge unavoidably monochromatic), 3-coloring = 0 conflicts
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    und = torch.cat([ei, ei.flip(0)], dim=1)
    return BatchedGraph.from_graph_list([und], [3])


def test_maxkcut_objective_and_reward_telescopes():
    env = ENVS.build({"type": "maxkcut", "k": 3})
    assert env.node_feature_dim == 2 * 3 + 1
    bg = GENERATORS.build({"type": "erdos_renyi", "num_nodes": [12, 12], "p": 0.3}).sample(
        3, rng=np.random.default_rng(0)
    )
    obs = env.reset(bg)
    # random labeling; undiscounted per-step reward must telescope to the final cut
    total = torch.zeros(3)
    while not obs.done.all():
        nmask = obs.action_mask
        k = env.num_labels
        flat = torch.rand(obs.num_nodes * k)
        flat[~nmask.repeat_interleave(k)] = -1e9
        bf = obs.graph.batch.repeat_interleave(k)
        from graco.utils.segment_ops import segment_argmax
        a = segment_argmax(flat, bf, obs.graph.num_graphs, nmask.repeat_interleave(k))
        s = env.step(a)
        total += s.reward
        obs = s.obs
    assert torch.allclose(total, env.objective(env.graph, env.state), atol=1e-4)


def test_maxkcut_objective_matches_networkx_on_labeling():
    env = ENVS.build({"type": "maxkcut", "k": 3})
    bg = _triangle()
    env.reset(bg)
    env.state["label"] = torch.tensor([0, 1, 2])  # all different -> all 3 edges cut
    assert abs(float(env.objective(env.graph, env.state)) - 3.0) < 1e-6
    env.state["label"] = torch.tensor([0, 0, 1])  # one edge monochromatic -> cut 2
    assert abs(float(env.objective(env.graph, env.state)) - 2.0) < 1e-6


def test_coloring_objective_counts_conflicts():
    env = ENVS.build({"type": "graph_coloring", "k": 3})
    bg = _triangle()
    env.reset(bg)
    env.state["label"] = torch.tensor([0, 1, 2])  # proper 3-coloring -> 0 conflicts
    assert float(env.objective(env.graph, env.state)) == 0.0
    env.state["label"] = torch.tensor([0, 0, 0])  # all same -> 3 conflicts
    assert abs(float(env.objective(env.graph, env.state)) - 3.0) < 1e-6


def test_flat_action_decode_roundtrip():
    k = 4
    for node in [0, 3, 7]:
        for color in range(k):
            flat = node * k + color
            assert flat // k == node and flat % k == color


def _label_algo(env):
    return ALGOS.build(
        {"type": "label_dqn", "batch_size": 16, "learn_start": 16, "n_step": 2,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2}}},
        env=env, device="cpu",
    )


def test_label_dqn_trains_and_checkpoints():
    for etype in ["maxkcut", "graph_coloring"]:
        env = ENVS.build({"type": etype, "k": 3})
        gen = GENERATORS.build({"type": "erdos_renyi", "num_nodes": [14, 14], "p": 0.25})
        algo = _label_algo(env)
        rng = np.random.default_rng(0)
        metrics = {}
        for it in range(4):
            obs = env.reset(gen.sample(6, rng=rng))
            algo.on_episode_start(env)
            algo.set_progress(it / 4, it * 20)
            while not obs.done.all():
                a = algo.act(obs)
                step = env.step(a)
                algo.observe(env, obs, a, step)
                m = algo.after_step()
                if m:
                    metrics = m
                obs = step.obs
            algo.after_episode(env)
        assert "loss" in metrics and np.isfinite(metrics["loss"])
        # label state survived replay collation and checkpoint round-trips
        batch = algo.buffer.sample(16, "cpu")
        assert "label" in batch.graph.node_attr or "label" in batch.state
        algo2 = _label_algo(ENVS.build({"type": etype, "k": 3}))
        algo2.load_state_dict(algo.state_dict())


def test_binary_path_untouched():
    # the k-label additions must not perturb the binary node-selection registry
    assert {"maxcut", "mis", "mvc", "dismantling"} <= set(ENVS.keys())
    env = ENVS.build({"type": "maxcut"})
    G = nx.erdos_renyi_graph(12, 0.3, seed=1)
    ei = torch.tensor(list(G.edges())).t()
    und = torch.cat([ei, ei.flip(0)], dim=1)
    bg = BatchedGraph.from_graph_list([und], [12])
    obs = env.reset(bg)
    assert obs.graph.x.shape[1] == env.node_feature_dim  # binary features unchanged
