"""Typed-graph training: node_type / edge_type / hypergraph incidence must
survive the replay/collate round-trip so typed encoders train, not fall back."""

import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.buffers  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS


def _train_typed(env_name, gen_cfg, encoder, iters=3):
    torch.manual_seed(0)
    np.random.seed(0)
    env = ENVS.build({"type": env_name})
    gen = GENERATORS.build(dict(gen_cfg))
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 2,
         "model": {"encoder": {"type": encoder, "hidden_dim": 16, "num_layers": 2}}},
        env=env, device="cpu",
    )
    rng = np.random.default_rng(0)
    for it in range(iters):
        g = gen.sample(8, device="cpu", rng=rng)
        obs = env.reset(g)
        algo.on_episode_start(env)
        algo.set_progress(it / iters, it * 20)
        while not obs.done.all():
            a = algo.act(obs, explore=True)
            s = env.step(a)
            algo.observe(env, obs, a, s)
            algo.after_step()
            obs = s.obs
        algo.after_episode(env)
    return algo.buffer.sample(16, "cpu")


def test_multiplex_rgcn_keeps_layers():
    b = _train_typed("dismantling", {"type": "multiplex", "num_nodes": [16, 20]}, "rgcn")
    assert b.graph.edge_type is not None
    assert b.graph.meta.get("num_layers") is not None


def test_hetero_hgt_keeps_types():
    b = _train_typed("mvc", {"type": "hetero", "num_nodes": [16, 20]}, "hgt")
    assert b.graph.node_type is not None and b.graph.edge_type is not None
    assert b.graph.meta.get("num_edge_types") is not None


def test_hypergraph_hgnn_keeps_incidence():
    b = _train_typed("dismantling", {"type": "hypergraph", "num_nodes": [16, 20]}, "hgnn")
    assert "inc_node" in b.graph.meta and "inc_hyperedge" in b.graph.meta
    assert b.graph.meta["inc_node"].numel() == b.graph.meta["inc_hyperedge"].numel()


def test_setcover_keeps_node_type():
    b = _train_typed("set_cover", {"type": "set_cover", "num_elements": [12, 16], "num_sets": 6}, "han")
    assert b.graph.node_type is not None
