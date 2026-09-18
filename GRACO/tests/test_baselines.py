"""FINDER / DIRAC baseline reproductions."""

import numpy as np
import torch

import graco.baselines  # noqa: F401  (registers finder + dirac)
import graco.buffers  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS


def _train(env_cfg, gen_cfg, algo_cfg, iters=3, num_envs=8):
    torch.manual_seed(0)
    np.random.seed(0)
    env = ENVS.build(dict(env_cfg))
    gen = GENERATORS.build(dict(gen_cfg))
    algo = ALGOS.build(dict(algo_cfg), env=env, device="cpu")
    rng = np.random.default_rng(0)
    for it in range(iters):
        g = gen.sample(num_envs, device="cpu", rng=rng)
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
    return env, algo


def test_finder_baseline_trains():
    env, algo = _train(
        {"type": "finder_dismantling"},
        {"type": "barabasi_albert", "num_nodes": [20, 30], "m": 4},
        {"type": "finder", "batch_size": 16, "learn_start": 16,
         "model": {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}},
    )
    # aux_feat is produced by the env and survives collation
    batch = algo.buffer.sample(16, "cpu")
    obs_graph = type(env).obs_graph(batch.graph, batch.state)
    assert "aux_feat" in obs_graph.meta
    assert obs_graph.meta["aux_feat"].shape == (16, 4)
    assert obs_graph.x.shape[1] == 2  # ones[n, 2]


def test_finder_reconstruction_loss_positive():
    env, algo = _train(
        {"type": "finder_dismantling"},
        {"type": "barabasi_albert", "num_nodes": [20, 30], "m": 4},
        {"type": "finder", "batch_size": 16, "learn_start": 16, "recon_alpha": 1.0,
         "model": {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}},
    )
    batch = algo.buffer.sample(16, "cpu")
    obs_t = type(env).obs_graph(batch.graph, batch.state)
    assert float(algo._aux_loss(obs_t, None)) > 0


def test_dirac_baseline_trains_with_coords():
    env, algo = _train(
        {"type": "dirac_spinglass"},
        {"type": "dirac_lattice", "dim": 3, "num_nodes": [27, 27], "weighted": True,
         "weight_dist": "normal"},
        {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 3,
         "bootstrap_clip_min": 0.0, "double": False,
         "model": {"encoder": {"type": "s2v", "hidden_dim": 16, "num_layers": 3}}},
    )
    batch = algo.buffer.sample(16, "cpu")
    # lattice coordinates survive replay collation and drive node features
    assert "node_pos" in batch.graph.meta
    obs_graph = type(env).obs_graph(batch.graph, batch.state)
    assert obs_graph.x.shape[1] == 4  # coords padded to 4
    assert env.node_feature_dim == 4 and env.edge_feature_dim == 4
