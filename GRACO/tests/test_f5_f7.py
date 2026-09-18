import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.buffers  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS
from graco.utils.segment_ops import segment_argmax


def _loop(env, algo, gen, iters=4, num_envs=6, seed=0):
    rng = np.random.default_rng(seed)
    for it in range(iters):
        g = gen.sample(num_envs, rng=rng)
        g = algo.on_sample(g)  # POMO multi-start replication (identity for others)
        obs = env.reset(g)
        algo.on_episode_start(env)
        algo.set_progress(it / iters, it * 20)
        while not obs.done.all():
            a = algo.act(obs)
            s = env.step(a)
            algo.observe(env, obs, a, s)
            algo.after_step()
            obs = s.obs
        algo.after_episode(env)


def test_pomo_replicates_and_trains():
    env = ENVS.build({"type": "maxcut"})
    gen = GENERATORS.build(
        {"type": "er", "num_nodes": [12, 16], "p": 0.4, "weighted": True, "weight_dist": "bimodal"}
    )
    algo = ALGOS.build(
        {"type": "pomo", "pomo_size": 4,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2}}},
        env=env, device="cpu",
    )
    g = gen.sample(4, rng=np.random.default_rng(0))
    assert algo.on_sample(g).num_graphs == 4 * 4  # multi-start replication
    _loop(env, algo, gen, num_envs=4)


def test_eco_env_invariants():
    env = ENVS.build({"type": "eco_maxcut", "horizon_frac": 2.0})
    gen = GENERATORS.build(
        {"type": "er", "num_nodes": [10, 10], "p": 0.4, "weighted": True, "weight_dist": "bimodal"}
    )
    g = gen.sample(4, rng=np.random.default_rng(1))
    obs = env.reset(g)
    best0 = env.objective(env.graph, env.state).clone()
    assert bool(obs.action_mask.all())  # every node is a legal (revisitable) flip
    steps = 0
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        s = env.step(a)
        assert (s.reward >= -1e-6).all()  # best-so-far shaping => non-negative reward
        obs = s.obs
        steps += 1
        assert steps < 60
    best_final = env.objective(env.graph, env.state)
    assert (best_final >= best0 - 1e-6).all()  # best-so-far is non-decreasing


def test_eco_dqn_trains():
    env = ENVS.build({"type": "eco_maxcut"})
    gen = GENERATORS.build(
        {"type": "er", "num_nodes": [12, 12], "p": 0.4, "weighted": True, "weight_dist": "bimodal"}
    )
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 2,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2}}},
        env=env, device="cpu",
    )
    _loop(env, algo, gen, num_envs=6)
