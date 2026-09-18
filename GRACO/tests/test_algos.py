import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS


def _train_a_bit(env_name, algo_name, algo_kw, iters=4, num_envs=8):
    torch.manual_seed(0)
    np.random.seed(0)
    env = ENVS.build({"type": env_name})
    gen = GENERATORS.build({"type": "er", "num_nodes": [10, 14], "p": 0.25, "weighted": True, "weight_dist": "bimodal"})
    algo = ALGOS.build({"type": algo_name, **algo_kw}, env=env, device="cpu")
    rng = np.random.default_rng(0)
    for it in range(iters):
        g = gen.sample(num_envs, device="cpu", rng=rng)
        obs = env.reset(g)
        algo.on_episode_start(env)
        algo.set_progress(it / iters, it * 20)
        while not obs.done.all():
            a = algo.act(obs, explore=True)
            step = env.step(a)
            algo.observe(env, obs, a, step)
            algo.after_step()
            obs = step.obs
        algo.after_episode(env)
    return env, algo


def test_dqn_trains_and_checkpoints():
    env, algo = _train_a_bit(
        "dismantling", "dqn",
        {"batch_size": 16, "learn_start": 16, "model": {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}},
    )
    sd = algo.state_dict()
    # a fresh algo can load the checkpoint
    env2 = ENVS.build({"type": "dismantling"})
    algo2 = ALGOS.build({"type": "dqn", "model": {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}}, env=env2, device="cpu")
    algo2.load_state_dict(sd)
    for p1, p2 in zip(algo.policy.parameters(), algo2.policy.parameters()):
        assert torch.allclose(p1, p2)


def test_dqn_dueling_per():
    _train_a_bit(
        "mvc", "dqn",
        {"batch_size": 16, "learn_start": 16, "dueling": True,
         "model": {"encoder": {"type": "gin", "hidden_dim": 16, "num_layers": 2}, "head": {"type": "dueling_q"}},
         "buffer": {"type": "prioritized_replay", "capacity": 1000}},
    )


def test_qrdqn():
    _train_a_bit(
        "dismantling", "qrdqn",
        {"batch_size": 16, "learn_start": 16, "num_quantiles": 8,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2}}},
    )


def test_ppo():
    _train_a_bit(
        "maxcut", "ppo",
        {"minibatch_size": 64, "ppo_epochs": 2,
         "model": {"encoder": {"type": "gatv2", "hidden_dim": 16, "num_layers": 2, "num_heads": 2}}},
    )


def test_a2c():
    _train_a_bit("mis", "a2c", {"model": {"encoder": {"type": "s2v", "hidden_dim": 16, "num_layers": 2}}})


def test_reinforce():
    _train_a_bit("mvc", "reinforce", {"model": {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}})


def test_dqn_greedy_action_valid():
    env, algo = _train_a_bit("maxcut", "dqn", {"batch_size": 16, "learn_start": 16, "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2}}})
    gen = GENERATORS.build({"type": "er", "num_nodes": 12, "p": 0.3})
    obs = env.reset(gen.sample(4, device="cpu", rng=np.random.default_rng(1)))
    a = algo.act(obs, explore=False)
    # greedy actions are legal nodes (or -1 if a graph is already terminal)
    for g in range(obs.graph.num_graphs):
        if a[g] >= 0:
            assert bool(obs.action_mask[a[g]])
