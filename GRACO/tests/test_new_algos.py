import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.buffers  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS

ENC = {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}


def _train(env_name, algo_name, kw, iters=5, num_envs=8):
    torch.manual_seed(0)
    np.random.seed(0)
    env = ENVS.build({"type": env_name})
    gen = GENERATORS.build({"type": "er", "num_nodes": [10, 14], "p": 0.25})
    algo = ALGOS.build({"type": algo_name, **kw}, env=env, device="cpu")
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


def test_c51():
    _train("dismantling", "c51", {"batch_size": 16, "learn_start": 16, "num_atoms": 21,
                                   "v_min": -2, "v_max": 0, "model": ENC})


def test_iqn():
    _train("mvc", "iqn", {"batch_size": 16, "learn_start": 16, "num_tau": 8, "model": ENC})


def test_munchausen():
    _train("dismantling", "munchausen_dqn", {"batch_size": 16, "learn_start": 16, "model": ENC})


def test_sac_and_checkpoint():
    env, algo = _train("dismantling", "sac", {"batch_size": 16, "learn_start": 16, "model": ENC})
    sd = algo.state_dict()
    env2 = ENVS.build({"type": "dismantling"})
    algo2 = ALGOS.build({"type": "sac", "batch_size": 16, "learn_start": 16, "model": ENC}, env=env2, device="cpu")
    algo2.load_state_dict(sd)


def test_reinforce_rollout():
    _train("mis", "reinforce_rollout", {"model": ENC})


def test_noisy_head_dqn():
    _train("dismantling", "dqn", {"batch_size": 16, "learn_start": 16, "eps_start": 0.0,
                                  "eps_end": 0.0,
                                  "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2},
                                            "head": {"type": "noisy_q"}}})
