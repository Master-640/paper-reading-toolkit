import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.buffers  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.eval.decoding import decode
from graco.registries import ALGOS, ENVS, GENERATORS


def _tiny_dqn():
    env = ENVS.build({"type": "maxcut"})
    gen = GENERATORS.build(
        {"type": "er", "num_nodes": [12, 14], "p": 0.4, "weighted": True, "weight_dist": "bimodal"}
    )
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 16, "learn_start": 16,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2}}},
        env=env, device="cpu",
    )
    rng = np.random.default_rng(0)
    for it in range(3):
        g = gen.sample(8, rng=rng)
        obs = env.reset(g)
        algo.on_episode_start(env)
        algo.set_progress(it / 3, it * 20)
        while not obs.done.all():
            a = algo.act(obs)
            s = env.step(a)
            algo.observe(env, obs, a, s)
            algo.after_step()
            obs = s.obs
        algo.after_episode(env)
    return env, algo, gen


def test_decode_strategies_shape_and_finite():
    env, algo, gen = _tiny_dqn()
    g = gen.sample(6, rng=np.random.default_rng(9))
    for strat in ["greedy", "sample", "multistart"]:
        o, r = decode(env, g.clone(), algo, strategy=strat, samples=8)
        assert o.shape == (6,) and r.shape == (6,)
        assert torch.isfinite(o).all() and torch.isfinite(r).all()


def test_more_samples_not_worse_take_best():
    env, algo, gen = _tiny_dqn()
    g = gen.sample(16, rng=np.random.default_rng(3))
    o1, _ = decode(env, g.clone(), algo, strategy="sample", samples=1,
                   generator=torch.Generator().manual_seed(1))
    o16, _ = decode(env, g.clone(), algo, strategy="sample", samples=16,
                    generator=torch.Generator().manual_seed(2))
    assert float(o16.mean()) >= float(o1.mean()) - 0.05  # take-best over more rollouts


def test_decode_unknown_strategy_raises():
    env, algo, gen = _tiny_dqn()
    g = gen.sample(2)
    try:
        decode(env, g, algo, strategy="nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
