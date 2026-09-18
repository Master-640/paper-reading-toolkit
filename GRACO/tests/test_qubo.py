import itertools

import numpy as np
import torch

import graco
import graco.algos  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.generators.qubo import QUBOGenerator
from graco.registries import ALGOS, ENVS, GENERATORS
from graco.utils.segment_ops import segment_argmax


def _brute_force_qubo(Q):
    """Exact min of x^T Q x over x in {0,1}^n (small n)."""
    n = Q.shape[0]
    best = float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        best = min(best, float(x @ Q @ x))
    return best


def test_qubo_energy_matches_matrix_form():
    rng = np.random.default_rng(0)
    Q = rng.uniform(-1, 1, size=(6, 6))
    Q = (Q + Q.T) / 2  # symmetric
    g = graco.qubo(Q)
    env = ENVS.build({"type": "qubo"})
    env.reset(g)
    # energy of a specific assignment must equal x^T Q x
    x = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)
    env.state["selected"] = x
    e_env = float(env._energy(env.graph, x))
    xf = x.numpy().astype(float)
    assert abs(e_env - float(xf @ Q @ xf)) < 1e-4


def test_qubo_reaches_optimum_by_flips_small():
    rng = np.random.default_rng(1)
    Q = rng.uniform(-1, 1, size=(7, 7))
    Q = (Q + Q.T) / 2
    opt = _brute_force_qubo(Q)
    g = graco.qubo(Q)
    env = ENVS.build({"type": "qubo", "horizon_frac": 6.0})
    obs = env.reset(g)
    # greedy best-flip policy should reach the exact optimum on a tiny instance
    while not obs.done.all():
        x = env.state["selected"].to(torch.float32)
        from graco.envs.qubo import _local_field

        gain = (1.0 - 2.0 * x) * _local_field(env.graph, env.state["selected"])  # ΔE per flip
        a = segment_argmax(-gain, obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        obs = env.step(a).obs
    assert float(env.objective(env.graph, env.state)) <= opt + 1e-4


def test_from_matrices_batches_and_carries_bias():
    Qs = [np.diag([1.0, -2.0, 3.0]), np.array([[0.0, 1.0], [1.0, 0.0]])]
    g = QUBOGenerator.from_matrices(Qs)
    assert g.num_graphs == 2 and g.num_nodes == 5
    assert torch.allclose(g.node_attr["q_diag"][:3], torch.tensor([1.0, -2.0, 3.0]))


def test_qubo_generator_emits_bias():
    gen = GENERATORS.build({"type": "qubo", "num_nodes": [12, 12], "base": "er", "p": 0.3})
    bg = gen.sample(4, rng=np.random.default_rng(0))
    assert bg.node_attr["q_diag"].shape[0] == bg.num_nodes
    assert bg.edge_weight is not None  # couplings present


def test_qubo_dqn_trains_and_bias_survives_replay():
    env = ENVS.build({"type": "qubo", "horizon_frac": 1.0})
    gen = GENERATORS.build({"type": "qubo", "num_nodes": [16, 16], "base": "er", "p": 0.2})
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 2, "dueling": True,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2},
                   "head": {"type": "dueling_q"}}},
        env=env, device="cpu",
    )
    rng = np.random.default_rng(0)
    for it in range(4):
        obs = env.reset(gen.sample(6, rng=rng))
        algo.on_episode_start(env)
        algo.set_progress(it / 4, it * 20)
        while not obs.done.all():
            a = algo.act(obs)
            step = env.step(a)
            algo.observe(env, obs, a, step)
            algo.after_step()
            obs = step.obs
        algo.after_episode(env)
    batch = algo.buffer.sample(16, "cpu")
    assert "q_diag" in batch.graph.node_attr  # per-node bias survived replay collation
