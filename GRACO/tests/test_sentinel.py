import math

import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.buffers  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.models.predictors import SentinelPredictor
from graco.registries import ALGOS, ENCODERS, ENVS, GENERATORS
from graco.utils.segment_ops import segment_argmax


def _gen(M=5, store_traj=True):
    return GENERATORS.build(
        {"type": "sentinel", "num_nodes": [24, 30], "base": "ba", "m": 3,
         "num_conditions": M, "timesteps": 20, "steady_window": 6, "store_traj": store_traj}
    )


def _gen_glv(M=5, store_traj=True, coupling=(0.2, 0.5), fixed=True):
    return GENERATORS.build(
        {"type": "sentinel", "dynamics": "glv", "fixed_graph": fixed, "graph_seed": 0,
         "num_nodes": [24, 24], "base": "ba", "m": 3, "num_conditions": M,
         "timesteps": 120, "steady_window": 40, "coupling_range": list(coupling),
         "r_range": [0.8, 1.2], "self_decay": 1.0, "dt": 0.2, "noise": 0.02,
         "store_traj": store_traj}
    )


# --------------------------------------------------------------- legacy (map)
def test_generator_emits_dynamics():
    bg = _gen().sample(4, rng=np.random.default_rng(0))
    assert bg.node_attr["dyn_mean"].shape[0] == bg.num_nodes
    assert bg.node_attr["dyn_mean"].shape[1] == 5
    assert bg.graph_attr["y"].shape == (4, 5)
    assert bg.node_attr["traj"].shape == (bg.num_nodes, 5, 20)


def test_env_error_reward_and_budget():
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 6})
    bg = _gen().sample(4, rng=np.random.default_rng(1))
    obs = env.reset(bg)
    e0 = env.objective(env.graph, env.state)  # empty-set baseline error
    steps = 0
    total = torch.zeros(4)
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        s = env.step(a)
        total += s.reward
        obs = s.obs
        steps += 1
        assert steps <= 6  # budget
    e_final = env.objective(env.graph, env.state)
    # sum of rewards telescopes to E0 - E_final (selection_cost=0)
    assert torch.allclose(total, (e0 - e_final), atol=1e-4)


def test_sentinel_encoder_uses_trajectory():
    bg = _gen().sample(3, rng=np.random.default_rng(2))
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5})
    obs = env.reset(bg)
    enc = ENCODERS.build(
        {"type": "sentinel_encoder", "hidden_dim": 16, "num_layers": 2, "dyn_dim": 8,
         "n_heads": 2, "transformer_layers": 1},
        in_dim=env.node_feature_dim, edge_dim=0,
    )
    out = enc(obs.graph)
    assert out.shape == (bg.num_nodes, enc.out_dim) and torch.isfinite(out).all()


def test_sentinel_dqn_trains_with_dynamics_through_replay():
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5})
    gen = _gen()
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 2, "dueling": True,
         "model": {"encoder": {"type": "sentinel_encoder", "hidden_dim": 16, "num_layers": 2,
                               "dyn_dim": 8, "n_heads": 2, "transformer_layers": 1},
                   "head": {"type": "dueling_q"}}},
        env=env, device="cpu",
    )
    rng = np.random.default_rng(0)
    for it in range(4):
        g = gen.sample(6, rng=rng)
        obs = env.reset(g)
        algo.on_episode_start(env)
        algo.set_progress(it / 4, it * 20)
        while not obs.done.all():
            a = algo.act(obs)
            s = env.step(a)
            algo.observe(env, obs, a, s)
            algo.after_step()
            obs = s.obs
        algo.after_episode(env)
    # dynamics data survived the replay collation
    batch = algo.buffer.sample(16, "cpu")
    assert "dyn_mean" in batch.graph.node_attr and "y" in batch.graph.graph_attr
    assert "traj" in batch.graph.node_attr


# ----------------------------------------------------------------- GLV + predictor
def test_glv_generator():
    bg = _gen_glv(M=5).sample(4, rng=np.random.default_rng(0))
    assert bg.node_attr["dyn_mean"].shape == (bg.num_nodes, 5)
    assert bg.graph_attr["y"].shape == (4, 5)
    assert bg.graph_attr["coupling"].shape == (4, 5)
    assert bg.graph_attr["converged"].shape == (4, 5)
    assert bg.graph_attr["converged"].dtype == torch.bool
    assert torch.isfinite(bg.node_attr["traj"]).all()
    c = bg.graph_attr["coupling"]
    assert (c >= 0.2).all() and (c <= 0.5).all()
    # fixed_graph + same graph_seed => same system (topology) across generators/samples
    bg2 = _gen_glv(M=5).sample(4, rng=np.random.default_rng(99))
    assert bg.num_nodes == bg2.num_nodes and torch.equal(bg.edge_index, bg2.edge_index)
    # weak-coupling stable regime => most conditions converge
    assert bg.graph_attr["converged"].float().mean() > 0.5


def test_robust_steady_state():
    gen = _gen_glv()
    T = 40
    torch.manual_seed(0)
    # noisy plateau at 2.0 -> converged, median ~= plateau (robust to noise)
    plateau = 2.0 + 0.02 * torch.randn(2, 1, T)
    a, conv = gen._steady_state(plateau, torch.tensor([0, 0]), 1)
    assert torch.allclose(a, torch.full_like(a, 2.0), atol=0.1)
    assert bool(conv[0, 0])
    # still-drifting ramp -> flagged non-converged
    ramp = torch.linspace(0.0, 4.0, T).view(1, 1, T).expand(2, 1, T).contiguous()
    _, conv2 = gen._steady_state(ramp, torch.tensor([0, 0]), 1)
    assert not bool(conv2[0, 0])


def test_predictor_reads_only_sentinels_and_is_finite():
    bg = _gen_glv(M=5).sample(3, rng=np.random.default_rng(2))
    pred = SentinelPredictor(num_conditions=5, d=16, n_heads=2, tf_layers=1)
    sel = torch.zeros(bg.num_nodes, dtype=torch.bool)
    g0 = (bg.batch == 0).nonzero(as_tuple=False).flatten()
    sel[g0[:2]] = True  # two sentinels in graph 0
    y0 = pred(bg, sel)
    assert y0.shape == (3, 5) and torch.isfinite(y0).all()
    # LEAKAGE: mutating a NON-selected node's trajectory must not change the prediction
    nonsel = (~sel).nonzero(as_tuple=False).flatten()[0]
    bg.node_attr["traj"][nonsel] += 5.0
    y1 = pred(bg, sel)
    assert torch.allclose(y0, y1, atol=1e-5)
    # empty set -> finite (learned E(empty) embedding); |S|=1..K all run
    assert torch.isfinite(pred(bg, torch.zeros_like(sel))).all()
    # gradients flow to the predictor
    pred(bg, sel).pow(2).mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in pred.parameters())


def test_env_learned_predictor_reward_and_telescoping():
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5})
    pred = SentinelPredictor(num_conditions=5, d=16, n_heads=2, tf_layers=1).eval()
    bg = _gen_glv(M=5).sample(3, rng=np.random.default_rng(3))
    # residual predictor starts at the analytic sentinel-mean anchor (correction≈0),
    # so an untrained predictor reproduces the analytic estimate (never worse baseline).
    env.reset(bg.clone())
    obj_analytic = env.objective(env.graph, env.state).clone()
    env.set_predictor(pred)
    obs = env.reset(bg.clone())
    obj_learned = env.objective(env.graph, env.state)
    assert obj_learned.shape == (3,) and torch.isfinite(obj_learned).all()
    assert torch.allclose(obj_learned, obj_analytic, atol=1e-4)  # residual anchor at init
    # with the predictor FROZEN across the episode, reward telescopes to E0 - E_final
    e0 = env.objective(env.graph, env.state)
    total = torch.zeros(3)
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        s = env.step(a)
        total += s.reward
        obs = s.obs
    e_final = env.objective(env.graph, env.state)
    assert torch.allclose(total, (e0 - e_final), atol=1e-4)
    # learned predictor also runs on a DISJOINT-range batch (extrapolation eval path)
    val = _gen_glv(M=5, coupling=(0.5, 0.8)).sample(3, rng=np.random.default_rng(4))
    env.reset(val.clone())
    assert torch.isfinite(env.objective(env.graph, env.state)).all()


def test_predictor_residual_correction_differs_from_anchor():
    # a trained/perturbed correction head moves the prediction off the analytic anchor
    bg = _gen_glv(M=5).sample(3, rng=np.random.default_rng(6))
    pred = SentinelPredictor(num_conditions=5, d=16, n_heads=2, tf_layers=1)
    sel = torch.zeros(bg.num_nodes, dtype=torch.bool)
    sel[(bg.batch == 0).nonzero(as_tuple=False).flatten()[:3]] = True
    anchor = pred._anchor(bg, sel)
    with torch.no_grad():  # perturb the correction head away from zero-init
        pred.head[-1].weight.normal_(std=0.5)
        pred.head[-1].bias.normal_(std=0.5)
    assert not torch.allclose(pred(bg, sel), anchor, atol=1e-3)
    # non-residual predictor predicts absolutely (no anchor)
    pred_abs = SentinelPredictor(num_conditions=5, d=16, n_heads=2, tf_layers=1, residual=False)
    assert torch.isfinite(pred_abs(bg, sel)).all()


def _sentinel_dqn(env):
    return ALGOS.build(
        {"type": "sentinel_dqn", "batch_size": 16, "learn_start": 16, "n_step": 2,
         "dueling": True, "pred_lr": 1e-3, "predictor": {"d": 16, "n_heads": 2, "tf_layers": 1},
         "model": {"encoder": {"type": "sentinel_encoder", "hidden_dim": 16, "num_layers": 2,
                               "dyn_dim": 8, "n_heads": 2, "transformer_layers": 1},
                   "head": {"type": "dueling_q"}}},
        env=env, device="cpu",
    )


def test_sentinel_dqn_smoke_and_checkpoint():
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5})
    gen = _gen_glv(M=5)
    algo = _sentinel_dqn(env)
    rng = np.random.default_rng(0)
    metrics = {}
    for it in range(5):
        g = gen.sample(6, rng=rng)
        obs = env.reset(g)
        algo.on_episode_start(env)
        algo.set_progress(it / 5, it * 20)
        while not obs.done.all():
            a = algo.act(obs)
            s = env.step(a)
            algo.observe(env, obs, a, s)
            m = algo.after_step()
            if m:
                metrics = m
            obs = s.obs
        algo.after_episode(env)
    assert "pred_loss" in metrics and math.isfinite(metrics["pred_loss"])
    # checkpoint round-trips policy + predictor + target predictor + optimizers
    sd = algo.state_dict()
    algo2 = _sentinel_dqn(ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5}))
    algo2.load_state_dict(sd)


def test_disjoint_stable_coupling_contract():
    train = _gen_glv(coupling=(0.2, 0.5))
    val = _gen_glv(coupling=(0.5, 0.8))
    # ranges disjoint (train strictly below val) and both on the stable side c < self_decay
    assert train.coupling_range[1] <= val.coupling_range[0]
    assert val.coupling_range[1] < train.self_decay


def test_residual_anchor_matches_env_analytic():
    # guard against silent drift between the predictor's anchor and the env's analytic
    # estimator (two copies of the sentinel-mean formula must stay bit-identical).
    bg = _gen_glv(M=5).sample(3, rng=np.random.default_rng(11))
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5})
    pred = SentinelPredictor(num_conditions=5, d=16, n_heads=2, tf_layers=1)
    sel = torch.zeros(bg.num_nodes, dtype=torch.bool)
    for gi in range(bg.num_graphs):  # a few sentinels per graph
        sel[(bg.batch == gi).nonzero(as_tuple=False).flatten()[:3]] = True
    assert torch.allclose(pred._anchor(bg, sel), env._estimate_analytic(bg, sel), atol=1e-6)


def test_eval_shadow_metric_is_generic():
    from graco.trainers.evaluator import Evaluator

    # sentinel env surfaces the analytic shadow metric alongside the learned objective
    env = ENVS.build({"type": "sentinel", "num_conditions": 5, "budget": 5})
    env.set_predictor(SentinelPredictor(num_conditions=5, d=16, n_heads=2, tf_layers=1).eval())
    val = [_gen_glv(M=5).sample(4, rng=np.random.default_rng(5))]

    class _Rand:
        def act(self, obs, explore=False):
            return segment_argmax(
                torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask
            )

    metrics = Evaluator(env, val, baselines=["random"]).evaluate(_Rand())
    assert "eval/objective" in metrics and "eval/objective_analytic" in metrics
    assert math.isfinite(metrics["eval/objective_analytic"])
    # generality: an unrelated env exposes NO extra metrics (hook defaults to none)
    assert ENVS.build({"type": "maxcut"}).eval_extra_metrics(None, {}) == {}
