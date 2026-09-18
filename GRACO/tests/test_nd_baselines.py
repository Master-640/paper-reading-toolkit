import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ALGOS, ENVS, GENERATORS
from graco.utils.segment_ops import segment_argmax


def test_line_graph_transform_matches_edge_count():
    import networkx as nx
    g = GENERATORS.build({"type": "line_graph", "base": {"type": "barabasi_albert", "num_nodes": [20, 20], "m": 3}})
    bg = g.sample(3, rng=np.random.default_rng(0))
    # each line-graph node == an edge of the base graph
    per = (bg.ptr[1:] - bg.ptr[:-1]).tolist()
    ref = nx.barabasi_albert_graph(20, 3, seed=1).number_of_edges()
    assert all(p == ref for p in per)  # BA(20,3) has a fixed edge count
    assert "orig_edge" in bg.node_attr and bg.meta["orig_num_nodes"] == 60


def test_edge_dismantling_measures_original_graph():
    g = GENERATORS.build({"type": "line_graph", "base": {"type": "barabasi_albert", "num_nodes": [15, 15], "m": 2}})
    bg = g.sample(2, rng=np.random.default_rng(0))
    env = ENVS.build({"type": "edge_dismantling", "metric": "pairwise"})
    env.reset(bg)
    full = env.objective(env.graph, env.state)  # no edges removed
    env.state["selected"] = torch.ones(bg.num_nodes, dtype=torch.bool)  # remove all edges
    empty = env.objective(env.graph, env.state)
    assert (full == 15 * 14 / 2).all() and (empty == 0).all()  # 15-node connected -> singletons


def test_pairwise_dismantling_telescopes():
    env = ENVS.build({"type": "pairwise_dismantling"})
    bg = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [20, 20], "m": 3}).sample(3, rng=np.random.default_rng(1))
    obs = env.reset(bg)
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        obs = env.step(a).obs
    assert (env.objective(env.graph, env.state) == 0).all()  # fully dismantled -> 0 connected pairs


def _smoke(cfg_env, gen):
    env = ENVS.build(cfg_env)
    gen = GENERATORS.build(gen)
    algo = ALGOS.build(
        {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 2, "dueling": True,
         "model": {"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2, "aggr": "sum"},
                   "head": {"type": "dueling_q"}}},
        env=env, device="cpu",
    )
    rng = np.random.default_rng(0)
    for it in range(3):
        obs = env.reset(gen.sample(4, rng=rng))
        algo.on_episode_start(env)
        algo.set_progress(it / 3, it * 20)
        while not obs.done.all():
            a = algo.act(obs)
            step = env.step(a)
            algo.observe(env, obs, a, step)
            algo.after_step()
            obs = step.obs
        algo.after_episode(env)
    assert len(algo.buffer) > 0


def test_nd_baselines_train_smoke():
    _smoke({"type": "pairwise_dismantling"}, {"type": "barabasi_albert", "num_nodes": [16, 16], "m": 3})
    _smoke({"type": "edge_dismantling", "metric": "edge_sq"},
           {"type": "line_graph", "base": {"type": "erdos_renyi", "num_nodes": [16, 16], "p": 0.2}})
    _smoke({"type": "dismantling"},  # HITTER substrate: dismantling on a hypergraph
           {"type": "hypergraph", "num_nodes": [16, 16], "num_hyperedges": 10, "hyperedge_size": 3})


def test_multiplex_gmcc_and_env():
    from graco.utils.graph_algos import mutually_connected_component_size
    # 3 nodes; layer0: 0-1-2 path, layer1: 0-1 only -> GMCC = {0,1} (size 2)
    ei = torch.tensor([[0, 1, 1, 2, 0, 1], [1, 0, 2, 1, 1, 0]])
    et = torch.tensor([0, 0, 0, 0, 1, 1])
    g = mutually_connected_component_size(ei, et, 2, 3, torch.zeros(3, dtype=torch.long), 1,
                                          torch.ones(3, dtype=torch.bool))
    assert float(g[0]) == 2.0
    _smoke({"type": "multiplex_dismantling"},
           {"type": "multiplex", "num_layers": 2, "base": "er", "p": 0.15, "num_nodes": [16, 16]})


def test_sis_risk_and_env():
    from graco.utils.graph_algos import sis_infection_risk
    bg = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [20, 20], "m": 3}).sample(2, rng=np.random.default_rng(0))
    full = sis_infection_risk(bg.edge_index, bg.num_nodes, bg.batch, bg.num_graphs, torch.ones(bg.num_nodes, dtype=torch.bool))
    none = sis_infection_risk(bg.edge_index, bg.num_nodes, bg.batch, bg.num_graphs, torch.zeros(bg.num_nodes, dtype=torch.bool))
    assert float(full.sum()) > 0 and float(none.sum()) == 0.0
    _smoke({"type": "sis_dismantling"}, {"type": "barabasi_albert", "num_nodes": [16, 16], "m": 3})


def test_mind_encoder_shape_and_smoke():
    from graco.registries import ENCODERS
    env = ENVS.build({"type": "dismantling"})
    bg = GENERATORS.build({"type": "erdos_renyi", "num_nodes": [24, 24], "p": 0.15}).sample(3, rng=np.random.default_rng(0))
    obs = env.reset(bg)
    enc = ENCODERS.build({"type": "mind", "hidden_dim": 32, "num_layers": 3}, in_dim=env.node_feature_dim, edge_dim=0)
    h = enc(obs.graph)
    assert tuple(h.shape) == (bg.num_nodes, enc.out_dim) and bool(torch.isfinite(h).all())
    # MIND pairs with SAC; smoke it via the shared DQN harness on the same encoder
    _smoke({"type": "dismantling"}, {"type": "erdos_renyi", "num_nodes": [16, 16], "p": 0.2})


def test_combat_operational_capability():
    from graco.utils.graph_algos import combat_operational_capability as cap
    # 2 sensors -> 1 decision -> 2 influencers => 2*2 = 4 combat chains
    und = torch.tensor([[0, 2], [1, 2], [2, 3], [2, 4]]).t()
    ei = torch.cat([und, und.flip(0)], dim=1)
    nt = torch.tensor([0, 0, 1, 2, 2])
    b = torch.zeros(5, dtype=torch.long)
    assert float(cap(ei, nt, 5, b, 1)[0]) == 4.0
    # removing the sole decision node destroys every chain
    dead = cap(ei, nt, 5, b, 1, torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool))
    assert float(dead[0]) == 0.0


def test_combat_dismantling_env_and_smoke():
    g = GENERATORS.build({"type": "combat", "num_nodes": [40, 50], "p_sd": 0.3, "p_di": 0.3})
    bg = g.sample(3, rng=np.random.default_rng(0))
    assert sorted(set(bg.node_type.tolist())) == [0, 1, 2] and bg.meta["num_node_types"] == 3
    env = ENVS.build({"type": "combat_dismantling"})
    obs = env.reset(bg)
    assert (env.objective(env.graph, env.state) > 0).all()  # network is initially operational
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        obs = env.step(a).obs
    assert (env.objective(env.graph, env.state) == 0).all()  # every combat chain destroyed
    _smoke({"type": "combat_dismantling"}, {"type": "combat", "num_nodes": [24, 24], "p_sd": 0.3, "p_di": 0.3})


def test_amh_improves_over_its_init():
    import graco.heuristics  # noqa: F401
    from graco.heuristics.base import HEURISTICS
    env = ENVS.build({"type": "dismantling"})
    bg = GENERATORS.build({"type": "erdos_renyi", "num_nodes": [40, 40], "p": 0.09}).sample(4, rng=np.random.default_rng(2))
    r0 = HEURISTICS.get("amh")(iters=0, seed=1).solve(env, bg.clone())
    rk = HEURISTICS.get("amh")(iters=500, seed=1).solve(env, bg.clone())
    # the adaptive search only accepts improving moves, so its return never drops below the init
    assert (rk.ret >= r0.ret - 1e-6).all() and rk.ret.sum() > r0.ret.sum()
    # gated to plain LCC dismantling only (specialized objectives redefine the metric)
    assert HEURISTICS.get("amh")().applicable(env)
    assert not HEURISTICS.get("amh")().applicable(ENVS.build({"type": "pairwise_dismantling"}))


def test_vns_encoder_shape_and_smoke():
    from graco.registries import ENCODERS
    env = ENVS.build({"type": "dismantling"})
    bg = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [24, 24], "m": 3}).sample(3, rng=np.random.default_rng(0))
    obs = env.reset(bg)
    enc = ENCODERS.build({"type": "vns", "hidden_dim": 32, "num_layers": 4}, in_dim=env.node_feature_dim, edge_dim=0)
    h = enc(obs.graph)
    assert tuple(h.shape) == (bg.num_nodes, enc.out_dim) and bool(torch.isfinite(h).all())
    _smoke({"type": "dismantling"}, {"type": "barabasi_albert", "num_nodes": [16, 16], "m": 3})


def test_hypergraph_cascade_and_env():
    from graco.utils.graph_algos import hypergraph_cascade_survivors as casc
    # e0={0,1,2}, e1={2,3}; unit load. Removing node 0 with tight capacity cascades to all.
    inc_n = torch.tensor([0, 1, 2, 2, 3])
    inc_e = torch.tensor([0, 0, 0, 1, 1])
    load = torch.ones(4)
    rem0 = torch.tensor([1, 0, 0, 0], dtype=torch.bool)
    loose = casc(inc_n, inc_e, 4, 2, load, torch.full((4,), 5.0), rem0)
    assert loose.tolist() == [False, True, True, True]  # only the removed node fails
    tight = casc(inc_n, inc_e, 4, 2, load, torch.full((4,), 1.2), rem0)
    assert int(tight.sum()) < int(loose.sum())  # tight capacity => a larger cascade
    # env telescopes to a shattered residual and trains
    env = ENVS.build({"type": "cascade_dismantling", "tolerance": 0.2})
    bg = GENERATORS.build({"type": "hypergraph", "num_nodes": [30, 30], "num_hyperedges": 15, "hyperedge_size": 4}).sample(2, rng=np.random.default_rng(0))
    obs = env.reset(bg.clone())
    assert (env.objective(env.graph, env.state) > 1).all()
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        obs = env.step(a).obs
    assert (env.objective(env.graph, env.state) <= 1).all()
    _smoke({"type": "cascade_dismantling"}, {"type": "hypergraph", "num_nodes": [20, 20], "num_hyperedges": 10, "hyperedge_size": 3})


def test_mhrl_improves_over_its_init():
    import graco.heuristics  # noqa: F401
    from graco.heuristics.base import HEURISTICS
    env = ENVS.build({"type": "dismantling"})
    bg = GENERATORS.build({"type": "erdos_renyi", "num_nodes": [40, 40], "p": 0.09}).sample(4, rng=np.random.default_rng(3))
    r0 = HEURISTICS.get("mhrl")(iters=0, seed=1).solve(env, bg.clone())
    rk = HEURISTICS.get("mhrl")(iters=500, seed=1).solve(env, bg.clone())
    assert (rk.ret >= r0.ret - 1e-6).all() and rk.ret.sum() > r0.ret.sum()  # Q-learned operator search improves


def test_grlnd_single_step_train_and_eval():
    import graco.algos  # noqa: F401
    env = ENVS.build({"type": "dismantling"})
    algo = ALGOS.build(
        {"type": "grlnd", "num_samples": 4,
         "model": {"encoder": {"type": "gcn", "hidden_dim": 32, "num_layers": 2}, "head": {"type": "q_node"}}},
        env=env, device="cpu",
    )
    assert algo.single_step
    bg = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [24, 24], "m": 3}).sample(4, rng=np.random.default_rng(0))
    obs = env.reset(bg)
    m = algo.learn_step(env, obs)
    assert set(m) >= {"loss", "reward", "removed_frac"} and all(np.isfinite(v) for v in m.values())
    assert env.state["selected"].shape[0] == bg.num_nodes  # greedy mask exposed for logging
    a = algo.act(obs, explore=False)  # greedy readout for evaluation
    assert tuple(a.shape) == (bg.num_graphs,)


def test_evaluator_bridges_score_heuristics_as_baselines():
    # corehd / collective_influence are ScoreHeuristics, not native trainer baselines,
    # but the evaluator bridge should resolve them
    from graco.trainers.baselines import baseline_score_fn
    env = ENVS.build({"type": "dismantling"})
    bg = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [20, 20], "m": 3}).sample(2, rng=np.random.default_rng(0))
    obs = env.reset(bg)
    for name in ("degree", "corehd", "collective_influence"):
        fn = baseline_score_fn(name, env)
        s = fn(obs)
        assert s.shape[0] == bg.num_nodes and bool(torch.isfinite(s).all())

