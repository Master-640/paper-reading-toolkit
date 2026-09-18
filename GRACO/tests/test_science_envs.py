import networkx as nx
import numpy as np
import torch

import graco.algos  # noqa: F401
import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.data.batch import BatchedGraph
from graco.registries import ALGOS, ENVS, GENERATORS
from graco.utils.segment_ops import segment_argmax


def _sbm_graph(seed=1):
    G = nx.stochastic_block_model([15, 15], [[0.5, 0.05], [0.05, 0.5]], seed=seed)
    ei = torch.tensor(list(G.edges())).t()
    und = torch.cat([ei, ei.flip(0)], dim=1)
    return G, BatchedGraph.from_graph_list([und], [30])


def test_modularity_matches_networkx():
    G, bg = _sbm_graph()
    env = ENVS.build({"type": "modularity"})
    sel = torch.zeros(30, dtype=torch.bool)
    sel[15:] = True  # the planted 2-block split
    q = float(env._modularity(bg, sel))
    q_nx = nx.algorithms.community.modularity(G, [set(range(15)), set(range(15, 30))])
    assert abs(q - q_nx) < 1e-4
    # all-one partition has modularity 0 (reward telescopes to final Q)
    assert abs(float(env._modularity(bg, torch.zeros(30, dtype=torch.bool)))) < 1e-6


def test_modularity_reward_telescopes():
    env = ENVS.build({"type": "modularity"})
    _, bg = _sbm_graph(seed=2)
    obs = env.reset(bg)
    total = torch.zeros(1)
    while not obs.done.all():
        a = segment_argmax(torch.rand(obs.num_nodes), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        s = env.step(a)
        total += s.reward
        obs = s.obs
    q_final = env.objective(env.graph, env.state)
    assert torch.allclose(total, q_final, atol=1e-4)  # Q(all-+1)=0 => Σ ΔQ = Q_final


def test_balanced_partition_cut_matches_networkx():
    G, bg = _sbm_graph(seed=3)
    env = ENVS.build({"type": "balanced_partition", "report": "cut"})
    sel = torch.zeros(30, dtype=torch.bool)
    sel[15:] = True  # balanced 15/15 split
    cut = float(env._cut(bg, sel))
    assert abs(cut - nx.cut_size(G, set(range(15)), set(range(15, 30)))) < 1e-4
    # a balanced split has zero imbalance penalty (cost == cut);
    # an imbalanced split (5/25) is penalized (cost > cut).
    assert abs(float(env._cost(bg, sel)) - cut) < 1e-4
    imbal = torch.zeros(30, dtype=torch.bool)
    imbal[25:] = True
    assert float(env._cost(bg, imbal)) > float(env._cut(bg, imbal)) + 1e-3


def test_new_envs_train_smoke():
    for etype, gtype, gkw in [
        ("modularity", "sbm", {"num_nodes": [24, 24]}),
        ("balanced_partition", "barabasi_albert", {"num_nodes": [24, 24], "m": 3}),
    ]:
        env = ENVS.build({"type": etype})
        gen = GENERATORS.build({"type": gtype, **gkw})
        algo = ALGOS.build(
            {"type": "dqn", "batch_size": 16, "learn_start": 16, "n_step": 2, "dueling": True,
             "model": {"encoder": {"type": "gcn", "hidden_dim": 16, "num_layers": 2},
                       "head": {"type": "dueling_q"}}},
            env=env, device="cpu",
        )
        rng = np.random.default_rng(0)
        for it in range(3):
            obs = env.reset(gen.sample(6, rng=rng))
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


def test_modularity_baselines_find_communities():
    from graco.heuristics.community import ModularityLocalSearch, ModularitySpectral
    env = ENVS.build({"type": "modularity"})
    gen = GENERATORS.build({"type": "sbm", "num_nodes": [40, 40], "num_blocks": 2,
                            "p_in": 0.4, "p_out": 0.03})
    bg = gen.sample(4, rng=np.random.default_rng(0))
    ls = ModularityLocalSearch(restarts=8).solve(env, bg.clone())
    sp = ModularitySpectral().solve(env, bg.clone())
    # both recover the planted 2-community structure (Q well above 0)
    assert (ls.objective > 0.1).all() and (sp.objective > 0.1).all()
    # greedy local search (with restarts) is competitive with spectral
    assert float(ls.objective.mean()) >= float(sp.objective.mean()) - 0.05
