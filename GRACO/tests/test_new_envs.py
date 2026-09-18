import itertools

import networkx as nx
import torch

import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
from graco.data.batch import BatchedGraph
from graco.registries import ENVS, GENERATORS
from graco.utils.segment_ops import segment_argmax


def _rollout(env, bg, score=None):
    obs = env.reset(bg.clone())
    steps = 0
    while not obs.done.all():
        sc = torch.rand(obs.num_nodes) if score is None else score(obs)
        a = segment_argmax(sc, obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        obs = env.step(a).obs
        steps += 1
        assert steps < 300
    return env


def _er():
    g = nx.erdos_renyi_graph(24, 0.35, seed=1)
    return BatchedGraph.from_networkx(g), nx.convert_node_labels_to_integers(g)


def test_maxclique_is_clique():
    bg, G = _er()
    env = _rollout(ENVS.build({"type": "maxclique"}), bg)
    S = [i for i in range(bg.num_nodes) if bool(env.state["selected"][i])]
    assert len(S) >= 1
    assert all(G.has_edge(u, v) for u, v in itertools.combinations(S, 2))


def test_mds_dominates():
    bg, G = _er()
    env = _rollout(ENVS.build({"type": "mds"}), bg)
    S = set(i for i in range(bg.num_nodes) if bool(env.state["selected"][i]))
    assert all((v in S) or any(u in S for u in G.neighbors(v)) for v in G.nodes())


def test_mwis_independent_and_weighted():
    bg, G = _er()
    env = _rollout(ENVS.build({"type": "mwis"}), bg)
    S = [i for i in range(bg.num_nodes) if bool(env.state["selected"][i])]
    assert all(not G.has_edge(u, v) for u, v in itertools.combinations(S, 2))
    obj = float(env.objective(env.graph, env.state)[0])
    assert obj >= len(S)  # degree+1 weights => obj >= count


def test_set_cover_covers_all():
    bg = GENERATORS.build(
        {"type": "set_cover", "num_elements": [15, 20], "num_sets": 8, "cover_size": 3}
    ).sample(3, device="cpu")
    env = ENVS.build({"type": "set_cover"})
    obs = env.reset(bg.clone())
    steps = 0
    while not obs.done.all():
        a = segment_argmax(obs.graph.degree(), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        obs = env.step(a).obs
        steps += 1
        assert steps < 100
    assert bool(obs.done.all())


def test_influence_budget_and_spread():
    bg = GENERATORS.build({"type": "ba", "num_nodes": [20, 25], "m": 3}).sample(3, device="cpu")
    env = ENVS.build({"type": "influence_max", "budget": 4, "mc_samples": 8})
    obs = env.reset(bg.clone())
    total = torch.zeros(3)
    steps = 0
    while not obs.done.all():
        a = segment_argmax(obs.graph.degree(), obs.graph.batch, obs.graph.num_graphs, obs.action_mask)
        s = env.step(a)
        total += s.reward
        obs = s.obs
        steps += 1
        assert steps <= 4  # budget respected
    assert (total > 0).all()  # positive expected spread
