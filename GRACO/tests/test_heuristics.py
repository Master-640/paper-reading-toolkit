import itertools

import networkx as nx
import torch

import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.heuristics  # noqa: F401  (registers all heuristics)
from graco.data.batch import BatchedGraph
from graco.heuristics.base import HEURISTICS
from graco.registries import ENVS, GENERATORS

EXPECTED = [
    "random", "degree", "adaptive_degree", "greedy_gain", "mvc_greedy", "min_degree",
    "weight_degree", "clique_greedy", "mds_greedy", "setcover_greedy",
    "collective_influence", "betweenness", "influence_degree",
    "sa_maxcut", "sa_mis", "local_search_maxcut", "mvc_2approx",
]


def test_all_heuristics_registered():
    keys = set(HEURISTICS.keys())
    for name in EXPECTED:
        assert name in keys, name


def _solve(hname, env_name, gen_cfg, b=4):
    env = ENVS.build({"type": env_name})
    gen = GENERATORS.build(dict(gen_cfg))
    g = gen.sample(b, device="cpu")
    res = HEURISTICS.build({"type": hname}).solve(env, g)
    assert torch.isfinite(res.objective).all()
    assert torch.isfinite(res.ret).all()
    return env, g, res


def test_greedy_gain_maxcut():
    _solve("greedy_gain", "maxcut", {"type": "er", "num_nodes": [20, 30], "p": 0.3,
                                     "weighted": True, "weight_dist": "bimodal"})


def test_sa_and_local_search_maxcut():
    _solve("sa_maxcut", "spinglass", {"type": "er", "num_nodes": [20, 30], "p": 0.3,
                                      "weighted": True, "weight_dist": "normal"})
    _solve("local_search_maxcut", "maxcut", {"type": "ba", "num_nodes": [20, 30], "m": 3,
                                             "weighted": True, "weight_dist": "bimodal"})


def test_mvc_heuristics_are_covers():
    # constructive heuristics step the env, so env.state reflects the solution
    for hname in ["mvc_greedy", "adaptive_degree"]:
        env, g, res = _solve(hname, "mvc", {"type": "er", "num_nodes": [24, 24], "p": 0.25})
        # verify the produced cover is valid on graph 0
        G = nx.Graph()
        n0 = int(g.ptr[1])
        em = g.edge_batch == 0
        for u, v in g.edge_index[:, em].t().tolist():
            G.add_edge(u, v)
        cover = {i for i in range(n0) if bool(env.state["selected"][i])}
        assert all((u in cover or v in cover) for u, v in G.edges())


def test_mis_greedy_independent():
    env, g, res = _solve("min_degree", "mis", {"type": "er", "num_nodes": [24, 24], "p": 0.2})
    n0 = int(g.ptr[1])
    G = nx.Graph()
    G.add_nodes_from(range(n0))
    em = g.edge_batch == 0
    for u, v in g.edge_index[:, em].t().tolist():
        G.add_edge(u, v)
    S = [i for i in range(n0) if bool(env.state["selected"][i])]
    assert all(not G.has_edge(u, v) for u, v in itertools.combinations(S, 2))


def test_clique_and_mds_and_dismantling():
    _solve("clique_greedy", "maxclique", {"type": "er", "num_nodes": [24, 24], "p": 0.35})
    _solve("mds_greedy", "mds", {"type": "er", "num_nodes": [24, 24], "p": 0.2})
    _solve("collective_influence", "dismantling", {"type": "ba", "num_nodes": [24, 30], "m": 3})
    _solve("betweenness", "dismantling", {"type": "ba", "num_nodes": [24, 30], "m": 3})


def test_setcover_and_influence():
    _solve("setcover_greedy", "set_cover",
           {"type": "set_cover", "num_elements": [15, 20], "num_sets": 8, "cover_size": 3})
    _solve("influence_degree", "influence_max", {"type": "ba", "num_nodes": [20, 25], "m": 3})


def test_benchmark_runner():
    from graco.eval.benchmark import run_benchmark
    from graco.utils.config import load_config

    res = run_benchmark(load_config("mvc_dqn"), num=16)
    assert len(res) >= 2
    for m, row in res.items():
        assert "objective_mean" in row and "return_mean" in row and "seconds" in row
    # method filter
    res2 = run_benchmark(load_config("finder"), num=16, methods=["degree", "random"])
    assert set(res2) == {"degree", "random"}
