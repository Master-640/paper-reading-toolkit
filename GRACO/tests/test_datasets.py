import networkx as nx
import numpy as np
import torch

import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
from graco.data.datasets import load_graph
from graco.registries import GENERATORS


def test_load_edgelist(tmp_path):
    p = tmp_path / "g.edges"
    p.write_text("# a comment\n0 1\n1 2\n2 0\n")
    bg = load_graph(str(p))
    assert bg.num_graphs == 1 and bg.num_nodes == 3
    assert bg.num_edges == 6  # 3 undirected -> 6 directed


def test_load_weighted_edgelist(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("0 1 2.5\n1 2 -1.0\n")
    bg = load_graph(str(p), weighted=True)
    assert bg.edge_weight is not None
    assert sorted(bg.edge_weight.tolist()) == sorted([2.5, 2.5, -1.0, -1.0])


def test_load_gset(tmp_path):
    p = tmp_path / "Gtest"
    p.write_text("3 2\n1 2 1\n2 3 -1\n")  # 1-indexed rudy format
    bg = load_graph(str(p), fmt="gset")
    assert bg.num_nodes == 3 and bg.num_edges == 4
    assert bg.edge_weight is not None


def test_load_gml(tmp_path):
    g = nx.karate_club_graph()
    p = tmp_path / "k.gml"
    nx.write_gml(g, str(p))
    bg = load_graph(str(p))
    assert bg.num_nodes == 34


def test_load_mtx(tmp_path):
    import scipy.sparse as sp
    from scipy.io import mmwrite

    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    p = tmp_path / "g.mtx"
    mmwrite(str(p), sp.coo_matrix(A), symmetry="symmetric")
    bg = load_graph(str(p))
    assert bg.num_nodes == 3 and bg.num_edges == 6


def test_file_dataset_generator(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "a.edges").write_text("0 1\n1 2\n")
    (d / "b.edges").write_text("0 1\n0 2\n0 3\n")
    gen_all = GENERATORS.build({"type": "dataset", "path": str(d), "mode": "all"})
    bg = gen_all.sample(99, device="cpu")
    assert bg.num_graphs == 2
    gen_rand = GENERATORS.build({"type": "dataset", "path": str(d), "mode": "random"})
    bg2 = gen_rand.sample(5, device="cpu", rng=np.random.default_rng(0))
    assert bg2.num_graphs == 5


def test_benchmark_on_real_graphs(tmp_path):
    from omegaconf import OmegaConf

    from graco.eval.benchmark import run_benchmark
    from graco.utils.config import load_config

    d = tmp_path / "nets"
    d.mkdir()
    for i in range(3):
        g = nx.barabasi_albert_graph(20, 3, seed=i)
        nx.write_edgelist(g, str(d / f"g{i}.edges"), data=False)
    cfg = load_config("dismantling_dqn")
    cfg.generator = OmegaConf.create({"type": "dataset", "path": str(d), "mode": "all"})
    res = run_benchmark(cfg, num=3)
    assert len(res) >= 2


def test_datasets_registry():
    from graco.data.benchmarks import DATASETS

    assert len(DATASETS) >= 3
    for d in DATASETS.values():
        assert "url" in d and "fmt" in d
