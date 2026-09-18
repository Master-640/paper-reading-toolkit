import numpy as np
import pytest

import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.heuristics  # noqa: F401
from graco.heuristics.base import HEURISTICS
from graco.registries import ENVS, GENERATORS


def _sample(cfg, n=4, seed=1):
    return GENERATORS.build(dict(cfg)).sample(n, rng=np.random.default_rng(seed))


def test_brute_force_maxcut_is_global_optimum():
    g = _sample({"type": "er", "num_nodes": [12, 14], "p": 0.4, "weighted": True, "weight_dist": "bimodal"})
    env = ENVS.build({"type": "maxcut"})
    bf = HEURISTICS.build({"type": "brute_force_maxcut"}).solve(env, g)
    for h in ["greedy_gain", "sa_maxcut", "local_search_maxcut"]:
        r = HEURISTICS.build({"type": h}).solve(env, g.clone())
        assert (bf.objective >= r.objective - 1e-3).all(), h  # exact dominates


def test_brute_force_skips_large_graphs():
    big = _sample({"type": "ba", "num_nodes": [40, 40], "m": 3})
    bf = HEURISTICS.build({"type": "brute_force_maxcut"})
    assert not bf.applicable(ENVS.build({"type": "maxcut"}), big)


def test_ilp_mvc_optimal():
    pytest.importorskip("pulp")
    g = _sample({"type": "er", "num_nodes": [14, 14], "p": 0.3}, seed=2)
    mvc = ENVS.build({"type": "mvc"})
    ilp = HEURISTICS.build({"type": "ilp"}).solve(mvc, g)
    greedy = HEURISTICS.build({"type": "mvc_greedy"}).solve(mvc, g.clone())
    assert (ilp.objective <= greedy.objective + 1e-6).all()  # exact cover <= greedy


def test_ilp_mis_optimal():
    pytest.importorskip("pulp")
    g = _sample({"type": "er", "num_nodes": [14, 14], "p": 0.3}, seed=3)
    mis = ENVS.build({"type": "mis"})
    ilp = HEURISTICS.build({"type": "ilp"}).solve(mis, g)
    greedy = HEURISTICS.build({"type": "min_degree"}).solve(mis, g.clone())
    assert (ilp.objective >= greedy.objective - 1e-6).all()  # exact IS >= greedy


def test_benchmark_reports_gap_column():
    from omegaconf import OmegaConf

    from graco.eval.benchmark import format_table, run_benchmark
    from graco.utils.config import load_config

    cfg = load_config("maxcut_ppo")
    cfg.generator = OmegaConf.create(
        {"type": "er", "num_nodes": [12, 12], "p": 0.4, "weighted": True, "weight_dist": "bimodal"}
    )
    res = run_benchmark(cfg, num=6, methods=["brute_force_maxcut", "greedy_gain", "random"])
    assert "brute_force_maxcut" in res
    tbl = format_table(res, ENVS.build({"type": "maxcut"}))
    assert "gap%" in tbl
