"""Acceleration infra: SpMM aggregation, cached adjacency, cache-sharing, flags."""

import numpy as np
import torch

import graco.envs  # noqa: F401
import graco.generators  # noqa: F401
import graco.models  # noqa: F401
from graco.registries import ENCODERS, ENVS, GENERATORS
from graco.utils import accel


def _graph(seed=0):
    bg = GENERATORS.build({"type": "barabasi_albert", "num_nodes": [80, 80], "m": 3}).sample(
        2, rng=np.random.default_rng(seed)
    )
    bg.x = torch.randn(bg.num_nodes, 5)
    return bg


def test_spmm_matches_scatter_aggregation():
    for aggr in ["mean", "sum"]:
        enc = ENCODERS.build(
            {"type": "graphsage", "hidden_dim": 16, "num_layers": 2, "aggr": aggr},
            in_dim=5, edge_dim=0,
        ).eval()
        bg = _graph()
        try:
            accel.configure(spmm=False)
            ref = enc(bg)
            bg._cache = {}
            accel.configure(spmm=True)
            got = enc(bg)
        finally:
            accel.configure(spmm=False)
        assert torch.allclose(ref, got, atol=1e-4), f"SpMM != scatter for aggr={aggr}"


def test_adj_norm_is_cached_and_sparse():
    bg = _graph()
    a = bg.adj_norm("mean")
    assert a.is_sparse and a.shape == (bg.num_nodes, bg.num_nodes)
    assert bg.adj_norm("mean") is a  # cached (built once, reused every step)


def test_obs_graph_shares_topology_cache():
    env = ENVS.build({"type": "maxcut"})
    bg = _graph()
    env.reset(bg)
    bg.degree()  # populate topology cache on the source graph
    obs1 = env.observe()
    # the obs graph shares the source topology cache -> degree/adj computed once
    assert obs1.graph._cache is env.graph._cache


def test_accelerate_toggles_flags_and_is_cpu_safe():
    try:
        accel.configure(compile=True, spmm=True)
        assert accel.compile_enabled() and accel.spmm_enabled()
        accel.configure(deterministic=True)  # disables autotuners; flags reset to defaults
        assert not accel.compile_enabled() and not accel.spmm_enabled()
    finally:
        accel.configure()  # reset


def test_compiled_policy_statedict_is_clean():
    from graco.models.build import build_q_policy

    env = ENVS.build({"type": "maxcut"})
    policy = build_q_policy({"encoder": {"type": "graphsage", "hidden_dim": 16, "num_layers": 2}}, env)
    try:
        accel.configure(compile=True)  # forward may fall back to eager; must not pollute state_dict
        bg = _graph()
        env.reset(bg)
        _ = policy(env.observe().graph)
        keys = list(policy.state_dict().keys())
        assert not any("_orig_mod" in k for k in keys)  # compiled alias stays out of state_dict
    finally:
        accel.configure()
