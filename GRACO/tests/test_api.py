"""Public top-level API + config-name resolution (developer-experience surface)."""

import pytest

import graco
from graco.utils.config import list_configs, resolve_config_path


def test_available_lists_components():
    allc = graco.available()
    assert {"envs", "encoders", "algos", "generators", "heads", "buffers", "heuristics"} <= set(allc)
    assert "maxcut" in allc["envs"] and "dqn" in allc["algos"]
    enc = graco.available("encoders")
    assert isinstance(enc, list) and enc == sorted(enc) and "gcn" in enc
    with pytest.raises(KeyError):
        graco.available("nope")


def test_make_helpers_build_components():
    env = graco.make_env("maxcut")
    assert type(env).__name__ == "MaxCutEnv"
    gen = graco.make_generator("barabasi_albert", num_nodes=[20, 20])
    bg = gen.sample(2)
    assert bg.num_graphs == 2
    algo = graco.make_algo("dqn", env=env, device="cpu")
    assert hasattr(algo, "act")


def test_config_short_name_resolution_and_overrides():
    # short name resolves to graco/configs/debug.yaml
    assert resolve_config_path("debug").endswith("debug.yaml")
    assert "debug" in list_configs()
    # key__sub=value sugar maps to the dotted override key.sub=value
    cfg = graco.load_config("debug", trainer__iterations=7)
    assert int(cfg.trainer.iterations) == 7
    with pytest.raises(FileNotFoundError):
        resolve_config_path("does_not_exist_xyz")
