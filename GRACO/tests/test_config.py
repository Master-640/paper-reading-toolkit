import textwrap

import pytest
import torch

from graco.registries import Registry
from graco.utils.config import instantiate, load_config, to_container


def test_registry_build_and_aliases():
    reg = Registry("thing")

    @reg.register("foo", aliases=["f"])
    class Foo:
        def __init__(self, a=1, b=2):
            self.a, self.b = a, b

    obj = reg.build({"type": "foo", "a": 5})
    assert obj.a == 5 and obj.b == 2
    assert reg.get("f") is Foo
    assert reg.build({"type": "f", "b": 9}).b == 9


def test_registry_dotted_path():
    reg = Registry("thing")
    # a class not registered, resolved by dotted path
    cls = reg.get("collections.OrderedDict")
    from collections import OrderedDict

    assert cls is OrderedDict


def test_registry_unknown_raises():
    reg = Registry("thing")
    with pytest.raises(KeyError):
        reg.get("nope")


def test_instantiate_target():
    obj = instantiate({"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 4})
    assert isinstance(obj, torch.nn.Linear)
    assert obj.weight.shape == (4, 3)


def test_load_config_include_and_override(tmp_path):
    (tmp_path / "base.yaml").write_text(
        textwrap.dedent(
            """
            a: 1
            nested: {x: 10, y: 20}
            """
        )
    )
    (tmp_path / "exp.yaml").write_text(
        textwrap.dedent(
            """
            base: base.yaml
            a: 2
            nested: {y: 99}
            """
        )
    )
    cfg = load_config(str(tmp_path / "exp.yaml"), overrides=["nested.x=42"])
    assert cfg.a == 2  # child overrides base
    assert cfg.nested.x == 42  # CLI override
    assert cfg.nested.y == 99  # child overrides base


def test_end_to_end_debug_config():
    cfg = load_config("debug", overrides=["trainer.iterations=3", "trainer.eval_every=0", "device=cpu"])
    from graco.trainers.trainer import Trainer

    trainer = Trainer(cfg)
    trainer.train()
    metrics = trainer.evaluate()
    assert "eval/objective" in metrics
