"""GRACO — a fully-vectorized, YAML-configurable Deep-RL framework for
combinatorial optimization on graphs (MaxCut, Spin Glass, MVC, MIS, network
dismantling, sentinel observability, ...).

Convenient top-level API::

    import graco

    # train from a shipped config (short name) with programmatic overrides
    trainer = graco.train("maxcut_ppo", trainer__iterations=5000, device="cuda")
    print(graco.evaluate("maxcut_ppo", ckpt="runs/maxcut_ppo/best.pt"))

    # discover what is registered / available
    graco.available()            # {"envs": [...], "encoders": [...], ...}
    graco.available("encoders")  # ["appnp", "arma", "chebnet", ...]

    # build individual components straight from the registries
    env = graco.make_env("maxcut")
    gen = graco.make_generator("barabasi_albert", num_nodes=[50, 100])

The ``key__subkey=value`` keyword form is sugar for OmegaConf dotted overrides
(``key.subkey=value``); pass ``overrides=[...]`` for full control.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

__version__ = "0.1.0"

from graco.registries import (  # noqa: E402
    ALGOS,
    BUFFERS,
    ENCODERS,
    ENVS,
    GENERATORS,
    HEADS,
)

__all__ = [
    "__version__",
    "GENERATORS", "ENVS", "ENCODERS", "HEADS", "ALGOS", "BUFFERS",
    "train", "evaluate", "load_config", "available",
    "make_env", "make_algo", "make_generator", "qubo", "accelerate",
]


def _dotlist(overrides: Optional[Sequence[str]], kwargs: Dict[str, Any]) -> List[str]:
    """Merge an explicit override list with ``key__subkey=value`` kwargs."""
    out = list(overrides or [])
    for k, v in kwargs.items():
        out.append(f"{k.replace('__', '.')}={v}")
    return out


def load_config(config: str, overrides: Optional[Sequence[str]] = None, **kwargs: Any):
    """Load a config by path or short name, applying dotted / ``key__sub=`` overrides."""
    from graco.utils.config import load_config as _load

    return _load(config, overrides=_dotlist(overrides, kwargs))


def train(config: str, *, ckpt: Optional[str] = None,
          overrides: Optional[Sequence[str]] = None, **kwargs: Any):
    """Train from a config (path or short name); returns the fitted ``Trainer``.

    Example: ``graco.train("maxcut_ppo", trainer__iterations=5000, device="cuda")``.
    """
    from graco.trainers.trainer import Trainer

    trainer = Trainer(load_config(config, overrides, **kwargs))
    if ckpt:
        trainer.load_checkpoint(ckpt)
    trainer.train()
    return trainer


def evaluate(config: str, *, ckpt: Optional[str] = None,
             overrides: Optional[Sequence[str]] = None, **kwargs: Any) -> Dict[str, float]:
    """Evaluate a (optionally checkpointed) config; returns the metrics dict."""
    from graco.trainers.trainer import Trainer

    trainer = Trainer(load_config(config, overrides, **kwargs))
    if ckpt:
        trainer.load_checkpoint(ckpt)
    return trainer.evaluate()


def _all_registries() -> Dict[str, Any]:
    # importing the subpackages runs the @register decorators
    import graco.algos  # noqa: F401
    import graco.buffers  # noqa: F401
    import graco.envs  # noqa: F401
    import graco.generators  # noqa: F401
    import graco.heuristics  # noqa: F401
    import graco.models  # noqa: F401
    try:
        import graco.baselines  # noqa: F401  (registers finder / dirac components)
    except Exception:
        pass
    from graco.heuristics.base import HEURISTICS

    return {"envs": ENVS, "encoders": ENCODERS, "algos": ALGOS,
            "generators": GENERATORS, "heads": HEADS, "buffers": BUFFERS,
            "heuristics": HEURISTICS}


def available(kind: Optional[str] = None) -> Union[List[str], Dict[str, List[str]]]:
    """List registered component names — all kinds, or just ``kind`` if given."""
    regs = _all_registries()
    if kind is not None:
        if kind not in regs:
            raise KeyError(f"unknown kind '{kind}'; choose from {sorted(regs)}")
        return sorted(regs[kind].keys())
    return {k: sorted(v.keys()) for k, v in regs.items()}


def make_env(name: str, **kwargs: Any):
    """Build a registered environment by name (e.g. ``graco.make_env('maxcut')``)."""
    import graco.envs  # noqa: F401

    return ENVS.build({"type": name, **kwargs})


def make_generator(name: str, **kwargs: Any):
    """Build a registered graph generator by name."""
    import graco.generators  # noqa: F401

    return GENERATORS.build({"type": name, **kwargs})


def make_algo(name: str, env, **kwargs: Any):
    """Build a registered algorithm by name, wired to ``env``."""
    import graco.algos  # noqa: F401

    return ALGOS.build({"type": name, **kwargs}, env=env)


def qubo(Q, device: str = "cpu"):
    """Wrap a QUBO matrix (or list of matrices) ``Q`` into a batched graph.

    Feed the result to the ``qubo`` environment (``graco.make_env('qubo')``) to
    minimize ``x^T Q x`` with a trained policy or any decoder::

        import numpy as np, graco
        g = graco.qubo(np.array([[ -1., 2.], [2., -1.]]))
        env = graco.make_env("qubo"); env.reset(g)
    """
    from graco.generators.qubo import QUBOGenerator

    mats = Q if isinstance(Q, (list, tuple)) else [Q]
    return QUBOGenerator.from_matrices(mats, device=device)


def accelerate(**kwargs: Any) -> None:
    """Enable the standard GPU speedups globally (TF32 / matmul precision / cuDNN).

    No-op on CPU; skipped under ``deterministic=True``. Kwargs: ``tf32``,
    ``matmul_precision``, ``cudnn_benchmark``, ``compile``, ``spmm``,
    ``deterministic``. The Trainer calls this automatically from the ``accel:``
    config block.
    """
    from graco.utils.accel import configure

    configure(**kwargs)
