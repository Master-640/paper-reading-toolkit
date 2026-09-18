"""Config loading and object instantiation (OmegaConf / Hydra-lite).

GRACO experiments are driven by a single YAML file.  This module provides:

* :func:`load_config` — read a YAML file (with optional ``defaults:``/``base:``
  includes), apply dotted-key CLI overrides, and resolve interpolations.
* :func:`instantiate` — Hydra-style recursive instantiation of arbitrary python
  objects from a ``_target_: pkg.mod.Cls`` mapping (used for optimizers,
  schedulers, and any custom object).

Framework components (envs, encoders, algos, ...) are built through their
:class:`~graco.utils.registry.Registry` (``type:`` key) — that path also
accepts dotted class paths, so **custom user classes drop in without editing
the framework**.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List, Optional, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

from graco.utils.registry import import_from_path


def resolve_config_path(name: str) -> str:
    """Resolve a config given as a file path **or** a short name.

    ``"configs/maxcut_ppo.yaml"`` (a real file) is returned as-is; a bare name
    like ``"maxcut_ppo"`` is looked up as ``<name>.yaml`` under a ``configs/``
    directory in the current working dir (user configs) or under the bundled
    ``graco/configs`` (shipped with the package), so users can write
    ``graco train -c maxcut_ppo`` instead of the full path.
    """
    p = Path(name)
    if p.is_file():
        return str(p)
    cand = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    roots = [
        Path.cwd() / "configs",  # user configs in the working dir (optional)
        Path(__file__).resolve().parent.parent / "configs",  # bundled: graco/configs
    ]
    for root in roots:
        f = root / cand
        if f.is_file():
            return str(f)
    if Path(cand).is_file():
        return str(cand)
    raise FileNotFoundError(
        f"config '{name}' not found — pass a path, or a name under configs/ "
        f"(searched: {', '.join(str(r) for r in roots)})"
    )


def list_configs() -> List[str]:
    """Names (without extension) of the bundled example configs, sorted."""
    roots = [
        Path.cwd() / "configs",  # user-provided configs in the working dir (optional)
        Path(__file__).resolve().parent.parent / "configs",  # bundled: graco/configs
    ]
    for root in roots:
        if root.is_dir():
            return sorted(p.stem for p in root.glob("*.yaml"))
    return []


def load_config(
    path: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
) -> DictConfig:
    """Load a config from ``path`` and apply ``key=value`` dotted overrides.

    ``path`` may be a file path or a short config name (see
    :func:`resolve_config_path`).  Supports a lightweight include mechanism: a
    config may declare ``base: other.yaml`` (single) or ``defaults: [a.yaml,
    b.yaml]`` (list, later ones win) resolved relative to the config's directory.
    """
    if path is not None:
        cfg = _load_with_includes(Path(resolve_config_path(path)))
    else:
        cfg = OmegaConf.create({})

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def _load_with_includes(path: Path, _seen: Optional[set] = None) -> DictConfig:
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular config include detected at {path}")
    _seen.add(path)
    raw = OmegaConf.load(str(path))

    includes: List[str] = []
    base = raw.pop("base", None) if isinstance(raw, DictConfig) else None
    defaults = raw.pop("defaults", None) if isinstance(raw, DictConfig) else None
    if base:
        includes.append(base)
    if defaults:
        includes.extend(list(defaults))

    merged = OmegaConf.create({})
    for inc in includes:
        inc_path = (path.parent / inc).resolve()
        merged = OmegaConf.merge(merged, _load_with_includes(inc_path, _seen))
    merged = OmegaConf.merge(merged, raw)
    return merged  # type: ignore[return-value]


def to_container(cfg: Any, resolve: bool = True) -> Any:
    """Convert an OmegaConf node to plain python containers."""
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=resolve)
    return cfg


def instantiate(cfg: Any, **extra: Any) -> Any:
    """Recursively instantiate an object graph from a ``_target_`` mapping.

    ``{"_target_": "torch.optim.Adam", "lr": 1e-3}`` -> a partial-less call.
    Nested mappings with their own ``_target_`` are instantiated first.  Pass
    positional-only extras via ``**extra`` (they override matching keys).
    """
    cfg = to_container(cfg, resolve=True)
    if isinstance(cfg, dict) and "_target_" in cfg:
        target = cfg.pop("_target_")
        partial = cfg.pop("_partial_", False)
        kwargs = {k: instantiate(v) for k, v in cfg.items()}
        kwargs.update(extra)
        fn = import_from_path(target)
        if partial:
            from functools import partial as _partial

            return _partial(fn, **kwargs)
        return fn(**kwargs)
    if isinstance(cfg, dict):
        return {k: instantiate(v) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [instantiate(v) for v in cfg]
    return cfg


def import_user_modules(modules: Optional[Sequence[str]]) -> None:
    """Import user modules so their ``@REGISTRY.register`` decorators run.

    Put ``imports: [my_project.custom_layers]`` at the top of your YAML to make
    custom components available by name.
    """
    for m in modules or []:
        importlib.import_module(m)
