"""A lightweight, decorator-based component registry.

Every pluggable family in GRACO (generators, environments, encoders, heads,
algorithms, buffers, ...) owns a :class:`Registry`.  Components register
themselves with a decorator and are later built *by name* from a YAML config::

    from graco.utils.registry import Registry

    ENVS = Registry("env")

    @ENVS.register("maxcut")
    class MaxCutEnv(VectorizedEnv):
        ...

    env = ENVS.build({"type": "maxcut", "num_envs": 128})

The ``type`` field selects the class; every other key becomes a constructor
keyword argument.  A dotted path (``my_pkg.my_module.MyEnv``) is also accepted
so users can plug in **custom** classes without touching the framework — see
:mod:`graco.utils.config`.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Optional


def import_from_path(path: str) -> Any:
    """Import an attribute from a dotted path, e.g. ``pkg.mod.Cls``."""
    if ":" in path:  # allow "pkg.mod:Cls" too
        module_name, _, attr = path.partition(":")
    else:
        module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ImportError(f"Cannot import '{path}': not a dotted path.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ImportError(f"Module '{module_name}' has no attribute '{attr}'.") from exc


class Registry:
    """A named mapping from string keys to classes/factories."""

    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Any] = {}

    # ------------------------------------------------------------------ register
    def register(
        self,
        name: Optional[Any] = None,
        obj: Optional[Any] = None,
        *,
        aliases: Optional[Iterable[str]] = None,
        override: bool = False,
    ) -> Any:
        """Register ``obj`` under ``name`` (defaults to ``obj.__name__``).

        Usable in three ways::

            @REG.register            # key = Cls.__name__.lower()
            @REG.register("my_key")  # explicit key
            REG.register("my_key", MyClass)  # imperative
        """

        def _do(o: Any, key: Optional[str]) -> Any:
            key = (key or getattr(o, "__name__")).lower()
            keys = [key] + [a.lower() for a in (aliases or [])]
            # also always expose the class' own name (lowercased) as a key
            cls_name = getattr(o, "__name__", "").lower()
            if cls_name and cls_name not in keys:
                keys.append(cls_name)
            for k in keys:
                if k in self._store and not override and self._store[k] is not o:
                    raise KeyError(
                        f"'{k}' already registered in registry '{self.name}' "
                        f"({self._store[k]!r}). Pass override=True to replace."
                    )
                self._store[k] = o
            return o

        # bare decorator: @REG.register
        if callable(name) and obj is None:
            return _do(name, None)
        # imperative: REG.register("k", obj)
        if obj is not None:
            return _do(obj, name if isinstance(name, str) else None)

        # parametrized decorator: @REG.register("k")
        def deco(o: Any) -> Any:
            return _do(o, name if isinstance(name, str) else None)

        return deco

    # --------------------------------------------------------------------- get
    def get(self, name: str) -> Any:
        key = str(name).lower()
        if key in self._store:
            return self._store[key]
        if "." in name or ":" in name:
            return import_from_path(name)
        raise KeyError(
            f"'{name}' not found in registry '{self.name}'. "
            f"Available: {sorted(self._store)}"
        )

    def __contains__(self, name: str) -> bool:
        return str(name).lower() in self._store or "." in str(name)

    def keys(self) -> List[str]:
        return sorted(self._store)

    # ------------------------------------------------------------------- build
    def build(self, cfg: Any = None, /, **kwargs: Any) -> Any:
        """Instantiate a component from a config mapping or ``type=`` kwargs.

        ``cfg`` may be a mapping containing a ``type`` (or ``name`` / ``_target_``)
        key plus constructor kwargs; extra ``**kwargs`` override/extend it.
        Nested mappings that themselves contain a ``type`` key are *not* auto
        expanded here — use :func:`graco.utils.config.instantiate` for
        recursive instantiation.
        """
        params: Dict[str, Any] = {}
        type_key: Optional[str] = None
        if cfg is not None:
            params = dict(cfg)  # shallow copy; works for dict & DictConfig
            type_key = params.pop("type", None) or params.pop("name", None) or params.pop(
                "_target_", None
            )
        params.update(kwargs)
        type_key = kwargs.pop("type", None) or type_key
        if type_key is None:
            raise ValueError(
                f"Registry '{self.name}'.build requires a 'type' key. Got: {cfg!r}"
            )
        cls = self.get(type_key)
        return cls(**params)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Registry(name={self.name!r}, entries={self.keys()})"
