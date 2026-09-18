"""Faithful reference-paper baselines built on the GRACO framework.

Importing this package registers the DIRAC and FINDER components (envs,
generators, head, algorithm) so they are selectable from YAML by name.
"""

from graco.baselines import dirac, finder  # noqa: F401

__all__ = ["dirac", "finder"]
