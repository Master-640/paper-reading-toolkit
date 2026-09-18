"""Seeding and determinism helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # opt-in; some ops have no deterministic impl
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve ``'auto'``/``None``/``'cuda'``/``'cpu'`` to a concrete device."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
