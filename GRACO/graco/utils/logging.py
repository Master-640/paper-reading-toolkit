"""Lightweight, dependency-optional experiment logging.

A :class:`Logger` fans metrics out to the console, TensorBoard and/or Weights &
Biases depending on what is installed and enabled in the config.  Everything
degrades gracefully: if ``tensorboard``/``wandb`` are missing we simply skip
those backends instead of crashing an experiment.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional


class AverageMeter:
    """Tracks a running window mean of a scalar."""

    def __init__(self, window: int = 100):
        self.vals: deque = deque(maxlen=window)

    def update(self, v: float) -> None:
        self.vals.append(float(v))

    @property
    def avg(self) -> float:
        return sum(self.vals) / len(self.vals) if self.vals else 0.0


class Logger:
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "graco",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        console: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.console = console
        self._t0 = time.time()
        self._meters: Dict[str, AverageMeter] = defaultdict(lambda: AverageMeter())
        self._jsonl = open(self.log_dir / "metrics.jsonl", "a")

        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.log_dir / "tb"))
            except Exception as exc:  # pragma: no cover
                print(f"[logger] TensorBoard unavailable ({exc}); skipping.")

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    name=run_name,
                    dir=str(self.log_dir),
                    config=config or {},
                )
            except Exception as exc:  # pragma: no cover
                print(f"[logger] W&B unavailable ({exc}); skipping.")

    def log(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        flat = {}
        for k, v in metrics.items():
            if v is None:
                continue
            key = f"{prefix}/{k}" if prefix else k
            v = float(v)
            flat[key] = v
            self._meters[key].update(v)
            if self.tb is not None:
                self.tb.add_scalar(key, v, step)
        if self.wandb is not None and flat:
            self.wandb.log(flat, step=step)
        rec = {"step": step, "wall": round(time.time() - self._t0, 2), **flat}
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()

    def console_line(self, step: int, keys=None) -> str:
        keys = keys or list(self._meters)
        parts = [f"step={step}"]
        for k in keys:
            if k in self._meters:
                parts.append(f"{k}={self._meters[k].avg:.4f}")
        line = " | ".join(parts)
        if self.console:
            print(line, flush=True)
        return line

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()
        self._jsonl.close()
