"""Performance configuration for GRACO.

One-call global GPU speedups — TF32 / ``set_float32_matmul_precision`` and cuDNN
autotune (CleanRL / Lightning / TorchRL style) — plus a fused-Adam builder and
opt-in AMP / ``torch.compile`` helpers.  Everything is a **no-op on CPU** and
leaves default numerics unchanged unless explicitly opted in, so the test-suite
and deterministic mode stay bit-stable.

Note on where speed actually comes from in GRACO: the scatter reductions
(``graco.utils.scatter``) are already memory-bandwidth-bound and near-optimal via
native ``scatter_reduce_``/``index_add_``; the training bottleneck on small graphs
is the *sequential per-step rollout* (many tiny kernels), which is launch-bound.
TF32/AMP help FLOP-bound cases (large graphs / wide hidden dims); kernel fusion
(``torch.compile``) is what helps the launch-bound case — but see the caveats on
:func:`maybe_compile`.  A routed ``torch_scatter`` backend is deliberately *not*
provided: it is not bit-equivalent (empty-bucket fill, unsorted-index segment
reductions, max backward tie-break, no deterministic fallback), so it would break
``segment_argmax`` semantics and deterministic mode for a marginal gain.
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Optional

import torch

_COMPILE = False  # opt-in: torch.compile the encoder forward (kernel fusion)
_SPMM = False  # opt-in: SpMM (sparse-matmul) neighbour aggregation


def compile_enabled() -> bool:
    return _COMPILE


def spmm_enabled() -> bool:
    return _SPMM


def configure(
    *,
    tf32: bool = True,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
    compile: bool = False,
    spmm: bool = False,
    deterministic: bool = False,
) -> None:
    """Enable the standard GPU speedups + opt-in infra accelerators (no-op on CPU
    for the TF32 bits).  ``compile`` fuses the encoder via ``torch.compile``;
    ``spmm`` runs neighbour aggregation as a cached sparse matmul.  Both are
    hardware-agnostic and fall back gracefully; ``deterministic=True`` disables the
    non-deterministic autotuners (TF32 / cuDNN benchmark).
    """
    global _COMPILE, _SPMM
    _COMPILE, _SPMM = bool(compile), bool(spmm)
    if deterministic:
        return
    try:
        torch.set_float32_matmul_precision(matmul_precision)  # TF32 matmuls (Ampere/Hopper)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def build_optimizer(params: Iterable, lr: float, weight_decay: float = 0.0):
    """Adam, using the fused CUDA implementation when available (fewer kernel launches)."""
    kw = dict(lr=lr, weight_decay=weight_decay)
    if torch.cuda.is_available():
        try:
            return torch.optim.Adam(params, fused=True, **kw)
        except (TypeError, RuntimeError, ValueError):
            pass  # older torch / unsupported params -> plain Adam
    return torch.optim.Adam(params, **kw)


def autocast(device: Optional[torch.device] = None, enabled: bool = False,
             dtype: torch.dtype = torch.bfloat16):
    """AMP autocast context (bf16 on CUDA); a no-op unless ``enabled`` (EXPERIMENTAL).

    Caveats you must respect if you enable this: autocast only downcasts
    matmul/linear/conv — it does **not** keep the hand-written scatter reductions in
    fp32, so wrap ``scatter_sum``/``softmax``/``logsumexp`` and the DQN TD/target/loss
    in fp32 yourself (the sensitive reductions in ``graco.utils.scatter`` already
    upcast internally). bf16 is used (no ``GradScaler`` needed); fp16 is unsupported
    here (would need a scaler).
    """
    if not enabled or device is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def maybe_compile(module, enabled: bool = False, **kw):
    """``torch.compile`` a module when opted in (EXPERIMENTAL); eager on failure.

    Uses ``dynamic=True`` (graph N/E vary each step, so per-size recompiles would
    dominate).  Compile the **encoder submodule**, not the top Q-policy — the policy
    flips ``train()``/``eval()`` every step which triggers Dynamo recompiles, and its
    ``state_dict`` gains an ``_orig_mod.`` prefix that breaks target-net sync /
    checkpoints.  On small graphs this is often net-negative (graph breaks on
    ``scatter_reduce_``); benchmark before relying on it.
    """
    if not enabled:
        return module
    kw.setdefault("dynamic", True)
    try:
        return torch.compile(module, **kw)
    except Exception as exc:  # pragma: no cover - environment dependent
        import warnings

        warnings.warn(f"torch.compile unavailable ({exc}); running eager.")
        return module
