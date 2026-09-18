"""Vectorized scatter / segment reductions.

These are the primitives behind every message-passing layer and every batched
environment reduction in GRACO.  They are implemented on top of native
PyTorch ops (``index_add_`` / ``scatter_reduce_``) so the framework has **no
hard dependency on torch-scatter or torch-geometric** — but if you have
torch-geometric installed the semantics match ``torch_geometric.utils.scatter``.

All functions operate on ``dim=0`` (the "node"/"edge" axis) which is the GRACO
convention: tensors are ``[N, ...]`` and ``index`` is ``[N]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def _broadcast_index(index: Tensor, src: Tensor) -> Tensor:
    """Expand a 1-D ``index`` [N] to match ``src`` [N, ...] for scatter ops."""
    if index.dim() == 1 and src.dim() > 1:
        shape = [index.shape[0]] + [1] * (src.dim() - 1)
        index = index.view(shape).expand_as(src)
    return index


def _out_shape(src: Tensor, dim_size: int) -> Tuple[int, ...]:
    return (dim_size, *src.shape[1:])


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``."""
    out = src.new_zeros(_out_shape(src, dim_size))
    idx = _broadcast_index(index, src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Mean of ``src`` rows per bucket (empty buckets -> 0)."""
    summed = scatter_sum(src, index, dim_size)
    count = scatter_sum(torch.ones_like(index, dtype=src.dtype), index, dim_size)
    count = count.clamp_(min=1)
    if summed.dim() > 1:
        count = count.view(-1, *([1] * (summed.dim() - 1)))
    return summed / count


def scatter_max(
    src: Tensor, index: Tensor, dim_size: int, fill_value: Optional[float] = None
) -> Tensor:
    """Max-reduce ``src`` per bucket. Empty buckets get ``fill_value`` (default 0)."""
    neg_inf = torch.finfo(src.dtype).min if src.is_floating_point() else torch.iinfo(src.dtype).min
    out = src.new_full(_out_shape(src, dim_size), neg_inf)
    idx = _broadcast_index(index, src)
    out.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)
    fill = 0.0 if fill_value is None else fill_value
    return out.masked_fill(out <= neg_inf, fill)


def scatter_min(
    src: Tensor, index: Tensor, dim_size: int, fill_value: Optional[float] = None
) -> Tensor:
    pos_inf = torch.finfo(src.dtype).max if src.is_floating_point() else torch.iinfo(src.dtype).max
    out = src.new_full(_out_shape(src, dim_size), pos_inf)
    idx = _broadcast_index(index, src)
    out.scatter_reduce_(0, idx, src, reduce="amin", include_self=True)
    fill = 0.0 if fill_value is None else fill_value
    return out.masked_fill(out >= pos_inf, fill)


def scatter_softmax(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Numerically-stable softmax within each bucket (used by attention convs).

    Accumulation is forced to fp32 so it stays correct under AMP/bf16 autocast.
    """
    in_dtype = src.dtype
    src = src.float()
    src_max = scatter_max(src, index, dim_size)
    src_max = src_max.index_select(0, index)
    out = (src - src_max).exp()
    denom = scatter_sum(out, index, dim_size).index_select(0, index)
    return (out / (denom + 1e-16)).to(in_dtype)


def scatter_logsumexp(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    in_dtype = src.dtype
    src = src.float()
    src_max = scatter_max(src, index, dim_size)
    gathered_max = src_max.index_select(0, index)
    out = (src - gathered_max).exp()
    summed = scatter_sum(out, index, dim_size)
    return (summed.clamp_min(1e-16).log() + src_max).to(in_dtype)


def segment_lengths_to_ids(lengths: Tensor) -> Tensor:
    """Convert per-graph node counts ``[B]`` to a ``batch`` vector ``[sum lengths]``.

    e.g. ``[3, 2] -> [0, 0, 0, 1, 1]``.  Fully vectorized.
    """
    return torch.repeat_interleave(
        torch.arange(lengths.numel(), device=lengths.device), lengths
    )


def lengths_to_ptr(lengths: Tensor) -> Tensor:
    """Convert per-graph counts ``[B]`` to CSR pointer ``[B+1]`` (prefix sum)."""
    ptr = lengths.new_zeros(lengths.numel() + 1)
    torch.cumsum(lengths, dim=0, out=ptr[1:])
    return ptr
