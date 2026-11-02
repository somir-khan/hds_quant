"""Asymmetric uniform group-wise RTN fake weight quantization (SPEC §2)."""

from __future__ import annotations

import torch

_MIN_SCALE = 1e-8


def quantize_weight_rtn(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Quantize-dequantize ``weight`` (``[d_out, d_in]``) with group-wise asymmetric RTN.

    Groups are ``group_size`` contiguous input channels within each output row.
    Per group (affine grid anchored at the group minimum, so single-sign groups such as
    all-positive weights still receive the full ``2^b`` levels):
    ``scale = (max - min) / (2^b - 1)`` (clamped to >= 1e-8),
    ``q = clamp(round((w - min) / scale), 0, 2^b - 1)`` and output ``q * scale + min``.

    ``clip_ratio`` shrinks the per-group [min, max] range symmetrically around its midpoint
    (1.0 = no clipping): ``half = 0.5 * (max - min) * clip_ratio``, ``[mid - half, mid + half]``.

    Zero-range (constant) groups are returned as that constant: a single level represents
    them exactly, and this avoids the huge ``w / 1e-8`` quotients a clamped scale would give.
    Math is done in float32; the result has the same dtype/device as ``weight``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [d_out, d_in], got shape {tuple(weight.shape)}")
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    d_out, d_in = weight.shape
    if d_in % group_size != 0:
        raise ValueError(f"d_in={d_in} is not divisible by group_size={group_size}")
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")

    orig_dtype = weight.dtype
    w = weight.detach().to(torch.float32).reshape(d_out, d_in // group_size, group_size)

    qmax = float(2**bits - 1)
    gmin = w.amin(dim=-1, keepdim=True)
    gmax = w.amax(dim=-1, keepdim=True)
    degenerate = (gmax - gmin) <= qmax * _MIN_SCALE
    constant = 0.5 * (gmin + gmax)

    if clip_ratio != 1.0:
        mid = 0.5 * (gmin + gmax)
        half = 0.5 * (gmax - gmin) * clip_ratio
        gmin = mid - half
        gmax = mid + half

    scale = ((gmax - gmin) / qmax).clamp(min=_MIN_SCALE)
    q = torch.clamp(torch.round((w - gmin) / scale), 0.0, qmax)
    deq = q * scale + gmin
    deq = torch.where(degenerate, constant.expand_as(deq), deq)

    return deq.reshape(d_out, d_in).to(orig_dtype)
