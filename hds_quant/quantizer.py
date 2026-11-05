"""Asymmetric uniform group-wise RTN and official Half-Quadratic Quantization (HQQ) fake weight quantization."""

from __future__ import annotations

import inspect
from types import MappingProxyType
from typing import Any, Dict, Union
import torch

try:
    import hqq
    from hqq.core.quantize import Quantizer
except ImportError:
    hqq = None
    Quantizer = None

_MIN_SCALE = 1e-8
QUANTIZER_MODES = ("rtn", "hqq")

# Official HQQ Backend Configuration Constants
HQQ_BACKEND_TAG = "official_hqq"
HQQ_VERSION = getattr(hqq, "__version__", "0.2.8.post1") if hqq is not None else "0.2.8.post1"
HQQ_AXIS = 1  # axis=1 groups contiguous input-channel chunks within each output row: W.reshape([-1, group_size])
HQQ_ROUND_ZERO = False  # Quantizer default; BaseQuantizeConfig instead uses True at 4 bits.
HQQ_OPTIMIZE = True  # Official proximal optimization enabled
HQQ_CHANNEL_WISE = True


def _get_official_hqq_optimizer_defaults() -> Dict[str, Union[float, int]]:
    """Derive and validate the actual pinned-package defaults for HQQ proximal optimization."""
    if Quantizer is not None and hasattr(Quantizer, "optimize_weights"):
        sig = inspect.signature(Quantizer.optimize_weights)
        default_opt = sig.parameters.get("opt_params")
        if default_opt is not None and isinstance(default_opt.default, dict):
            return dict(default_opt.default)
    return {"lp_norm": 0.7, "beta": 10.0, "kappa": 1.01, "iters": 20}


def get_effective_hqq_config(
    group_size: int = 128,
    device: Union[torch.device, str, None] = None,
    dtype: Union[torch.dtype, None] = None,
    round_zero: bool = HQQ_ROUND_ZERO,
    optimize: bool = HQQ_OPTIMIZE,
    axis: int = HQQ_AXIS,
) -> Dict[str, Any]:
    """Construct an immutable, exact dictionary of the effective HQQ configuration.

    Reflects the actual execution settings, tensor device, and computation precision.
    """
    if axis not in (0, 1):
        raise ValueError("HQQ axis must be 0 or 1")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    dev_type = dev.type
    # Official HQQ optimizer uses float16 on CUDA and float32 on CPU
    optimizer_dtype = "float16" if dev_type == "cuda" else "float32"

    if dev_type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dequantize_compute_dtype = "float32"
    elif dtype is not None:
        dequantize_compute_dtype = str(dtype).replace("torch.", "")
    else:
        dequantize_compute_dtype = "float32" if dev_type == "cpu" else "float16"

    return {
        "backend": HQQ_BACKEND_TAG,
        "version": HQQ_VERSION,
        "axis": int(axis),
        "group_size": int(group_size),
        "channel_wise": HQQ_CHANNEL_WISE,
        "optimize": bool(optimize),
        "round_zero": bool(round_zero),
        "device_type": dev_type,
        "optimizer_dtype": optimizer_dtype,
        "dequantize_compute_dtype": dequantize_compute_dtype,
        "optimizer_defaults": _get_official_hqq_optimizer_defaults(),
    }


# Immutable default configuration for backward compatibility
HQQ_CONFIG = MappingProxyType(get_effective_hqq_config(group_size=128))


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


@torch.no_grad()
def quantize_weight_hqq(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
    clip_ratio: float = 1.0,
    round_zero: bool = HQQ_ROUND_ZERO,
    optimize: bool = HQQ_OPTIMIZE,
    axis: int = HQQ_AXIS,
) -> torch.Tensor:
    """Quantize and dequantize ``weight`` using the official HQQ backend (hqq.core.quantize.Quantizer).

    Official grouping: axis=1 reshapes W to [-1, group_size]; axis=0 reshapes W
    to [group_size, -1]. Min/max and optimization reduce along the selected axis.
    Zero-points are kept as floating-point values (round_zero=False), matching the
    low-level Quantizer default. The high-level BaseQuantizeConfig uses True at 4 bits.

    Note: This is simulated (fake) weight quantization/dequantization for mixed-precision
    sensitivity estimation and perplexity evaluation. It does not by itself demonstrate
    packed low-bit inference speedup or runtime memory savings.

    Rejects clip_ratio != 1.0 with a clear ValueError, as standard HQQ does not support range clipping.
    Returns dequantized weights with the original shape, dtype, and device.
    """
    if Quantizer is None:
        raise ImportError(
            "The official 'hqq' package is not installed. Please install it via 'pip install hqq==0.2.8.post1'."
        )
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [d_out, d_in], got shape {tuple(weight.shape)}")
    if bits not in Quantizer.SUPPORTED_BITS:
        raise ValueError(f"bits={bits} is not supported by HQQ. Supported: {Quantizer.SUPPORTED_BITS}")
    if axis not in (0, 1):
        raise ValueError("HQQ axis must be 0 or 1")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    d_out, d_in = weight.shape
    if weight.numel() % group_size != 0:
        raise ValueError(f"weight.numel()={weight.numel()} is not divisible by group_size={group_size}")
    if clip_ratio != 1.0:
        raise ValueError(
            f"Official HQQ quantizer does not support clip_ratio != 1.0 (got clip_ratio={clip_ratio}). "
            f"Range clipping is only supported with the 'rtn' quantizer."
        )

    orig_dtype = weight.dtype
    orig_device = weight.device
    device_str = str(orig_device)

    # Invoke official HQQ Quantizer
    W_q, meta = Quantizer.quantize(
        weight,
        nbits=bits,
        channel_wise=True,
        group_size=group_size,
        optimize=optimize,
        round_zero=round_zero,
        axis=axis,
        bitpack=False,
        device=device_str,
    )

    # Explicit compute precision for dequantization: float32 on CPU prevents float16 emulation issues
    meta["compute_dtype"] = (
        torch.float32
        if (orig_device.type == "cpu" and orig_dtype in (torch.float16, torch.bfloat16))
        else orig_dtype
    )
    deq = Quantizer.dequantize(W_q, meta)

    return deq.reshape(d_out, d_in).to(device=orig_device, dtype=orig_dtype)


def quantize_weight(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
    clip_ratio: float = 1.0,
    quantizer: str = "rtn",
    hqq_axis: int = HQQ_AXIS,
) -> torch.Tensor:
    """Unified weight quantization dispatcher supporting group-wise RTN and official HQQ.

    Simulated fake quantization: quantizes and dequantizes weights to evaluate task accuracy
    and perplexity at candidate bit allocations. Does not evaluate packed low-bit runtime speed.
    """
    if quantizer == "rtn":
        return quantize_weight_rtn(weight, bits, group_size=group_size, clip_ratio=clip_ratio)
    elif quantizer == "hqq":
        return quantize_weight_hqq(weight, bits, group_size=group_size, clip_ratio=clip_ratio, axis=hqq_axis)
    else:
        raise ValueError(f"quantizer={quantizer!r} is invalid; expected one of {QUANTIZER_MODES}")
