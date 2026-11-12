"""Sensitivity estimators computed from a local residual matrix (SPEC §6).

Vector-valued features are returned as float32 CPU tensors, scalar estimators as Python floats.
Ridge readouts are fit downstream; this module only builds the features.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Union

import torch

from .config import PROJ_FAMILIES
from .hds_encoder import HDSEncoder

ESTIMATOR_NAMES = (
    "Scalar-MSE",
    "Scalar-L1",
    "Scalar-TrimmedMean",
    "Scalar-WinsorL2",
    "Scalar-Q99",
    "Ordinal-Hist",
    "RandProj-JL",
    "HDS-Rank",
    "HDS-Mag-Strict",
    "HDS-Hybrid",
    "Scalar-Meta-Ridge",
    "Coord-Rank-Hist",
    "Coord-Rank-Exact",
)

_LOG_EPS = 1e-8
_STD_EPS = 1e-6
COORD_HIST_BUCKETS = 16  # B_c_hist: canonical coordinate buckets of the explicit (non-hyperdimensional) control


@torch.no_grad()
def compute_selected_estimator(R_pb, fp16_col_norms, proj_family_idx, encoder,
                               estimator, coord_binding="salience_bucket"):
    """Compute ONLY one limited-label benchmark feature, matching compute_estimators."""
    if estimator not in ("Scalar-MSE-Ridge", "Scalar-Meta-Ridge", "Coord-Rank-Hist", "Coord-Rank-Exact", "HDS-Hybrid", "Coverage-Primary"):
        raise ValueError(f"Unsupported standalone estimator: {estimator}")
    R = R_pb.to(device=encoder.device, dtype=torch.float32)
    mse = float(R.pow(2).mean())
    if estimator == "Scalar-MSE-Ridge":
        return {"Scalar-MSE": mse}
    log_mse = math.log(mse + _LOG_EPS)
    q99 = _sorted_quantile(torch.sort(R.reshape(-1)).values, 0.99)
    one_hot = torch.zeros(len(PROJ_FAMILIES), dtype=torch.float32, device=R.device)
    one_hot[proj_family_idx] = 1.0
    meta = torch.cat([R.new_tensor([log_mse, math.log(q99 + _LOG_EPS)]), one_hot])
    if estimator == "Scalar-Meta-Ridge":
        return {estimator: meta.cpu()}
    F = encoder.level_frequencies(R)
    if estimator == "Coverage-Primary":
        return {
            "Scalar-Meta-Ridge": meta.cpu(),
            "Coord-Rank-Hist": torch.cat([_coord_rank_hist(F, fp16_col_norms, encoder, coord_binding), meta]).cpu(),
            "HDS-Hybrid": torch.cat([encoder.bind(F, fp16_col_norms, coord_binding), meta]).cpu(),
        }
    if estimator == "HDS-Hybrid":
        vector = encoder.bind(F, fp16_col_norms, coord_binding)
    elif estimator == "Coord-Rank-Exact":
        vector = _coord_rank_exact(F, fp16_col_norms, encoder, coord_binding)
    else:
        vector = _coord_rank_hist(F, fp16_col_norms, encoder, coord_binding)
    return {estimator: torch.cat([vector, meta]).cpu()}


class _Clock:
    """Lap timer that synchronizes CUDA first (so asynchronous kernels are charged to the right segment)."""

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device = device
        self.enabled = enabled
        self.last = self._now()

    def _now(self) -> float:
        if not self.enabled:
            return 0.0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def lap(self) -> float:
        if not self.enabled:
            return 0.0
        now = self._now()
        dt = now - self.last
        self.last = now
        return dt


def _coord_rank_hist(
    F: torch.Tensor,
    fp16_col_norms: torch.Tensor,
    encoder: HDSEncoder,
    coord_binding: str,
    num_buckets: int = COORD_HIST_BUCKETS,
) -> torch.Tensor:
    """Explicit coordinate-bucket x rank-level histogram ``[num_buckets * num_levels]`` (no hypervector binding).

    Rows of the level-frequency matrix ``F`` ``[d_out, M]`` are averaged inside ``num_buckets`` canonical
    buckets: salience-rank buckets of ``fp16_col_norms`` for ``salience_bucket`` binding, and raw-index buckets
    ``(j * num_buckets) // d_out`` for ``raw_index_strict``. Every bucket row is a distribution over rank levels.
    """
    d_out = F.shape[0]
    ar = torch.arange(d_out, device=F.device)
    if coord_binding == "salience_bucket":
        norms = fp16_col_norms.to(device=F.device, dtype=torch.float32)
        order = torch.sort(norms, descending=True, stable=True).indices
        salience_rank = torch.empty_like(order)
        salience_rank[order] = ar
        bucket = (salience_rank * num_buckets) // d_out
    else:
        bucket = (ar * num_buckets) // d_out
    F_bucket = torch.zeros(num_buckets, F.shape[1], dtype=F.dtype, device=F.device)
    F_bucket.index_add_(0, bucket, F)
    counts = torch.bincount(bucket, minlength=num_buckets).clamp(min=1).to(F.dtype).unsqueeze(1)
    return (F_bucket / counts).reshape(-1)


def _coord_rank_exact(
    F: torch.Tensor,
    fp16_col_norms: torch.Tensor,
    encoder: HDSEncoder,
    coord_binding: str,
) -> torch.Tensor:
    """Full-resolution coordinate-bucket x rank-level histogram [512 * num_levels] (no hypervector binding).

    Preserves the exact same 512-bucket coordinate resolution as HDS:
      * For coord_binding == 'salience_bucket': bucket = encoder.bucket_indices(fp16_col_norms) (512 buckets).
      * For coord_binding == 'raw_index_strict': bucket = (ar * 512) // d_out (512 canonical index buckets).
    Rows of F are averaged inside each of the 512 buckets so each bucket row sums to 1.
    """
    d_out = F.shape[0]
    num_buckets = encoder.num_buckets
    if coord_binding == "salience_bucket":
        bucket = encoder.bucket_indices(fp16_col_norms)
    else:
        ar = torch.arange(d_out, device=F.device)
        bucket = (ar * num_buckets) // d_out
    F_bucket = torch.zeros(num_buckets, F.shape[1], dtype=F.dtype, device=F.device)
    F_bucket.index_add_(0, bucket, F)
    counts = torch.bincount(bucket, minlength=num_buckets).clamp(min=1).to(F.dtype).unsqueeze(1)
    return (F_bucket / counts).reshape(-1)


def _sorted_quantile(sorted_flat: torch.Tensor, q: float) -> float:
    """Linear-interpolated quantile of an ascending 1D tensor (matches numpy/torch default).

    ``torch.quantile`` is avoided because it rejects inputs with more than 16M elements.
    """
    pos = q * (sorted_flat.numel() - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_flat[lo]) * (1.0 - frac) + float(sorted_flat[hi]) * frac


@torch.no_grad()
def compute_estimators(
    R_pb: torch.Tensor,
    fp16_col_norms: torch.Tensor,
    proj_family_idx: int,
    encoder: HDSEncoder,
    coord_binding: str = "raw_index_strict",
    hist_bins: int = 32,
    hist_max: float = 8.0,
    trim_frac: float = 0.10,
    winsor_quantile: float = 0.99,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, Union[float, torch.Tensor]]:
    """Compute features for every estimator in SPEC §6 (except ``Oracle-Loss``, which is the label).

    Args:
        R_pb: ``[T, d_out]`` non-negative residual matrix ``|Y_ref - Y_b|``.
        fp16_col_norms: ``[d_out]`` FP16 calibration output L2 norms ``||Y_p^(16)[:, j]||_2``.
        proj_family_idx: 0..6, index into ``PROJ_FAMILIES``.
        encoder: ``HDSEncoder`` (supplies codebooks and the JL matrix).
        coord_binding: ``raw_index_strict`` or ``salience_bucket`` for the HDS features.
        timings: optional dict; if given, the standalone feature-extraction seconds of every estimator (and of
            ``Scalar-MSE-Ridge``, whose feature is ``log(MSE)``) are *added* to ``timings[name]``. Computations
            shared between estimators (the sort, the level-frequency pass, MSE, Q99) are charged in full to every
            estimator that would need them on its own.

    Returns a dict keyed by ``ESTIMATOR_NAMES``:
        scalars (float): Scalar-MSE, Scalar-L1, Scalar-TrimmedMean, Scalar-WinsorL2, Scalar-Q99
        vectors (float32 CPU): Ordinal-Hist ``[hist_bins + 1]``, RandProj-JL ``[D]``, HDS-Rank ``[D]``,
        HDS-Mag-Strict ``[D + 1]``, HDS-Hybrid ``[D + 1 + 1 + 7]``,
        Scalar-Meta-Ridge ``[log_mse, log_q99, one_hot(7)]`` (the HDS-Hybrid metadata *without* ``v``),
        Coord-Rank-Hist ``[16 * num_levels + 9]`` (explicit coordinate-bucket x rank-level histogram from the same
        ``F`` that HDS binds, plus the same 9 metadata dims; no random hypervector binding).

    Conventions:
        * Scalar-L1 = mean(R); Scalar-TrimmedMean drops ``trim_frac`` of entries from each tail of the
          flattened R; Scalar-Q99 = linear-interpolated 99th percentile of R.
        * Scalar-WinsorL2 = RMS of R clipped at its 99th percentile, i.e. ``||clip(R, q99)||_2 / sqrt(T * d_out)``
          (size-normalized so projections with different d_out are comparable).
        * Ordinal-Hist = normalized histogram of ``R / (std(R) + 1e-6)`` over ``[0, hist_max]`` (overflow goes
          to the last bin) followed by ``log(MSE + 1e-8)``.
        * RandProj-JL = mean_t(ranks / d_out) @ Pi (same as projecting each token then averaging, by linearity).
    """
    if R_pb.ndim != 2:
        raise ValueError(f"R_pb must be [T, d_out], got {tuple(R_pb.shape)}")
    if not 0 <= int(proj_family_idx) < len(PROJ_FAMILIES):
        raise ValueError(f"proj_family_idx must be in [0, {len(PROJ_FAMILIES) - 1}]")
    T, d_out = R_pb.shape
    R = R_pb.to(device=encoder.device, dtype=torch.float32)
    clock = _Clock(R.device, enabled=timings is not None)

    # ---- scalar estimators (one shared sort of the flattened residuals)
    sorted_flat = torch.sort(R.reshape(-1)).values
    t_sort = clock.lap()
    n = sorted_flat.numel()
    mse_t = R.pow(2).mean()
    mse = float(mse_t)
    t_mse = clock.lap()
    l1 = float(R.mean())
    t_l1 = clock.lap()
    k = int(trim_frac * n)
    trimmed = float(sorted_flat[k : n - k].mean())
    t_trim = clock.lap()
    q99 = _sorted_quantile(sorted_flat, 0.99)
    t_q99 = clock.lap()
    q_win = _sorted_quantile(sorted_flat, winsor_quantile)
    winsor_l2 = float(R.clamp(max=q_win).pow(2).mean().sqrt())
    t_win = clock.lap()

    log_mse = math.log(mse + _LOG_EPS)
    log_q99 = math.log(q99 + _LOG_EPS)

    # ---- Ordinal-Hist
    z = (R / (R.std() + _STD_EPS)).clamp(max=hist_max)
    hist = torch.histc(z, bins=hist_bins, min=0.0, max=hist_max)
    hist = hist / hist.sum().clamp(min=1.0)
    ordinal_hist = torch.cat([hist, hist.new_tensor([log_mse])])
    t_hist = clock.lap()

    # ---- rank-based features from a single sort pass
    F, mean_rank = encoder.level_frequencies(R, return_mean_rank=True)
    t_freq = clock.lap()
    jl = mean_rank @ encoder.jl_matrix(d_out)
    t_jl = clock.lap()
    v = encoder.bind(F, fp16_col_norms, coord_binding)
    t_bind = clock.lap()

    one_hot = torch.zeros(len(PROJ_FAMILIES), dtype=torch.float32, device=v.device)
    one_hot[int(proj_family_idx)] = 1.0
    scal = v.new_tensor([log_mse])
    meta_vec = torch.cat([v.new_tensor([log_mse, log_q99]), one_hot])  # [log_mse, log_q99, one_hot(7)]
    hybrid = torch.cat([v, scal, v.new_tensor([log_q99]), one_hot])
    t_meta = clock.lap()

    # ---- non-HDS controls
    coord_hist = _coord_rank_hist(F, fp16_col_norms, encoder, coord_binding)
    coord_rank_hist = torch.cat([coord_hist, meta_vec])
    t_coord = clock.lap()
    coord_exact = _coord_rank_exact(F, fp16_col_norms, encoder, coord_binding)
    coord_rank_exact = torch.cat([coord_exact, meta_vec])
    t_coord_exact = clock.lap()

    if timings is not None:
        shared_meta = t_mse + t_sort + t_q99 + t_meta
        charged = {
            "Scalar-MSE": t_mse,
            "Scalar-L1": t_l1,
            "Scalar-TrimmedMean": t_sort + t_trim,
            "Scalar-WinsorL2": t_sort + t_win,
            "Scalar-Q99": t_sort + t_q99,
            "Scalar-MSE-Ridge": t_mse,
            "Scalar-Meta-Ridge": shared_meta,
            "Ordinal-Hist": t_mse + t_hist,
            "RandProj-JL": t_freq + t_jl,
            "HDS-Rank": t_freq + t_bind,
            "HDS-Mag-Strict": t_freq + t_bind + t_mse,
            "HDS-Hybrid": t_freq + t_bind + shared_meta,
            "Coord-Rank-Hist": t_freq + t_coord + shared_meta,
            "Coord-Rank-Exact": t_freq + t_coord_exact + shared_meta,
        }
        for name, secs in charged.items():
            timings[name] = timings.get(name, 0.0) + float(secs)

    return {
        "Scalar-MSE": mse,
        "Scalar-L1": l1,
        "Scalar-TrimmedMean": trimmed,
        "Scalar-WinsorL2": winsor_l2,
        "Scalar-Q99": q99,
        "Ordinal-Hist": ordinal_hist.cpu(),
        "RandProj-JL": jl.cpu(),
        "HDS-Rank": v.cpu(),
        "HDS-Mag-Strict": torch.cat([v, scal]).cpu(),
        "HDS-Hybrid": hybrid.cpu(),
        "Scalar-Meta-Ridge": meta_vec.cpu(),
        "Coord-Rank-Hist": coord_rank_hist.cpu(),
        "Coord-Rank-Exact": coord_rank_exact.cpu(),
    }
