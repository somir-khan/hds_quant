"""Experiment configuration (see SPEC.md)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple

import numpy as np

# Order defines `proj_family_idx` (0..6) used for one-hot features.
PROJ_FAMILIES: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

RESIDUAL_ANCHORS = ("fp16_strict", "int4_aligned")
COORD_BINDINGS = ("raw_index_strict", "salience_bucket")
QUANTIZERS = ("rtn", "hqq")
# pooled: one ridge readout over everything; per_bit: one per candidate bit-width b;
# per_family: one per projection family; per_bit_family: one per (b, family).
READOUT_GROUPINGS = ("pooled", "per_bit", "per_family", "per_bit_family")
# Transform applied to the delta_loss regression target before fitting the readout.
TARGET_TRANSFORMS = ("raw", "signed_log1p", "rank")

DEFAULT_RIDGE_ALPHAS: Tuple[float, ...] = tuple(float(x) for x in np.logspace(-2, 4, 25))


@dataclass
class ExperimentConfig:
    # --- Target model (SPEC §1) ---
    model_name: str = "meta-llama/Llama-3.2-1B"
    num_layers: int = 16
    proj_families: Tuple[str, ...] = PROJ_FAMILIES

    # --- Quantizer (SPEC §2) ---
    quantizer: str = "rtn"
    quant_group_size: int = 128
    candidate_bits: Tuple[int, ...] = (2, 3, 4)
    reference_bit: int = 4
    clip_ratio: float = 1.0

    # --- Calibration data ---
    max_encoded_tokens: int = 4096  # T: subsampled tokens for R_{p,b}
    calib_num_sequences: int = 128
    calib_seq_len: int = 512

    # --- HDS encoding (SPEC §5) ---
    hd_dim: int = 4096
    num_levels: int = 32
    max_coords: int = 8192
    num_buckets: int = 512
    seeds: Tuple[int, ...] = (42,)  # seeds[0] drives token selection / codebooks
    device: str = "cuda"

    # --- Estimator hyperparameters (SPEC §6) ---
    hist_bins: int = 32
    hist_max: float = 8.0  # upper edge of standardized-residual histogram (overflow -> last bin)
    trim_frac: float = 0.10
    winsor_quantile: float = 0.99
    ridge_alphas: Tuple[float, ...] = DEFAULT_RIDGE_ALPHAS
    cv_folds: int = 4

    # --- Ablation toggles ---
    residual_anchor: str = "fp16_strict"  # SPEC §4: fp16_strict | int4_aligned
    coord_binding: str = "raw_index_strict"  # SPEC §5.3: raw_index_strict | salience_bucket
    readout_grouping: str = "pooled"
    target_transform: str = "raw"

    def __post_init__(self) -> None:
        _check("quantizer", self.quantizer, QUANTIZERS)
        _check("residual_anchor", self.residual_anchor, RESIDUAL_ANCHORS)
        _check("coord_binding", self.coord_binding, COORD_BINDINGS)
        _check("readout_grouping", self.readout_grouping, READOUT_GROUPINGS)
        _check("target_transform", self.target_transform, TARGET_TRANSFORMS)
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if len(self.ridge_alphas) == 0 or any(a <= 0 for a in self.ridge_alphas):
            raise ValueError("ridge_alphas must be a non-empty sequence of positive floats")
        if self.reference_bit not in self.candidate_bits:
            raise ValueError("reference_bit must be one of candidate_bits")
        if len(self.seeds) == 0:
            raise ValueError("seeds must contain at least one seed")
        if self.hd_dim < 2 * (self.num_levels - 1):
            raise ValueError("hd_dim too small for the correlated level codebook")

    @property
    def num_projections(self) -> int:
        return self.num_layers * len(self.proj_families)

    @property
    def marginal_bits(self) -> Tuple[int, ...]:
        """Bit-widths that receive a delta_loss label (everything except the reference)."""
        return tuple(b for b in self.candidate_bits if b != self.reference_bit)

    def to_dict(self) -> dict:
        return asdict(self)


def _check(name: str, value: str, allowed: Tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is invalid; expected one of {allowed}")
