"""HDS-Quant: hyperdimensional sensitivity estimation for mixed-precision quantization."""

from .config import ExperimentConfig, PROJ_FAMILIES
from .quantizer import quantize_weight_rtn
from .hds_encoder import HDSEncoder
from .estimators import ESTIMATOR_NAMES, compute_estimators
from .data_utils import get_calibration_batches, get_eval_tokens
from .label_collector import (
    collect_features_single_pass,
    collect_marginal_loss_labels,
    discover_target_projections,
)
from .readout_cv import (
    cross_validate_estimators,
    evaluate_fixed_k_layer_acquisition,
    select_stratified_k_layers,
)
from .allocator import solve_bit_allocation_ilp
from .evaluator import (
    compute_ranking_metrics,
    evaluate_perplexity,
    aggregate_seed_stability,
)

__all__ = [
    "ExperimentConfig",
    "PROJ_FAMILIES",
    "quantize_weight_rtn",
    "HDSEncoder",
    "ESTIMATOR_NAMES",
    "compute_estimators",
    "get_calibration_batches",
    "get_eval_tokens",
    "collect_features_single_pass",
    "collect_marginal_loss_labels",
    "discover_target_projections",
    "cross_validate_estimators",
    "evaluate_fixed_k_layer_acquisition",
    "select_stratified_k_layers",
    "solve_bit_allocation_ilp",
    "compute_ranking_metrics",
    "evaluate_perplexity",
    "aggregate_seed_stability",
]
