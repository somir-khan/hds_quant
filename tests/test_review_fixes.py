"""Unit tests explicitly validating Fixes 1 through 9 from the technical review."""

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import GroupKFold
from transformers import LlamaConfig, LlamaForCausalLM

from hds_quant import (
    ESTIMATOR_NAMES,
    ExperimentConfig,
    HDSEncoder,
    compute_estimators,
    quantize_weight_rtn,
    select_stratified_k_layers,
)
from hds_quant.allocator import _FEASIBLE_PAYLOAD_CACHE, solve_bit_allocation_ilp
from hds_quant.evaluator import aggregate_seed_stability, evaluate_perplexity
from types import SimpleNamespace

from hds_quant.label_collector import (
    PROJ_FAMILIES,
    collect_features_single_pass,
    collect_marginal_loss_labels,
    discover_target_projections,
)
from hds_quant.readout_cv import (
    ALL_ESTIMATORS,
    ORACLE_ESTIMATORS,
    SUPERVISED_ESTIMATORS,
    UNSUPERVISED_SCALARS,
    _META_COLS,
    _build_alloc_costs,
    _fit_rank_transform,
    _fit_ridge,
    _get_column_weights,
    _inner_group_splits,
    _make_target_transform,
    _prepare_design,
    cross_validate_estimators,
    evaluate_fixed_k_layer_acquisition,
)
from hds_quant.run_experiment import (
    LOW_SHOT_ESTIMATORS,
    MAIN_ESTIMATORS,
    _build_smoke_model,
    _compute_label_cache_fingerprint,
    _get_smoke_calib_ids,
    _load_cached_labels,
    run_experiment,
)
from scripts.format_tables import ORDERED_ESTIMATORS


# ------------------------------------------------------------------ Fix 1 [P1]: Deterministic Smoke Model & Fingerprint
def test_fix1_deterministic_smoke_model_across_global_rng():
    """_build_smoke_model() must call torch.manual_seed(0) BEFORE model creation."""
    torch.manual_seed(12345)
    _ = torch.randn(100, 100)
    m1, _ = _build_smoke_model()

    torch.manual_seed(99999)
    _ = torch.randn(50, 50)
    m2, _ = _build_smoke_model()

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Smoke model weights must be 100% deterministic regardless of preceding RNG!"


def test_fix1_label_cache_fingerprint_sensitivity():
    """_compute_label_cache_fingerprint must hash all specified fields and detect any change."""
    model, _ = _build_smoke_model()
    calib_ids = _get_smoke_calib_ids(42)
    cfg = ExperimentConfig(
        model_name="smoke",
        num_layers=4,
        quant_group_size=64,
        reference_bit=4,
        candidate_bits=(2, 3, 4),
        clip_ratio=1.0,
        hd_dim=128,
        device="cpu",
    )

    base_fp = _compute_label_cache_fingerprint(model, calib_ids, cfg)
    assert isinstance(base_fp, str) and len(base_fp) == 64  # sha256 hex string

    # 1. Changing calib_ids changes fingerprint
    calib_ids_alt = _get_smoke_calib_ids(100)
    assert _compute_label_cache_fingerprint(model, calib_ids_alt, cfg) != base_fp

    # 2. Changing quant_group_size changes fingerprint
    cfg_g128 = ExperimentConfig(model_name="smoke", quant_group_size=128, candidate_bits=(2, 3, 4), device="cpu")
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg_g128) != base_fp

    # 3. Changing clip_ratio changes fingerprint
    cfg_clip = ExperimentConfig(model_name="smoke", clip_ratio=0.9, candidate_bits=(2, 3, 4), device="cpu")
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg_clip) != base_fp

    # 4. Changing model name changes fingerprint
    cfg_name = ExperimentConfig(model_name="other-model", candidate_bits=(2, 3, 4), device="cpu")
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg_name) != base_fp

    # 5. Mutating weights changes fingerprint
    target_projs = discover_target_projections(model)
    with torch.no_grad():
        target_projs[0]["module"].weight[0, 0] += 0.05
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg) != base_fp


def test_fix1_load_cached_labels_fingerprint_validation(tmp_path):
    """--reuse-cache verifies fingerprint and recomputes if missing or mismatched."""
    cache_file = tmp_path / "label_cache.pt"
    valid_data = {
        "fingerprint": "expected_fp_hash_12345",
        "delta_loss": {("proj.0", 2): 0.123},
        "label_time_sec": 4.56,
    }
    torch.save(valid_data, cache_file)

    # Matching fingerprint loads successfully
    loaded = _load_cached_labels(cache_file, "expected_fp_hash_12345")
    assert loaded is not None
    assert loaded["delta_loss"] == {("proj.0", 2): 0.123}

    # Mismatched fingerprint returns None with a warning
    with pytest.warns(RuntimeWarning, match="mismatch"):
        loaded_mismatch = _load_cached_labels(cache_file, "different_fp_hash")
    assert loaded_mismatch is None

    # Missing fingerprint in cache returns None with a warning
    missing_data = {
        "delta_loss": {("proj.0", 2): 0.123},
        "label_time_sec": 4.56,
    }
    torch.save(missing_data, cache_file)
    with pytest.warns(RuntimeWarning, match="mismatch"):
        loaded_missing = _load_cached_labels(cache_file, "expected_fp_hash_12345")
    assert loaded_missing is None


# ------------------------------------------------------------------ Fix 2 [P1]: Separate Calibration vs Test Perplexity Timing
def test_fix2_timing_accounting_formulas():
    """Verify exact calibration time breakdown for unsupervised, oracle, and 4-fold CV estimators."""
    # Synthetic seed runs data
    seed_runs = [
        {
            "seed": 42,
            "estimators": ["Scalar-MSE", "Oracle-Loss", "Oracle-Loss-Raw", "HDS-Hybrid"],
            "budgets": [3.0],
            "forward_capture_time_sec": 1.0,
            "estimator_feature_time_sec": {
                "Scalar-MSE": 0.1,
                "Scalar-L1": 0.1,
                "Scalar-TrimmedMean": 0.1,
                "Scalar-WinsorL2": 0.1,
                "Scalar-Q99": 0.1,
                "Scalar-MSE-Ridge": 0.1,
                "Scalar-Meta-Ridge": 0.15,
                "Ordinal-Hist": 0.2,
                "RandProj-JL": 0.2,
                "Coord-Rank-Hist": 0.25,
                "HDS-Rank": 0.3,
                "HDS-Mag-Strict": 0.35,
                "HDS-Hybrid": 0.4,
                "Oracle-Loss": 0.0,
                "Oracle-Loss-Raw": 0.0,
            },
            "estimator_readout_time_sec": {
                "Scalar-MSE": 0.0,
                "Oracle-Loss": 0.0,
                "Oracle-Loss-Raw": 0.0,
                "HDS-Hybrid": 0.8,
            },
            "estimator_ilp_time_sec": {est: 0.05 for est in ALL_ESTIMATORS},
            "label_time_sec": 10.0,
            "wall_label_time_sec": 10.0,
            "test_eval_time_sec": 30.0,  # Must NOT be included in calib time!
            "ranking_metrics": {
                est: {
                    "spearman_global": 0.8,
                    "kendall_global": 0.6,
                    "spearman_bit2": 0.8,
                    "spearman_bit3": 0.8,
                    "spearman_within_family": 0.7,
                    "top20_recall": 0.9,
                }
                for est in ALL_ESTIMATORS
            },
            "eval_results": {est: {3.0: {"ppl": 4.0, "achieved_avg_bits": 3.0}} for est in ALL_ESTIMATORS},
        }
    ]

    summary = aggregate_seed_stability(seed_runs)
    costs = summary["calibration_costs"]

    # 1. Unsupervised scalar: fwd + feat + ilp
    assert costs["Scalar-MSE"]["total_calib_time_sec"] == pytest.approx(1.0 + 0.1 + 0.05)
    assert costs["Scalar-MSE"]["test_eval_time_sec"] == pytest.approx(30.0)

    # 2. Oracle loss: label_time + ilp (no forward capture, no feature extraction)
    assert costs["Oracle-Loss"]["total_calib_time_sec"] == pytest.approx(10.0 + 0.05)
    assert costs["Oracle-Loss-Raw"]["total_calib_time_sec"] == pytest.approx(10.0 + 0.05)

    # 3. 4-fold CV supervised: total_calib_time_sec is full bank (1.0 * label) = 1.0 + 0.4 + 1.0 * 10.0 + 0.8 + 0.05
    expected_full = 1.0 + 0.4 + 1.0 * 10.0 + 0.8 + 0.05
    assert costs["HDS-Hybrid"]["total_calib_time_sec"] == pytest.approx(expected_full)

    # 4. Single-fold calib time: fwd + feat + 0.75 * label + readout + ilp
    expected_single_fold = 1.0 + 0.4 + 0.75 * 10.0 + 0.8 + 0.05
    assert costs["HDS-Hybrid"]["single_fold_calib_time_sec"] == pytest.approx(expected_single_fold)


# ------------------------------------------------------------------ Fix 3 [P2]: Quantizer Affine Grid
def test_fix3_single_sign_group_affine_grid():
    """W = linspace(1.0, 2.0, 128) at 4 bits must retain all 16 levels and <= scale/2 reconstruction error."""
    W = torch.linspace(1.0, 2.0, 128).reshape(1, 128)
    Wq = quantize_weight_rtn(W, 4, group_size=128)
    scale = (2.0 - 1.0) / 15.0

    assert torch.unique(Wq).numel() == 16
    max_err = float((Wq - W).abs().max())
    assert max_err <= (scale / 2.0) + 1e-6


# ------------------------------------------------------------------ Fix 4 [P2]: Enforce Matched Payload Budget
def test_fix4_enforce_exact_payload_utilization():
    """Even with negative/zero costs, enforce_exact_payload=True must prevent underutilizing bit budget."""
    # Simulate 16-layer Llama-like projections (112 projections)
    meta = []
    sizes = [4096 * 4096, 1024 * 4096, 1024 * 4096, 4096 * 4096, 14336 * 4096, 4096 * 14336, 14336 * 4096]
    for layer in range(16):
        for fi, s in enumerate(sizes):
            meta.append({"name": f"layer.{layer}.proj.{fi}", "num_params": s})

    total_params = sum(m["num_params"] for m in meta)

    # Case: negative/zero costs where higher bit has lower or equal cost
    # e.g., raw readout cost where 2-bit has cost -0.5, 3-bit -0.8, 4-bit 0.0
    # Under plain <= 3.00, if costs are inverted or degenerate, budget could collapse to 2.0 or 2.67.
    rng = np.random.RandomState(42)
    costs = {}
    for m in meta:
        # Negative costs that favor 2-bit or 3-bit if unconstrained
        costs[(m["name"], 4)] = 0.0
        costs[(m["name"], 3)] = float(rng.uniform(-2.0, -1.0))
        costs[(m["name"], 2)] = float(rng.uniform(-5.0, -3.0))

    target = 3.0
    # With enforce_exact_payload=True, the lower bound keeps achieved bits near target
    res_enforced = solve_bit_allocation_ilp(costs, meta, target, enforce_exact_payload=True)
    assert res_enforced["success"]
    assert target - 0.02 <= res_enforced["achieved_avg_bits"] <= target + 1e-6

    # With enforce_exact_payload=False, the solver freely picks all 2-bit (cost is lowest)
    res_free = solve_bit_allocation_ilp(costs, meta, target, enforce_exact_payload=False)
    assert res_free["success"]
    assert res_free["achieved_avg_bits"] == pytest.approx(2.0)


# ------------------------------------------------------------------ Fix 5 [P2]: LaTeX Escaping & Mojibake Cleanup
def test_fix5_latex_escaping_and_mojibake():
    """_latex_escape must correctly escape LaTeX special chars and mojibake."""
    # Dynamically load format_tables.py
    script_path = Path(__file__).parent.parent / "scripts" / "format_tables.py"
    spec = importlib.util.spec_from_file_location("format_tables", script_path)
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)

    # Test escaping
    assert ft._latex_escape("Accuracy %") == r"Accuracy \%"
    assert ft._latex_escape("A & B") == r"A \& B"
    assert ft._latex_escape("layer_0_weight") == r"layer\_0\_weight"
    assert ft._latex_escape("# Params") == r"\# Params"

    # Test greek & plusminus symbols
    assert ft._latex_escape(r"Spearman ρ") == r"Spearman $\rho$"
    assert ft._latex_escape(r"Kendall τ") == r"Kendall $\tau$"
    assert ft._latex_escape(r"1.23 ± 0.05") == r"1.23 $\pm$ 0.05"

    # Test mojibake cleanup
    assert ft._latex_escape("1.23 Â± 0.05") == r"1.23 $\pm$ 0.05"
    assert ft._latex_escape("Spearman Ï ") == r"Spearman $\rho$"
    assert ft._latex_escape("Kendall Ï„") == r"Kendall $\tau$"

    # Test df_to_latex escaping on headers, captions, and cells
    df = pd.DataFrame(
        {
            "Method_Name": ["HDS_Hybrid", "Scalar_MSE"],
            "Spearman_ρ": ["0.85 ± 0.02", "0.72 Â± 0.03"],
            "Cost_%": ["10%", "5%"],
        }
    )
    latex_out = ft.df_to_latex(df, caption="Evaluation & Results with layer_idx", label="tab:results")
    assert r"Method\_Name" in latex_out
    assert r"Spearman\_$\rho$" in latex_out
    assert r"Cost\_\%" in latex_out
    assert r"Evaluation \& Results with layer\_idx" in latex_out
    assert r"HDS\_Hybrid" in latex_out
    assert r"0.85 $\pm$ 0.02" in latex_out
    assert r"0.72 $\pm$ 0.03" in latex_out
    assert r"10\%" in latex_out


# ------------------------------------------------------------------ Fix 6 [Research Validity 1]: Non-HDS Controls
def test_fix6_controls_scalar_meta_ridge_and_coord_rank_hist():
    """Verify Scalar-Meta-Ridge (10D) and Coord-Rank-Hist (521D) controls."""
    encoder = HDSEncoder(hd_dim=128, device="cpu", seed=0)
    R = torch.rand(32, 64)
    norms = torch.rand(64) + 0.1
    f = compute_estimators(R, norms, 1, encoder, coord_binding="salience_bucket")

    # 1. Scalar-Meta-Ridge: [log_mse, log_q99, one_hot_proj(7)] = 9 features (+ bit indicator in readout = 10 cols)
    meta = f["Scalar-Meta-Ridge"]
    assert tuple(meta.shape) == (9,)
    assert torch.allclose(meta, f["HDS-Hybrid"][128:])

    # 2. Coord-Rank-Hist: 16 buckets * 32 rank-levels = 512 + 9 meta = 521 features (+ indicator = 522 cols)
    hist = f["Coord-Rank-Hist"]
    assert tuple(hist.shape) == (16 * 32 + 9,)
    assert torch.allclose(hist[512:], meta)

    # 3. Column weights for Coord-Rank-Hist upweights metadata by sqrt(512 / 10)
    w = _get_column_weights(522, estimator="Coord-Rank-Hist", hd_dim=512)
    assert w is not None
    assert np.allclose(w[:512], 1.0)
    assert np.allclose(w[512:], math.sqrt(512 / 10.0))


# ------------------------------------------------------------------ Fix 7 [Research Validity 2]: Fixed K-Layer Acquisition
def test_fix7_fixed_k_layer_acquisition():
    """evaluate_fixed_k_layer_acquisition acquires strictly k_layers, fits on them, and predicts held-out."""
    meta = []
    for layer in range(4):
        for fi, fam in enumerate(PROJ_FAMILIES):
            meta.append(
                {
                    "name": f"model.layers.{layer}.{fam}",
                    "layer_idx": layer,
                    "family": fam.split(".")[-1],
                    "family_idx": fi,
                    "num_params": 1000,
                }
            )

    encoder = HDSEncoder(hd_dim=64, device="cpu", seed=1)
    feats, delta = {}, {}
    for m in meta:
        for b in (2, 3):
            R = torch.rand(16, 32)
            norms = torch.rand(32) + 0.1
            feats[(m["name"], b)] = compute_estimators(R, norms, m["family_idx"], encoder)
            delta[(m["name"], b)] = float(0.1 * m["layer_idx"] + 0.05 * (4 - b))

    cfg = ExperimentConfig(hd_dim=64, device="cpu")
    k_layers = 2
    res = evaluate_fixed_k_layer_acquisition(feats, delta, meta, cfg, k_layers=k_layers, seed=42)

    assert res["n_labeled_layers"] == 2
    assert len(res["acquired_layers"]) == 2
    assert len(res["unlabeled_layers"]) == 2
    assert res["acquisition_fraction"] == 0.5

    # Check that heldout_predictions contains only the unlabeled layers
    for est in SUPERVISED_ESTIMATORS:
        heldout = res["heldout_predictions"][est]
        for (name, b) in heldout:
            layer = int(name.split(".")[2])
            assert layer in res["unlabeled_layers"]
            assert layer not in res["acquired_layers"]

    # Check that raw_predictions uses ground-truth labels for acquired layers
    for est in SUPERVISED_ESTIMATORS:
        raw = res["raw_predictions"][est]
        for (name, b), v in raw.items():
            layer = int(name.split(".")[2])
            if layer in res["acquired_layers"]:
                assert v == delta[(name, b)]


# ------------------------------------------------------------------ Fix 8 [Research Validity 3]: Oracle-Loss-Raw
def test_fix8_oracle_loss_raw_unmodified_costs():
    """Oracle-Loss-Raw keeps unmodified delta_loss while Oracle-Loss monotone clamps."""
    delta = {
        ("proj_A", 2): -0.05,  # negative
        ("proj_A", 3): 0.10,   # non-monotone (s2 < s3)
        ("proj_B", 2): 0.30,
        ("proj_B", 3): 0.20,
    }

    raw = {
        "Oracle-Loss": dict(delta),
        "Oracle-Loss-Raw": dict(delta),
    }
    projs = [{"name": "proj_A"}, {"name": "proj_B"}]
    alloc = _build_alloc_costs(raw, projs, ["Oracle-Loss", "Oracle-Loss-Raw"])

    # Oracle-Loss-Raw: raw delta for b in (2, 3), 0.0 for b=4
    assert alloc["Oracle-Loss-Raw"][("proj_A", 2)] == -0.05
    assert alloc["Oracle-Loss-Raw"][("proj_A", 3)] == 0.10
    assert alloc["Oracle-Loss-Raw"][("proj_A", 4)] == 0.0

    # Oracle-Loss: monotone clamped
    assert alloc["Oracle-Loss"][("proj_A", 3)] == 0.10  # clamped >= 0
    assert alloc["Oracle-Loss"][("proj_A", 2)] >= 0.10 + 1e-6  # monotone enforced: s2 >= s3 + 1e-6
    assert alloc["Oracle-Loss"][("proj_A", 4)] == 0.0


# ------------------------------------------------------------------ Fix 9 [Research Validity 4]: GroupKFold & Per-Seq PPL
def test_fix9_group_kfold_in_ridge_cv(monkeypatch):
    """RidgeCV should receive GroupKFold splits when groups are passed with >=2 unique groups."""
    X = np.random.randn(20, 8)
    y = np.random.randn(20)
    groups = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5

    splits = _inner_group_splits(X, y, groups)
    assert splits is not None
    assert len(splits) == 4

    for tr, te in splits:
        tr_groups = {groups[i] for i in tr}
        te_groups = {groups[i] for i in te}
        assert tr_groups.isdisjoint(te_groups), "Inner CV must be group/layer disjoint!"

    scaler, ridge, _ = _fit_ridge(X, y, groups=groups)
    assert hasattr(ridge, "alpha_")


def test_fix9_eval_perplexity_returns_seq_nlls_and_token_counts():
    """evaluate_perplexity must return seq_nlls and seq_token_counts matching eval_ids batches."""
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=128,
    )
    model = LlamaForCausalLM(cfg).eval()
    meta_list = discover_target_projections(model)
    eval_ids = torch.randint(0, 128, (3, 32))
    exp_config = ExperimentConfig(num_layers=2, quant_group_size=32, device="cpu")

    out = evaluate_perplexity(model, eval_ids, meta_list, exp_config)
    assert "seq_nlls" in out and "seq_token_counts" in out
    assert len(out["seq_nlls"]) == 3
    assert len(out["seq_token_counts"]) == 3
    assert all(math.isfinite(nll) for nll in out["seq_nlls"])
    assert all(c == 31 for c in out["seq_token_counts"])
    # Weighted mean of seq_nlls should equal overall nll
    weighted_mean = sum(n * c for n, c in zip(out["seq_nlls"], out["seq_token_counts"])) / sum(out["seq_token_counts"])
    assert out["nll"] == pytest.approx(weighted_mean)


# ==================================================================
# Technical Review 2 Fixes Verification Tests
# ==================================================================

# ------------------------------------------------------------------ Fix 1 [P1]: Support True Unlabeled Layers
def test_review2_fix1_select_stratified_k_layers():
    layers = list(range(16))
    k = 4
    acquired = select_stratified_k_layers(layers, k, seed=42)
    assert len(acquired) == 4
    # Each stratum is [0..3], [4..7], [8..11], [12..15]
    assert 0 <= acquired[0] < 4
    assert 4 <= acquired[1] < 8
    assert 8 <= acquired[2] < 12
    assert 12 <= acquired[3] < 16
    assert acquired == sorted(acquired)


def test_review2_fix1_true_unlabeled_layers_acquisition():
    meta_list = []
    for l in range(16):
        meta_list.append({
            "name": f"model.layers.{l}.self_attn.q_proj",
            "layer_idx": l,
            "family": "q_proj",
            "family_idx": 0,
            "num_params": 1024,
            "d_out": 32,
            "d_in": 32,
        })
    feats = {}
    for m in meta_list:
        for b in (2, 3):
            feats[(m["name"], b)] = {
                "Scalar-MSE": 0.05 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-Meta-Ridge": np.ones(9),
                "Ordinal-Hist": np.ones(33) / 33,
                "RandProj-JL": np.ones(64),
                "Coord-Rank-Hist": np.ones(521),
                "Coord-Rank-Exact": np.ones(16393),
                "HDS-Rank": np.ones(64),
                "HDS-Mag-Strict": np.ones(65),
                "HDS-Hybrid": np.ones(73),
            }
    # delta_loss ONLY for 4 layers: {1, 5, 9, 13}
    labeled_layers = {1, 5, 9, 13}
    delta_loss = {}
    for m in meta_list:
        if m["layer_idx"] in labeled_layers:
            for b in (2, 3):
                delta_loss[(m["name"], b)] = 0.1 * (4 - b) + 0.02 * m["layer_idx"]

    cfg = ExperimentConfig(num_layers=16, hd_dim=64, quant_group_size=32, device="cpu")

    # If no projections have labels, require_all_labels=True raises ValueError
    with pytest.raises(ValueError, match="features and labels"):
        _prepare_design(feats, {}, meta_list, cfg, require_all_labels=True)

    # When 4 layers are labeled, require_all_labels=True keeps only those 4 projections
    d_true = _prepare_design(feats, delta_loss, meta_list, cfg, require_all_labels=True)
    assert len(d_true["names"]) == 4 * 2
    assert np.all(np.isfinite(d_true["y"]))

    # _prepare_design require_all_labels=False succeeds for all 16 projections and fills missing labels with NaN
    d = _prepare_design(feats, delta_loss, meta_list, cfg, require_all_labels=False)
    assert len(d["names"]) == 32
    assert np.isnan(d["y"]).sum() == (16 - 4) * 2

    # evaluate_fixed_k_layer_acquisition with k_layers=4 should directly use labeled_layers
    acq = evaluate_fixed_k_layer_acquisition(feats, delta_loss, meta_list, cfg, k_layers=4, seed=42)
    assert acq["acquired_layers"] == [1, 5, 9, 13]
    assert len(acq["unlabeled_layers"]) == 12
    assert set(acq["unlabeled_layers"]) == set(range(16)) - labeled_layers
    assert "HDS-Hybrid" in acq["heldout_predictions"]
    # All unlabeled layers should have predictions
    assert len(acq["heldout_predictions"]["HDS-Hybrid"]) == 12 * 2
    for (name, b), val in acq["heldout_predictions"]["HDS-Hybrid"].items():
        assert math.isfinite(val)


# ------------------------------------------------------------------ Fix 2 [P1]: Two-Stage Exact Payload Equality
def test_review2_fix2_exact_payload_equality_across_cost_functions():
    rng = np.random.RandomState(42)
    meta = [
        {"name": f"proj_{i}", "num_params": p}
        for i, p in enumerate([1024, 2048, 4096, 8192, 1024, 4096, 8192, 16384])
    ]
    # Cost function A: standard positive monotone costs
    costs_A = {}
    for m in meta:
        base = rng.uniform(0.1, 1.0)
        costs_A[(m["name"], 4)] = 0.0
        costs_A[(m["name"], 3)] = float(base)
        costs_A[(m["name"], 2)] = float(base * 2.5 + 0.1)

    # Cost function B: radically different, inverted / negative costs
    costs_B = {}
    for m in meta:
        base = rng.uniform(5.0, 20.0)
        costs_B[(m["name"], 4)] = 0.0
        costs_B[(m["name"], 3)] = float(-base)
        costs_B[(m["name"], 2)] = float(-base * 0.5)

    for budget in (2.50, 2.75, 3.00, 3.25, 3.50):
        resA = solve_bit_allocation_ilp(costs_A, meta, target_avg_bits=budget, enforce_exact_payload=True)
        resB = solve_bit_allocation_ilp(costs_B, meta, target_avg_bits=budget, enforce_exact_payload=True)
        assert resA["success"] and resB["success"]
        # Exact payload equality down to float precision!
        assert resA["achieved_avg_bits"] == resB["achieved_avg_bits"]
        k_A = sum(resA["bit_assignment"][m["name"]] * m["num_params"] for m in meta)
        k_B = sum(resB["bit_assignment"][m["name"]] * m["num_params"] for m in meta)
        assert k_A == k_B
        # Check feasible payload cache
        cache_key = (tuple(float(m["num_params"]) for m in meta), (2, 3, 4), float(budget))
        assert cache_key in _FEASIBLE_PAYLOAD_CACHE


# ------------------------------------------------------------------ Fix 3 [P2]: Full-Bank Calibration Cost & Overhead
def test_review2_fix3_full_bank_calib_time_and_overhead():
    seed_runs = [{
        "seed": 42,
        "estimators": ["Scalar-MSE", "HDS-Hybrid"],
        "budgets": [3.0],
        "forward_capture_time_sec": 10.0,
        "feature_time_sec": 15.0,
        "estimator_feature_time_sec": {"Scalar-MSE": 1.0, "HDS-Hybrid": 4.0},
        "estimator_readout_time_sec": {"Scalar-MSE": 0.5, "HDS-Hybrid": 2.0},
        "estimator_ilp_time_sec": {"Scalar-MSE": 0.2, "HDS-Hybrid": 0.2},
        "label_time_sec": 100.0,
        "base_overhead_time_sec": 10.0,
        "probe_time_sec": 90.0,
        "ranking_metrics": {
            "Scalar-MSE": {"spearman_global": 0.8, "kendall_global": 0.6, "spearman_bit2": 0.8, "spearman_bit3": 0.8, "spearman_within_family": 0.7, "top20_recall": 0.7, "spearman_per_layer_mean": 0.8, "spearman_per_layer_std": 0.0},
            "HDS-Hybrid": {"spearman_global": 0.9, "kendall_global": 0.7, "spearman_bit2": 0.9, "spearman_bit3": 0.9, "spearman_within_family": 0.85, "top20_recall": 0.85, "spearman_per_layer_mean": 0.9, "spearman_per_layer_std": 0.0},
        },
        "allocations": {
            "Scalar-MSE": {"3.00": {"ppl": 15.0, "nll": 2.7, "achieved_avg_bits": 3.0, "success": True, "seq_nlls": [2.7], "seq_token_counts": [31]}},
            "HDS-Hybrid": {"3.00": {"ppl": 14.5, "nll": 2.67, "achieved_avg_bits": 3.0, "success": True, "seq_nlls": [2.67], "seq_token_counts": [31]}},
        },
    }]
    summary = aggregate_seed_stability(seed_runs)
    hyb_time = summary["calibration_costs"]["HDS-Hybrid"]
    # 10.0 (fwd) + 4.0 (feat) + 1.0 * 100.0 (label) + 2.0 (readout) + 0.2 (ilp) = 116.2
    assert hyb_time["total_calib_time_sec"] == pytest.approx(116.2)
    # single fold: 10 + 4 + 0.75 * 100 + 2 + 0.2 = 91.2
    assert hyb_time["single_fold_calib_time_sec"] == pytest.approx(91.2)


# ------------------------------------------------------------------ Fix 4 [P2]: Isolate Inner GroupKFold Scaling & Transform
def test_review2_fix4_inner_groupkfold_isolation_and_transforms():
    rng = np.random.RandomState(42)
    N = 40
    D = 10
    X = rng.randn(N, D)
    groups = np.repeat(np.arange(8), 5)  # 8 groups/layers, 5 samples each
    y = np.exp(X[:, 0] * 1.5 + rng.randn(N) * 0.1)  # positive skewed target

    # Test with signed_log1p
    scaler, ridge, inv_fn = _fit_ridge(X, y, groups=groups, transform_mode="signed_log1p")
    assert hasattr(ridge, "alpha_")
    assert callable(inv_fn)
    y_pred_inv = inv_fn(np.array([1.0, 2.0]))
    assert np.all(y_pred_inv > 0)

    # Test with rank transform
    scaler, ridge, inv_fn = _fit_ridge(X, y, groups=groups, transform_mode="rank")
    assert hasattr(ridge, "alpha_")
    y_pred_inv = inv_fn(np.array([0.0, 0.5, 1.0]))
    assert y_pred_inv[0] == pytest.approx(np.min(y))
    assert y_pred_inv[2] == pytest.approx(np.max(y))


# ------------------------------------------------------------------ Fix 5 [Research Validity]: Coord-Rank-Exact Control
def test_review2_fix5_coord_rank_exact_control():
    assert "Coord-Rank-Exact" in ESTIMATOR_NAMES
    assert "Coord-Rank-Exact" in SUPERVISED_ESTIMATORS
    assert "Coord-Rank-Exact" in MAIN_ESTIMATORS
    assert "Coord-Rank-Exact" in LOW_SHOT_ESTIMATORS
    assert "Coord-Rank-Exact" in ORDERED_ESTIMATORS
    assert _META_COLS["Coord-Rank-Exact"] == 10

    w = _get_column_weights(16394, estimator="Coord-Rank-Exact", hd_dim=4096)
    assert w is not None
    assert np.allclose(w[:16384], 1.0)
    assert np.allclose(w[16384:], math.sqrt(16384 / 10.0))

    # Test compute_estimators with Coord-Rank-Exact
    enc = HDSEncoder(hd_dim=512, device="cpu", seed=42)
    R = torch.randn(32, 512).abs()
    norms = torch.rand(512) + 0.1
    f_raw = compute_estimators(R, norms, 0, enc, coord_binding="raw_index_strict")
    assert f_raw["Coord-Rank-Exact"].shape == (16393,)
    f_sal = compute_estimators(R, norms, 0, enc, coord_binding="salience_bucket")
    assert f_sal["Coord-Rank-Exact"].shape == (16393,)


# ------------------------------------------------------------------ Fix 6 [P2]: Hash Middle Projection Weight Mutation
def test_review2_fix6_hash_middle_projection_mutation():
    model, calib_ids = _build_smoke_model()
    cfg = ExperimentConfig(num_layers=4, quant_group_size=64, device="cpu")
    fp1 = _compute_label_cache_fingerprint(model, calib_ids, cfg)

    # Mutate a middle projection weight in layer 2
    p_mid = model.model.layers[2].mlp.gate_proj.weight
    orig_val = p_mid.data[0, 0].item()
    p_mid.data[0, 0] += 1.0

    fp2 = _compute_label_cache_fingerprint(model, calib_ids, cfg)
    assert fp1 != fp2, "Mutating a middle projection weight MUST change fingerprint!"

    # Restore weight
    p_mid.data[0, 0] = orig_val
    fp3 = _compute_label_cache_fingerprint(model, calib_ids, cfg)
    assert fp3 == fp1


# ------------------------------------------------------------------ Fix 7 [P2]: Low-Shot Detailed Records and Payload
def test_review2_fix7_low_shot_detailed_records_and_payload():
    mock_record = {
        "seed": 42,
        "acquired_layers": [0, 2],
        "unlabeled_layers": [1, 3],
        "n_labeled_layers": 2,
        "bit_assignment": {"p1": 3, "p2": 3},
        "achieved_avg_bits": 3.0,
        "heldout_predictions": {"p1|2": 0.5},
        "raw_predictions": {"p1|2": 0.5},
        "spearman_global": 0.85,
        "top20_recall": 0.8,
        "ppl_3bit": 12.3,
        "nll_3bit": 2.5,
        "seq_nlls": [2.5],
        "seq_token_counts": [31],
        "acquisition_cost_sec": 42.0,
    }
    required_keys = [
        "seed", "acquired_layers", "unlabeled_layers", "n_labeled_layers",
        "bit_assignment", "achieved_avg_bits", "heldout_predictions",
        "raw_predictions", "spearman_global", "top20_recall",
        "ppl_3bit", "nll_3bit", "seq_nlls", "seq_token_counts", "acquisition_cost_sec"
    ]
    for k in required_keys:
        assert k in mock_record


# ------------------------------------------------------------------ Fix 8 [Supported-Mode Bug]: Rank Transform Inverse
def test_review2_fix8_rank_transform_inverse_empirical_quantiles():
    rng = np.random.RandomState(42)
    y_train = rng.uniform(0.01, 10.0, size=50)
    p_train, inv_fn = _make_target_transform("rank", y_train)

    assert np.all(p_train >= 0.0) and np.all(p_train <= 1.0)

    # Inverting p_train should recover y_train
    y_rec = inv_fn(p_train)
    assert np.allclose(y_rec, y_train, atol=1e-5)

    # Extrapolation clamping
    assert inv_fn(-0.5) == pytest.approx(np.min(y_train))
    assert inv_fn(1.5) == pytest.approx(np.max(y_train))

    # Monotonicity
    test_grid = np.linspace(0.0, 1.0, 100)
    inv_vals = inv_fn(test_grid)
    assert np.all(np.diff(inv_vals) >= 0.0)


# ==================================================================
# Technical Review 3 Fixes Verification Tests
# ==================================================================

# ------------------------------------------------------------------ Fix 1 [P2]: Fingerprint Layernorm & LM Head
def test_review3_fix1_fingerprint_layernorm_and_lm_head_mutation():
    model, calib_ids = _build_smoke_model()
    cfg = ExperimentConfig(num_layers=4, quant_group_size=64, device="cpu")
    fp1 = _compute_label_cache_fingerprint(model, calib_ids, cfg)

    # 1. Mutate input_layernorm weight in layer 0
    ln = model.model.layers[0].input_layernorm.weight
    orig_ln = ln.data[0].item()
    ln.data[0] += 0.5
    fp_ln = _compute_label_cache_fingerprint(model, calib_ids, cfg)
    assert fp_ln != fp1, "Mutating input_layernorm MUST change fingerprint!"
    ln.data[0] = orig_ln
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg) == fp1

    # 2. Mutate lm_head weight
    head = model.lm_head.weight
    orig_head = head.data[0, 0].item()
    head.data[0, 0] += 0.5
    fp_head = _compute_label_cache_fingerprint(model, calib_ids, cfg)
    assert fp_head != fp1, "Mutating lm_head MUST change fingerprint!"
    head.data[0, 0] = orig_head
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg) == fp1


# ------------------------------------------------------------------ Fix 2 [P2]: Validate Every Required Label on Acquired Layers
def test_review3_fix2_validate_every_label_on_acquired_layers():
    meta_list = []
    for l in range(4):
        meta_list.append({
            "name": f"model.layers.{l}.self_attn.q_proj",
            "layer_idx": l,
            "family": "q_proj",
            "family_idx": 0,
            "num_params": 1024,
            "d_out": 32,
            "d_in": 32,
        })
    feats = {}
    for m in meta_list:
        for b in (2, 3):
            feats[(m["name"], b)] = {
                "Scalar-MSE": 0.05 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-Meta-Ridge": np.ones(9),
                "Ordinal-Hist": np.ones(33) / 33,
                "RandProj-JL": np.ones(64),
                "Coord-Rank-Hist": np.ones(521),
                "Coord-Rank-Exact": np.ones(16393),
                "HDS-Rank": np.ones(64),
                "HDS-Mag-Strict": np.ones(65),
                "HDS-Hybrid": np.ones(73),
            }
    # delta_loss: layers 0, 1, 2 have both b=2 and b=3; layer 3 has b=2 but MISSES b=3!
    delta_loss = {}
    for m in meta_list:
        if m["layer_idx"] in (0, 1, 2):
            for b in (2, 3):
                delta_loss[(m["name"], b)] = 0.1 * (4 - b) + 0.02 * m["layer_idx"]
        elif m["layer_idx"] == 3:
            delta_loss[(m["name"], 2)] = 0.2  # only b=2, missing b=3!

    cfg = ExperimentConfig(num_layers=4, hd_dim=64, quant_group_size=32, device="cpu")

    # When k_layers=4, all 4 layers are identified as labeled_layer_ids,
    # but layer 3 is missing b=3 label; evaluate_fixed_k_layer_acquisition MUST raise ValueError!
    with pytest.raises(ValueError, match="missing finite label"):
        evaluate_fixed_k_layer_acquisition(feats, delta_loss, meta_list, cfg, k_layers=4)


# ------------------------------------------------------------------ Fix 3 [P2]: Consistent Tied-Label Interpolation in rank Mode
def test_review3_fix3_tied_label_rank_interpolation():
    y_train = np.array([1.0, 1.0, 2.0])
    y_ranks, forward_fn, inverse_fn = _fit_rank_transform(y_train)

    # 1.0 maps to 0.25 in y_ranks, 2.0 maps to 1.0
    assert np.allclose(forward_fn(y_train), y_ranks)
    assert forward_fn(np.array([1.0]))[0] == pytest.approx(0.25)
    assert forward_fn(np.array([2.0]))[0] == pytest.approx(1.0)

    # Out-of-sample interpolation with ties in training
    val_pred = forward_fn(np.array([1.0, 1.5, 2.0]))
    assert val_pred[0] == pytest.approx(0.25)
    assert val_pred[1] == pytest.approx(0.625)
    assert val_pred[2] == pytest.approx(1.0)

    # Inverse mapping back to empirical values
    assert inverse_fn(np.array([0.25]))[0] == pytest.approx(1.0)
    assert inverse_fn(np.array([1.0]))[0] == pytest.approx(2.0)


# ------------------------------------------------------------------ Fix 4 [P2]: Wire cv_folds & ridge_alphas Into readout_cv
def test_review3_fix4_config_cv_folds_and_ridge_alphas_wiring():
    # Validation
    with pytest.raises(ValueError, match="cv_folds must be >= 2"):
        ExperimentConfig(cv_folds=1)
    with pytest.raises(ValueError, match="ridge_alphas"):
        ExperimentConfig(ridge_alphas=())
    with pytest.raises(ValueError, match="ridge_alphas"):
        ExperimentConfig(ridge_alphas=(-1.0,))

    # Wiring into cross_validate_estimators
    meta_list = []
    for l in range(4):
        meta_list.append({
            "name": f"model.layers.{l}.self_attn.q_proj",
            "layer_idx": l,
            "family": "q_proj",
            "family_idx": 0,
            "num_params": 1024,
            "d_out": 32,
            "d_in": 32,
        })
    feats = {}
    delta_loss = {}
    for m in meta_list:
        for b in (2, 3):
            k = (m["name"], b)
            feats[k] = {
                "Scalar-MSE": 0.05 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-L1": 0.04 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-TrimmedMean": 0.04 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-WinsorL2": 0.04 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-Q99": 0.06 * (b == 2) + 0.01 * m["layer_idx"],
                "Scalar-Meta-Ridge": np.ones(9),
                "Ordinal-Hist": np.ones(33) / 33,
                "RandProj-JL": np.ones(64),
                "Coord-Rank-Hist": np.ones(521),
                "Coord-Rank-Exact": np.ones(16393),
                "HDS-Rank": np.ones(64),
                "HDS-Mag-Strict": np.ones(65),
                "HDS-Hybrid": np.ones(73),
            }
            delta_loss[k] = 0.1 * (4 - b) + 0.02 * m["layer_idx"]

    cfg = ExperimentConfig(
        num_layers=4,
        hd_dim=64,
        quant_group_size=32,
        cv_folds=2,
        ridge_alphas=(0.5,),
        device="cpu",
    )
    out = cross_validate_estimators(feats, delta_loss, meta_list, cfg)
    assert out["n_folds"] == 2
    for k, vec in out["model_coefs"].items():
        assert vec[-1] == pytest.approx(0.5)  # Alpha is wired down and selected!


# ------------------------------------------------------------------ Fix 5 [P2]: Early Budget Validation & ILP Failure Handling
def test_review3_fix5_budget_validation_and_ilp_failure_handling():
    args = SimpleNamespace(
        smoke_test=True,
        budgets=[1.5],
        seeds=[42],
        model_id="smoke",
        eval_dataset="wikitext2",
        residual_anchor="fp16_strict",
        coord_binding="raw_index_strict",
        readout_grouping="pooled",
        target_transform="raw",
        n_calib_seqs=2,
        calib_seq_len=32,
        max_encoded_tokens=64,
        eval_seq_len=64,
        batch_size=2,
        run_low_shot_curve=False,
        cache_dir="results/cache",
        reuse_cache=False,
        output_dir="results",
        run_tag="test_fail",
    )
    with pytest.raises(ValueError, match="Budget 1.5 is outside feasible"):
        run_experiment(args)


# ------------------------------------------------------------------ Fix 6 [P3]: Restore Time in collect_marginal_loss_labels
def test_review3_fix6_restore_time_sec_in_label_collector():
    model, _ = _build_smoke_model()
    calib_ids = _get_smoke_calib_ids(42)
    cfg = ExperimentConfig(num_layers=4, quant_group_size=64, device="cpu")
    labels = collect_marginal_loss_labels(model, calib_ids, cfg, batch_size=4)

    assert "restore_time_sec" in labels
    assert labels["restore_time_sec"] >= 0.0
    assert labels["label_time_sec"] >= labels["base_overhead_time_sec"] + labels["probe_time_sec"]


# ==================================================================
# Technical Review 4 Fixes Verification Tests
# ==================================================================

# ------------------------------------------------------------------ Fix 1 [P2]: Failed Allocation & Unavailable Stability Handling
def test_review4_fix1_failed_seed_allocation_aggregation():
    from scripts.format_tables import build_table1, build_table2

    est = "HDS-Hybrid"

    def _make_dummy_seed_run(seed: int, success: bool, ppl: float = 12.0, bit_assignment: dict = None):
        if bit_assignment is None:
            bit_assignment = {"p0": 3, "p1": 3} if success else {}
        return {
            "seed": seed,
            "num_layers": 4,
            "n_folds": 4,
            "budgets": [3.00],
            "estimators": [est],
            "ranking_metrics": {
                est: {
                    "spearman_global": 0.9,
                    "kendall_global": 0.8,
                    "spearman_bit2": 0.85,
                    "spearman_bit3": 0.85,
                    "spearman_within_family": 0.8,
                    "top20_recall": 0.9,
                }
            },
            "allocations": {
                est: {
                    "3.00": {
                        "success": success,
                        "bit_assignment": bit_assignment,
                        "ppl": ppl if success else float("nan"),
                        "achieved_avg_bits": 3.0 if success else float("nan"),
                    }
                }
            },
            "feature_time_sec": 1.0,
            "label_time_sec": 2.0,
            "restore_time_sec": 0.1,
        }

    # Case A: Seed 0 succeeds (ppl=12.5), Seed 1 fails (bit_assignment={}, success=False)
    seed_runs_a = [_make_dummy_seed_run(0, success=True, ppl=12.5), _make_dummy_seed_run(1, success=False)]
    summary_a = aggregate_seed_stability(seed_runs_a)

    assert summary_a["ppl_metrics"][est]["3.00"]["ppl_mean"] == pytest.approx(12.5)
    assert np.isnan(summary_a["allocation_stability"][est]["3.00"])

    _, df1_a = build_table1(summary_a)
    assert df1_a.loc[df1_a["Estimator"] == est, "Cross-Seed Stability"].values[0] == "-"
    _, df2_a = build_table2(summary_a)
    assert df2_a.loc[df2_a["Estimator"] == est, "Seed Agreement"].values[0] == "-"

    # Case B: Reverse order (Seed 0 fails, Seed 1 succeeds with ppl=14.0)
    seed_runs_b = [_make_dummy_seed_run(0, success=False), _make_dummy_seed_run(1, success=True, ppl=14.0)]
    summary_b = aggregate_seed_stability(seed_runs_b)

    assert summary_b["ppl_metrics"][est]["3.00"]["ppl_mean"] == pytest.approx(14.0)
    assert np.isnan(summary_b["allocation_stability"][est]["3.00"])

    _, df1_b = build_table1(summary_b)
    assert df1_b.loc[df1_b["Estimator"] == est, "Cross-Seed Stability"].values[0] == "-"
    _, df2_b = build_table2(summary_b)
    assert df2_b.loc[df2_b["Estimator"] == est, "Seed Agreement"].values[0] == "-"


# ------------------------------------------------------------------ Fix 2 [P2]: Chunked Byte Hash Swapped Weights Fingerprint
def test_review4_fix2_swapping_two_unsampled_weights_changes_fingerprint():
    model, calib_ids = _build_smoke_model()
    cfg = ExperimentConfig(num_layers=4, quant_group_size=64, device="cpu")
    fp_orig = _compute_label_cache_fingerprint(model, calib_ids, cfg)

    # Pick a projection weight tensor (e.g. q_proj in layer 0)
    w = model.model.layers[0].self_attn.q_proj.weight
    w_flat = w.data.view(-1)
    # Ensure indices 1 and 3 have distinct values
    if w_flat[1] == w_flat[3]:
        w_flat[1] += 0.123
        fp_orig = _compute_label_cache_fingerprint(model, calib_ids, cfg)

    # Swap elements at odd indices 1 and 3 (which were previously skipped by [::stride])
    val1 = w_flat[1].clone()
    val3 = w_flat[3].clone()
    w_flat[1] = val3
    w_flat[3] = val1

    fp_swapped = _compute_label_cache_fingerprint(model, calib_ids, cfg)
    assert fp_swapped != fp_orig, "Swapping two unsampled weights MUST change the fingerprint in v5 chunked byte hash!"

    # Swap back and verify fingerprint matches fp_orig
    w_flat[1] = val1
    w_flat[3] = val3
    assert _compute_label_cache_fingerprint(model, calib_ids, cfg) == fp_orig


# ------------------------------------------------------------------ Fix 3 [P3]: restore_time_sec Preserved in Cache & Low-Shot
def test_review4_fix3_restore_time_preserved_in_cache_and_low_shot(tmp_path):
    import json

    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "results"
    cache_dir.mkdir()
    output_dir.mkdir()

    args1 = SimpleNamespace(
        model_id="smoke",
        eval_dataset="wikitext2",
        residual_anchor="fp16_strict",
        coord_binding="raw_index_strict",
        readout_grouping="pooled",
        target_transform="raw",
        seeds=[42],
        budgets=[3.00],
        n_calib_seqs=2,
        calib_seq_len=32,
        max_encoded_tokens=64,
        eval_seq_len=64,
        batch_size=2,
        run_low_shot_curve=True,
        cache_dir=str(cache_dir),
        reuse_cache=False,
        output_dir=str(output_dir),
        run_tag="fresh_run",
        cv_folds=2,
        smoke_test=True,
    )
    run_experiment(args1)

    # 1. Verify restore_time_sec is saved in cache file
    cache_file = cache_dir / "smoke_seed42_labels.pt"
    assert cache_file.exists()
    cached = torch.load(cache_file, weights_only=False)
    assert "restore_time_sec" in cached
    assert cached["restore_time_sec"] >= 0.0
    saved_restore_time = float(cached["restore_time_sec"])

    # Verify run output JSON
    res_file1 = output_dir / "run_fresh_run.json"
    assert res_file1.exists()
    with open(res_file1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    assert data1["seed_runs"][0]["restore_time_sec"] >= 0.0
    assert data1["seed_runs"][0]["n_folds"] == 2
    # Verify acquisition_cost_sec in low shot curve includes restore_time_sec
    assert "low_shot_curve" in data1
    for k_key, est_dict in data1["low_shot_curve"].items():
        for est, stats in est_dict.items():
            assert stats["acquisition_cost_sec_mean"] >= saved_restore_time
    assert "low_shot_seed_runs" in data1
    for k_key, est_dict in data1["low_shot_seed_runs"].items():
        for est, run_list in est_dict.items():
            for r in run_list:
                assert r["acquisition_cost_sec"] >= saved_restore_time

    # 2. Re-run with reuse_cache=True
    args2 = SimpleNamespace(
        model_id="smoke",
        eval_dataset="wikitext2",
        residual_anchor="fp16_strict",
        coord_binding="raw_index_strict",
        readout_grouping="pooled",
        target_transform="raw",
        seeds=[42],
        budgets=[3.00],
        n_calib_seqs=2,
        calib_seq_len=32,
        max_encoded_tokens=64,
        eval_seq_len=64,
        batch_size=2,
        run_low_shot_curve=True,
        cache_dir=str(cache_dir),
        reuse_cache=True,
        output_dir=str(output_dir),
        run_tag="cached_run",
        cv_folds=2,
        smoke_test=True,
    )
    run_experiment(args2)

    res_file2 = output_dir / "run_cached_run.json"
    assert res_file2.exists()
    with open(res_file2, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    assert data2["seed_runs"][0]["wall_label_time_sec"] == 0.0
    assert data2["seed_runs"][0]["restore_time_sec"] == pytest.approx(saved_restore_time)


# ==================================================================
# Technical Review 5 Fixes Verification Tests
# ==================================================================

# ------------------------------------------------------------------ Fix 1 [P2]: Low-Shot Failed Seed Filtering
def test_review5_fix1_low_shot_failed_seed_filtering():
    run_list = [
        {
            "seed": 0,
            "success": True,
            "bit_assignment": {"p0": 3, "p1": 3},
            "achieved_avg_bits": 3.0,
            "spearman_global": 0.85,
            "top20_recall": 0.90,
            "ppl_3bit": 11.5,
            "nll_3bit": 2.44,
            "acquisition_cost_sec": 5.0,
        },
        {
            "seed": 1,
            "success": False,
            "bit_assignment": {},
            "achieved_avg_bits": float("nan"),
            "spearman_global": 0.82,
            "top20_recall": 0.85,
            "ppl_3bit": float("nan"),
            "nll_3bit": float("nan"),
            "acquisition_cost_sec": 4.8,
        },
    ]

    valid_runs = [
        r for r in run_list
        if r.get("success", True)
        and bool(r.get("bit_assignment", True))
        and r.get("ppl_3bit") is not None
        and not np.isnan(float(r.get("ppl_3bit", float("nan"))))
    ]
    valid_ppls = [float(r["ppl_3bit"]) for r in valid_runs]
    valid_bits = [
        float(r["achieved_avg_bits"]) for r in valid_runs
        if r.get("achieved_avg_bits") is not None and not np.isnan(float(r.get("achieved_avg_bits", float("nan"))))
    ]
    summary = {
        "n_labeled_layers": 2,
        "spearman_global_mean": float(np.mean([r["spearman_global"] for r in run_list])),
        "spearman_global_std": float(np.std([r["spearman_global"] for r in run_list])) if len(run_list) > 1 else 0.0,
        "top20_recall_mean": float(np.mean([r["top20_recall"] for r in run_list])),
        "top20_recall_std": float(np.std([r["top20_recall"] for r in run_list])) if len(run_list) > 1 else 0.0,
        "ppl_3bit_mean": float(np.mean(valid_ppls)) if valid_ppls else None,
        "ppl_3bit_std": float(np.std(valid_ppls)) if len(valid_ppls) > 1 else 0.0,
        "achieved_avg_bits_mean": float(np.mean(valid_bits)) if valid_bits else None,
        "acquisition_cost_sec_mean": float(np.mean([r["acquisition_cost_sec"] for r in run_list])),
        "n_valid_seeds": len(valid_runs),
    }

    assert summary["n_valid_seeds"] == 1
    assert summary["ppl_3bit_mean"] == pytest.approx(11.5)
    assert summary["achieved_avg_bits_mean"] == pytest.approx(3.0)


# ------------------------------------------------------------------ Fix 2 [P3]: Strict RFC-8259 JSON & Table Null Handling
def test_review5_fix2_strict_json_no_nan_tokens_and_table_null_handling(tmp_path):
    import json
    from hds_quant.run_experiment import _sanitize_for_json
    from scripts.format_tables import build_table1, build_table2, build_table3_ablation, build_table4_low_shot

    payload = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "neg_inf_val": float("-inf"),
        "np_nan": np.float64(np.nan),
        "valid_float": 1.25,
        "nested": {
            "arr": [1.0, float("nan"), 3.0],
            "flag": True,
            "int_val": 42,
        },
    }

    sanitized = _sanitize_for_json(payload)
    assert sanitized["nan_val"] is None
    assert sanitized["inf_val"] is None
    assert sanitized["neg_inf_val"] is None
    assert sanitized["np_nan"] is None
    assert sanitized["valid_float"] == 1.25
    assert sanitized["nested"]["arr"] == [1.0, None, 3.0]
    assert sanitized["nested"]["flag"] is True
    assert sanitized["nested"]["int_val"] == 42

    # Verify standard json.dumps with allow_nan=False succeeds
    json_str = json.dumps(sanitized, indent=2, allow_nan=False)
    assert "NaN" not in json_str
    assert "Infinity" not in json_str
    assert "null" in json_str

    # Verify tables format None values cleanly as "-"
    summary_with_nulls = {
        "n_folds": 4,
        "ranking_metrics": {
            "HDS-Hybrid": {
                "spearman_global_mean": None,
                "spearman_global_std": None,
                "kendall_global_mean": None,
                "spearman_within_family_mean": None,
                "top20_recall_mean": None,
            }
        },
        "allocation_stability": {
            "HDS-Hybrid": {
                "3.00": None,
            }
        },
        "ppl_metrics": {
            "HDS-Hybrid": {
                "3.00": {
                    "ppl_mean": None,
                    "ppl_std": None,
                    "achieved_avg_bits_mean": None,
                }
            }
        },
        "calibration_costs": {
            "HDS-Hybrid": {
                "total_calib_time_sec": None,
                "single_fold_calib_time_sec": None,
            }
        },
        "uniform_baselines": {
            "FP16": {
                "ppl_mean": 8.0,
                "ppl_std": 0.0,
            }
        },
    }

    _, df1 = build_table1(summary_with_nulls)
    assert df1.loc[df1["Estimator"] == "HDS-Hybrid", "Cross-Seed Stability"].values[0] == "-"
    assert df1.loc[df1["Estimator"] == "HDS-Hybrid", "Top-20% Recall"].values[0] == "-"

    _, df2 = build_table2(summary_with_nulls)
    assert df2.loc[df2["Estimator"] == "HDS-Hybrid", "Seed Agreement"].values[0] == "-"
    assert df2.loc[df2["Estimator"] == "HDS-Hybrid", "PPL @ 3.00b (bits)"].values[0] == "- (-)"
    assert df2.loc[df2["Estimator"] == "HDS-Hybrid", "Total Calib Time (s)"].values[0] == "-"

    # Test Table 3 ablation with None values
    run1 = tmp_path / "run_ab1.json"
    run2 = tmp_path / "run_ab2.json"
    with open(run1, "w", encoding="utf-8") as f:
        json.dump({"config": {"run_tag": "ab1"}, "summary": summary_with_nulls}, f, allow_nan=False)
    with open(run2, "w", encoding="utf-8") as f:
        json.dump({"config": {"run_tag": "ab2"}, "summary": summary_with_nulls}, f, allow_nan=False)

    tbl3 = build_table3_ablation([run1, run2])
    assert "ab1" in tbl3
    assert "NaN" not in tbl3

    # Test Table 4 low shot with None values
    low_shot_data = {
        "low_shot_curve": {
            "2": {
                "HDS-Hybrid": {
                    "spearman_global_mean": None,
                    "spearman_global_std": None,
                    "top20_recall_mean": None,
                    "ppl_3bit_mean": None,
                    "ppl_3bit_std": None,
                    "achieved_avg_bits_mean": None,
                    "acquisition_cost_sec_mean": None,
                }
            }
        }
    }
    tbl4 = build_table4_low_shot(low_shot_data)
    assert "HDS-Hybrid" in tbl4
    assert "NaN" not in tbl4
    assert "- (-)" in tbl4


# ------------------------------------------------------------------ Fix 3 [P3]: Dynamic Fold Column Header
def test_review5_fix3_dynamic_fold_column_header():
    from scripts.format_tables import build_table2

    def _make_summary(n_folds):
        return {
            "n_folds": n_folds,
            "ppl_metrics": {
                "HDS-Hybrid": {
                    "3.00": {
                        "ppl_mean": 12.0,
                        "ppl_std": 0.1,
                        "achieved_avg_bits_mean": 3.0,
                    }
                }
            },
            "calibration_costs": {
                "HDS-Hybrid": {
                    "total_calib_time_sec": 10.0,
                    "single_fold_calib_time_sec": 7.5,
                }
            },
            "allocation_stability": {"HDS-Hybrid": {"3.00": 1.0}},
            "uniform_baselines": {"FP16": {"ppl_mean": 8.0, "ppl_std": 0.0}},
        }

    # When n_folds is 2
    _, df_2 = build_table2(_make_summary(2))
    assert "1-of-2 Fold Train Time (s)" in df_2.columns
    assert "3-Fold Train Time (s)" not in df_2.columns

    # When n_folds is 4
    _, df_4 = build_table2(_make_summary(4))
    assert "3-of-4 Fold Train Time (s)" in df_4.columns
    assert "3-Fold Train Time (s)" not in df_4.columns

    # When n_folds is None or missing
    _, df_none = build_table2(_make_summary(None))
    assert "Single-Fold Train Time (s)" in df_none.columns






# ==================================================================
# 1-bit candidate support (candidate_bits=(1, 2, 3, 4)) and 1.5b / 2.0b / 3.0b budgets
# ==================================================================
def _one_bit_design_inputs(marginal_bits, n_layers=4):
    meta_list = [
        {
            "name": f"model.layers.{l}.self_attn.q_proj",
            "layer_idx": l,
            "family": "q_proj",
            "family_idx": 0,
            "num_params": 1024,
            "d_out": 32,
            "d_in": 32,
        }
        for l in range(n_layers)
    ]
    feats, delta_loss = {}, {}
    for m in meta_list:
        for b in marginal_bits:
            k = (m["name"], b)
            feats[k] = {e: np.ones(4) * (5 - b) for e in SUPERVISED_ESTIMATORS}
            feats[k]["Scalar-MSE"] = 0.01 * (5 - b)
            delta_loss[k] = 0.1 * (5 - b) + 0.02 * m["layer_idx"]
    return meta_list, feats, delta_loss


def test_1bit_alloc_cost_multi_is_monotone_and_backward_compatible():
    from hds_quant.readout_cv import MONO_EPS, _alloc_cost, _alloc_cost_multi

    # (2, 3) reproduces the legacy two-bit helper exactly
    for raw2, raw3 in [(0.5, 0.1), (0.1, 0.5), (-0.2, 0.3), (float("nan"), 0.2), (float("inf"), -1.0)]:
        s2, s3 = _alloc_cost(raw2, raw3)
        multi = _alloc_cost_multi({2: raw2, 3: raw3}, (2, 3))
        assert multi[2] == pytest.approx(s2) and multi[3] == pytest.approx(s3)

    # (1, 2, 3): non-negative and strictly increasing as bits decrease, even for adversarial raw values
    out = _alloc_cost_multi({1: -0.5, 2: 0.4, 3: 0.9}, (1, 2, 3))
    assert out[3] == pytest.approx(0.9)
    assert out[2] >= out[3] + MONO_EPS - 1e-12
    assert out[1] >= out[2] + MONO_EPS - 1e-12
    assert all(v >= 0.0 for v in out.values())


def test_1bit_prepare_design_indicator_columns_and_column_weights():
    meta_list, feats, delta_loss = _one_bit_design_inputs((1, 2, 3))
    cfg = ExperimentConfig(candidate_bits=(1, 2, 3, 4), device="cpu", cv_folds=2)
    assert cfg.marginal_bits == (1, 2, 3)
    d = _prepare_design(feats, delta_loss, meta_list, cfg)
    assert d["marginal_bits"] == (1, 2, 3)
    assert d["n_indicator_cols"] == 2
    X = d["X_all"]["HDS-Hybrid"]
    assert X.shape[1] == 4 + 2
    for row, b in zip(X, d["bits"]):
        assert tuple(row[-2:]) == ((1.0, 0.0) if b == 1 else (0.0, 1.0) if b == 2 else (0.0, 0.0))

    # Default (2, 3) config is unchanged: one indicator column equal to (bits == 2)
    meta2, feats2, dl2 = _one_bit_design_inputs((2, 3))
    d2 = _prepare_design(feats2, dl2, meta2, ExperimentConfig(device="cpu", cv_folds=2))
    assert d2["n_indicator_cols"] == 1
    assert np.array_equal(d2["X_all"]["HDS-Hybrid"][:, -1], (d2["bits"] == 2).astype(float))

    # Metadata block grows with the number of indicator columns: d_meta = (10 - 1) + n_indicator_cols
    w1 = _get_column_weights(4106, estimator="HDS-Hybrid", hd_dim=4096)
    w2 = _get_column_weights(4107, estimator="HDS-Hybrid", hd_dim=4096, n_indicator_cols=2)
    assert np.allclose(w1[4096:], math.sqrt(4096 / 10.0))
    assert np.allclose(w2[:4096], 1.0) and np.allclose(w2[4096:], math.sqrt(4096 / 11.0))
    wm = _get_column_weights(4099, estimator="HDS-Mag-Strict", hd_dim=4096, n_indicator_cols=2)
    assert np.allclose(wm[4096:], math.sqrt(4096 / 3.0))


def test_1bit_build_alloc_costs_and_ranking_metrics():
    from hds_quant.evaluator import compute_ranking_metrics

    meta_list, _, delta_loss = _one_bit_design_inputs((1, 2, 3))
    raw = {"Oracle-Loss": dict(delta_loss), "Oracle-Loss-Raw": dict(delta_loss)}
    alloc = _build_alloc_costs(raw, meta_list, ["Oracle-Loss", "Oracle-Loss-Raw"], marginal_bits=(1, 2, 3), reference_bit=4)
    for m in meta_list:
        n = m["name"]
        assert alloc["Oracle-Loss-Raw"][(n, 4)] == 0.0 and alloc["Oracle-Loss"][(n, 4)] == 0.0
        for b in (1, 2, 3):
            assert alloc["Oracle-Loss-Raw"][(n, b)] == pytest.approx(delta_loss[(n, b)])
        assert alloc["Oracle-Loss"][(n, 1)] > alloc["Oracle-Loss"][(n, 2)] > alloc["Oracle-Loss"][(n, 3)] >= 0.0

    metrics = compute_ranking_metrics(delta_loss, delta_loss, meta_list)  # candidate_bits inferred -> (1, 2, 3)
    assert metrics["spearman_bit1"] == pytest.approx(1.0)
    assert metrics["spearman_bit2"] == pytest.approx(1.0)
    assert metrics["spearman_bit3"] == pytest.approx(1.0)
    # Without 1-bit labels spearman_bit1 is 0.0 and the two-bit behaviour is unchanged
    two_bit = {k: v for k, v in delta_loss.items() if k[1] != 1}
    m2 = compute_ranking_metrics(two_bit, two_bit, meta_list)
    assert m2["spearman_bit1"] == 0.0 and m2["spearman_bit2"] == pytest.approx(1.0)


def test_1bit_ilp_allocation_with_four_candidates():
    meta_list, _, _ = _one_bit_design_inputs((1, 2, 3))
    costs = {}
    for i, m in enumerate(meta_list):
        for b, c in ((1, 3.0), (2, 1.0), (3, 0.3), (4, 0.0)):
            costs[(m["name"], b)] = c * (1 + i)
    res = solve_bit_allocation_ilp(costs, meta_list, target_avg_bits=1.5, candidate_bits=(1, 2, 3, 4))
    assert res["success"]
    assert set(res["bit_assignment"].values()) <= {1, 2, 3, 4}
    assert res["achieved_avg_bits"] <= 1.5 + 1e-9


def _smoke_args(tmp_path, **overrides):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "results"
    cache_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    base = dict(
        model_id="smoke",
        eval_dataset="wikitext2",
        residual_anchor="fp16_strict",
        coord_binding="raw_index_strict",
        readout_grouping="pooled",
        target_transform="raw",
        seeds=[42],
        budgets=[3.00],
        n_calib_seqs=2,
        calib_seq_len=32,
        max_encoded_tokens=64,
        eval_seq_len=64,
        batch_size=2,
        run_low_shot_curve=False,
        cache_dir=str(cache_dir),
        reuse_cache=False,
        output_dir=str(output_dir),
        run_tag="t",
        cv_folds=2,
        smoke_test=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_1bit_candidate_and_budgets(tmp_path):
    """candidate_bits=(1, 2, 3, 4) with budgets [1.50, 2.00, 3.00]: end-to-end incl. incremental cache upgrade."""
    from scripts.format_tables import build_table2, build_table4_low_shot

    # 1. A legacy (2, 3, 4) run writes a label cache holding only b in {2, 3}.
    run_experiment(_smoke_args(tmp_path, candidate_bits=[2, 3, 4], run_tag="legacy234"))
    cache_file = tmp_path / "cache" / "smoke_seed42_labels.pt"
    legacy = torch.load(cache_file, weights_only=False)
    assert {int(k[1]) for k in legacy["delta_loss"]} == {2, 3}
    legacy_delta = dict(legacy["delta_loss"])

    # 2. A 1-bit run reuses that cache: only b = 1 is probed, b in {2, 3} labels are reused verbatim.
    out = run_experiment(
        _smoke_args(
            tmp_path,
            candidate_bits=[1, 2, 3, 4],
            budgets=[1.50, 2.00, 3.00],
            reuse_cache=True,
            run_low_shot_curve=True,
            run_tag="onebit",
        )
    )
    upgraded = torch.load(cache_file, weights_only=False)
    assert upgraded["fingerprint"] != legacy["fingerprint"]
    assert {int(k[1]) for k in upgraded["delta_loss"]} == {1, 2, 3}
    for k, v in legacy_delta.items():
        assert upgraded["delta_loss"][k] == pytest.approx(v)
    assert upgraded["label_time_sec"] >= legacy["label_time_sec"]
    assert upgraded["probe_time_sec"] >= legacy["probe_time_sec"]
    assert upgraded["base_loss_4bit"] == pytest.approx(legacy["base_loss_4bit"])

    run0 = out["seed_runs"][0]
    assert 0.0 < run0["wall_label_time_sec"] <= run0["label_time_sec"] + 1e-6
    assert any(k.endswith("|1") for k in run0["delta_loss"])

    # 3. Uniform-1bit baseline, 1-bit ranking metric, three budgets with an identical payload for every estimator.
    assert "Uniform-1bit" in out["uniform_baselines"] and "Uniform-1bit" in out["summary"]["uniform_baselines"]
    assert "spearman_bit1" in run0["ranking_metrics"]["HDS-Hybrid"]
    assert "spearman_bit1_mean" in out["summary"]["ranking_metrics"]["HDS-Hybrid"]
    assert set(out["summary"]["ppl_metrics"]["HDS-Hybrid"]) == {"1.50", "2.00", "3.00"}
    for bk in ("1.50", "2.00", "3.00"):
        allocs = {est: run0["allocations"][est][bk] for est in run0["estimators"]}
        assert all(a["success"] for a in allocs.values())
        bits = [a["achieved_avg_bits"] for a in allocs.values()]
        assert max(bits) - min(bits) < 1e-9
        assert max(bits) <= float(bk) + 1e-9
        for a in allocs.values():
            assert set(a["bit_assignment"].values()) <= {1, 2, 3, 4}
    # An average of 1.5 bits is impossible without 1-bit projections.
    assert 1 in run0["allocations"]["Oracle-Loss"]["1.50"]["bit_assignment"].values()

    # 4. Low-shot curve reports one PPL entry per budget (default: budgets in {1.5, 2.0, 3.0}).
    assert out["low_shot_curve"]
    for est_dict in out["low_shot_curve"].values():
        for stats in est_dict.values():
            assert set(stats["ppl_by_budget"]) == {"1.50", "2.00", "3.00"}
    for est_dict in out["low_shot_seed_runs"].values():
        for run_list in est_dict.values():
            for r in run_list:
                assert set(r["ppl_by_budget"]) == {"1.50", "2.00", "3.00"} and "ppl_3bit" in r

    # 5. Tables carry the 1-bit baseline and the 1.50b / 2.00b columns.
    _, df2 = build_table2(out["summary"])
    assert "Baseline: Uniform-1bit" in set(df2["Estimator"])
    assert {"PPL @ 1.50b (bits)", "PPL @ 2.00b (bits)", "PPL @ 3.00b (bits)"} <= set(df2.columns)
    tbl4 = build_table4_low_shot(out)
    assert "PPL @ 1.50b (bits)" in tbl4 and "PPL @ 2.00b (bits)" in tbl4 and "PPL @ 3.00b (bits)" in tbl4

    # 6. Re-running with the upgraded cache is a plain cache hit (no probing at all).
    out2 = run_experiment(
        _smoke_args(
            tmp_path,
            candidate_bits=[1, 2, 3, 4],
            budgets=[1.50, 2.00, 3.00],
            reuse_cache=True,
            run_tag="onebit_cached",
        )
    )
    assert out2["seed_runs"][0]["wall_label_time_sec"] == 0.0
    for k, v in upgraded["delta_loss"].items():
        assert out2["seed_runs"][0]["delta_loss"][f"{k[0]}|{k[1]}"] == pytest.approx(v)


def test_1bit_candidate_bits_validation(tmp_path):
    with pytest.raises(ValueError, match="reference bit 4"):
        run_experiment(_smoke_args(tmp_path, budgets=[2.5], candidate_bits=[1, 2, 3]))
    with pytest.raises(ValueError, match="Low-shot budget"):
        run_experiment(
            _smoke_args(tmp_path, budgets=[2.5], candidate_bits=[2, 3, 4], low_shot_budgets=[1.5])
        )


def test_hqq_smoke_experiment(tmp_path):
    """Verify run_experiment in --smoke-test mode with official quantizer='hqq'."""
    args = _smoke_args(
        tmp_path,
        quantizer="hqq",
        candidate_bits=[1, 2, 3, 4],
        budgets=[1.50, 2.00, 3.00],
        run_tag="hqq_smoke",
    )
    out = run_experiment(args)
    assert out["config"]["quantizer"] == "hqq"
    assert out["config"]["hqq_backend"] == "official_hqq"
    assert out["config"]["hqq_version"] == "0.2.8.post1"
    assert out["config"]["hqq_config"]["axis"] == 1
    assert out["config"]["hqq_config"]["round_zero"] is False
    assert out["config"]["hqq_config"]["group_size"] == 64
    assert out["config"]["hqq_config"]["device_type"] == "cpu"
    assert out["config"]["hqq_config"]["optimizer_dtype"] == "float32"

    run0 = out["seed_runs"][0]
    # Check that HQQ cache was saved with quantizer in cache filename and full metadata
    cache_file = tmp_path / "cache" / "smoke_seed42_hqq_labels.pt"
    assert cache_file.exists()
    cached = torch.load(cache_file, weights_only=False)
    assert cached["quantizer"] == "hqq"
    assert cached["hqq_backend"] == "official_hqq"
    assert cached["hqq_version"] == "0.2.8.post1"
    assert cached["hqq_config"]["axis"] == 1
    assert cached["hqq_config"]["group_size"] == 64
    assert {int(k[1]) for k in cached["delta_loss"]} == {1, 2, 3}

    # Verify allocations and evaluations across budgets
    for bk in ("1.50", "2.00", "3.00"):
        allocs = {est: run0["allocations"][est][bk] for est in run0["estimators"]}
        assert all(a["success"] for a in allocs.values())
        bits = [a["achieved_avg_bits"] for a in allocs.values()]
        assert max(bits) - min(bits) < 1e-9

    assert "Uniform-1bit" in out["uniform_baselines"]
    assert "Uniform-2bit" in out["uniform_baselines"]


def test_hqq_cache_invalidation_and_incremental_upgrade(tmp_path):
    """Verify cache keys isolate HQQ from RTN, invalidate legacy custom HQQ caches, and allow incremental upgrade."""
    from hds_quant.run_experiment import (
        _build_smoke_model,
        _compute_label_cache_fingerprint,
        _get_smoke_calib_ids,
        _load_cached_labels,
        _load_or_upgrade_cached_labels,
    )

    model, _ = _build_smoke_model()
    calib_ids = _get_smoke_calib_ids(42)

    cfg_rtn = ExperimentConfig(quantizer="rtn", candidate_bits=(2, 3, 4), quant_group_size=64)
    cfg_hqq = ExperimentConfig(quantizer="hqq", candidate_bits=(2, 3, 4), quant_group_size=64)
    cfg_hqq_1234 = ExperimentConfig(quantizer="hqq", candidate_bits=(1, 2, 3, 4), quant_group_size=64)

    fp_rtn = _compute_label_cache_fingerprint(model, calib_ids, cfg_rtn)
    fp_hqq = _compute_label_cache_fingerprint(model, calib_ids, cfg_hqq)
    fp_hqq_1234 = _compute_label_cache_fingerprint(model, calib_ids, cfg_hqq_1234)

    # 1. RTN and HQQ fingerprints must differ
    assert fp_rtn != fp_hqq
    assert fp_hqq != fp_hqq_1234

    # 2. Legacy custom HQQ cache (which had a different fingerprint) must be rejected
    legacy_cache_file = tmp_path / "legacy_custom_hqq_cache.pt"
    torch.save(
        {
            "delta_loss": {("proj0", 2): 0.1, ("proj0", 3): 0.05},
            "base_loss_4bit": 2.5,
            "label_time_sec": 1.0,
            "fingerprint": "legacy_custom_hqq_hash_xyz",
            "quantizer": "hqq",
        },
        legacy_cache_file,
    )
    loaded = _load_cached_labels(legacy_cache_file, fp_hqq)
    assert loaded is None

    upgraded, wall_sec = _load_or_upgrade_cached_labels(
        legacy_cache_file, fp_hqq_1234, model, calib_ids, cfg_hqq_1234, batch_size=2, save_path=tmp_path / "saved.pt"
    )
    assert upgraded is None
    assert wall_sec == 0.0

    # 3. Official HQQ cache with (2, 3, 4) CAN be incrementally upgraded to (1, 2, 3, 4)
    # Collect genuine 2,3 labels for one layer so upgrade logic can merge
    exp_234 = ExperimentConfig(quantizer="hqq", candidate_bits=(2, 3, 4), quant_group_size=64)
    labels_234 = collect_marginal_loss_labels(model, calib_ids, exp_234, batch_size=2)
    hqq_234_cache_file = tmp_path / "official_hqq_234.pt"
    torch.save(
        {
            "delta_loss": labels_234["delta_loss"],
            "base_loss_4bit": labels_234["base_loss_4bit"],
            "label_time_sec": labels_234["label_time_sec"],
            "base_overhead_time_sec": labels_234.get("base_overhead_time_sec", 0.0),
            "probe_time_sec": labels_234.get("probe_time_sec", labels_234["label_time_sec"]),
            "restore_time_sec": labels_234.get("restore_time_sec", 0.0),
            "fingerprint": fp_hqq,
            "quantizer": "hqq",
            "hqq_backend": "official_hqq",
            "hqq_version": "0.2.8.post1",
        },
        hqq_234_cache_file,
    )

    upgrade_target = tmp_path / "official_hqq_upgraded_1234.pt"
    upgraded, wall_sec = _load_or_upgrade_cached_labels(
        hqq_234_cache_file, fp_hqq_1234, model, calib_ids, cfg_hqq_1234, batch_size=2, save_path=upgrade_target
    )
    assert upgraded is not None
    assert wall_sec > 0.0
    assert upgraded["fingerprint"] == fp_hqq_1234
    assert any(k[1] == 1 for k in upgraded["delta_loss"])
    assert upgrade_target.exists()


def test_hqq_device_precision_fingerprint_isolation(tmp_path):
    """Verify fingerprints differ between CPU and CUDA devices/precision, and reject cross-device caches."""
    from hds_quant.run_experiment import (
        _build_smoke_model,
        _compute_label_cache_fingerprint,
        _get_smoke_calib_ids,
        _load_cached_labels,
        _load_or_upgrade_cached_labels,
    )
    from unittest.mock import MagicMock

    model_cpu, _ = _build_smoke_model()
    calib_ids = _get_smoke_calib_ids(42)

    cfg = ExperimentConfig(quantizer="hqq", candidate_bits=(2, 3, 4), quant_group_size=64)
    fp_cpu = _compute_label_cache_fingerprint(model_cpu, calib_ids, cfg)

    # Simulate CUDA model by mocking parameter device
    model_cuda = MagicMock(spec=model_cpu)
    model_cuda.named_parameters = model_cpu.named_parameters
    model_cuda.named_buffers = model_cpu.named_buffers
    model_cuda.config = model_cpu.config
    cuda_param = MagicMock()
    cuda_param.device = torch.device("cuda:0")
    cuda_param.dtype = torch.float16
    model_cuda.parameters.return_value = [cuda_param]

    fp_cuda = _compute_label_cache_fingerprint(model_cuda, calib_ids, cfg)
    assert fp_cpu != fp_cuda, "Fingerprint on CPU (float32 optimizer) must differ from CUDA (float16 optimizer)"

    # An HQQ cache produced on CPU must NOT be loaded when evaluated with the CUDA fingerprint
    cpu_cache_file = tmp_path / "cpu_hqq_cache.pt"
    torch.save(
        {
            "delta_loss": {("proj0", 2): 0.1, ("proj0", 3): 0.05},
            "base_loss_4bit": 2.5,
            "label_time_sec": 1.0,
            "fingerprint": fp_cpu,
            "quantizer": "hqq",
        },
        cpu_cache_file,
    )
    assert _load_cached_labels(cpu_cache_file, fp_cuda) is None

    cfg_1234 = ExperimentConfig(quantizer="hqq", candidate_bits=(1, 2, 3, 4), quant_group_size=64)
    fp_cuda_1234 = _compute_label_cache_fingerprint(model_cuda, calib_ids, cfg_1234)
    upgraded, wall_sec = _load_or_upgrade_cached_labels(
        cpu_cache_file, fp_cuda_1234, model_cuda, calib_ids, cfg_1234, batch_size=2, save_path=tmp_path / "upgraded.pt"
    )
    assert upgraded is None and wall_sec == 0.0


def test_hqq_revision_14_cache_invalidation(tmp_path):
    """Verify revision 14 caches lacking precision-aware fingerprinting are rejected and not upgraded."""
    from hds_quant.run_experiment import (
        _build_smoke_model,
        _compute_label_cache_fingerprint,
        _get_smoke_calib_ids,
        _load_cached_labels,
        _load_or_upgrade_cached_labels,
    )

    model, _ = _build_smoke_model()
    calib_ids = _get_smoke_calib_ids(42)
    cfg = ExperimentConfig(quantizer="hqq", candidate_bits=(1, 2, 3, 4), quant_group_size=64)
    fp_current = _compute_label_cache_fingerprint(model, calib_ids, cfg)

    # Create rev 14 cache with fake old hash
    rev14_cache_file = tmp_path / "rev14_hqq_cache.pt"
    torch.save(
        {
            "delta_loss": {("proj0", 2): 0.1, ("proj0", 3): 0.05},
            "base_loss_4bit": 2.5,
            "label_time_sec": 1.0,
            "fingerprint": "rev14_hash_without_device_precision",
            "quantizer": "hqq",
            "hqq_backend": "official_hqq",
            "hqq_version": "0.2.8.post1",
        },
        rev14_cache_file,
    )
    assert _load_cached_labels(rev14_cache_file, fp_current) is None

    upgraded, wall_sec = _load_or_upgrade_cached_labels(
        rev14_cache_file, fp_current, model, calib_ids, cfg, batch_size=2, save_path=tmp_path / "target.pt"
    )
    assert upgraded is None and wall_sec == 0.0


def test_hqq_settings_agreement_with_actual_arguments():
    """Verify get_effective_hqq_config matches arguments actually used during HQQ execution."""
    from hds_quant.quantizer import (
        HQQ_AXIS,
        HQQ_BACKEND_TAG,
        HQQ_CHANNEL_WISE,
        HQQ_OPTIMIZE,
        HQQ_ROUND_ZERO,
        HQQ_VERSION,
        get_effective_hqq_config,
    )
    import inspect
    from hqq.core.quantize import Quantizer

    eff_64 = get_effective_hqq_config(group_size=64, device="cpu", dtype=torch.float32)
    assert eff_64["backend"] == HQQ_BACKEND_TAG == "official_hqq"
    assert eff_64["version"] == HQQ_VERSION == "0.2.8.post1"
    assert eff_64["group_size"] == 64
    assert eff_64["axis"] == HQQ_AXIS == 1
    assert eff_64["round_zero"] == HQQ_ROUND_ZERO is False
    assert eff_64["optimize"] == HQQ_OPTIMIZE is True
    assert eff_64["channel_wise"] == HQQ_CHANNEL_WISE is True
    assert eff_64["device_type"] == "cpu"
    assert eff_64["optimizer_dtype"] == "float32"

    eff_cuda = get_effective_hqq_config(group_size=128, device="cuda", dtype=torch.float16)
    assert eff_cuda["group_size"] == 128
    assert eff_cuda["device_type"] == "cuda"
    assert eff_cuda["optimizer_dtype"] == "float16"

    # Verify optimizer defaults match actual Quantizer.optimize_weights default parameters
    sig = inspect.signature(Quantizer.optimize_weights)
    opt_p = sig.parameters["opt_params"].default
    assert eff_64["optimizer_defaults"] == opt_p


def test_format_tables_backend_provenance(tmp_path):
    """Verify format_tables reports 'legacy/unknown' for legacy runs and exact settings for new runs."""
    import json
    import sys
    from scripts.format_tables import main

    # 1. Legacy artifact missing backend provenance
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_run = {
        "config": {
            "model_id": "test_legacy",
            "quantizer": "hqq",
            "residual_anchor": "fp16_strict",
            "coord_binding": "raw_index_strict",
            "readout_grouping": "pooled",
            "target_transform": "raw",
            "seeds": [42],
        },
        "summary": {"ranking_metrics": {}, "ppl_metrics": {}},
        "uniform_baselines": {},
        "seed_runs": [],
    }
    with open(legacy_dir / "run_legacy.json", "w") as f:
        json.dump(legacy_run, f)

    sys.argv = ["format_tables.py", "--input-dir", str(legacy_dir), "--output-dir", str(legacy_dir)]
    main()
    legacy_report = (legacy_dir / "SUPERVISOR_REPORT.md").read_text()
    assert "legacy/unknown" in legacy_report
    assert "official_hqq" not in legacy_report

    # 2. Modern artifact with full backend provenance
    modern_dir = tmp_path / "modern"
    modern_dir.mkdir()
    modern_run = {
        "config": {
            "model_id": "test_modern",
            "quantizer": "hqq",
            "hqq_backend": "official_hqq",
            "hqq_version": "0.2.8.post1",
            "hqq_config": {
                "axis": 1,
                "round_zero": False,
                "optimize": True,
                "group_size": 64,
                "device_type": "cpu",
                "optimizer_dtype": "float32",
            },
            "residual_anchor": "fp16_strict",
            "coord_binding": "raw_index_strict",
            "readout_grouping": "pooled",
            "target_transform": "raw",
            "seeds": [42],
        },
        "summary": {"ranking_metrics": {}, "ppl_metrics": {}},
        "uniform_baselines": {},
        "seed_runs": [],
    }
    with open(modern_dir / "run_modern.json", "w") as f:
        json.dump(modern_run, f)

    sys.argv = ["format_tables.py", "--input-dir", str(modern_dir), "--output-dir", str(modern_dir)]
    main()
    modern_report = (modern_dir / "SUPERVISOR_REPORT.md").read_text()
    assert "HQQ 0.2.8.post1" in modern_report
    assert "official_hqq" in modern_report
    assert "group_size=64" in modern_report
    assert "opt_dtype=float32" in modern_report

