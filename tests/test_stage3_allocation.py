"""Stage-3 tests: layer-disjoint readout CV and the exact ILP bit allocator."""

import itertools
import math

import numpy as np
import pytest
import torch

from pathlib import Path

from hds_quant import ExperimentConfig, HDSEncoder, compute_estimators
from hds_quant.allocator import solve_bit_allocation_ilp
from hds_quant.config import PROJ_FAMILIES
from hds_quant.readout_cv import (
    ALL_ESTIMATORS,
    SUPERVISED_ESTIMATORS,
    UNSUPERVISED_SCALARS,
    cross_validate_estimators,
    _get_column_weights,
    _fit_ridge,
    _fit_predict_grouped,
)

NUM_LAYERS = 4
HD_DIM = 128
GROUPINGS = ("pooled", "per_bit", "per_family", "per_bit_family")
TRANSFORMS = ("raw", "signed_log1p", "rank")


# ------------------------------------------------------------------ synthetic data built with the real estimators
def _build_meta():
    meta = []
    for layer in range(NUM_LAYERS):
        for fi, fam in enumerate(PROJ_FAMILIES):
            d_out = 64 if fi in (1, 2) else 96
            meta.append(
                {
                    "name": f"model.layers.{layer}.{fam}",
                    "layer_idx": layer,
                    "family": fam.split(".")[-1],
                    "family_idx": fi,
                    "num_params": d_out * 128,
                    "d_out": d_out,
                    "d_in": 128,
                }
            )
    return meta


@pytest.fixture(scope="module")
def data():
    enc = HDSEncoder(hd_dim=HD_DIM, device="cpu", seed=0)
    rng = np.random.RandomState(0)
    meta = _build_meta()
    feats, delta = {}, {}
    for m in meta:
        sens = float(rng.uniform(0.5, 2.0))
        for b in (2, 3):
            scale = sens * (3.0 if b == 2 else 1.0)
            R = torch.from_numpy(np.abs(rng.randn(48, m["d_out"])) * scale).float()
            norms = torch.from_numpy(rng.rand(m["d_out"]) + 0.5).float()
            f = compute_estimators(R, norms, m["family_idx"], enc)
            feats[(m["name"], b)] = f
            delta[(m["name"], b)] = float(f["Scalar-MSE"] * 0.01 + 0.002 * rng.randn())
    return feats, delta, meta


def make_config(**kw):
    return ExperimentConfig(hd_dim=HD_DIM, device="cpu", **kw)


def folds_of(meta, fold_layers):
    return {m["name"] for m in meta if m["layer_idx"] in fold_layers}


# ------------------------------------------------------------------ 1. layer-fold isolation
@pytest.mark.parametrize("grouping", ("pooled", "per_bit_family"))
@pytest.mark.parametrize("transform", ("raw", "signed_log1p"))
def test_no_label_leakage_from_held_out_fold(data, grouping, transform):
    feats, delta, meta = data
    cfg = make_config(readout_grouping=grouping, target_transform=transform)
    clean = cross_validate_estimators(feats, delta, meta, cfg)

    # corrupt every label of Fold 0 (= layer 0 with 4 layers)
    fold0 = folds_of(meta, {0})
    corrupted_delta = dict(delta)
    for (name, b), v in delta.items():
        if name in fold0:
            corrupted_delta[(name, b)] = v * 1e3 + 5.0
    dirty = cross_validate_estimators(feats, corrupted_delta, meta, cfg)

    for est in ALL_ESTIMATORS:
        if est in ("Oracle-Loss", "Oracle-Loss-Raw"):
            continue
        for key, v in clean["raw_predictions"][est].items():
            if key[0] in fold0:
                assert dirty["raw_predictions"][est][key] == v, (est, key)  # exactly identical
                assert clean["alloc_costs"][est][key] == dirty["alloc_costs"][est][key]
    for est in SUPERVISED_ESTIMATORS:
        assert np.array_equal(clean["model_coefs"][(est, 0)], dirty["model_coefs"][(est, 0)]), est

    # sanity: the corruption does reach models that legitimately train on Fold 0 labels
    assert not np.array_equal(clean["model_coefs"][("HDS-Hybrid", 1)], dirty["model_coefs"][("HDS-Hybrid", 1)])
    assert clean["raw_predictions"]["Oracle-Loss"] != dirty["raw_predictions"]["Oracle-Loss"]
    assert clean["raw_predictions"]["Oracle-Loss-Raw"] != dirty["raw_predictions"]["Oracle-Loss-Raw"]


def test_folds_are_layer_disjoint_and_complete(data):
    feats, delta, meta = data
    out = cross_validate_estimators(feats, delta, meta, make_config())
    assert out["n_folds"] == 4
    assert out["fold_of_layer"] == {0: 0, 1: 1, 2: 2, 3: 3}
    for k, layers in out["train_layers_used"].items():
        assert k not in layers and sorted(layers) == [l for l in range(4) if l != k]
    n = len(meta) * 2
    for est in ALL_ESTIMATORS:
        assert len(out["raw_predictions"][est]) == n  # every (proj, b) predicted exactly once


def test_fold_assignment_for_16_layers_and_tiny_models():
    # fold_id = layer_idx % 4 for 16 layers; n_folds = min(4, num_layers) for tiny models
    assert [l for l in range(16) if l % 4 == 0] == [0, 4, 8, 12]
    assert [l for l in range(16) if l % 4 == 1] == [1, 5, 9, 13]


# ------------------------------------------------------------------ 2. grouping x transform modes
@pytest.mark.parametrize("grouping", GROUPINGS)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_all_grouping_and_transform_modes(data, grouping, transform):
    feats, delta, meta = data
    out = cross_validate_estimators(feats, delta, meta, make_config(readout_grouping=grouping, target_transform=transform))
    assert set(out["raw_predictions"]) == set(ALL_ESTIMATORS) == set(out["alloc_costs"])
    assert out["readout_grouping"] == grouping and out["target_transform"] == transform
    names = [m["name"] for m in meta]
    for est in ALL_ESTIMATORS:
        raw, alloc = out["raw_predictions"][est], out["alloc_costs"][est]
        assert set(raw) == {(n, b) for n in names for b in (2, 3)}
        assert set(alloc) == {(n, b) for n in names for b in (2, 3, 4)}
        assert all(isinstance(v, float) and math.isfinite(v) for v in raw.values()), est
        for n in names:
            assert alloc[(n, 4)] == 0.0
            if est == "Oracle-Loss-Raw":
                assert alloc[(n, 3)] == delta[(n, 3)]
                assert alloc[(n, 2)] == delta[(n, 2)]
            else:
                assert alloc[(n, 3)] >= 0.0
                assert alloc[(n, 2)] >= alloc[(n, 3)] + 1e-6 - 1e-12


def test_unsupervised_scalars_and_oracle_pass_through(data):
    feats, delta, meta = data
    out = cross_validate_estimators(feats, delta, meta, make_config())
    for key in delta:
        for est in UNSUPERVISED_SCALARS:
            assert out["raw_predictions"][est][key] == pytest.approx(float(feats[key][est]))
        assert out["raw_predictions"]["Oracle-Loss"][key] == delta[key]
        assert out["raw_predictions"]["Oracle-Loss-Raw"][key] == delta[key]


def test_supervised_estimators_learn_signal(data):
    """Labels are ~linear in MSE, so held-out supervised predictions must correlate with the truth."""
    from scipy.stats import spearmanr

    feats, delta, meta = data
    out = cross_validate_estimators(feats, delta, meta, make_config(readout_grouping="pooled"))
    keys = sorted(delta)
    truth = [delta[k] for k in keys]
    for est in ("Scalar-MSE-Ridge", "HDS-Mag-Strict", "HDS-Hybrid"):
        rho = spearmanr([out["raw_predictions"][est][k] for k in keys], truth).statistic
        assert rho > 0.5, (est, rho)


def test_rank_transform_predicts_unit_interval_scores(data):
    feats, delta, meta = data
    out = cross_validate_estimators(feats, delta, meta, make_config(target_transform="rank"))
    for est in SUPERVISED_ESTIMATORS:
        vals = np.array(list(out["raw_predictions"][est].values()))
        assert vals.min() > -1.0 and vals.max() < 2.0, est


def test_missing_labels_are_skipped(data):
    feats, delta, meta = data
    partial = {k: v for k, v in delta.items() if not k[0].startswith("model.layers.3.")}
    out = cross_validate_estimators(feats, partial, meta, make_config())
    assert not any(k[0].startswith("model.layers.3.") for k in out["raw_predictions"]["HDS-Rank"])
    assert out["n_folds"] == 3
    with pytest.raises(ValueError):
        cross_validate_estimators(feats, {}, meta, make_config())


# ------------------------------------------------------------------ 3. low-shot subsampling
def test_low_shot_two_train_layers(data):
    feats, delta, meta = data
    cfg = make_config(readout_grouping="per_bit_family")
    a = cross_validate_estimators(feats, delta, meta, cfg, num_train_layers=2, seed=7)
    b = cross_validate_estimators(feats, delta, meta, cfg, num_train_layers=2, seed=7)
    c = cross_validate_estimators(feats, delta, meta, cfg, num_train_layers=2, seed=8)
    for k, layers in a["train_layers_used"].items():
        assert len(layers) == 2 and k not in layers
    assert a["train_layers_used"] == b["train_layers_used"]  # deterministic given the seed
    assert a["raw_predictions"] == b["raw_predictions"]
    assert a["train_layers_used"] != c["train_layers_used"]
    for est in ALL_ESTIMATORS:
        assert all(math.isfinite(v) for v in a["raw_predictions"][est].values())
    full = cross_validate_estimators(feats, delta, meta, cfg)
    assert full["raw_predictions"]["HDS-Rank"] != a["raw_predictions"]["HDS-Rank"]


def test_low_shot_larger_than_available_uses_all_layers(data):
    feats, delta, meta = data
    cfg = make_config()
    a = cross_validate_estimators(feats, delta, meta, cfg, num_train_layers=99)
    b = cross_validate_estimators(feats, delta, meta, cfg)
    assert a["raw_predictions"] == b["raw_predictions"]


# ------------------------------------------------------------------ 4. knapsack ILP
def toy_instance(P, seed=0, sizes=None):
    rng = np.random.RandomState(seed)
    meta = [
        {"name": f"p{i}", "num_params": int(sizes[i] if sizes is not None else rng.randint(1, 20) * 1000)}
        for i in range(P)
    ]
    costs = {}
    for m in meta:
        c3 = float(rng.uniform(0.01, 1.0))
        costs[(m["name"], 4)] = 0.0
        costs[(m["name"], 3)] = c3
        costs[(m["name"], 2)] = c3 + float(rng.uniform(0.01, 2.0))
    return costs, meta


def brute_force(costs, meta, budget):
    total = sum(m["num_params"] for m in meta)
    best, best_assign = math.inf, None
    for assign in itertools.product((2, 3, 4), repeat=len(meta)):
        avg = sum(b * m["num_params"] for b, m in zip(assign, meta)) / total
        if avg > budget + 1e-12:
            continue
        cost = sum(costs[(m["name"], b)] for b, m in zip(assign, meta))
        if cost < best:
            best, best_assign = cost, assign
    return best, best_assign


BUDGETS = [2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00]


def test_ilp_budget_and_monotone_cost():
    costs, meta = toy_instance(40, seed=1)
    total_params = sum(m["num_params"] for m in meta)
    results = [solve_bit_allocation_ilp(costs, meta, t) for t in BUDGETS]
    prev = math.inf
    for t, r in zip(BUDGETS, results):
        assert r["success"]
        assert r["achieved_avg_bits"] <= t + 1e-6
        assert set(r["bit_assignment"]) == {m["name"] for m in meta}
        assert set(r["bit_assignment"].values()) <= {2, 3, 4}
        assert sum(r["bit_counts"].values()) == len(meta)
        recomputed = sum(b * m["num_params"] for m in meta for b in [r["bit_assignment"][m["name"]]]) / total_params
        assert r["achieved_avg_bits"] == pytest.approx(recomputed)
        assert r["total_predicted_cost"] == pytest.approx(sum(costs[(n, b)] for n, b in r["bit_assignment"].items()))
        assert r["total_predicted_cost"] <= prev + 1e-9  # looser budget never costs more
        prev = r["total_predicted_cost"]
    assert results[0]["total_predicted_cost"] > results[-1]["total_predicted_cost"]


def test_ilp_full_budget_keeps_everything_at_4_bits():
    costs, meta = toy_instance(30, seed=2)
    r = solve_bit_allocation_ilp(costs, meta, 4.0)
    assert r["success"] and set(r["bit_assignment"].values()) == {4}
    assert r["total_predicted_cost"] == 0.0
    assert r["bit_counts"] == {2: 0, 3: 0, 4: 30}
    assert r["achieved_avg_bits"] == pytest.approx(4.0)


def test_ilp_minimum_budget_forces_all_2_bit_and_infeasible_below():
    costs, meta = toy_instance(10, seed=3)
    r = solve_bit_allocation_ilp(costs, meta, 2.0)
    assert r["success"] and r["bit_counts"] == {2: 10, 3: 0, 4: 0}
    r = solve_bit_allocation_ilp(costs, meta, 1.5)
    assert r["success"] is False


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("budget", BUDGETS)
def test_ilp_matches_brute_force_on_six_projections(seed, budget):
    costs, meta = toy_instance(6, seed=seed)
    r = solve_bit_allocation_ilp(costs, meta, budget, enforce_exact_payload=False)
    best, _ = brute_force(costs, meta, budget)
    assert r["success"]
    assert r["total_predicted_cost"] == pytest.approx(best, abs=1e-9)
    assert r["achieved_avg_bits"] <= budget + 1e-6


def test_ilp_matches_brute_force_with_nonmonotone_costs_and_skewed_sizes():
    rng = np.random.RandomState(11)
    sizes = [1000, 50000, 700, 20000, 3000, 90000]
    meta = [{"name": f"q{i}", "num_params": s} for i, s in enumerate(sizes)]
    costs = {(m["name"], b): float(rng.uniform(0, 3)) for m in meta for b in (2, 3, 4)}
    for budget in (2.4, 3.1, 3.6):
        r = solve_bit_allocation_ilp(costs, meta, budget, enforce_exact_payload=False)
        best, _ = brute_force(costs, meta, budget)
        assert r["total_predicted_cost"] == pytest.approx(best, abs=1e-9)


def test_ilp_end_to_end_with_cv_costs(data):
    feats, delta, meta = data
    out = cross_validate_estimators(feats, delta, meta, make_config())
    for est in ("Scalar-MSE", "HDS-Hybrid", "Oracle-Loss"):
        r = solve_bit_allocation_ilp(out["alloc_costs"][est], meta, 3.0)
        assert r["success"] and r["achieved_avg_bits"] <= 3.0 + 1e-6


def test_ilp_missing_cost_entry_raises():
    costs, meta = toy_instance(3, seed=5)
    del costs[("p1", 3)]
    with pytest.raises(KeyError):
        solve_bit_allocation_ilp(costs, meta, 3.0)


# ------------------------------------------------------------------ 5. Bug 3 verification tests
def test_metadata_column_upweighting_math():
    w_mag = _get_column_weights(4098, estimator="HDS-Mag-Strict", hd_dim=4096)
    assert w_mag is not None
    assert np.allclose(w_mag[:4096], 1.0)
    assert np.allclose(w_mag[4096:], math.sqrt(4096 / 2.0))

    w_hyb = _get_column_weights(4106, estimator="HDS-Hybrid", hd_dim=4096)
    assert w_hyb is not None
    assert np.allclose(w_hyb[:4096], 1.0)
    assert np.allclose(w_hyb[4096:], math.sqrt(4096 / 10.0))

    # Auto-detection from column counts
    w_auto_mag = _get_column_weights(4098)
    assert np.allclose(w_auto_mag, w_mag)
    w_auto_hyb = _get_column_weights(4106)
    assert np.allclose(w_auto_hyb, w_hyb)


def test_metadata_upweighting_prevents_scalar_anchor_dilution():
    rng = np.random.RandomState(42)
    N = 30
    D = 4096
    X_hd = rng.randn(N, D)
    log_mse = rng.randn(N, 1) * 2.0
    indicator = rng.choice([0.0, 1.0], size=(N, 1))
    X = np.hstack([X_hd, log_mse, indicator])
    y = np.ravel(log_mse * 3.0 + 0.1 * rng.randn(N, 1))

    # Fit unweighted vs weighted
    scaler_unw, ridge_unw, _ = _fit_ridge(X, y, col_weights=np.ones(X.shape[1]))
    scaler_w, ridge_w, _ = _fit_ridge(X, y, estimator="HDS-Mag-Strict", hd_dim=D)

    coef_unweighted_log_mse = abs(ridge_unw.coef_[D])
    coef_weighted_log_mse = abs(ridge_w.coef_[D] * scaler_w.col_weights_[D])
    assert coef_weighted_log_mse > coef_unweighted_log_mse * 2.0


def test_fit_predict_grouped_fallback_threshold_eight():
    rng = np.random.RandomState(0)
    # Group A has 4 samples (< 8), Group B has 10 samples (>= 8)
    keys_tr = ["A"] * 4 + ["B"] * 10
    X_tr = rng.randn(14, 8)
    y_tr = rng.randn(14)
    keys_te = ["A", "B"]
    X_te = rng.randn(2, 8)

    preds, _ = _fit_predict_grouped(X_tr, y_tr, keys_tr, X_te, keys_te)
    assert len(preds) == 2
    assert all(math.isfinite(p) for p in preds)


def test_scripts_run_pilot_a100_uses_per_bit():
    script_path = Path(__file__).parent.parent / "scripts" / "run_pilot_a100.sh"
    content = script_path.read_text(encoding="utf-8")
    assert '--readout-grouping "per_bit"' in content
    assert '--readout-grouping "per_bit_family"' not in content
