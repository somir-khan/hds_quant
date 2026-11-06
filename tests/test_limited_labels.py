"""Guard against acquiring unselected labels or timing unrelated estimators."""
import copy
from types import SimpleNamespace

import pytest
import torch

from hds_quant.config import ExperimentConfig
from hds_quant.estimators import compute_estimators, compute_selected_estimator
from hds_quant.hds_encoder import HDSEncoder
from hds_quant.label_collector import collect_features_single_pass, collect_marginal_loss_labels, discover_target_projections
from hds_quant.readout_cv import evaluate_fixed_k_layer_acquisition
from hds_quant.run_experiment import _build_smoke_model, _get_smoke_calib_ids
import hds_quant.run_limited_labels as runner
from scripts.summarize_limited_labels import summarize


@pytest.mark.parametrize("method", runner.METHODS)
@pytest.mark.parametrize("binding", ["raw_index_strict", "salience_bucket"])
def test_selective_features_match_existing(method, binding):
    torch.manual_seed(19)
    r, norms = torch.rand(12, 32), torch.rand(32)
    encoder = HDSEncoder(hd_dim=512, max_coords=64, num_buckets=32, device="cpu")
    expected = compute_estimators(r, norms, 2, encoder, coord_binding=binding)
    actual = compute_selected_estimator(r, norms, 2, encoder, method, binding)
    key = "Scalar-MSE" if method == "Scalar-MSE-Ridge" else method
    torch.testing.assert_close(torch.as_tensor(actual[key]), torch.as_tensor(expected[key]), atol=0, rtol=0)


def test_partial_fit_matches_full_feature_path():
    model, _ = _build_smoke_model()
    ids = _get_smoke_calib_ids(42)
    cfg = ExperimentConfig(num_layers=4, hd_dim=512, device="cpu", quant_group_size=64,
                           max_encoded_tokens=64, seeds=(42,), readout_grouping="per_bit",
                           residual_anchor="int4_aligned", coord_binding="salience_bucket")
    encoder = HDSEncoder(hd_dim=512, device="cpu", seed=42)
    labels = collect_marginal_loss_labels(model, ids, cfg, target_layers={0, 2})["delta_loss"]
    meta = discover_target_projections(model)
    assert len(labels) == 2 * 7 * 2
    assert {m["layer_idx"] for m in meta if (m["name"], 2) in labels} == {0, 2}
    all_features = collect_features_single_pass(model, ids, cfg, encoder)["features"]
    full = evaluate_fixed_k_layer_acquisition(all_features, labels, meta, cfg, 2)
    for method in runner.METHODS:
        selective = collect_features_single_pass(model, ids, cfg, encoder, feature_estimator=method)["features"]
        actual = evaluate_fixed_k_layer_acquisition(selective, labels, meta, cfg, 2, estimator_names=(method,))
        assert actual["raw_predictions"][method] == pytest.approx(full["raw_predictions"][method])
        assert tuple(actual["estimators"]) == (method,)


def test_complete_runner_acquires_only_k_and_separates_reference(tmp_path, monkeypatch):
    args = SimpleNamespace(smoke_test=True, model_id="smoke", eval_dataset="wikitext2", k_layers=[2],
        seeds=[42], methods=list(runner.METHODS), budgets=[3.0, 3.5], candidate_bits=[1, 2, 3, 4],
        n_calib_seqs=8, calib_seq_len=64, max_encoded_tokens=64, eval_seq_len=128, batch_size=4,
        output_dir=str(tmp_path))
    real = runner.collect_marginal_loss_labels
    calls = []
    def guarded(*a, **kw):
        calls.append(kw.get("target_layers"))
        return real(*a, **kw)
    monkeypatch.setattr(runner, "collect_marginal_loss_labels", guarded)
    d = runner.run(args)
    assert len(calls) == 4 and calls[-1] is None
    assert all(c == calls[0] and len(c) == 2 for c in calls[:-1])
    assert d["complete"] and len(d["trials"]) == 3
    for trial in d["trials"]:
        assert trial["probe_label_count"] == 42
        assert trial["calibration_wall_sec"] >= trial["calibration_shared_sec"] > 0
        assert trial["feature_payload_bytes"] > 0
        assert trial["peak_cuda_allocated_bytes"] is None
        if trial["method"] != "HDS-Hybrid":
            assert trial["encoder_codebook_bytes"] == 0
        assert "offline_ranking" in trial
        assert len(trial["allocations"]) == 2
    assert (tmp_path / "limited_labels.json").is_file()
    summary = summarize(tmp_path, make_plots=False)
    assert len(summary) == 8  # three limited methods + full reference at two budgets
    assert all(r["n_seeds"] == 1 and r["calibration_sec_mean"] > 0 for r in summary)
    assert (tmp_path / "trials.csv").is_file()
