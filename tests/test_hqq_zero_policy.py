"""Validate bit-specific zero rounding and a complete paired ablation."""
import copy
from dataclasses import replace

import pytest
import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from hds_quant.config import ExperimentConfig
from hds_quant.quantizer import quantize_weight, get_effective_hqq_config, resolve_hqq_round_zero
from hds_quant.run_experiment import _build_smoke_model, _compute_label_cache_fingerprint, parse_args, run_experiment
from scripts.compare_hqq_zero_policies import compare


@pytest.mark.parametrize("policy", ["floating", "round4"])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_policy_parity(policy, bits, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(17)
    layer = torch.nn.Linear(64, 32, bias=False).to(device=device, dtype=dtype)
    config = BaseQuantizeConfig(nbits=bits, group_size=128, axis=0)
    config["weight_quant_params"]["round_zero"] = policy == "round4" and bits == 4
    official = HQQLinear(copy.deepcopy(layer), config, compute_dtype=dtype, device=device)
    actual = quantize_weight(layer.weight, bits, quantizer="hqq", hqq_axis=0, hqq_zero_policy=policy)
    torch.testing.assert_close(actual, official.dequantize(), atol=0, rtol=0)


def test_policy_isolation():
    model, ids = _build_smoke_model()
    cfg = ExperimentConfig(quantizer="hqq", hqq_axis=0)
    assert _compute_label_cache_fingerprint(model, ids, cfg) != _compute_label_cache_fingerprint(model, ids, replace(cfg, hqq_zero_policy="round4"))
    torch.manual_seed(17)
    w = torch.randn(32, 64)
    for bits in (1, 2, 3):
        assert torch.equal(quantize_weight(w, bits, quantizer="hqq"), quantize_weight(w, bits, quantizer="hqq", hqq_zero_policy="round4"))
    assert not torch.equal(quantize_weight(w, 4, quantizer="hqq"), quantize_weight(w, 4, quantizer="hqq", hqq_zero_policy="round4"))
    assert get_effective_hqq_config(zero_policy="round4")["round_zero"] == "4-bit only"
    with pytest.raises(ValueError):
        resolve_hqq_round_zero(4, "invalid")


def test_paired_smoke_run_and_comparison(tmp_path, monkeypatch):
    for policy in ("floating", "round4"):
        out = tmp_path / policy
        monkeypatch.setattr("sys.argv", ["run_experiment", "--smoke-test", "--quantizer", "hqq",
            "--hqq-axis", "0", "--hqq-zero-policy", policy, "--candidate-bits", "1", "2", "3", "4",
            "--seeds", "42", "--budgets", "3", "3.5", "--residual-anchor", "int4_aligned",
            "--coord-binding", "salience_bucket", "--readout-grouping", "per_bit",
            "--estimators", "HDS-Hybrid", "Coord-Rank-Hist", "Oracle-Loss",
            "--output-dir", str(out), "--cache-dir", str(out / "cache"), "--run-tag", policy])
        result = run_experiment(parse_args())
        assert set(result["summary"]["ppl_metrics"]) == {"HDS-Hybrid", "Coord-Rank-Hist", "Oracle-Loss"}
        assert result["config"]["hqq_zero_policy"] == policy
        cache = torch.load(out / "cache" / "smoke_seed42_hqq_labels.pt", weights_only=False)
        assert cache["hqq_config"]["round_zero"] == (False if policy == "floating" else "4-bit only")
    rows = compare(tmp_path)
    assert len(rows) == 11  # five uniform baselines + three methods at two budgets
    assert all(r["relative_change_percent"] == 0 for r in rows if r["method"] in ("FP16", "Uniform-1bit", "Uniform-2bit", "Uniform-3bit"))
