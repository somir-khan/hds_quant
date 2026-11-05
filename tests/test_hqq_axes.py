"""Independent packed HQQLinear parity and axis propagation checks."""
import copy
from dataclasses import replace

import pytest
import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from hds_quant.config import ExperimentConfig
from hds_quant.evaluator import evaluate_perplexity
from hds_quant.label_collector import discover_target_projections
from hds_quant.quantizer import quantize_weight, quantize_weight_hqq
from hds_quant.run_experiment import _build_smoke_model, _compute_label_cache_fingerprint


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("group_size", [16, 128])
def test_packed_hqqlinear_weights_and_outputs(axis, bits, group_size):
    torch.manual_seed(7)
    linear = torch.nn.Linear(64, 32, bias=True)
    original = linear.weight.detach().clone()
    cfg = BaseQuantizeConfig(nbits=bits, group_size=group_size, axis=axis)
    # The high-level helper rounds zero points at 4 bits; our recorded policy
    # deliberately keeps them floating point at every bit width.
    cfg["weight_quant_params"]["round_zero"] = False
    official = HQQLinear(copy.deepcopy(linear), cfg, compute_dtype=torch.float32, device="cpu")
    actual = quantize_weight(original, bits, group_size=group_size, quantizer="hqq", hqq_axis=axis)
    torch.testing.assert_close(actual, official.dequantize(), rtol=0, atol=0)
    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(torch.nn.functional.linear(x, actual, linear.bias), official(x), rtol=1e-6, atol=1e-6)
    assert torch.equal(original, linear.weight)


def test_axis_changes_fingerprint_and_reconstruction():
    model, ids = _build_smoke_model()
    cfg = ExperimentConfig(quantizer="hqq", hqq_axis=1, quant_group_size=64)
    assert _compute_label_cache_fingerprint(model, ids, cfg) != _compute_label_cache_fingerprint(model, ids, replace(cfg, hqq_axis=0))
    w = torch.randn(32, 64)
    assert not torch.equal(quantize_weight_hqq(w, 2, axis=0), quantize_weight_hqq(w, 2, axis=1))


@pytest.mark.parametrize("axis", [0, 1])
def test_whole_model_perplexity_matches_packed_hqqlinear(axis):
    model, ids = _build_smoke_model()
    cfg = ExperimentConfig(quantizer="hqq", hqq_axis=axis, quant_group_size=64)
    meta = discover_target_projections(model)
    original = {m["name"]: m["module"].weight.detach().clone() for m in meta}
    expected = evaluate_perplexity(model, ids[:2], meta, cfg, uniform_bit=3)
    for m in meta:
        assert torch.equal(m["module"].weight, original[m["name"]])
        official = HQQLinear(copy.deepcopy(m["module"]), BaseQuantizeConfig(nbits=3, group_size=64, axis=axis), compute_dtype=torch.float32, device="cpu")
        model.set_submodule(m["name"], official)
    actual = evaluate_perplexity(model, ids[:2], [], cfg)
    assert actual["nll"] == pytest.approx(expected["nll"], abs=1e-6)


def test_invalid_axis_rejected():
    with pytest.raises(ValueError, match="axis"):
        quantize_weight_hqq(torch.randn(8, 128), 3, axis=2)
    with pytest.raises(ValueError, match="axis"):
        ExperimentConfig(hqq_axis=2)


def test_four_bit_high_level_default_differs_from_recorded_policy():
    cfg = BaseQuantizeConfig(nbits=4, group_size=128, axis=0)
    assert cfg["weight_quant_params"]["round_zero"] is True
    torch.manual_seed(7)
    layer = torch.nn.Linear(64, 32, bias=False)
    official = HQQLinear(copy.deepcopy(layer), cfg, compute_dtype=torch.float32, device="cpu")
    expected = quantize_weight_hqq(layer.weight, 4, axis=0, round_zero=True)
    torch.testing.assert_close(expected, official.dequantize(), rtol=0, atol=0)


def test_axis_zero_accepts_official_total_size_grouping():
    # HQQ requires total size divisibility, not input width divisibility.
    w = torch.randn(16, 24)
    result = quantize_weight_hqq(w, 3, group_size=128, axis=0)
    assert result.shape == w.shape and torch.isfinite(result).all()
