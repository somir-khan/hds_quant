"""Stage-2 tests: projection discovery, feature collection, marginal-loss labels, data utilities."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from hds_quant import ESTIMATOR_NAMES, ExperimentConfig, HDSEncoder, quantize_weight_rtn
from hds_quant.data_utils import get_calibration_batches, get_eval_tokens
from hds_quant.label_collector import (
    PROJ_FAMILIES,
    _select_token_positions,
    collect_features_single_pass,
    collect_marginal_loss_labels,
    discover_target_projections,
)

HD_DIM = 512
NUM_LAYERS = 4
ANCHORS = ("fp16_strict", "int4_aligned")
BINDINGS = ("raw_index_strict", "salience_bucket")


def make_config(**kw) -> ExperimentConfig:
    base = dict(quant_group_size=64, hd_dim=HD_DIM, max_encoded_tokens=256, device="cpu")
    base.update(kw)
    return ExperimentConfig(**base)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def calib_ids():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 256, (8, 64), generator=g)


@pytest.fixture(scope="module")
def encoder():
    return HDSEncoder(hd_dim=HD_DIM, device="cpu", seed=42)


@pytest.fixture(scope="module")
def labels(model, calib_ids):
    return collect_marginal_loss_labels(model, calib_ids, make_config(), batch_size=3)


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def assert_identical(model, snap):
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), snap[n]), f"parameter {n} changed"


# ------------------------------------------------------------------ (a) discovery
def test_discover_finds_28_projections(model):
    projs = discover_target_projections(model)
    assert len(projs) == NUM_LAYERS * 7 == 28
    assert len({p["name"] for p in projs}) == 28
    assert all("embed_tokens" not in p["name"] and "lm_head" not in p["name"] for p in projs)
    assert [(p["layer_idx"], p["family_idx"]) for p in projs] == [(l, f) for l in range(NUM_LAYERS) for f in range(7)]
    for p in projs:
        assert p["family"] == PROJ_FAMILIES[p["family_idx"]]
        assert p["name"].endswith(p["family"])
        assert p["module"] is model.get_submodule(p["name"])
        assert p["num_params"] == p["module"].weight.numel() == p["d_out"] * p["d_in"]
        assert (p["d_out"], p["d_in"]) == (p["module"].out_features, p["module"].in_features)
    dims = {p["family"]: (p["d_out"], p["d_in"]) for p in projs}
    assert dims["q_proj"] == (128, 128) and dims["k_proj"] == (64, 128)
    assert dims["gate_proj"] == (256, 128) and dims["down_proj"] == (128, 256)


# ------------------------------------------------------------------ (b) features
@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("binding", BINDINGS)
def test_collect_features_finite_for_all_pairs(model, calib_ids, encoder, anchor, binding):
    snap = snapshot(model)
    cfg = make_config(residual_anchor=anchor, coord_binding=binding)
    out = collect_features_single_pass(model, calib_ids, cfg, encoder, batch_size=3)

    assert set(out) == {
        "features",
        "meta",
        "feature_time_sec",
        "forward_capture_time_sec",
        "estimator_feature_time_sec",
    }
    assert out["feature_time_sec"] > 0
    # timing breakdown: shared forward capture + per-estimator feature extraction, both inside the total
    assert 0 < out["forward_capture_time_sec"] <= out["feature_time_sec"] + 1e-6
    assert set(out["estimator_feature_time_sec"]) == set(ESTIMATOR_NAMES) | {"Scalar-MSE-Ridge"}
    assert all(v > 0 for v in out["estimator_feature_time_sec"].values())
    assert len(out["meta"]) == 28 and all("module" not in m for m in out["meta"])
    names = [m["name"] for m in out["meta"]]
    assert set(out["features"]) == {(n, b) for n in names for b in (2, 3)}
    assert len(out["features"]) == 56
    for key, feats in out["features"].items():
        assert len(feats) == len(ESTIMATOR_NAMES), key
        for est, val in feats.items():
            arr = torch.as_tensor(val)
            assert torch.isfinite(arr).all(), (key, est)
    assert out["features"][(names[0], 2)]["HDS-Rank"].shape == (HD_DIM,)
    assert out["features"][(names[0], 2)]["HDS-Hybrid"].shape == (HD_DIM + 9,)
    # features never modify the model and leave no hooks behind
    assert_identical(model, snap)
    assert all(len(m._forward_pre_hooks) == 0 for m in model.modules())


def test_features_are_deterministic_and_batch_size_invariant(model, calib_ids, encoder):
    cfg = make_config()
    a = collect_features_single_pass(model, calib_ids, cfg, encoder, batch_size=8)["features"]
    b = collect_features_single_pass(model, calib_ids, cfg, encoder, batch_size=3)["features"]
    for key in a:
        assert a[key]["Scalar-MSE"] == pytest.approx(b[key]["Scalar-MSE"], rel=1e-3)
        assert torch.allclose(a[key]["HDS-Rank"], b[key]["HDS-Rank"], atol=5e-3)


@pytest.mark.parametrize("anchor", ANCHORS)
def test_features_match_manual_residual(model, calib_ids, encoder, anchor):
    """Layer-0 q_proj: rebuild X and R by hand and compare the MSE / L1 estimators."""
    cfg = make_config(residual_anchor=anchor)
    feats = collect_features_single_pass(model, calib_ids, cfg, encoder, batch_size=3)["features"]

    layer0 = model.model.layers[0]
    with torch.no_grad():
        hidden = layer0.input_layernorm(model.model.embed_tokens(calib_ids))
        X = hidden.reshape(-1, hidden.shape[-1])[_select_token_positions(calib_ids, cfg)]
        W = layer0.self_attn.q_proj.weight
        Y16 = F.linear(X, W)
        Yref = F.linear(X, quantize_weight_rtn(W, 4, 64)) if anchor == "int4_aligned" else Y16
        for b in (2, 3):
            R = (Yref - F.linear(X, quantize_weight_rtn(W, b, 64))).abs()
            f = feats[("model.layers.0.self_attn.q_proj", b)]
            assert f["Scalar-MSE"] == pytest.approx(float(R.pow(2).mean()), rel=1e-3)
            assert f["Scalar-L1"] == pytest.approx(float(R.mean()), rel=1e-3)
    # 2-bit damage exceeds 3-bit damage on average
    assert feats[("model.layers.0.self_attn.q_proj", 2)]["Scalar-MSE"] > feats[("model.layers.0.self_attn.q_proj", 3)]["Scalar-MSE"]


def test_features_reject_mismatched_encoder(model, calib_ids):
    with pytest.raises(ValueError):
        collect_features_single_pass(model, calib_ids, make_config(), HDSEncoder(hd_dim=1024, device="cpu"))


# ------------------------------------------------------------------ (c) labels
def test_labels_finite_for_all_pairs(model, labels):
    names = [p["name"] for p in discover_target_projections(model)]
    d = labels["delta_loss"]
    assert set(d) == {(n, b) for n in names for b in (2, 3)}
    assert len(d) == 56
    assert all(math.isfinite(v) for v in d.values())
    assert math.isfinite(labels["base_loss_4bit"]) and labels["base_loss_4bit"] > 0
    assert labels["label_time_sec"] > 0
    assert labels["base_overhead_time_sec"] > 0
    assert labels["label_time_sec"] >= labels["base_overhead_time_sec"] + labels["probe_time_sec"]
    assert labels["restore_time_sec"] >= 0.0
    assert any(abs(v) > 0 for v in d.values())


def test_labels_target_layers_subset(model, calib_ids, labels):
    out = collect_marginal_loss_labels(model, calib_ids, make_config(), batch_size=3, target_layers={0, 2})
    d = out["delta_loss"]
    assert len(d) == 14 * 2 == 28
    layer_of = {p["name"]: p["layer_idx"] for p in discover_target_projections(model)}
    assert {layer_of[n] for n, _ in d} == {0, 2}
    assert out["base_loss_4bit"] == pytest.approx(labels["base_loss_4bit"], abs=1e-9)
    for k, v in d.items():
        assert v == pytest.approx(labels["delta_loss"][k], abs=1e-9)


def test_label_matches_manual_swap_and_batching(model, calib_ids, labels):
    """Independent re-implementation: all projections at 4 bit, swap one to b bits, full-batch CE."""
    target = "model.layers.1.mlp.down_proj"
    m = copy.deepcopy(model)

    def loss(mm):
        with torch.no_grad():
            logits = mm(calib_ids).logits[:, :-1]
            return float(F.cross_entropy(logits.reshape(-1, 256), calib_ids[:, 1:].reshape(-1)))

    projs = discover_target_projections(m)
    with torch.no_grad():
        for p in projs:
            p["module"].weight.copy_(quantize_weight_rtn(p["module"].weight, 4, 64))
        base = loss(m)
        mod = m.get_submodule(target)
        w4 = mod.weight.detach().clone()
        for b in (2, 3):
            mod.weight.copy_(quantize_weight_rtn(model.get_submodule(target).weight, b, 64))
            expected = loss(m) - base
            mod.weight.copy_(w4)
            assert labels["delta_loss"][(target, b)] == pytest.approx(expected, abs=1e-5)
    assert labels["base_loss_4bit"] == pytest.approx(base, abs=1e-5)


# ------------------------------------------------------------------ (d) weight restoration
def test_weights_bit_identical_after_label_collection(model, calib_ids):
    snap = snapshot(model)
    collect_marginal_loss_labels(model, calib_ids, make_config(), batch_size=4)
    assert_identical(model, snap)
    collect_marginal_loss_labels(model, calib_ids, make_config(), batch_size=4, target_layers={1})
    assert_identical(model, snap)


def test_weights_restored_when_evaluation_fails(model, calib_ids):
    snap = snapshot(model)
    bad_ids = torch.full((2, 8), 10_000, dtype=torch.long)  # out-of-range token ids -> embedding error
    with pytest.raises(Exception):
        collect_marginal_loss_labels(model, bad_ids, make_config())
    assert_identical(model, snap)


def test_training_mode_is_preserved(model, calib_ids, encoder):
    model.train()
    try:
        collect_marginal_loss_labels(model, calib_ids, make_config(), target_layers={0})
        collect_features_single_pass(model, calib_ids, make_config(), encoder)
        assert model.training
    finally:
        model.eval()


# ------------------------------------------------------------------ data utils (offline)
class FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[ord(c) % 256 for c in text]], dtype=torch.long)
        return SimpleNamespace(input_ids=ids)


def fake_loader(dataset_name, role):
    base = "the quick brown fox jumps over the lazy dog 0123456789 " * (200 if role == "calib" else 40)
    return base + ("EVAL" if role == "eval" else "")


def test_calibration_batches_shape_dtype_determinism():
    tok = FakeTokenizer()
    a = get_calibration_batches(tok, n_seqs=6, seq_len=100, seed=3, text_loader=fake_loader)
    b = get_calibration_batches(tok, n_seqs=6, seq_len=100, seed=3, text_loader=fake_loader)
    c = get_calibration_batches(tok, n_seqs=6, seq_len=100, seed=4, text_loader=fake_loader)
    assert a.shape == (6, 100) and a.dtype == torch.long
    assert torch.equal(a, b) and not torch.equal(a, c)
    stream = tok(fake_loader("wikitext2", "calib")).input_ids[0]
    flat = stream.tolist()
    assert all(any(flat[s : s + 100] == row for s in range(len(flat) - 99)) for row in a.tolist())


def test_eval_tokens_are_non_overlapping_chunks():
    tok = FakeTokenizer()
    out = get_eval_tokens(tok, seq_len=128, text_loader=fake_loader)
    stream = tok(fake_loader("wikitext2", "eval")).input_ids[0]
    n = stream.numel() // 128
    assert out.shape == (n, 128) and out.dtype == torch.long
    assert torch.equal(out.reshape(-1), stream[: n * 128])


def test_data_utils_validation():
    tok = FakeTokenizer()
    with pytest.raises(ValueError):
        get_calibration_batches(tok, dataset_name="nope")
    with pytest.raises(ValueError):
        get_eval_tokens(tok, seq_len=10**6, text_loader=fake_loader)


def test_stream_cache_tokenizes_once(monkeypatch):
    import hds_quant.data_utils as du
    tok = FakeTokenizer()
    load_count = 0

    def mock_load(dataset_name, role):
        nonlocal load_count
        load_count += 1
        return "the quick brown fox jumps over the lazy dog " * 50

    monkeypatch.setattr(du, "_load_text", mock_load)

    # First call - populates cache
    b1 = du.get_calibration_batches(tok, n_seqs=2, seq_len=16, seed=42)
    assert load_count == 1

    # Second call with same tokenizer, different seed - hits cache
    b2 = du.get_calibration_batches(tok, n_seqs=2, seq_len=16, seed=100)
    assert load_count == 1
    assert not torch.equal(b1, b2)

    # Third call with seed 42 - reproduces b1 exactly
    b3 = du.get_calibration_batches(tok, n_seqs=2, seq_len=16, seed=42)
    assert load_count == 1
    assert torch.equal(b1, b3)

