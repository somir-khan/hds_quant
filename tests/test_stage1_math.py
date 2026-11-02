"""Stage-1 math tests: quantizer, HDS encoder, estimators."""

import math
import time

import numpy as np
import pytest
import torch
from scipy.stats import trim_mean

from hds_quant import ESTIMATOR_NAMES, ExperimentConfig, HDSEncoder, compute_estimators, quantize_weight_rtn

HAS_CUDA = torch.cuda.is_available()
BINDINGS = ("raw_index_strict", "salience_bucket")


@pytest.fixture(scope="module")
def encoder():
    return HDSEncoder(device="cpu", seed=42)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp(min=1e-12))


# ------------------------------------------------------------------ (a) quantizer
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_rtn_discrete_levels_and_no_nan(bits):
    torch.manual_seed(0)
    g = 128
    W = torch.randn(64, 512) * 0.05
    Wq = quantize_weight_rtn(W, bits, group_size=g)
    assert Wq.shape == W.shape and Wq.dtype == W.dtype
    assert not torch.isnan(Wq).any() and not torch.isinf(Wq).any()

    Wg = W.reshape(64, -1, g).double()
    Qg = Wq.reshape(64, -1, g).double()
    scale = (Wg.amax(-1) - Wg.amin(-1)) / (2**bits - 1)
    for r in range(Qg.shape[0]):
        for c in range(Qg.shape[1]):
            vals = Qg[r, c]
            assert torch.unique(vals).numel() <= 2**bits
            # Levels are an integer lattice with spacing `scale`.
            steps = (vals - vals.min()) / scale[r, c]
            assert torch.allclose(steps, steps.round(), atol=1e-3)
            # Reconstruction stays within about one step of the original.
            assert (vals - Wg[r, c]).abs().max() <= 1.001 * scale[r, c]


def test_rtn_error_monotone_in_bits():
    torch.manual_seed(1)
    W = torch.randn(256, 1024) * 0.02
    errs = {b: float((quantize_weight_rtn(W, b) - W).pow(2).mean()) for b in (2, 3, 4)}
    assert errs[4] < errs[3] < errs[2]


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_rtn_zero_range_groups_are_safe(bits):
    W = torch.randn(8, 256) * 0.1
    W[0, :128] = 0.0  # all-zero group
    W[1, :128] = 0.37  # constant non-zero group
    W[2, 128:] = -1.5  # constant negative group
    Wq = quantize_weight_rtn(W, bits)
    assert not torch.isnan(Wq).any() and not torch.isinf(Wq).any()
    assert torch.equal(Wq[0, :128], torch.zeros(128))
    assert torch.allclose(Wq[1, :128], torch.full((128,), 0.37))
    assert torch.allclose(Wq[2, 128:], torch.full((128,), -1.5))
    assert not torch.isnan(quantize_weight_rtn(torch.zeros(4, 128), bits)).any()


def test_rtn_input_validation_and_dtype():
    with pytest.raises(ValueError):
        quantize_weight_rtn(torch.randn(4, 100), 4, group_size=128)
    with pytest.raises(ValueError):
        quantize_weight_rtn(torch.randn(128), 4)
    W = (torch.randn(16, 256) * 0.05).to(torch.bfloat16)
    assert quantize_weight_rtn(W, 3).dtype == torch.bfloat16
    # clip_ratio < 1 still works and never produces NaN
    assert not torch.isnan(quantize_weight_rtn(W.float(), 2, clip_ratio=0.9)).any()


def test_rtn_single_sign_group_keeps_full_grid():
    """All-positive / all-negative groups used to collapse onto a boundary level (zero-point clamped to 0 / qmax)."""
    W = torch.linspace(1.0, 2.0, 128).reshape(1, 128)
    Wq = quantize_weight_rtn(W, 4, group_size=128)
    scale = (2.0 - 1.0) / 15
    assert torch.unique(Wq).numel() == 16
    assert float((Wq - W).abs().max()) <= scale / 2 + 1e-6
    # mirrored: strictly negative group
    Wn = -W
    Wqn = quantize_weight_rtn(Wn, 4, group_size=128)
    assert torch.unique(Wqn).numel() == 16
    assert float((Wqn - Wn).abs().max()) <= scale / 2 + 1e-6
    # two-valued positive group from the review ([1.0, 2.0]) stays exactly representable
    W2 = torch.tensor([[1.0, 2.0] * 64])
    assert torch.allclose(quantize_weight_rtn(W2, 2, group_size=128), W2, atol=1e-6)


def test_rtn_clip_ratio_shrinks_range_around_midpoint():
    W = torch.linspace(1.0, 2.0, 128).reshape(1, 128)
    Wq = quantize_weight_rtn(W, 4, group_size=128, clip_ratio=0.5)
    # range [1, 2] clipped around its midpoint 1.5 -> [1.25, 1.75]
    assert float(Wq.min()) == pytest.approx(1.25, abs=1e-6)
    assert float(Wq.max()) == pytest.approx(1.75, abs=1e-6)
    assert torch.unique(Wq).numel() == 16


# ------------------------------------------------------------------ (b) encoder identity
@pytest.mark.parametrize("binding", BINDINGS)
def test_encode_matches_naive_reference_small(encoder, binding):
    torch.manual_seed(2)
    R = torch.rand(8, 16)
    norms = torch.rand(16) + 0.1
    fast = encoder.encode(R, norms, binding)
    slow = encoder.encode_naive_reference(R, norms, binding)
    assert fast.shape == (4096,)
    assert rel_err(fast, slow) < 1e-5


@pytest.mark.parametrize("binding", BINDINGS)
def test_encode_matches_naive_reference_with_ties(encoder, binding):
    torch.manual_seed(3)
    R = torch.randint(0, 4, (8, 16)).float()  # heavy ties
    norms = torch.randint(0, 3, (16,)).float()  # tied salience too
    assert rel_err(encoder.encode(R, norms, binding), encoder.encode_naive_reference(R, norms, binding)) < 1e-5


@pytest.mark.parametrize("binding", BINDINGS)
def test_encode_matches_naive_reference_medium(encoder, binding):
    torch.manual_seed(4)
    R = torch.rand(16, 600).abs()
    norms = torch.rand(600)
    assert rel_err(encoder.encode(R, norms, binding), encoder.encode_naive_reference(R, norms, binding)) < 1e-5


def test_encode_chunking_is_invariant():
    torch.manual_seed(5)
    R, norms = torch.rand(100, 64), torch.rand(64)
    a = HDSEncoder(device="cpu", token_chunk=100).encode(R, norms, "raw_index_strict")
    b = HDSEncoder(device="cpu", token_chunk=7).encode(R, norms, "raw_index_strict")
    assert rel_err(b, a) < 1e-6


def test_frequency_matrix_rows_sum_to_one(encoder):
    R = torch.rand(32, 100)
    F = encoder.level_frequencies(R)
    assert F.shape == (100, 32)
    assert torch.allclose(F.sum(1), torch.ones(100))
    # every token places exactly #{k : (k * M) // d_out == m} coordinates into level m
    expected = torch.bincount((torch.arange(100) * 32) // 100, minlength=32).float()
    assert torch.allclose(F.sum(0), expected)

def test_encode_speed_budget():
    """SPEC target: encoding a [4096, 8192] residual matrix in <5 ms (GPU only)."""
    R = torch.rand(4096, 8192)
    norms = torch.rand(8192)
    if not HAS_CUDA:
        pytest.skip("5 ms budget applies to CUDA; no GPU available")
    enc = HDSEncoder(device="cuda")
    R = R.cuda()
    norms = norms.cuda()
    # Warmup iterations (both bindings) to stabilize GPU state
    for _ in range(5):
        enc.encode(R, norms, "raw_index_strict")
        enc.encode(R, norms, "salience_bucket")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        enc.encode(R, norms, "raw_index_strict")
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    assert ms < 8.0, f"encode took {ms:.2f} ms"


@pytest.mark.skipif(not HAS_CUDA, reason="requires CUDA")
def test_cuda_matches_cpu():
    R, norms = torch.rand(64, 256), torch.rand(256)
    cpu = HDSEncoder(device="cpu").encode(R, norms, "salience_bucket")
    gpu = HDSEncoder(device="cuda").encode(R, norms, "salience_bucket").cpu()
    assert rel_err(gpu, cpu) < 1e-5


# ------------------------------------------------------------------ (c) level codebook
def test_level_hypervector_properties(encoder):
    L = encoder.L
    M, D = L.shape
    assert (M, D) == (32, 4096)
    assert set(L.unique().tolist()) == {-1.0, 1.0}
    cos = (L @ L.T) / D
    k = D // (2 * (M - 1))
    for m1 in range(M):
        prev = 1.0 + 1e-9
        for dist in range(max(m1, M - 1 - m1) + 1):
            ms = [m for m in (m1 - dist, m1 + dist) if 0 <= m < M]
            sims = [float(cos[m1, m]) for m in ms]
            # strictly decreasing with |m1 - m2|; both sides at equal distance agree
            assert max(sims) - min(sims) < 1e-6
            assert sims[0] < prev - 1e-9 or dist == 0
            prev = sims[0]
            # exact closed form: disjoint flips => cos = 1 - 2 * k * |dm| / D
            assert abs(sims[0] - (1 - 2 * k * dist / D)) < 1e-6
    assert abs(float(cos[0, M - 1])) < 0.01  # L[0] and L[M-1] ~ orthogonal


def test_codebooks_shapes_and_seed_reproducibility():
    a = HDSEncoder(device="cpu", seed=7)
    b = HDSEncoder(device="cpu", seed=7)
    c = HDSEncoder(device="cpu", seed=8)
    assert a.C_raw.shape == (8192, 4096) and a.C_bucket.shape == (512, 4096)
    assert torch.equal(a.L, b.L) and torch.equal(a.C_raw, b.C_raw) and torch.equal(a.C_bucket, b.C_bucket)
    assert not torch.equal(a.L, c.L)
    assert set(a.C_bucket.unique().tolist()) == {-1.0, 1.0}


# ------------------------------------------------------------------ (d) estimators
@pytest.mark.parametrize("d_out", [512, 2048, 8192])
@pytest.mark.parametrize("binding", BINDINGS)
def test_estimators_across_d_out(encoder, d_out, binding):
    torch.manual_seed(d_out)
    T = 128
    R = torch.randn(T, d_out).abs() * (1 + torch.rand(d_out))
    norms = torch.rand(d_out) * 10
    feats = compute_estimators(R, norms, proj_family_idx=3, encoder=encoder, coord_binding=binding)
    assert set(feats) == set(ESTIMATOR_NAMES)
    D = encoder.hd_dim
    shapes = {
        "Ordinal-Hist": (33,),
        "RandProj-JL": (D,),
        "HDS-Rank": (D,),
        "HDS-Mag-Strict": (D + 1,),
        "HDS-Hybrid": (D + 1 + 1 + 7,),
        "Scalar-Meta-Ridge": (1 + 1 + 7,),
        "Coord-Rank-Hist": (16 * 32 + 1 + 1 + 7,),
        "Coord-Rank-Exact": (512 * 32 + 1 + 1 + 7,),
    }
    for name, val in feats.items():
        if name in shapes:
            assert isinstance(val, torch.Tensor) and tuple(val.shape) == shapes[name], name
            assert torch.isfinite(val).all(), name
        else:
            assert isinstance(val, float) and math.isfinite(val), name
    assert math.isclose(feats["Scalar-MSE"], float(R.pow(2).mean()), rel_tol=1e-5)
    assert math.isclose(feats["Scalar-L1"], float(R.mean()), rel_tol=1e-5)
    assert feats["Scalar-TrimmedMean"] <= feats["Scalar-Q99"]
    assert feats["Scalar-WinsorL2"] <= float(R.pow(2).mean().sqrt()) + 1e-6
    assert math.isclose(float(feats["Ordinal-Hist"][:-1].sum()), 1.0, rel_tol=1e-5)
    assert torch.equal(feats["HDS-Mag-Strict"][:D], feats["HDS-Rank"])
    h = feats["HDS-Hybrid"]
    assert torch.equal(h[:D], feats["HDS-Rank"])
    assert float(h[D]) == pytest.approx(float(feats["HDS-Mag-Strict"][D]))
    assert float(h[D + 1]) == pytest.approx(math.log(feats["Scalar-Q99"] + 1e-8), rel=1e-5)
    one_hot = h[D + 2 :]
    assert one_hot.sum() == 1 and one_hot[3] == 1


def test_estimators_match_reference_implementations(encoder):
    torch.manual_seed(11)
    R = torch.randn(64, 512).abs()
    f = compute_estimators(R, torch.rand(512), 0, encoder)
    flat = R.reshape(-1).numpy()
    assert f["Scalar-Q99"] == pytest.approx(float(np.quantile(flat, 0.99)), rel=1e-5)
    assert f["Scalar-TrimmedMean"] == pytest.approx(float(trim_mean(flat, 0.1)), rel=1e-5)
    q = np.quantile(flat, 0.99)
    assert f["Scalar-WinsorL2"] == pytest.approx(float(np.sqrt(np.mean(np.minimum(flat, q) ** 2))), rel=1e-5)
    # HDS-Rank equals the encoder output
    v = encoder.encode(R, torch.ones(512), "raw_index_strict")
    assert rel_err(f["HDS-Rank"], v) < 1e-6
    # JL feature equals "project each token's normalized ranks, then average"
    order = torch.sort(R, dim=1, stable=True).indices
    ranks = torch.sort(order, dim=1).indices.float() / 512
    jl = (ranks @ encoder.jl_matrix(512)).mean(0)
    assert rel_err(f["RandProj-JL"], jl) < 1e-5


def test_estimators_scale_behaviour(encoder):
    torch.manual_seed(12)
    R, norms = torch.rand(64, 512), torch.rand(512)
    a = compute_estimators(R, norms, 1, encoder)
    b = compute_estimators(2 * R, norms, 1, encoder)
    assert b["Scalar-MSE"] == pytest.approx(4 * a["Scalar-MSE"], rel=1e-5)
    assert b["Scalar-Q99"] == pytest.approx(2 * a["Scalar-Q99"], rel=1e-5)
    # rank-based features are invariant to monotone rescaling
    assert torch.allclose(a["HDS-Rank"], b["HDS-Rank"])
    assert torch.allclose(a["RandProj-JL"], b["RandProj-JL"])


def test_estimators_all_zero_residual_is_finite(encoder):
    f = compute_estimators(torch.zeros(16, 512), torch.ones(512), 6, encoder)
    for val in f.values():
        assert np.isfinite(val if isinstance(val, float) else val.numpy()).all()


def test_estimators_reject_bad_inputs(encoder):
    with pytest.raises(ValueError):
        compute_estimators(torch.rand(4, 512), torch.rand(512), 7, encoder)
    with pytest.raises(ValueError):
        compute_estimators(torch.rand(512), torch.rand(512), 0, encoder)


# ------------------------------------------------------------------ non-HDS controls (Fix 6)
@pytest.mark.parametrize("binding", BINDINGS)
def test_control_estimators_share_metadata_and_frequencies(encoder, binding):
    torch.manual_seed(21)
    R = torch.rand(64, 512)
    norms = torch.rand(512) + 0.1
    f = compute_estimators(R, norms, 2, encoder, coord_binding=binding)
    D = encoder.hd_dim
    meta = f["Scalar-Meta-Ridge"]
    assert tuple(meta.shape) == (9,)
    # exactly the HDS-Hybrid scalar + family metadata, without the hypervector
    assert torch.allclose(meta, f["HDS-Hybrid"][D:])
    assert float(meta[0]) == pytest.approx(math.log(f["Scalar-MSE"] + 1e-8), rel=1e-5)
    assert float(meta[1]) == pytest.approx(math.log(f["Scalar-Q99"] + 1e-8), rel=1e-5)
    assert meta[2:].sum() == 1 and meta[2 + 2] == 1

    c = f["Coord-Rank-Hist"]
    assert tuple(c.shape) == (16 * 32 + 9,)
    assert torch.allclose(c[16 * 32 :], meta)
    hist = c[: 16 * 32].reshape(16, 32)
    assert torch.allclose(hist.sum(1), torch.ones(16), atol=1e-5)  # each bucket row is a rank-level distribution


def test_coord_rank_hist_matches_manual_bucket_average(encoder):
    torch.manual_seed(22)
    R = torch.rand(48, 480)
    norms = torch.rand(480) + 0.1
    F = encoder.level_frequencies(R)  # the exact matrix HDS binds, before codebook binding
    d_out = 480

    f_raw = compute_estimators(R, norms, 0, encoder, coord_binding="raw_index_strict")
    expected = torch.zeros(16, 32)
    for bkt in range(16):
        idx = [j for j in range(d_out) if (j * 16) // d_out == bkt]
        expected[bkt] = F[idx].mean(0)
    assert torch.allclose(f_raw["Coord-Rank-Hist"][: 16 * 32].reshape(16, 32), expected, atol=1e-6)

    f_sal = compute_estimators(R, norms, 0, encoder, coord_binding="salience_bucket")
    order = sorted(range(d_out), key=lambda j: (-norms[j].item(), j))
    rank = {j: r for r, j in enumerate(order)}
    expected = torch.zeros(16, 32)
    for bkt in range(16):
        idx = [j for j in range(d_out) if (rank[j] * 16) // d_out == bkt]
        expected[bkt] = F[idx].mean(0)
    assert torch.allclose(f_sal["Coord-Rank-Hist"][: 16 * 32].reshape(16, 32), expected, atol=1e-6)


# ------------------------------------------------------------------ per-estimator timings (Fix 2)
def test_compute_estimators_timings_cover_every_estimator(encoder):
    torch.manual_seed(23)
    R, norms = torch.rand(64, 512), torch.rand(512) + 0.1
    timings = {}
    f_timed = compute_estimators(R, norms, 0, encoder, timings=timings)
    assert set(timings) == set(ESTIMATOR_NAMES) | {"Scalar-MSE-Ridge"}
    assert all(v >= 0.0 for v in timings.values())
    # superset estimators are charged at least what their building blocks cost
    assert timings["HDS-Hybrid"] >= timings["HDS-Mag-Strict"] >= timings["HDS-Rank"]
    assert timings["Coord-Rank-Hist"] >= timings["Scalar-Meta-Ridge"]
    # timings accumulate across calls and never change the features
    before = dict(timings)
    compute_estimators(R, norms, 0, encoder, timings=timings)
    assert all(timings[k] >= before[k] for k in before)
    f_plain = compute_estimators(R, norms, 0, encoder)
    for name in ESTIMATOR_NAMES:
        a, b = f_timed[name], f_plain[name]
        assert (a == b) if isinstance(a, float) else torch.equal(a, b)


# ------------------------------------------------------------------ config
def test_config_validation():
    cfg = ExperimentConfig()
    assert cfg.num_projections == 112 and cfg.marginal_bits == (2, 3)
    with pytest.raises(ValueError):
        ExperimentConfig(residual_anchor="nope")
    with pytest.raises(ValueError):
        ExperimentConfig(coord_binding="nope")
    ExperimentConfig(residual_anchor="int4_aligned", coord_binding="salience_bucket",
                     readout_grouping="per_bit_family", target_transform="signed_log1p")
