"""Resolution parity, label-free LIM, exact payloads and isolated end-to-end results."""
import copy
import csv
import json
from pathlib import Path

import pytest
import torch

from hds_quant.estimators import compute_estimators, compute_selected_estimator
from hds_quant.hds_encoder import HDSEncoder
from hds_quant.lim_baseline import allocate_lim_blocks, collect_lim_scores, METHOD
from hds_quant.run_representation_controls import run_config, validate_spec, METHODS
import hds_quant.run_limited_labels as limited
from scripts.summarize_representation_controls import summarize, validate_result

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "representation_controls"


@pytest.mark.parametrize("binding", ["salience_bucket", "raw_index_strict"])
def test_exact_selective_matches_full_without_codebooks(binding, monkeypatch):
    generator = torch.Generator().manual_seed(19)
    residual = torch.rand(12, 32, generator=generator)
    norms = torch.rand(32, generator=generator)
    full_encoder = HDSEncoder(hd_dim=512, max_coords=64, num_buckets=32, device="cpu")
    expected = compute_estimators(residual, norms, 2, full_encoder, coord_binding=binding)["Coord-Rank-Exact"]
    def forbidden(*args, **kwargs):
        raise AssertionError("Exact must not build or bind hypervector codebooks")
    monkeypatch.setattr(HDSEncoder, "_build_level_codebook", forbidden)
    monkeypatch.setattr(HDSEncoder, "bind", forbidden)
    encoder = HDSEncoder(hd_dim=512, max_coords=64, num_buckets=32, device="cpu", build_codebooks=False)
    actual = compute_selected_estimator(residual, norms, 2, encoder, "Coord-Rank-Exact", binding)
    torch.testing.assert_close(actual["Coord-Rank-Exact"], expected, rtol=0, atol=0)
    assert expected.shape == (32 * 32 + 9,)
    assert encoder.L is encoder.C_raw is encoder.C_bucket is None


class Block(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def forward(self, hidden_states):
        if self.operation == "swap":
            return hidden_states.flip(-1),
        if self.operation == "negate":
            return -hidden_states
        if self.operation == "scale":
            return hidden_states * 2
        return hidden_states


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(s) for s in ("identity", "swap", "negate", "scale")])

    def forward(self, input_ids, **kwargs):
        x = torch.stack((input_ids.float() + 1, torch.ones_like(input_ids).float()), dim=-1)
        for block in self.layers:
            x = block(hidden_states=x)  # Exercise kwargs input hooks.
            x = x[0] if isinstance(x, tuple) else x
        return x


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(7.0))
        self.model = Backbone()

    def forward(self, *args, **kwargs):
        raise AssertionError("LIM needs only the backbone, not the LM head")


def test_lim_known_cosines_token_weighting_and_state_restoration(monkeypatch):
    model = Model().train()
    ids = torch.arange(15).reshape(5, 3)
    before = copy.deepcopy(model.state_dict())
    expected_swap = float(-(2 * (ids.double() + 1) / ((ids.double() + 1).square() + 1)).mean())
    for batch_size in (1, 2, 4, 5):
        scores, counts = collect_lim_scores(model, ids, batch_size)
        assert scores == pytest.approx({0: -1, 1: expected_swap, 2: 1, 3: -1}, abs=1e-7)
        assert counts == [15] * 4
        assert model.training and all(not b._forward_hooks for b in model.model.layers)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")
    monkeypatch.setattr(model.model.layers[2], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        collect_lim_scores(model, ids)
    assert model.training and all(not b._forward_hooks for b in model.model.layers)


def make_meta():
    families = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return [{"name": f"model.layers.{i}.{family}", "layer_idx": i, "family": family,
             "num_params": (j + 1) * 10} for i in range(4) for j, family in enumerate(families)]


def test_lim_ranking_native_block_bits_and_exact_weighted_payload():
    meta = make_meta()
    scores = {0: -0.9, 1: -0.7, 2: -0.5, 3: -0.7}
    a = allocate_lim_blocks(scores, meta, 3.0)
    b = allocate_lim_blocks(scores, meta, 3.5)
    assert a["high_precision_blocks"] == [1, 2]  # Larger LIM wins; tie resolves to earlier block.
    assert b["high_precision_blocks"] == [1, 2, 3]
    assert a["block_ranking"] == [2, 1, 3, 0]
    assert a["achieved_avg_bits"] == 3.0 and b["achieved_avg_bits"] == 3.5
    for allocation in (a, b):
        for index in range(4):
            assert len({allocation["bit_assignment"][p["name"]] for p in meta if p["layer_idx"] == index}) == 1
    with pytest.raises(ValueError, match="exactly meet"):
        allocate_lim_blocks(scores, meta, 3.1)
    different = copy.deepcopy(meta)
    different[0]["num_params"] += 1
    with pytest.raises(ValueError, match="equal"):
        allocate_lim_blocks(scores, different, 3)
    with pytest.raises(ValueError, match="seven"):
        allocate_lim_blocks(scores, meta[1:], 3)
    with pytest.raises(ValueError, match="Nonfinite"):
        allocate_lim_blocks({**scores, 0: float("nan")}, meta, 3)


def test_frozen_configs_match_previous_data_and_keep_new_protocol_separate():
    for path in CONFIGS.glob("*.json"):
        spec = json.loads(path.read_text())
        validate_spec(spec)
        old = json.loads((ROOT / "configs/generalization" / path.name).read_text())
        for key in ("model_id", "model_revision", "dataset", "seeds", "probe_bits", "allocation_bits",
                    "budgets", "calibration", "eval_seq_len", "batch_size"):
            assert spec[key] == old[key]
        assert spec["experiment_id"] != old["experiment_id"] and spec["k_layers"] == [4]
        assert not spec["full_reference"]
        changed = copy.deepcopy(spec)
        changed["external_baseline"]["allocation_bits"] = [2, 3, 4]
        with pytest.raises(ValueError, match="policy"):
            validate_spec(changed)
    assert "Coord-Rank-Exact" not in limited.METHODS  # Historical defaults retained.


def test_offline_control_run_and_paired_tables(tmp_path, monkeypatch):
    spec = json.loads((CONFIGS / "llama31_8b_wikitext.json").read_text())
    spec.update(seeds=[42], k_layers=[2], batch_size=4)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    original = path.read_bytes()
    real_acquire = limited.collect_marginal_loss_labels
    calls = []
    def acquire(*args, **kwargs):
        calls.append(kwargs.get("target_layers"))
        return real_acquire(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Control experiment must not run full-bank acquisition")
    monkeypatch.setattr(limited, "collect_marginal_loss_labels", acquire)
    monkeypatch.setattr(limited, "run_full_reference", forbidden)
    output = tmp_path / "isolated_result"
    result = run_config(path, output, smoke_test=True)
    assert result["complete"] and len(result["trials"]) == 4 and not result["references"]
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2] and len(calls[0]) == 2
    trials = {t["method"]: t for t in result["trials"]}
    assert set(trials) == {*METHODS, METHOD}
    assert trials[METHOD]["probe_label_count"] == 0 and trials[METHOD]["tokens_per_block"] == [512] * 4
    assert "heldout_predictions" not in trials[METHOD]
    assert all(trials[m]["probe_label_count"] == 42 for m in METHODS)
    assert trials["Coord-Rank-Exact"]["encoder_codebook_bytes"] == 0
    assert trials["Coord-Rank-Exact"]["feature_payload_bytes"] > trials["Coord-Rank-Hist"]["feature_payload_bytes"]
    assert all(p["passed"] for p in result["projection_parity"])
    assert path.read_bytes() == original
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["complete"] and manifest["smoke_test"]
    assert "hds_quant/lim_baseline.py" in manifest["code_sha256"]
    assert "scripts/load_hf_env.sh" in manifest["code_sha256"]
    assert all(".env" != name for name in manifest["code_sha256"])
    summary = summarize(output, make_plots=False)
    assert len(summary) == 8
    with (output / "paired_trials.csv").open() as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 10
    first = pairs[0]
    a = trials[first["method_a"]]["allocations"][f"{float(first['budget']):.2f}"]
    b = trials[first["method_b"]]["allocations"][f"{float(first['budget']):.2f}"]
    assert float(first["ppl_a_minus_b"]) == pytest.approx(a["evaluation"]["ppl"] - b["evaluation"]["ppl"])
    assert "different feasible sets" in (output / "SUMMARY.txt").read_text()
    with pytest.raises(FileExistsError):
        run_config(path, output, smoke_test=True)
    bad = copy.deepcopy(result)
    bad["trials"][0]["allocations"]["3.00"]["achieved_avg_bits"] = 2.9
    with pytest.raises(ValueError, match="budget"):
        validate_result(bad)
    bad = copy.deepcopy(result)
    bad["trials"].append(bad["trials"][0])
    with pytest.raises(ValueError, match="duplicate"):
        validate_result(bad)
    bad = copy.deepcopy(result)
    bad["trials"][0]["allocations"]["3.00"]["evaluation"]["seq_token_counts"][0] += 1
    with pytest.raises(ValueError, match="token counts"):
        validate_result(bad)
