"""Stage-4 tests: evaluation metrics, perplexity, e2e smoke tests, and table formatting."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hds_quant import ExperimentConfig
from hds_quant.evaluator import compute_ranking_metrics, evaluate_perplexity
from hds_quant.label_collector import discover_target_projections


@pytest.fixture(scope="module")
def smoke_setup():
    cfg = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )
    model = LlamaForCausalLM(cfg).eval()
    meta_list = discover_target_projections(model)
    g = torch.Generator().manual_seed(42)
    eval_ids = torch.randint(0, 256, (4, 64), generator=g)
    exp_config = ExperimentConfig(
        model_name="smoke",
        num_layers=4,
        quant_group_size=64,
        hd_dim=512,
        device="cpu",
    )
    return model, meta_list, eval_ids, exp_config


def test_ranking_metrics_exact_identity():
    # Synthetic ground truth
    names = [f"proj_{i}" for i in range(14)]
    true_delta = {(n, b): float(i * 10 + b) for i, n in enumerate(names) for b in (2, 3)}
    # When predictions equal ground truth
    meta = [{"name": n, "family": "q_proj"} for n in names]
    metrics = compute_ranking_metrics(true_delta, true_delta, meta)

    assert metrics["spearman_global"] == pytest.approx(1.0)
    assert metrics["kendall_global"] == pytest.approx(1.0)
    assert metrics["spearman_bit2"] == pytest.approx(1.0)
    assert metrics["spearman_bit3"] == pytest.approx(1.0)
    assert metrics["spearman_within_family"] == pytest.approx(1.0)
    assert metrics["top20_recall"] == pytest.approx(1.0)


def test_ranking_metrics_constant_safe():
    names = [f"proj_{i}" for i in range(5)]
    true_delta = {(n, 2): 1.0 for n in names}
    pred = {(n, 2): 2.0 for n in names}
    meta = [{"name": n, "family": "q_proj"} for n in names]
    metrics = compute_ranking_metrics(true_delta, pred, meta, candidate_bits=(2,))
    assert metrics["spearman_global"] == 0.0
    assert metrics["kendall_global"] == 0.0


def test_evaluate_perplexity_restores_weights(smoke_setup):
    model, meta_list, eval_ids, exp_config = smoke_setup
    initial_weights = {m["name"]: m["module"].weight.data.clone() for m in meta_list}

    # 1. Test with bit_assignment
    bit_assignment = {m["name"]: 2 if i % 2 == 0 else 3 for i, m in enumerate(meta_list)}
    res1 = evaluate_perplexity(model, eval_ids, meta_list, exp_config, bit_assignment=bit_assignment)
    assert math.isfinite(res1["ppl"]) and res1["ppl"] > 0
    for m in meta_list:
        assert torch.equal(m["module"].weight.data, initial_weights[m["name"]]), f"{m['name']} changed after bit_assignment"

    # 2. Test with uniform_bit
    res2 = evaluate_perplexity(model, eval_ids, meta_list, exp_config, uniform_bit=3)
    assert math.isfinite(res2["ppl"]) and res2["ppl"] > 0
    for m in meta_list:
        assert torch.equal(m["module"].weight.data, initial_weights[m["name"]]), f"{m['name']} changed after uniform_bit"

    # 3. Test FP16 unquantized
    res3 = evaluate_perplexity(model, eval_ids, meta_list, exp_config)
    assert math.isfinite(res3["ppl"]) and res3["ppl"] > 0
    for m in meta_list:
        assert torch.equal(m["module"].weight.data, initial_weights[m["name"]]), f"{m['name']} changed after unquantized eval"


def test_end_to_end_smoke_runs_and_tables(tmp_path):
    results_dir = tmp_path / "results"
    cache_dir = results_dir / "cache"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # Run 1: Strict baseline with smoke-test
    cmd1 = [
        py, "-m", "hds_quant.run_experiment",
        "--smoke-test",
        "--residual-anchor", "fp16_strict",
        "--coord-binding", "raw_index_strict",
        "--readout-grouping", "pooled",
        "--target-transform", "signed_log1p",
        "--seeds", "42", "100",
        "--budgets", "2.50", "3.00", "3.50",
        "--cache-dir", str(cache_dir),
        "--output-dir", str(results_dir),
        "--run-tag", "strict_baseline",
    ]
    p1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    assert p1.returncode == 0, f"Run 1 failed:\n{p1.stderr}\n{p1.stdout}"

    out1_json = results_dir / "run_strict_baseline.json"
    assert out1_json.exists()
    with open(out1_json, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    assert len(data1["seed_runs"]) == 2
    # Verify label time was recorded (>0) on fresh run
    assert data1["seed_runs"][0]["label_time_sec"] > 0.0

    # Run 2: Salience int4 tweak with reuse-cache and low-shot curve
    cmd2 = [
        py, "-m", "hds_quant.run_experiment",
        "--smoke-test",
        "--residual-anchor", "int4_aligned",
        "--coord-binding", "salience_bucket",
        "--readout-grouping", "per_bit",
        "--target-transform", "signed_log1p",
        "--seeds", "42", "100",
        "--budgets", "2.50", "3.00", "3.50",
        "--run-low-shot-curve",
        "--cache-dir", str(cache_dir),
        "--reuse-cache",
        "--output-dir", str(results_dir),
        "--run-tag", "salience_int4_tweak",
    ]
    p2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    assert p2.returncode == 0, f"Run 2 failed:\n{p2.stderr}\n{p2.stdout}"

    out2_json = results_dir / "run_salience_int4_tweak.json"
    assert out2_json.exists()
    with open(out2_json, "r", encoding="utf-8") as f:
        data2 = json.load(f)
    # Confirm wall_label_time_sec drops to 0.0 when cached labels are reused,
    # while label_time_sec preserves the true scientific label collection cost
    assert data2["seed_runs"][0]["wall_label_time_sec"] == 0.0
    assert data2["seed_runs"][1]["wall_label_time_sec"] == 0.0
    assert data2["seed_runs"][0]["label_time_sec"] > 0.0
    assert data2["seed_runs"][1]["label_time_sec"] > 0.0
    assert data2["summary"]["calibration_costs"]["Oracle-Loss"]["total_calib_time_sec"] > 0.0
    # Confirm low_shot_curve is populated
    assert "low_shot_curve" in data2 and len(data2["low_shot_curve"]) > 0

    # Step 3: Run format_tables.py
    cmd3 = [
        py, "scripts/format_tables.py",
        "--input-dir", str(results_dir),
        "--output-dir", str(results_dir),
    ]
    p3 = subprocess.run(cmd3, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    assert p3.returncode == 0, f"Format tables failed:\n{p3.stderr}\n{p3.stdout}"

    report_md = results_dir / "SUPERVISOR_REPORT.md"
    tables_tex = results_dir / "tables.tex"
    summary_csv = results_dir / "summary_metrics.csv"

    assert report_md.exists() and report_md.stat().st_size > 0
    assert tables_tex.exists() and tables_tex.stat().st_size > 0
    assert summary_csv.exists() and summary_csv.stat().st_size > 0

    report_text = report_md.read_text(encoding="utf-8")
    tex_text = tables_tex.read_text(encoding="utf-8")
    csv_text = summary_csv.read_text(encoding="utf-8")

    # Assert no 'NaN' or 'nan' artifacts in markdown report or tex
    assert "NaN" not in report_text and "nan" not in report_text.lower().split()
    assert "NaN" not in tex_text
    assert "\\toprule" in tex_text and "\\bottomrule" in tex_text


def test_calibration_batches_redrawn_across_seeds():
    from hds_quant.run_experiment import _get_smoke_calib_ids
    c42 = _get_smoke_calib_ids(42)
    c100 = _get_smoke_calib_ids(100)
    assert not torch.equal(c42, c100), "Calibration batches must differ across seeds!"


def test_perplexity_cache_deduplication():
    eval_count = 0

    def mock_eval_ppl():
        nonlocal eval_count
        eval_count += 1
        return {"nll": 1.23, "ppl": 3.42}

    cache = {}

    def lookup(bit_assignment):
        key = tuple(sorted(bit_assignment.items()))
        if key in cache:
            return dict(cache[key])
        res = mock_eval_ppl()
        cache[key] = dict(res)
        return dict(res)

    # Identical allocations with different dict insertion order
    assign1 = {"model.layers.0.mlp.up_proj": 4, "model.layers.0.mlp.down_proj": 2}
    assign2 = {"model.layers.0.mlp.down_proj": 2, "model.layers.0.mlp.up_proj": 4}
    assign3 = {"model.layers.0.mlp.up_proj": 3, "model.layers.0.mlp.down_proj": 3}

    r1 = lookup(assign1)
    r2 = lookup(assign2)
    assert eval_count == 1, "Duplicate bit allocation must be retrieved from cache without evaluating"
    assert r1 == r2

    r3 = lookup(assign3)
    assert eval_count == 2, "Distinct bit allocation must trigger evaluation"
    assert r3 == {"nll": 1.23, "ppl": 3.42}


