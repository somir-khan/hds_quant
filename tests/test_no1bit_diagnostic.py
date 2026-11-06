"""Allocation-only intervention must preserve saved predictions and exact payload."""
import copy
import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from hds_quant.config import ExperimentConfig
from hds_quant.evaluator import evaluate_perplexity
from hds_quant.label_collector import discover_target_projections
from hds_quant.quantizer import get_effective_hqq_config
from hds_quant.run_experiment import _build_smoke_model
import hds_quant.run_no1bit_diagnostic as diagnostic


@pytest.fixture(scope="module")
def source():
    model, ids = _build_smoke_model()
    meta = discover_target_projections(model)
    cfg = ExperimentConfig(model_name="synthetic", num_layers=4, hd_dim=512, device="cpu",
        quantizer="hqq", hqq_axis=0, quant_group_size=64, candidate_bits=(1, 2, 3, 4))
    raw = {(m["name"], b): (4-b) * (0.01 + i / 10000)
           for i, m in enumerate(meta) for b in cfg.marginal_bits}
    trial = {"seed": 100, "method": "HDS-Hybrid", "k_layers": 2, "acquired_layers": [0, 2],
        "measured_labels": {f"{m['name']}|{b}": raw[(m["name"], b)] for m in meta if m["layer_idx"] in (0, 2) for b in cfg.marginal_bits},
        "heldout_predictions": {f"{m['name']}|{b}": raw[(m["name"], b)] for m in meta if m["layer_idx"] not in (0, 2) for b in cfg.marginal_bits}}
    costs = diagnostic.reconstruct_costs(trial, meta, cfg)
    # gate/up/down have equal numbers of weights: (1,4,4) preserves the payload
    # of (3,3,3). This isolates payload correctness without expensive fitting.
    original = {m["name"]: 3 for m in meta}
    for m in meta:
        if m["layer_idx"] == 0 and m["family"] in ("gate_proj", "up_proj", "down_proj"):
            original[m["name"]] = 1 if m["family"] == "gate_proj" else 4
    def allocation(assignment):
        return {"bit_assignment": assignment, "achieved_avg_bits": 3.0,
            "total_predicted_cost": sum(costs[(n, b)] for n, b in assignment.items()),
            "bit_counts": dict(Counter(assignment.values())), "success": True, "payload_lower_bound": 3.0,
            "evaluation": evaluate_perplexity(model, ids, meta, cfg, bit_assignment=assignment, batch_size=4)}
    trial["allocations"] = {"3.00": allocation(original)}
    control = copy.deepcopy(trial)
    control["method"] = "Coord-Rank-Hist"
    control["allocations"] = {"3.00": allocation({m["name"]: 3 for m in meta})}
    doc = {"schema": "true_limited_labels_v1", "complete": True, "config": cfg.to_dict(),
        "hqq_config": get_effective_hqq_config(64, "cpu", next(model.parameters()).dtype, axis=0),
        "args": {"smoke_test": True, "batch_size": 4, "model_id": "synthetic", "eval_dataset": "wikitext2", "eval_seq_len": 128},
        "trials": [trial, control], "uniform_baselines": {
            "FP16": evaluate_perplexity(model, ids, meta, cfg, batch_size=4),
            "Uniform-3bit": evaluate_perplexity(model, ids, meta, cfg, uniform_bit=3, batch_size=4)}}
    return doc, cfg, meta


def test_exact_payload_and_no_mutation(source):
    doc, cfg, meta = source
    original = copy.deepcopy(doc)
    trial = doc["trials"][0]
    plan = diagnostic.plan_intervention(trial, "3.00", meta, cfg)
    assert len(plan["original_one_bit_projections"]) == 1
    assert all(b in (2, 3, 4) for b in plan["restricted_allocation"]["bit_assignment"].values())
    assert diagnostic.payload(trial["allocations"]["3.00"]["bit_assignment"], meta) == plan["matched_integer_payload"]
    assert plan["changed_projections"]
    assert doc == original


def test_unchanged_control_does_not_resolve(source, monkeypatch):
    doc, cfg, meta = source
    def forbidden(*a, **kw):
        raise AssertionError("Unchanged control must not be re-solved")
    monkeypatch.setattr(diagnostic, "solve_bit_allocation_ilp", forbidden)
    plan = diagnostic.plan_intervention(doc["trials"][1], "3.00", meta, cfg)
    assert plan["changed_projections"] == []
    assert plan["restricted_allocation"]["bit_assignment"] == doc["trials"][1]["allocations"]["3.00"]["bit_assignment"]


def test_corrupt_costs_or_missing_prediction_rejected(source):
    doc, cfg, meta = source
    trial = copy.deepcopy(doc["trials"][0])
    trial["allocations"]["3.00"]["total_predicted_cost"] += 1
    with pytest.raises(ValueError, match="objective"):
        diagnostic.plan_intervention(trial, "3.00", meta, cfg)
    trial = copy.deepcopy(doc["trials"][0])
    trial["heldout_predictions"].pop(next(iter(trial["heldout_predictions"])))
    with pytest.raises(ValueError, match="cover"):
        diagnostic.reconstruct_costs(trial, meta, cfg)


def test_complete_replay_run_and_provenance(source, tmp_path):
    doc, _, _ = source
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(doc), encoding="utf-8")
    before = source_path.read_bytes()
    args = SimpleNamespace(source=str(source_path), output_dir=str(tmp_path / "diagnostic"),
        seeds=[100], k_layers=[2], methods=None, budgets=[3.0], nll_atol=1e-6, replay_all=True)
    report = diagnostic.run(args)
    assert report["complete"] and len(report["comparisons"]) == 2
    assert report["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert source_path.read_bytes() == before
    assert all(r["original_replay"]["passed"] for r in report["comparisons"])
    assert report["comparisons"][0]["restricted_evaluation_origin"] == "new_evaluation"
    assert report["comparisons"][1]["restricted_evaluation_origin"] == "replayed_unchanged_assignment"
    assert (tmp_path / "diagnostic" / "comparisons.csv").is_file()
    with pytest.raises(FileExistsError):
        diagnostic.run(args)


def test_environment_replay_mismatch_stops(source, tmp_path):
    doc = copy.deepcopy(source[0])
    doc["uniform_baselines"]["FP16"]["nll"] += 0.1
    path = tmp_path / "source.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    args = SimpleNamespace(source=str(path), output_dir=str(tmp_path / "diagnostic"),
        seeds=None, k_layers=None, methods=None, budgets=None, nll_atol=1e-6, replay_all=True)
    with pytest.raises(ValueError, match="replay failed"):
        diagnostic.run(args)
    partial = json.loads((tmp_path / "diagnostic" / "diagnostic.json").read_text())
    assert not partial["complete"] and not partial["comparisons"]
