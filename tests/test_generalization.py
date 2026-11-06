"""Frozen C4 selection, distinct allocation bits and independent experiment manifests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hds_quant.generalization_data import select_c4_texts, prepare_tokens, prepare_dataset
from hds_quant.run_generalization import run_config, validate_spec
import hds_quant.run_limited_labels as limited
from scripts.summarize_limited_labels import summarize

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "generalization"


class Tokenizer:
    def __call__(self, text, **kw):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 250 for c in text]]))


def test_c4_disjoint_document_ranges_and_deterministic_tokens():
    records = [{"text": f"DOCUMENT {i} " + "words " * 12} for i in range(7)]
    c, e = select_c4_texts(iter(records), 3, 2)
    assert "DOCUMENT 2" in c and "DOCUMENT 3" not in c
    assert "DOCUMENT 3" in e and "DOCUMENT 2" not in e and "DOCUMENT 5" not in e
    args = SimpleNamespace(eval_chunks=2, eval_seq_len=16, calib_seq_len=16, n_calib_seqs=2, seeds=[42, 100])
    a = prepare_tokens(Tokenizer(), c, e, args, {"dataset_id": "test"})
    b = prepare_tokens(Tokenizer(), c, e, args, {"dataset_id": "test"})
    assert torch.equal(a["eval_ids"], b["eval_ids"])
    assert a["eval_ids"].shape == (2, 16)
    assert a["manifest"]["scored_tokens"] == 30
    assert a["manifest"] == b["manifest"]
    for seed in args.seeds:
        assert torch.equal(a["calib_by_seed"][seed], b["calib_by_seed"][seed])
    with pytest.raises(ValueError, match="range"):
        select_c4_texts(records[:1], 3, 2)
    args.eval_chunks = 100
    with pytest.raises(ValueError, match="fewer"):
        prepare_tokens(Tokenizer(), c, e, args, {})


def test_streaming_loader_uses_pinned_shard(monkeypatch):
    calls = []
    def fake_load(*a, **kw):
        calls.append((a, kw))
        return iter({"text": f"doc {i} " + "words "*20} for i in range(8))
    monkeypatch.setattr("datasets.load_dataset", fake_load)
    args = SimpleNamespace(eval_chunks=2, eval_seq_len=16, calib_seq_len=16, n_calib_seqs=2, seeds=[42])
    result = prepare_dataset(Tokenizer(), args, {"name": "c4", "calibration_docs": 3, "evaluation_docs": 2}, "abc123")
    assert calls[0][0] == ("json",)
    assert calls[0][1]["streaming"] is True
    assert "/resolve/abc123/en/c4-validation.00000-of-00008.json.gz" in calls[0][1]["data_files"]["validation"]
    assert result["manifest"]["calibration_document_range"] == [0, 3]
    assert result["manifest"]["evaluation_document_range"] == [3, 5]


def test_all_configurations_and_policy_validation():
    for path in CONFIGS.glob("*.json"):
        validate_spec(json.loads(path.read_text()))
    spec = json.loads((CONFIGS / "c4_1b.json").read_text())
    spec["allocation_bits"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="protocol"):
        validate_spec(spec)


@pytest.mark.parametrize("preflight", [False, True])
def test_offline_end_to_end_separate_probe_and_allocation_bits(tmp_path, monkeypatch, preflight):
    name = "llama31_8b_preflight.json" if preflight else "c4_1b.json"
    spec = json.loads((CONFIGS / name).read_text())
    spec.update(seeds=[42], k_layers=[2])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    if preflight:
        def forbidden(*a, **kw):
            raise AssertionError("Preflight must not acquire a full label bank")
        monkeypatch.setattr(limited, "run_full_reference", forbidden)
    source = path.read_bytes()
    output = tmp_path / "result"
    result = run_config(path, output, smoke_test=True)
    assert result["complete"]
    assert result["allocation_candidate_bits"] == [2, 3, 4]
    assert result["config"]["candidate_bits"] == [1, 2, 3, 4] or result["config"]["candidate_bits"] == (1, 2, 3, 4)
    assert all(t["probe_label_count"] == 42 for t in result["trials"])
    assert all(b >= 2 for t in result["trials"] + result["references"] for a in t["allocations"].values() for b in a["bit_assignment"].values())
    assert all(r["passed"] for r in result["projection_parity"])
    assert bool(result["references"]) is not preflight
    assert path.read_bytes() == source
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["complete"] and manifest["smoke_test"]
    summarize(output, make_plots=False)
    if preflight:
        assert "PREFLIGHT ONLY" in (output / "SUMMARY.txt").read_text()
    with pytest.raises(FileExistsError):
        run_config(path, output, smoke_test=True)
