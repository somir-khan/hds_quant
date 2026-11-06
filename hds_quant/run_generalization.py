"""Configuration-driven C4 benchmark and larger-model preflight, using shared engines."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

from .generalization_data import prepare_dataset
from .label_collector import discover_target_projections
from .quantizer import quantize_weight_hqq
from .run_limited_labels import METHODS, run as run_limited


def validate_spec(spec):
    required = {"schema", "experiment_id", "purpose", "model_id", "model_revision", "dataset", "seeds",
                "k_layers", "methods", "probe_bits", "allocation_bits", "budgets", "calibration",
                "eval_seq_len", "batch_size", "full_reference"}
    if set(spec) != required or spec["schema"] != "generalization_v1":
        raise ValueError("Unexpected configuration fields or schema")
    if not re.fullmatch(r"[a-z0-9_]+", spec["experiment_id"]):
        raise ValueError("experiment_id must be lowercase letters, numbers, underscores")
    if spec["purpose"] not in ("benchmark", "preflight"):
        raise ValueError("Purpose must be benchmark or preflight")
    ds = spec["dataset"]
    if ds["name"] not in ("c4", "wikitext2") or ds["eval_chunks"] < 1 or not ds["revision"]:
        raise ValueError("Invalid dataset specification")
    if ds["name"] == "c4" and (ds["calibration_docs"] < 1 or ds["evaluation_docs"] < 1):
        raise ValueError("Invalid C4 document ranges")
    if spec["probe_bits"] != [1, 2, 3, 4] or spec["allocation_bits"] != [2, 3, 4]:
        raise ValueError("This protocol fixes probe bits to {1,2,3,4} and allocation bits to {2,3,4}")
    if not spec["methods"] or set(spec["methods"]) - set(METHODS):
        raise ValueError("Invalid methods")
    if not spec["k_layers"] or min(spec["k_layers"]) < 2 or not spec["seeds"]:
        raise ValueError("Invalid K/seeds")
    if not spec["budgets"] or any(v < 2 or v > 4 for v in spec["budgets"]):
        raise ValueError("Budget outside allocation range")
    if min(spec["batch_size"], spec["eval_seq_len"], *spec["calibration"].values()) < 1:
        raise ValueError("Lengths and batch size must be positive")
    if spec["purpose"] == "preflight" and (len(spec["seeds"]) != 1 or spec["k_layers"] != [2] or spec["full_reference"]):
        raise ValueError("Preflight requires one seed, K=2, and no full-bank reference")


@torch.no_grad()
def check_projections(model):
    """Check real q/up projection weights and outputs against packed HQQLinear."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    group = 64 if device.type == "cpu" else 128
    meta = discover_target_projections(model)
    chosen = [next(m for m in meta if m["family"] == family) for family in ("q_proj", "up_proj")]
    rows = []
    for m in chosen:
        layer = m["module"]
        for bits in (3, 4):
            config = BaseQuantizeConfig(nbits=bits, group_size=group, axis=0)
            config["weight_quant_params"]["round_zero"] = False
            packed = HQQLinear(copy.deepcopy(layer), config, compute_dtype=dtype, device=str(device))
            fake = quantize_weight_hqq(layer.weight, bits, group_size=group, axis=0)
            torch.testing.assert_close(fake, packed.dequantize(), rtol=0, atol=0)
            generator = torch.Generator(device=device).manual_seed(17)
            x = torch.randn(1, 4, layer.in_features, device=device, dtype=dtype, generator=generator)
            a, b = torch.nn.functional.linear(x, fake, layer.bias), packed(x)
            torch.testing.assert_close(a, b, rtol=1e-3 if device.type == "cuda" else 1e-6,
                                       atol=1e-3 if device.type == "cuda" else 1e-6)
            rows.append({"projection": m["name"], "shape": list(layer.weight.shape), "bits": bits,
                         "weight_max_abs_difference": float((fake-packed.dequantize()).abs().max()),
                         "output_max_abs_difference": float((a-b).abs().max()), "passed": True})
            del packed, fake, x, a, b
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def run_config(config_path, output_dir, smoke_test=False):
    raw = Path(config_path).read_bytes()
    spec = json.loads(raw)
    validate_spec(spec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment_config.json").write_bytes(raw)
    manifest = {"experiment_id": spec["experiment_id"], "purpose": spec["purpose"],
                "config_sha256": hashlib.sha256(raw).hexdigest(), "smoke_test": smoke_test, "complete": False,
                "packages": {}, "code_sha256": {},
                "execution": {key: os.environ.get(key) for key in
                              ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    for package in ("torch", "transformers", "hqq", "datasets", "scipy", "scikit-learn"):
        manifest["packages"][package] = importlib.metadata.version(package)
    for name in ("run_generalization.py", "generalization_data.py", "run_limited_labels.py", "estimators.py",
                 "readout_cv.py", "label_collector.py", "quantizer.py", "allocator.py", "evaluator.py", "hds_encoder.py"):
        path = Path(__file__).with_name(name)
        manifest["code_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = output / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if smoke_test:
        model_revision = dataset_revision = "synthetic_offline"
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        model_revision = api.model_info(spec["model_id"], revision=spec["model_revision"]).sha
        dataset_id = "allenai/c4" if spec["dataset"]["name"] == "c4" else "Salesforce/wikitext"
        dataset_revision = api.dataset_info(dataset_id, revision=spec["dataset"]["revision"]).sha
    manifest.update(resolved_model_revision=model_revision, resolved_dataset_revision=dataset_revision)
    resolved_spec = copy.deepcopy(spec)
    resolved_spec["model_revision"] = model_revision
    resolved_spec["dataset"]["revision"] = dataset_revision
    (output / "resolved_experiment_config.json").write_text(json.dumps(resolved_spec, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cal = spec["calibration"]
    args = SimpleNamespace(model_id=spec["model_id"], model_revision=model_revision,
        experiment_id=spec["experiment_id"], experiment_purpose=spec["purpose"],
        eval_dataset=spec["dataset"]["name"], seeds=spec["seeds"], k_layers=spec["k_layers"], methods=spec["methods"],
        budgets=spec["budgets"], candidate_bits=spec["probe_bits"], allocation_bits=spec["allocation_bits"],
        n_calib_seqs=cal["n_sequences"], calib_seq_len=cal["seq_len"], max_encoded_tokens=cal["max_encoded_tokens"],
        eval_seq_len=spec["eval_seq_len"], eval_chunks=spec["dataset"]["eval_chunks"], batch_size=spec["batch_size"],
        skip_full_reference=not spec["full_reference"], output_dir=str(output), smoke_test=smoke_test)
    print(f"Experiment {spec['experiment_id']}: {spec['purpose']}; fixed probes={args.candidate_bits}, allocations={args.allocation_bits}", flush=True)
    def data_preparer(tokenizer):
        prepared = prepare_dataset(tokenizer, args, spec["dataset"], dataset_revision)
        (output / "data_manifest.json").write_text(json.dumps(prepared["manifest"], indent=2), encoding="utf-8")
        return prepared
    result = run_limited(args, data_preparer=data_preparer, model_check=check_projections)
    manifest.update(complete=True, evaluated_chunks=result["evaluation_chunks"],
                    model_resolved_revision=result["model_resolved_revision"],
                    evaluation_token_sha256=result["evaluation_token_sha256"],
                    projection_parity=result["projection_parity"])
    if "data_manifest" in result:
        (output / "data_manifest.json").write_text(json.dumps(result["data_manifest"], indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    run_config(args.config, args.output_dir, args.smoke_test)
