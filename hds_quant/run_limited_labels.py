"""True K-block acquisition with independent, directly timed method executions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .allocator import _FEASIBLE_PAYLOAD_CACHE, solve_bit_allocation_ilp
from .config import ExperimentConfig
from .data_utils import get_calibration_batches, get_eval_tokens
from .evaluator import compute_ranking_metrics, evaluate_perplexity
from .hds_encoder import HDSEncoder
from .label_collector import collect_features_single_pass, collect_marginal_loss_labels, discover_target_projections
from .quantizer import get_effective_hqq_config
from .readout_cv import _build_alloc_costs, evaluate_fixed_k_layer_acquisition, select_stratified_k_layers
from .run_experiment import _build_smoke_model, _get_smoke_calib_ids, _pair_dict_to_json

METHODS = ("HDS-Hybrid", "Coord-Rank-Hist", "Scalar-MSE-Ridge")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def storage_bytes(value):
    """Numeric feature payload only: tensors + scalar values, not Python-object overhead."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(storage_bytes(v) for v in value.values())
    if isinstance(value, (float, int)):
        return 8
    return 0


def memory_start(device):
    gc.collect()
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def memory_finish(device, baseline):
    if baseline is None:
        return {"peak_cuda_allocated_bytes": None, "peak_cuda_increment_bytes": None}
    peak = torch.cuda.max_memory_allocated(device)
    return {"peak_cuda_allocated_bytes": peak, "peak_cuda_increment_bytes": peak - baseline}


def run_trial(model, calib_ids, eval_ids, cfg, method, k, budgets, seed, batch_size):
    """No label or feature caches: everything needed by this method is executed."""
    device = next(model.parameters()).device
    meta = discover_target_projections(model)
    _FEASIBLE_PAYLOAD_CACHE.clear()
    memory_baseline = memory_start(device)
    sync(device)
    start = time.perf_counter()
    selected = select_stratified_k_layers(sorted({m["layer_idx"] for m in meta}), k, seed)
    labels = collect_marginal_loss_labels(model, calib_ids, cfg, batch_size=batch_size,
                                         target_layers=set(selected))
    expected = {(m["name"], bit) for m in meta if m["layer_idx"] in selected for bit in cfg.marginal_bits}
    if set(labels["delta_loss"]) != expected:
        raise RuntimeError("Acquisition included missing or unselected probe labels")
    sync(device)
    labels_done = time.perf_counter()
    encoder = HDSEncoder(hd_dim=cfg.hd_dim, num_levels=cfg.num_levels, max_coords=cfg.max_coords,
                         num_buckets=cfg.num_buckets, seed=seed, device=device,
                         build_codebooks=method == "HDS-Hybrid")
    features = collect_features_single_pass(model, calib_ids, cfg, encoder,
                                           batch_size=batch_size, feature_estimator=method)
    sync(device)
    features_done = time.perf_counter()
    fit = evaluate_fixed_k_layer_acquisition(features["features"], labels["delta_loss"], meta,
                                            cfg, k_layers=k, seed=seed, estimator_names=(method,))
    sync(device)
    fit_done = time.perf_counter()
    allocations = {}
    for budget in budgets:
        t0 = time.perf_counter()
        allocation = solve_bit_allocation_ilp(fit["alloc_costs"][method], meta, budget,
                                             candidate_bits=cfg.candidate_bits)
        sync(device)
        elapsed = time.perf_counter() - t0
        if not allocation["success"]:
            raise RuntimeError(f"Allocation failed: {method}, K={k}, budget={budget}")
        allocations[f"{budget:.2f}"] = dict(allocation, allocation_sec=elapsed,
                                            calibration_to_budget_sec=fit_done-start+elapsed)
    sync(device)
    wall = time.perf_counter() - start
    memory = memory_finish(device, memory_baseline)
    result = {"seed": seed, "method": method, "k_layers": k, "acquired_layers": selected,
              "probe_label_count": len(expected), "label_acquisition_sec": labels_done-start,
              "feature_extraction_sec": features_done-labels_done, "fit_sec": fit_done-features_done,
              "calibration_shared_sec": fit_done-start, "calibration_wall_sec": wall,
              "feature_payload_bytes": storage_bytes(features["features"]),
              "encoder_codebook_bytes": storage_bytes({n: getattr(encoder, n) for n in ("L", "C_raw", "C_bucket")}),
              "heldout_predictions": _pair_dict_to_json(fit["heldout_predictions"][method]),
              "measured_labels": _pair_dict_to_json(labels["delta_loss"]),
              "allocations": allocations, **memory}
    # Drop calibration objects before measuring held-out evaluation.
    del features, fit, encoder, labels
    gc.collect()
    for allocation in allocations.values():
        allocation["evaluation"] = evaluate_perplexity(model, eval_ids, meta, cfg,
            bit_assignment=allocation["bit_assignment"], batch_size=batch_size)
    return result


def run_full_reference(model, calib_ids, eval_ids, cfg, budgets, seed, batch_size):
    """Measured-loss baseline; no unnecessary feature extraction or regression."""
    device = next(model.parameters()).device
    meta = discover_target_projections(model)
    _FEASIBLE_PAYLOAD_CACHE.clear()
    baseline = memory_start(device)
    sync(device)
    start = time.perf_counter()
    labels = collect_marginal_loss_labels(model, calib_ids, cfg, batch_size=batch_size)
    costs = _build_alloc_costs({"Oracle-Loss": labels["delta_loss"]}, meta, ("Oracle-Loss",),
                              marginal_bits=cfg.marginal_bits, reference_bit=cfg.reference_bit)["Oracle-Loss"]
    sync(device)
    shared = time.perf_counter() - start
    allocations = {}
    for budget in budgets:
        t0 = time.perf_counter()
        a = solve_bit_allocation_ilp(costs, meta, budget, candidate_bits=cfg.candidate_bits)
        sync(device)
        elapsed = time.perf_counter() - t0
        if not a["success"]:
            raise RuntimeError("Measured-loss reference allocation failed")
        allocations[f"{budget:.2f}"] = dict(a, allocation_sec=elapsed, calibration_to_budget_sec=shared+elapsed)
    sync(device)
    result = {"seed": seed, "method": "Oracle-Loss", "k_layers": cfg.num_layers,
              "calibration_shared_sec": shared, "calibration_wall_sec": time.perf_counter()-start,
              "probe_label_count": len(labels["delta_loss"]), "feature_payload_bytes": 0,
              "encoder_codebook_bytes": 0, "allocations": allocations, **memory_finish(device, baseline)}
    for a in allocations.values():
        a["evaluation"] = evaluate_perplexity(model, eval_ids, meta, cfg,
                                             bit_assignment=a["bit_assignment"], batch_size=batch_size)
    return result, labels["delta_loss"]


def run(args):
    if not args.smoke_test and not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA; --smoke-test is available offline")
    if any(k < 2 for k in args.k_layers):
        raise ValueError("Use K >= 2 for layer-grouped regression tuning")
    if not args.k_layers or not args.methods or not args.seeds:
        raise ValueError("K, methods and seeds must be non-empty")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.k_layers)) != len(args.k_layers):
        raise ValueError("Seeds and K values must be unique")
    if 4 not in args.candidate_bits or len(set(args.candidate_bits)) < 2:
        raise ValueError("Candidates must contain reference bit 4 and a lower precision")
    if not args.budgets or any(b < min(args.candidate_bits) or b > 4 for b in args.budgets):
        raise ValueError("Budget outside candidate precision range")
    if args.smoke_test:
        model, eval_ids = _build_smoke_model()
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to("cuda").eval()
        eval_ids = get_eval_tokens(tokenizer, args.eval_dataset, args.eval_seq_len)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    if any(k >= n_layers for k in args.k_layers):
        raise ValueError(f"Every K must be smaller than the model's {n_layers} blocks")
    report = {"schema": "true_limited_labels_v1", "args": vars(args),
              "torch_version": torch.__version__, "trials": [], "references": [], "calibration_token_sha256": {},
              "timing_scope": "Direct synchronized execution; excludes model/data loading, warmup, JSON I/O, and test evaluation. No label/feature caches. Both-budget wall time and individual-budget times are separate.",
              "memory_scope": "CUDA allocated tensors during calibration, including resident model; increment is above trial-start allocation. CPU feature payload excludes object overhead. No inference memory/speed claim."}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def save():
        path = output / "limited_labels.json"
        temp = output / "limited_labels.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    for seed in args.seeds:
        cfg = ExperimentConfig(model_name=args.model_id, num_layers=n_layers, quantizer="hqq",
            hqq_axis=0, hqq_zero_policy="floating", quant_group_size=64 if args.smoke_test else 128,
            candidate_bits=tuple(args.candidate_bits), device=str(device), hd_dim=512 if args.smoke_test else 4096,
            max_encoded_tokens=256 if args.smoke_test else args.max_encoded_tokens, seeds=(seed,),
            calib_num_sequences=8 if args.smoke_test else args.n_calib_seqs,
            calib_seq_len=64 if args.smoke_test else args.calib_seq_len,
            residual_anchor="int4_aligned", coord_binding="salience_bucket", readout_grouping="per_bit",
            target_transform="signed_log1p")
        report["config"] = cfg.to_dict()
        report["config"]["seeds"] = args.seeds
        report["hqq_config"] = get_effective_hqq_config(cfg.quant_group_size, device, next(model.parameters()).dtype, axis=0)
        calib = _get_smoke_calib_ids(seed) if args.smoke_test else get_calibration_batches(
            tokenizer, args.n_calib_seqs, args.calib_seq_len, seed, args.eval_dataset)
        report["calibration_token_sha256"][str(seed)] = hashlib.sha256(calib.numpy().tobytes()).hexdigest()
        # Warm model inference only; no sensitivity labels or features are acquired here.
        with torch.inference_mode():
            warm = model(input_ids=calib[:1].to(device), use_cache=False)
            del warm
        sync(device)
        if "uniform_baselines" not in report:
            meta = discover_target_projections(model)
            report["uniform_baselines"] = {"FP16": evaluate_perplexity(model, eval_ids, meta, cfg, batch_size=args.batch_size)}
            for bit in (3, 4):
                report["uniform_baselines"][f"Uniform-{bit}bit"] = evaluate_perplexity(model, eval_ids, meta, cfg, uniform_bit=bit, batch_size=args.batch_size)
        for k in args.k_layers:
            order = list(args.methods)
            random.Random(seed + k).shuffle(order)
            for method in order:
                print(f"Starting seed={seed} K={k} method={method}; fresh selected-block acquisition", flush=True)
                trial = run_trial(model, calib, eval_ids, cfg, method, k, args.budgets, seed, args.batch_size)
                report["trials"].append(trial)
                save()
                print(f"Done: calibration={trial['calibration_wall_sec']:.2f}s, labels={trial['probe_label_count']}", flush=True)
        # Full labels are collected AFTER all limited-label decisions, and never supplied to them.
        ref, full_labels = run_full_reference(model, calib, eval_ids, cfg, args.budgets, seed, args.batch_size)
        report["references"].append(ref)
        for trial in (t for t in report["trials"] if t["seed"] == seed):
            heldout = {key: value for key, value in full_labels.items()
                       if int(key[0].split("layers.")[1].split(".")[0]) not in trial["acquired_layers"]}
            # Ranking is an offline diagnostic and does not enter allocation or calibration timings.
            preds = {key: trial["heldout_predictions"][f"{key[0]}|{key[1]}"] for key in heldout}
            trial["offline_ranking"] = compute_ranking_metrics(heldout, preds, discover_target_projections(model))
        save()
    report["complete"] = True
    save()
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--eval-dataset", choices=("wikitext2", "c4"), default="wikitext2")
    p.add_argument("--k-layers", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--budgets", type=float, nargs="+", default=[3.0, 3.5])
    p.add_argument("--candidate-bits", type=int, choices=(1, 2, 3, 4), nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--n-calib-seqs", type=int, default=64)
    p.add_argument("--calib-seq-len", type=int, default=1024)
    p.add_argument("--max-encoded-tokens", type=int, default=4096)
    p.add_argument("--eval-seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-dir", default="results_hqq_limited_labels")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
