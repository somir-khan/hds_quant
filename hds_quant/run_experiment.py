"""Main experiment runner for HDS-Quant."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from .allocator import solve_bit_allocation_ilp
from .config import ExperimentConfig
from .data_utils import get_calibration_batches, get_eval_tokens
from .evaluator import aggregate_seed_stability, compute_ranking_metrics, evaluate_perplexity
from .hds_encoder import HDSEncoder
from .label_collector import (
    collect_features_single_pass,
    collect_marginal_loss_labels,
    discover_target_projections,
)
from .readout_cv import ALL_ESTIMATORS, cross_validate_estimators, evaluate_fixed_k_layer_acquisition

MAIN_ESTIMATORS = [
    "Scalar-MSE",
    "Scalar-WinsorL2",
    "Scalar-Q99",
    "Scalar-MSE-Ridge",
    "Scalar-Meta-Ridge",
    "Ordinal-Hist",
    "RandProj-JL",
    "Coord-Rank-Hist",
    "Coord-Rank-Exact",
    "HDS-Rank",
    "HDS-Mag-Strict",
    "HDS-Hybrid",
    "Oracle-Loss",
    "Oracle-Loss-Raw",
]

# Estimators reported in the fixed-K layer-acquisition (low-shot) curve.
LOW_SHOT_ESTIMATORS = (
    "HDS-Hybrid",
    "HDS-Rank",
    "Ordinal-Hist",
    "Scalar-Meta-Ridge",
    "Coord-Rank-Hist",
    "Coord-Rank-Exact",
)


def _build_smoke_model():
    cfg = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
    )
    # Seed BEFORE construction so separate processes (e.g. Run 1 / Run 2 of the pilot) build identical weights;
    # otherwise the label-cache fingerprint could never match across processes.
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(0)
    eval_ids = torch.randint(0, 256, (4, 128), generator=g)
    return model, eval_ids


def _get_smoke_calib_ids(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (8, 64), generator=g)


def _compute_label_cache_fingerprint(model: nn.Module, calib_ids: torch.Tensor, exp_config: ExperimentConfig) -> str:
    """SHA-256 fingerprint of everything the marginal-loss labels depend on.

    Covers schema version, model name, quantizer settings, calibration token tensor, and complete
    chunked byte contents across all parameters and buffers in the model.
    """
    h = hashlib.sha256()

    def upd(tag: str, value) -> None:
        h.update(tag.encode("utf-8"))
        h.update(repr(value).encode("utf-8"))
        h.update(b"|")

    param = next(iter(model.parameters()), None)
    model_device = param.device if param is not None else calib_ids.device
    model_dtype = param.dtype if param is not None else torch.float32

    upd("cache_schema_version", "v5_complete_chunked_byte_hash")
    upd("model_name", exp_config.model_name)
    upd("quant_group_size", int(exp_config.quant_group_size))
    upd("reference_bit", int(exp_config.reference_bit))
    upd("candidate_bits", tuple(int(b) for b in exp_config.candidate_bits))
    upd("clip_ratio", float(exp_config.clip_ratio))
    if getattr(exp_config, "quantizer", "rtn") != "rtn":
        upd("quantizer", exp_config.quantizer)
        if exp_config.quantizer == "hqq":
            from .quantizer import get_effective_hqq_config
            eff_cfg = get_effective_hqq_config(
                group_size=exp_config.quant_group_size,
                device=model_device,
                dtype=model_dtype,
            )
            upd("hqq_fingerprint_schema", "v2_precision_aware")
            upd("hqq_backend", eff_cfg["backend"])
            upd("hqq_version", eff_cfg["version"])
            upd("hqq_device_type", eff_cfg["device_type"])
            upd("hqq_optimizer_dtype", eff_cfg["optimizer_dtype"])
            upd("hqq_dequantize_compute_dtype", eff_cfg["dequantize_compute_dtype"])
            for k in sorted(eff_cfg.keys()):
                upd(f"hqq_effective_{k}", eff_cfg[k])
    upd("calib_shape", tuple(int(s) for s in calib_ids.shape))
    upd("calib_dtype", str(calib_ids.dtype))
    upd("calib_sha256", hashlib.sha256(calib_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest())

    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and hasattr(model_cfg, "to_dict") and callable(model_cfg.to_dict):
        upd("model_config_json", json.dumps(model_cfg.to_dict(), sort_keys=True, default=str))

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        h.update(name.encode("utf-8"))
        upd(f"{name}_shape", tuple(tensor.shape))
        upd(f"{name}_dtype", str(tensor.dtype))
        t_flat = tensor.detach().reshape(-1)
        chunk_elems = 1_048_576
        for start in range(0, t_flat.numel(), chunk_elems):
            chunk_cpu = t_flat[start : start + chunk_elems].cpu().contiguous()
            h.update(chunk_cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _load_cached_labels(cache_file: Path, fp_hash: str) -> Optional[dict]:
    """Load a label cache; return ``None`` (and warn) if its fingerprint is missing or does not match."""
    cached = torch.load(cache_file, weights_only=False)
    if cached.get("fingerprint") != fp_hash:
        msg = (
            f"Label cache {cache_file} has a missing or mismatching fingerprint "
            f"(cached={cached.get('fingerprint')!r}, expected={fp_hash!r}); ignoring it and recomputing labels."
        )
        print(f"[WARNING] {msg}")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return None
    return cached


# Candidate set of label caches written before 1-bit support; they can be upgraded in place by probing only
# the newly requested bit-widths (see ``_load_or_upgrade_cached_labels``).
_LEGACY_CANDIDATE_BITS = (2, 3, 4)


def _load_or_upgrade_cached_labels(
    cache_file: Path,
    fp_hash: str,
    model: nn.Module,
    calib_ids: torch.Tensor,
    exp_config: ExperimentConfig,
    batch_size: int,
    save_path: Path,
) -> "tuple[Optional[dict], float]":
    """Load a label cache, incrementally upgrading a ``(2, 3, 4)`` cache to the requested ``candidate_bits``.

    Returns ``(cached_or_None, wall_label_time_sec)`` where ``wall_label_time_sec`` is the time actually spent in
    this call probing missing bit-widths (0.0 for a plain cache hit).

    Upgrade rule: if the cache fingerprint does not match the requested config but DOES match the same
    model / calibration tokens / quantizer settings with ``candidate_bits=(2, 3, 4)``, and the request needs
    bit-widths absent from the cached ``delta_loss`` (e.g. ``{1}``), only those bit-widths are probed. The
    cached ``(2, 3)`` labels are reused, probe/label times are summed, and the merged cache is saved under the
    new fingerprint to ``save_path``.
    """
    cached = torch.load(cache_file, weights_only=False)
    if cached.get("fingerprint") == fp_hash:
        return cached, 0.0

    ref = int(exp_config.reference_bit)
    if tuple(exp_config.candidate_bits) != _LEGACY_CANDIDATE_BITS and ref in _LEGACY_CANDIDATE_BITS:
        exp_config_234 = dataclasses.replace(exp_config, candidate_bits=_LEGACY_CANDIDATE_BITS)
        fp_hash_234 = _compute_label_cache_fingerprint(model, calib_ids, exp_config_234)
        if cached.get("fingerprint") == fp_hash_234:
            cached_bits = {int(k[1]) for k in cached["delta_loss"]}
            missing_bits = set(int(b) for b in exp_config.marginal_bits) - cached_bits
            if missing_bits:
                print(
                    f"Upgrading cached labels {cache_file}: reusing bits {sorted(cached_bits)}, "
                    f"probing only missing bits {sorted(missing_bits)}..."
                )
                exp_config_missing = dataclasses.replace(
                    exp_config, candidate_bits=tuple(sorted(missing_bits | {ref}))
                )
                out = collect_marginal_loss_labels(model, calib_ids, exp_config_missing, batch_size=batch_size)
                # Re-reference the new labels to the CACHED 4-bit baseline so all bit-widths share one baseline.
                base_shift = float(out["base_loss_4bit"]) - float(cached["base_loss_4bit"])
                delta_loss = dict(cached["delta_loss"])
                for k, v in out["delta_loss"].items():
                    delta_loss[k] = float(v) + base_shift

                def _prev(key: str) -> float:
                    return float(cached.get(key, 0.0))

                base_overhead = _prev("base_overhead_time_sec")
                restore_time = _prev("restore_time_sec")
                probe_time = float(cached.get("probe_time_sec", _prev("label_time_sec"))) + float(
                    out.get("probe_time_sec", out["label_time_sec"])
                )
                label_time = base_overhead + probe_time + restore_time
                merged = {
                    "delta_loss": delta_loss,
                    "base_loss_4bit": float(cached["base_loss_4bit"]),
                    "label_time_sec": label_time,
                    "base_overhead_time_sec": base_overhead,
                    "probe_time_sec": probe_time,
                    "restore_time_sec": restore_time,
                    "fingerprint": fp_hash,
                    "quantizer": exp_config.quantizer,
                }
                if exp_config.quantizer == "hqq":
                    from .quantizer import get_effective_hqq_config
                    param = next(iter(model.parameters()), None)
                    dev = param.device if param is not None else calib_ids.device
                    dt = param.dtype if param is not None else torch.float32
                    eff_cfg = get_effective_hqq_config(
                        group_size=exp_config.quant_group_size,
                        device=dev,
                        dtype=dt,
                    )
                    merged["hqq_backend"] = eff_cfg["backend"]
                    merged["hqq_version"] = eff_cfg["version"]
                    merged["hqq_config"] = eff_cfg
                torch.save(merged, save_path)
                print(f"Upgraded label cache saved to {save_path} (+{float(out['label_time_sec']):.2f}s).")
                return merged, float(out["label_time_sec"])

    return _load_cached_labels(cache_file, fp_hash), 0.0


def _pair_dict_to_json(d: Dict) -> Dict[str, float]:
    """``{(proj_name, b): v}`` -> ``{"proj_name|b": v}`` (JSON object keys must be strings)."""
    return {f"{name}|{b}": float(v) for (name, b), v in d.items()}


def _sanitize_for_json(obj):
    """Recursively convert NumPy scalars, NaN/Inf floats, and tuples to standard RFC-8259 JSON objects."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="HDS-Quant end-to-end experiment runner.")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--eval-dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--quantizer", type=str, default="rtn", choices=["rtn", "hqq"])
    parser.add_argument("--residual-anchor", type=str, default="fp16_strict", choices=["fp16_strict", "int4_aligned"])
    parser.add_argument("--coord-binding", type=str, default="raw_index_strict", choices=["raw_index_strict", "salience_bucket"])
    parser.add_argument("--readout-grouping", type=str, default="per_bit_family", choices=["pooled", "per_bit", "per_family", "per_bit_family"])
    parser.add_argument("--target-transform", type=str, default="signed_log1p", choices=["raw", "signed_log1p", "rank"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 100, 2026])
    parser.add_argument("--budgets", nargs="+", type=float, default=[2.50, 2.75, 3.00, 3.25, 3.50])
    parser.add_argument(
        "--candidate-bits",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Candidate weight bit-widths (must include the reference bit 4), e.g. `1 2 3 4` to enable 1-bit.",
    )
    parser.add_argument(
        "--low-shot-budgets",
        nargs="+",
        type=float,
        default=None,
        help="Budgets for the low-shot curve. Default: the --budgets entries in {1.5, 2.0, 3.0}, "
        "else the middle budget.",
    )
    parser.add_argument("--n-calib-seqs", type=int, default=64)
    parser.add_argument("--calib-seq-len", type=int, default=1024)
    parser.add_argument("--max-encoded-tokens", type=int, default=4096)
    parser.add_argument("--eval-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--run-low-shot-curve", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="results/cache")
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-tag", type=str, default="default")
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_experiment(args):
    device = "cuda" if torch.cuda.is_available() and not args.smoke_test else "cpu"
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    candidate_bits = tuple(sorted(set(int(b) for b in (getattr(args, "candidate_bits", None) or [2, 3, 4]))))
    if 4 not in candidate_bits or len(candidate_bits) < 2:
        raise ValueError(f"candidate_bits={candidate_bits} must contain the reference bit 4 and at least one lower bit")
    min_b, max_b = min(candidate_bits), max(candidate_bits)
    for b in args.budgets:
        if not (min_b <= float(b) <= max_b):
            raise ValueError(f"Budget {b} is outside feasible candidate_bits range [{min_b}, {max_b}]")
    for b in getattr(args, "low_shot_budgets", None) or []:
        if not (min_b <= float(b) <= max_b):
            raise ValueError(f"Low-shot budget {b} is outside feasible candidate_bits range [{min_b}, {max_b}]")

    if args.smoke_test:
        print("[Smoke Test Mode] Initializing synthetic 4-layer model and token streams on CPU...")
        model, eval_ids = _build_smoke_model()
        tokenizer = None
        num_layers = 4
        quant_group_size = 64
        hd_dim = 512
        seeds = [s for s in args.seeds if s in (42, 100)] or [42, 100]
        smoke_budgets = (2.50, 3.00, 3.50) + ((1.50, 2.00) if 1 in candidate_bits else ())
        budgets = [b for b in args.budgets if b in smoke_budgets] or [2.50, 3.00, 3.50]
        max_encoded_tokens = 256
    else:
        print(f"Loading model {args.model_id} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
        model.eval()
        num_layers = getattr(model.config, "num_hidden_layers", 16)
        quant_group_size = 128
        hd_dim = 4096
        seeds = args.seeds
        budgets = args.budgets
        max_encoded_tokens = args.max_encoded_tokens

        print(f"Preparing {args.eval_dataset} eval token stream...")
        eval_ids = get_eval_tokens(tokenizer, dataset_name=args.eval_dataset, seq_len=args.eval_seq_len)

    # Budgets at which the low-shot (fixed-K layer acquisition) curve is evaluated.
    if getattr(args, "low_shot_budgets", None):
        low_shot_budgets = sorted(set(float(b) for b in args.low_shot_budgets))
    else:
        low_shot_budgets = [b for b in budgets if any(math.isclose(float(b), t) for t in (1.50, 2.00, 3.00))]
        if not low_shot_budgets:
            low_shot_budgets = [sorted(budgets)[len(budgets) // 2]]
        low_shot_budgets = sorted(float(b) for b in low_shot_budgets)

    meta_list = discover_target_projections(model)
    orig_weights_cpu = {m["name"]: m["module"].weight.data.detach().cpu().clone() for m in meta_list}

    # Evaluate uniform baselines ONCE on eval_ids (deterministic & seed-independent)
    print("Evaluating uniform baselines on eval tokens (once)...")
    base_eval_config = ExperimentConfig(
        model_name=args.model_id,
        num_layers=num_layers,
        quantizer=getattr(args, "quantizer", "rtn"),
        quant_group_size=quant_group_size,
        hd_dim=hd_dim,
        candidate_bits=candidate_bits,
        device=device,
    )
    uniform_baselines = {
        "FP16": evaluate_perplexity(model, eval_ids, meta_list, base_eval_config, orig_weights_cpu=orig_weights_cpu, batch_size=args.batch_size),
        "Uniform-4bit": evaluate_perplexity(model, eval_ids, meta_list, base_eval_config, uniform_bit=4, orig_weights_cpu=orig_weights_cpu, batch_size=args.batch_size),
        "Uniform-3bit": evaluate_perplexity(model, eval_ids, meta_list, base_eval_config, uniform_bit=3, orig_weights_cpu=orig_weights_cpu, batch_size=args.batch_size),
        "Uniform-2bit": evaluate_perplexity(model, eval_ids, meta_list, base_eval_config, uniform_bit=2, orig_weights_cpu=orig_weights_cpu, batch_size=args.batch_size),
    }
    if 1 in candidate_bits:
        uniform_baselines["Uniform-1bit"] = evaluate_perplexity(
            model, eval_ids, meta_list, base_eval_config, uniform_bit=1, orig_weights_cpu=orig_weights_cpu, batch_size=args.batch_size
        )

    seed_runs = []
    low_shot_curve_data = {}
    ppl_cache: dict[tuple, dict] = {}
    test_eval_clock = {"sec": 0.0}  # held-out perplexity wall time of the current seed (never calibration cost)

    def _eval_or_lookup_ppl(bit_assignment: dict[str, int], config: ExperimentConfig) -> dict:
        t_eval = time.perf_counter()
        alloc_key = tuple(sorted(bit_assignment.items()))
        if alloc_key in ppl_cache:
            res = dict(ppl_cache[alloc_key])
        else:
            res = evaluate_perplexity(
                model,
                eval_ids,
                meta_list,
                config,
                bit_assignment=bit_assignment,
                orig_weights_cpu=orig_weights_cpu,
                batch_size=args.batch_size,
            )
            ppl_cache[alloc_key] = dict(res)
            res = dict(res)
        test_eval_clock["sec"] += time.perf_counter() - t_eval
        return res

    for seed in seeds:
        print(f"\n================ Running Seed {seed} ================")
        exp_config = ExperimentConfig(
            model_name=args.model_id,
            num_layers=num_layers,
            quantizer=getattr(args, "quantizer", "rtn"),
            quant_group_size=quant_group_size,
            candidate_bits=candidate_bits,
            hd_dim=hd_dim,
            max_encoded_tokens=max_encoded_tokens,
            device=device,
            residual_anchor=args.residual_anchor,
            coord_binding=args.coord_binding,
            readout_grouping=args.readout_grouping,
            target_transform=args.target_transform,
            cv_folds=getattr(args, "cv_folds", 4),
            seeds=(seed,),
        )
        test_eval_clock["sec"] = 0.0

        # Draw calibration batches specific to this seed
        if args.smoke_test:
            calib_ids = _get_smoke_calib_ids(seed)
        else:
            print(f"Drawing calibration batches for seed {seed} from {args.eval_dataset}...")
            calib_ids = get_calibration_batches(
                tokenizer,
                n_seqs=args.n_calib_seqs,
                seq_len=args.calib_seq_len,
                seed=seed,
                dataset_name=args.eval_dataset,
            )

        # 1. Marginal loss labels (cached with a fingerprint of everything they depend on)
        fp_hash = _compute_label_cache_fingerprint(model, calib_ids, exp_config)
        cache_key = f"{args.model_id.replace('/', '_')}_seed{seed}" if not args.smoke_test else f"smoke_seed{seed}"
        q_suffix = f"_{exp_config.quantizer}" if exp_config.quantizer != "rtn" else ""
        primary_cache_file = Path(args.cache_dir) / f"{cache_key}{q_suffix}_labels.pt"
        cache_file = primary_cache_file
        if not cache_file.exists():
            alt_cache = Path(args.cache_dir) / f"labels_seed{seed}{q_suffix}.pt"
            if alt_cache.exists():
                cache_file = alt_cache

        cached = None
        upgraded_wall_label_time_sec = 0.0
        if args.reuse_cache and cache_file.exists():
            print(f"Loading cached marginal loss labels from {cache_file}...")
            cached, upgraded_wall_label_time_sec = _load_or_upgrade_cached_labels(
                cache_file,
                fp_hash,
                model,
                calib_ids,
                exp_config,
                batch_size=args.batch_size,
                save_path=primary_cache_file,
            )

        if cached is not None:
            delta_loss = cached["delta_loss"]
            base_loss_4bit = cached["base_loss_4bit"]
            label_time_sec = float(cached.get("label_time_sec", 0.0))
            base_overhead_time_sec = float(cached.get("base_overhead_time_sec", 0.0))
            probe_time_sec = float(cached.get("probe_time_sec", label_time_sec))
            restore_time_sec = float(
                cached.get("restore_time_sec", max(0.0, label_time_sec - base_overhead_time_sec - probe_time_sec))
            )
            wall_label_time_sec = float(upgraded_wall_label_time_sec)  # >0 only when bits were added incrementally
        else:
            print(f"Collecting marginal loss labels for seed {seed}...")
            label_out = collect_marginal_loss_labels(model, calib_ids, exp_config, batch_size=args.batch_size)
            delta_loss = label_out["delta_loss"]
            base_loss_4bit = label_out["base_loss_4bit"]
            label_time_sec = float(label_out["label_time_sec"])
            base_overhead_time_sec = float(label_out.get("base_overhead_time_sec", 0.0))
            probe_time_sec = float(label_out.get("probe_time_sec", label_time_sec))
            restore_time_sec = float(label_out.get("restore_time_sec", 0.0))
            wall_label_time_sec = label_time_sec
            cache_payload = {
                "delta_loss": delta_loss,
                "base_loss_4bit": base_loss_4bit,
                "label_time_sec": label_time_sec,
                "base_overhead_time_sec": base_overhead_time_sec,
                "probe_time_sec": probe_time_sec,
                "restore_time_sec": restore_time_sec,
                "fingerprint": fp_hash,
                "quantizer": exp_config.quantizer,
            }
            if exp_config.quantizer == "hqq":
                from .quantizer import get_effective_hqq_config
                param = next(iter(model.parameters()), None)
                dev = param.device if param is not None else calib_ids.device
                dt = param.dtype if param is not None else torch.float32
                eff_cfg = get_effective_hqq_config(
                    group_size=exp_config.quant_group_size,
                    device=dev,
                    dtype=dt,
                )
                cache_payload["hqq_backend"] = eff_cfg["backend"]
                cache_payload["hqq_version"] = eff_cfg["version"]
                cache_payload["hqq_config"] = eff_cfg
            torch.save(cache_payload, primary_cache_file)
            print(f"Cached labels saved to {primary_cache_file} ({label_time_sec:.2f}s).")

        # 2. Single-pass features
        print(f"Collecting HDS features with anchor={args.residual_anchor}, binding={args.coord_binding}...")
        hds_encoder = HDSEncoder(
            hd_dim=exp_config.hd_dim,
            num_levels=exp_config.num_levels,
            max_coords=exp_config.max_coords,
            num_buckets=exp_config.num_buckets,
            seed=seed,
            device=device,
        )
        feat_out = collect_features_single_pass(model, calib_ids, exp_config, hds_encoder, batch_size=args.batch_size)
        features = feat_out["features"]
        feature_time_sec = float(feat_out["feature_time_sec"])
        forward_capture_time_sec = float(feat_out["forward_capture_time_sec"])
        estimator_feature_time_sec = dict(feat_out["estimator_feature_time_sec"])

        # 3. Cross-validate readouts (calibration cost: timed per estimator inside cross_validate_estimators)
        t_cv = time.perf_counter()
        cv_out = cross_validate_estimators(features, delta_loss, meta_list, exp_config, seed=seed)
        cv_time_sec = float(time.perf_counter() - t_cv)
        raw_preds = cv_out["raw_predictions"]
        alloc_costs = cv_out["alloc_costs"]
        estimator_readout_time_sec = dict(cv_out["estimator_readout_time_sec"])

        # Ranking metrics
        ranking_metrics = {}
        for est in cv_out["estimators"]:
            ranking_metrics[est] = compute_ranking_metrics(delta_loss, raw_preds[est], meta_list)

        # 4. Bit allocation & held-out PPL across budgets
        print(f"Solving ILP bit allocations and evaluating PPL across budgets {budgets}...")
        allocations = {}
        estimator_ilp_time_sec = {est: 0.0 for est in cv_out["estimators"]}

        for est in cv_out["estimators"]:
            allocations[est] = {}
            for b in budgets:
                b_key = f"{b:.2f}"
                t_ilp = time.perf_counter()
                ilp_res = solve_bit_allocation_ilp(
                    alloc_costs[est], meta_list, target_avg_bits=b, candidate_bits=candidate_bits
                )
                estimator_ilp_time_sec[est] += time.perf_counter() - t_ilp
                if ilp_res["success"] and ilp_res.get("bit_assignment"):
                    ppl_res = _eval_or_lookup_ppl(ilp_res["bit_assignment"], exp_config)  # NOT calibration cost
                else:
                    ppl_res = {"nll": float("nan"), "ppl": float("nan"), "eval_time_sec": 0.0, "seq_nlls": [], "seq_token_counts": []}
                allocations[est][b_key] = {
                    "bit_assignment": ilp_res.get("bit_assignment", {}),
                    "achieved_avg_bits": float(ilp_res.get("achieved_avg_bits", float("nan"))),
                    "total_predicted_cost": float(ilp_res.get("total_predicted_cost", float("nan"))),
                    "bit_counts": ilp_res.get("bit_counts", {}),
                    "success": ilp_res.get("success", False),
                    "nll": ppl_res["nll"],
                    "ppl": ppl_res["ppl"],
                    "seq_nlls": ppl_res["seq_nlls"],
                    "seq_token_counts": ppl_res["seq_token_counts"],
                }

        test_eval_time_sec = float(test_eval_clock["sec"])
        # Calibration-side readout + allocation time: CV fitting plus all ILP solves; excludes held-out evaluation.
        readout_alloc_time_sec = float(cv_time_sec + sum(estimator_ilp_time_sec.values()))

        seed_run_data = {
            "seed": seed,
            "num_layers": num_layers,
            "n_folds": int(cv_out["n_folds"]),
            "budgets": budgets,
            "estimators": cv_out["estimators"],
            "base_loss_4bit": float(base_loss_4bit),
            "base_overhead_time_sec": base_overhead_time_sec,
            "probe_time_sec": probe_time_sec,
            "restore_time_sec": restore_time_sec,
            "feature_time_sec": feature_time_sec,
            "forward_capture_time_sec": forward_capture_time_sec,
            "estimator_feature_time_sec": estimator_feature_time_sec,
            "estimator_readout_time_sec": estimator_readout_time_sec,
            "estimator_ilp_time_sec": estimator_ilp_time_sec,
            "test_eval_time_sec": test_eval_time_sec,
            "label_time_sec": label_time_sec,
            "wall_label_time_sec": wall_label_time_sec,
            "readout_alloc_time_sec": readout_alloc_time_sec,
            "ranking_metrics": ranking_metrics,
            "uniform_baselines": uniform_baselines,
            "allocations": allocations,
            "delta_loss": _pair_dict_to_json(delta_loss),
            "raw_predictions": {est: _pair_dict_to_json(raw_preds[est]) for est in cv_out["estimators"]},
        }
        seed_runs.append(seed_run_data)

        # 6. Low-shot curve: ONE fixed depth-stratified set of K labeled layers (K layers acquired in total)
        if args.run_low_shot_curve:
            print("Evaluating fixed-K layer-acquisition (low-shot) curve...")
            candidate_layer_counts = [k for k in (2, 4, 8, 12) if k < num_layers]
            if not candidate_layer_counts and num_layers > 1:
                candidate_layer_counts = [num_layers - 1]

            for k in candidate_layer_counts:
                k_key = str(k)
                if k_key not in low_shot_curve_data:
                    low_shot_curve_data[k_key] = {}
                acq = evaluate_fixed_k_layer_acquisition(
                    features, delta_loss, meta_list, exp_config, k_layers=k, seed=seed
                )
                for est in LOW_SHOT_ESTIMATORS:
                    if est not in low_shot_curve_data[k_key]:
                        low_shot_curve_data[k_key][est] = []
                    heldout_pred = acq["heldout_predictions"][est]
                    heldout_true = {key: delta_loss[key] for key in heldout_pred}
                    r_metrics = compute_ranking_metrics(heldout_true, heldout_pred, meta_list)

                    # One ILP solve + (cached) held-out PPL per low-shot budget.
                    by_budget: dict = {}
                    ilp_k_times = []
                    for lb in low_shot_budgets:
                        t_ilp = time.perf_counter()
                        ilp_k = solve_bit_allocation_ilp(
                            acq["alloc_costs"][est], meta_list, target_avg_bits=lb, candidate_bits=candidate_bits
                        )
                        ilp_k_times.append(float(time.perf_counter() - t_ilp))
                        if ilp_k["success"] and ilp_k.get("bit_assignment"):
                            ppl_b = _eval_or_lookup_ppl(ilp_k["bit_assignment"], exp_config)
                        else:
                            ppl_b = {"nll": float("nan"), "ppl": float("nan"), "eval_time_sec": 0.0, "seq_nlls": [], "seq_token_counts": []}
                        by_budget[f"{lb:.2f}"] = {
                            "success": bool(ilp_k.get("success", False)),
                            "bit_assignment": ilp_k.get("bit_assignment", {}),
                            "achieved_avg_bits": float(ilp_k.get("achieved_avg_bits", float("nan"))),
                            "ppl": ppl_b["ppl"],
                            "nll": ppl_b["nll"],
                            "seq_nlls": ppl_b.get("seq_nlls", []),
                            "seq_token_counts": ppl_b.get("seq_token_counts", []),
                        }
                    # The acquisition cost covers allocating for ONE target budget (mean ILP time over budgets).
                    ilp_k_time_sec = float(np.mean(ilp_k_times))
                    acquisition_cost_sec = float(
                        forward_capture_time_sec
                        + estimator_feature_time_sec.get(est, 0.0)
                        + base_overhead_time_sec
                        + (k / num_layers) * probe_time_sec
                        + restore_time_sec
                        + acq["estimator_readout_time_sec"].get(est, 0.0)
                        + ilp_k_time_sec
                    )
                    # Legacy top-level fields: the 3.00b budget when evaluated, else the first low-shot budget.
                    legacy_key = next((bk for bk in by_budget if math.isclose(float(bk), 3.0)), None)
                    primary = by_budget[legacy_key if legacy_key is not None else next(iter(by_budget))]
                    low_shot_curve_data[k_key][est].append({
                        "seed": seed,
                        "success": primary["success"],
                        "acquired_layers": acq["acquired_layers"],
                        "unlabeled_layers": acq["unlabeled_layers"],
                        "n_labeled_layers": acq["n_labeled_layers"],
                        "bit_assignment": primary["bit_assignment"],
                        "achieved_avg_bits": primary["achieved_avg_bits"],
                        "heldout_predictions": _pair_dict_to_json(heldout_pred),
                        "raw_predictions": _pair_dict_to_json(acq["raw_predictions"][est]),
                        "spearman_global": r_metrics["spearman_global"],
                        "top20_recall": r_metrics["top20_recall"],
                        "ppl_3bit": by_budget[legacy_key]["ppl"] if legacy_key is not None else float("nan"),
                        "nll_3bit": by_budget[legacy_key]["nll"] if legacy_key is not None else float("nan"),
                        "ppl_by_budget": {bk: v["ppl"] for bk, v in by_budget.items()},
                        "budgets": by_budget,
                        "seq_nlls": primary["seq_nlls"],
                        "seq_token_counts": primary["seq_token_counts"],
                        "acquisition_cost_sec": acquisition_cost_sec,
                    })

    # 7. Aggregate stability
    print("Aggregating stability across seeds...")
    summary = aggregate_seed_stability(seed_runs, uniform_baselines=uniform_baselines)

    # Aggregate low-shot curve across seeds
    low_shot_summary = {}
    for k_key, est_dict in low_shot_curve_data.items():
        low_shot_summary[k_key] = {}
        for est, run_list in est_dict.items():
            valid_runs = [
                r for r in run_list
                if r.get("success", True)
                and bool(r.get("bit_assignment", True))
                and r.get("ppl_3bit") is not None
                and not np.isnan(float(r.get("ppl_3bit", float("nan"))))
            ]
            valid_ppls = [float(r["ppl_3bit"]) for r in valid_runs]
            valid_bits = [
                float(r["achieved_avg_bits"]) for r in valid_runs
                if r.get("achieved_avg_bits") is not None and not np.isnan(float(r.get("achieved_avg_bits", float("nan"))))
            ]
            # Per-budget aggregation (every low-shot budget evaluated for this (K, estimator)).
            ppl_by_budget_summary = {}
            for bk in sorted({bk for r in run_list for bk in r.get("budgets", {})}, key=float):
                b_runs = [
                    r["budgets"][bk] for r in run_list
                    if bk in r.get("budgets", {})
                    and r["budgets"][bk].get("success", True)
                    and bool(r["budgets"][bk].get("bit_assignment", True))
                    and r["budgets"][bk].get("ppl") is not None
                    and not np.isnan(float(r["budgets"][bk]["ppl"]))
                ]
                b_ppls = [float(r["ppl"]) for r in b_runs]
                b_bits = [
                    float(r["achieved_avg_bits"]) for r in b_runs
                    if r.get("achieved_avg_bits") is not None and not np.isnan(float(r["achieved_avg_bits"]))
                ]
                ppl_by_budget_summary[bk] = {
                    "ppl_mean": float(np.mean(b_ppls)) if b_ppls else None,
                    "ppl_std": float(np.std(b_ppls)) if len(b_ppls) > 1 else 0.0,
                    "achieved_avg_bits_mean": float(np.mean(b_bits)) if b_bits else None,
                    "n_valid_seeds": len(b_runs),
                }
            low_shot_summary[k_key][est] = {
                "n_labeled_layers": int(k_key),
                "spearman_global_mean": float(np.mean([r["spearman_global"] for r in run_list])),
                "spearman_global_std": float(np.std([r["spearman_global"] for r in run_list])) if len(run_list) > 1 else 0.0,
                "top20_recall_mean": float(np.mean([r["top20_recall"] for r in run_list])),
                "top20_recall_std": float(np.std([r["top20_recall"] for r in run_list])) if len(run_list) > 1 else 0.0,
                "ppl_3bit_mean": float(np.mean(valid_ppls)) if valid_ppls else None,
                "ppl_3bit_std": float(np.std(valid_ppls)) if len(valid_ppls) > 1 else 0.0,
                "achieved_avg_bits_mean": float(np.mean(valid_bits)) if valid_bits else None,
                "ppl_by_budget": ppl_by_budget_summary,
                "acquisition_cost_sec_mean": float(np.mean([r["acquisition_cost_sec"] for r in run_list])),
                "n_valid_seeds": len(valid_runs),
            }

    cfg_dict = dict(vars(args))
    if cfg_dict.get("quantizer") == "hqq":
        from .quantizer import get_effective_hqq_config
        param = next(iter(model.parameters()), None)
        dev = param.device if param is not None else torch.device(device)
        dt = param.dtype if param is not None else torch.float32
        eff_cfg = get_effective_hqq_config(
            group_size=quant_group_size,
            device=dev,
            dtype=dt,
        )
        cfg_dict["hqq_backend"] = eff_cfg["backend"]
        cfg_dict["hqq_version"] = eff_cfg["version"]
        cfg_dict["hqq_config"] = eff_cfg

    output_payload = {
        "config": cfg_dict,
        "uniform_baselines": uniform_baselines,
        "summary": summary,
        "seed_runs": seed_runs,
        "low_shot_curve": low_shot_summary,
        "low_shot_seed_runs": low_shot_curve_data,
    }

    output_payload = _sanitize_for_json(output_payload)
    out_file = Path(args.output_dir) / f"run_{args.run_tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, allow_nan=False)
    print(f"\n[Done] Successfully saved experiment results to {out_file}")
    return output_payload


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
