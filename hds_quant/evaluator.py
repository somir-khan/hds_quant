"""Evaluation metrics: ranking quality, held-out perplexity, and multi-seed stability."""

from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

from .config import ExperimentConfig
from .quantizer import quantize_weight
from .readout_cv import ORACLE_ESTIMATORS, SUPERVISED_ESTIMATORS, UNSUPERVISED_SCALARS

FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    res = spearmanr(x, y).statistic
    return float(0.0 if np.isnan(res) else res)


def _safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    res = kendalltau(x, y).statistic
    return float(0.0 if np.isnan(res) else res)


def compute_ranking_metrics(
    true_delta_loss: Dict[Tuple[str, int], float],
    raw_predictions: Dict[Tuple[str, int], float],
    meta_list: List[dict],
    candidate_bits: Optional[Tuple[int, ...]] = None,
) -> dict:
    """Compute ranking accuracy metrics between true_delta_loss and raw_predictions.

    ``candidate_bits`` defaults to the set of bit-widths present in ``true_delta_loss`` (``(2, 3)`` if empty).

    Returns:
        spearman_global, kendall_global, spearman_bit1, spearman_bit2, spearman_bit3,
        spearman_within_family, top20_recall.  ``spearman_bitK`` is 0.0 when no b=K labels are present.
    """
    if candidate_bits is None:
        candidate_bits = tuple(sorted({int(k[1]) for k in true_delta_loss})) or (2, 3)
    keys = [
        (m["name"], b)
        for m in meta_list
        for b in candidate_bits
        if (m["name"], b) in true_delta_loss and (m["name"], b) in raw_predictions
    ]
    if not keys:
        return {
            "spearman_global": 0.0,
            "kendall_global": 0.0,
            "spearman_bit1": 0.0,
            "spearman_bit2": 0.0,
            "spearman_bit3": 0.0,
            "spearman_within_family": 0.0,
            "top20_recall": 0.0,
        }

    y_true = np.array([float(true_delta_loss[k]) for k in keys], dtype=np.float64)
    y_pred = np.array([float(raw_predictions[k]) for k in keys], dtype=np.float64)

    spearman_global = _safe_spearman(y_pred, y_true)
    kendall_global = _safe_kendall(y_pred, y_true)

    # Bit-stratified
    b1_idx = [i for i, k in enumerate(keys) if k[1] == 1]
    b2_idx = [i for i, k in enumerate(keys) if k[1] == 2]
    b3_idx = [i for i, k in enumerate(keys) if k[1] == 3]
    spearman_bit1 = _safe_spearman(y_pred[b1_idx], y_true[b1_idx]) if b1_idx else 0.0
    spearman_bit2 = _safe_spearman(y_pred[b2_idx], y_true[b2_idx]) if b2_idx else 0.0
    spearman_bit3 = _safe_spearman(y_pred[b3_idx], y_true[b3_idx]) if b3_idx else 0.0

    # Within-family (across 7 canonical families)
    family_of = {m["name"]: m.get("family", m["name"].split(".")[-1]) for m in meta_list}
    fam_rhos = []
    for fam in FAMILIES:
        f_idx = [i for i, k in enumerate(keys) if family_of.get(k[0]) == fam]
        if len(f_idx) >= 2:
            fam_rhos.append(_safe_spearman(y_pred[f_idx], y_true[f_idx]))
    spearman_within_family = float(np.mean(fam_rhos)) if fam_rhos else 0.0

    # Top-20% recall: most damaging ground truth vs highest predicted
    k_top = max(1, int(round(0.20 * len(keys))))
    top_true_idx = set(np.argsort(-y_true)[:k_top])
    top_pred_idx = set(np.argsort(-y_pred)[:k_top])
    top20_recall = float(len(top_true_idx & top_pred_idx) / k_top)

    return {
        "spearman_global": spearman_global,
        "kendall_global": kendall_global,
        "spearman_bit1": spearman_bit1,
        "spearman_bit2": spearman_bit2,
        "spearman_bit3": spearman_bit3,
        "spearman_within_family": spearman_within_family,
        "top20_recall": top20_recall,
    }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    eval_input_ids: torch.Tensor,
    meta_list: List[dict],
    config: ExperimentConfig,
    bit_assignment: Optional[Dict[str, int]] = None,
    uniform_bit: Optional[int] = None,
    batch_size: int = 8,
    orig_weights_cpu: Optional[Dict[str, torch.Tensor]] = None,
) -> dict:
    """Evaluate held-out autoregressive perplexity under bit_assignment, uniform_bit, or FP16 unquantized.

    Guarantees original weights are restored on exit. Besides the pooled ``nll`` / ``ppl`` it returns, for every
    evaluation sequence (chunk), ``seq_nlls`` (mean next-token NLL of that chunk) and ``seq_token_counts`` (number of
    scored tokens), so paired bootstrap confidence intervals can be computed offline
    (chunk NLL sum = ``seq_nlls[i] * seq_token_counts[i]``).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    def get_module(p_dict):
        if "module" in p_dict and isinstance(p_dict["module"], nn.Linear):
            return p_dict["module"]
        return model.get_submodule(p_dict["name"])

    projs = [(m["name"], get_module(m)) for m in meta_list]
    if orig_weights_cpu is None:
        orig_weights_cpu = {name: mod.weight.data.detach().cpu().clone() for name, mod in projs}

    t0 = time.perf_counter()
    seq_nll_sums: List[float] = []
    seq_token_counts: List[int] = []
    try:
        if bit_assignment is not None:
            for name, mod in projs:
                b = bit_assignment[name]
                w_q = quantize_weight(
                    orig_weights_cpu[name].to(device),
                    bits=b,
                    group_size=config.quant_group_size,
                    clip_ratio=config.clip_ratio,
                    quantizer=config.quantizer,
                )
                mod.weight.data.copy_(w_q.to(device=mod.weight.data.device, dtype=mod.weight.data.dtype))
        elif uniform_bit is not None:
            for name, mod in projs:
                w_q = quantize_weight(
                    orig_weights_cpu[name].to(device),
                    bits=uniform_bit,
                    group_size=config.quant_group_size,
                    clip_ratio=config.clip_ratio,
                    quantizer=config.quantizer,
                )
                mod.weight.data.copy_(w_q.to(device=mod.weight.data.device, dtype=mod.weight.data.dtype))

        total_loss = 0.0
        total_tokens = 0
        n_chunks = eval_input_ids.shape[0]

        for start in range(0, n_chunks, batch_size):
            batch = eval_input_ids[start : start + batch_size].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            shift_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
            shift_labels = batch[:, 1:].reshape(-1)
            tok_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none").reshape(batch.shape[0], -1)
            per_seq = tok_loss.double().sum(dim=1)
            total_loss += float(per_seq.sum())
            total_tokens += shift_labels.numel()
            seq_nll_sums.extend(per_seq.tolist())
            seq_token_counts.extend([int(tok_loss.shape[1])] * int(batch.shape[0]))

        mean_nll = total_loss / max(1, total_tokens)
        ppl = float(math.exp(min(mean_nll, 50.0)))
    finally:
        for name, mod in projs:
            mod.weight.data.copy_(orig_weights_cpu[name].to(device=mod.weight.data.device, dtype=mod.weight.data.dtype))
        model.train(was_training)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "nll": float(mean_nll),
        "ppl": ppl,
        "eval_time_sec": float(time.perf_counter() - t0),
        "seq_nlls": [s / max(1, c) for s, c in zip(seq_nll_sums, seq_token_counts)],
        "seq_token_counts": seq_token_counts,
    }


def aggregate_seed_stability(seed_runs: List[dict], uniform_baselines: Optional[dict] = None) -> dict:
    """Aggregate multiple seed runs into summary statistics."""
    if not seed_runs:
        return {}

    n_seeds = len(seed_runs)
    estimators = seed_runs[0]["estimators"]
    budgets = [float(b) for b in seed_runs[0]["budgets"]]
    num_layers = seed_runs[0].get("num_layers", 16)
    n_folds = int(seed_runs[0].get("n_folds", 4))
    single_fold_frac = (n_folds - 1.0) / n_folds if n_folds > 1 else 1.0

    # 1. Ranking metrics aggregation
    ranking_agg = {}
    metric_keys = ["spearman_global", "kendall_global", "spearman_bit2", "spearman_bit3", "spearman_within_family", "top20_recall"]
    if any("spearman_bit1" in seed_runs[0].get("ranking_metrics", {}).get(est, {}) for est in estimators):
        metric_keys.insert(2, "spearman_bit1")
    metric_keys = tuple(metric_keys)

    for est in estimators:
        ranking_agg[est] = {}
        for mk in metric_keys:
            vals = [run["ranking_metrics"][est][mk] for run in seed_runs if mk in run.get("ranking_metrics", {}).get(est, {})]
            ranking_agg[est][f"{mk}_mean"] = float(np.mean(vals)) if vals else 0.0
            ranking_agg[est][f"{mk}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # 2. PPL & achieved bits aggregation per (estimator, budget)
    ppl_agg = {}
    for est in estimators:
        ppl_agg[est] = {}
        for b in budgets:
            b_key = f"{b:.2f}"
            valid_allocs = [
                run["allocations"][est][b_key]
                for run in seed_runs
                if est in run.get("allocations", {})
                and b_key in run["allocations"][est]
                and run["allocations"][est][b_key].get("success", True)
                and bool(run["allocations"][est][b_key].get("bit_assignment"))
            ]
            ppls = [a["ppl"] for a in valid_allocs if not np.isnan(a.get("ppl", float("nan")))]
            bits = [a["achieved_avg_bits"] for a in valid_allocs if not np.isnan(a.get("achieved_avg_bits", float("nan")))]
            ppl_agg[est][b_key] = {
                "ppl_mean": float(np.mean(ppls)) if ppls else float("nan"),
                "ppl_std": float(np.std(ppls)) if len(ppls) > 1 else 0.0,
                "achieved_avg_bits_mean": float(np.mean(bits)) if bits else float("nan"),
                "achieved_avg_bits_std": float(np.std(bits)) if len(bits) > 1 else 0.0,
            }

    # Uniform baselines aggregation
    uniform_agg = {}
    uniform_keys = ("FP16", "Uniform-4bit", "Uniform-3bit", "Uniform-2bit", "Uniform-1bit")
    if uniform_baselines is not None:
        for u_key in uniform_keys:
            if u_key in uniform_baselines:
                uniform_agg[u_key] = {
                    "ppl_mean": float(uniform_baselines[u_key]["ppl"]),
                    "ppl_std": 0.0,
                }
    else:
        for u_key in uniform_keys:
            u_vals = [run["uniform_baselines"][u_key]["ppl"] for run in seed_runs if "uniform_baselines" in run and u_key in run["uniform_baselines"]]
            if u_vals:
                uniform_agg[u_key] = {
                    "ppl_mean": float(np.mean(u_vals)),
                    "ppl_std": float(np.std(u_vals)) if len(u_vals) > 1 else 0.0,
                }

    # 3. Pairwise bit allocation stability across seeds
    stability_agg = {}
    for est in estimators:
        stability_agg[est] = {}
        for b in budgets:
            b_key = f"{b:.2f}"
            valid_assigns = [
                run["allocations"][est][b_key]["bit_assignment"]
                for run in seed_runs
                if est in run.get("allocations", {})
                and b_key in run["allocations"][est]
                and run["allocations"][est][b_key].get("success", True)
                and bool(run["allocations"][est][b_key].get("bit_assignment"))
            ]
            if len(valid_assigns) == 0:
                # No successful allocation on any seed -> stability is unavailable (NaN, NOT 1.0!)
                stability_agg[est][b_key] = float("nan")
            elif n_seeds == 1 and len(valid_assigns) == 1:
                stability_agg[est][b_key] = 1.0
            elif len(valid_assigns) < 2:
                # Multi-seed run where fewer than 2 seeds succeeded -> pairwise agreement is unavailable (NaN, NOT 1.0!)
                stability_agg[est][b_key] = float("nan")
            else:
                match_fractions = []
                for a1, a2 in itertools.combinations(valid_assigns, 2):
                    common_keys = sorted(set(a1.keys()) & set(a2.keys()))
                    all_keys = sorted(set(a1.keys()) | set(a2.keys()))
                    if all_keys and len(common_keys) == len(all_keys):
                        matches = sum(a1[p] == a2[p] for p in all_keys)
                        match_fractions.append(matches / len(all_keys))
                stability_agg[est][b_key] = float(np.mean(match_fractions)) if match_fractions else float("nan")

    # 4. Calibration cost breakdown (held-out perplexity evaluation is NEVER part of calibration cost)
    def _mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def _mean_per_est(key: str, est: str) -> float:
        return _mean([float(run.get(key, {}).get(est, 0.0)) for run in seed_runs])

    legacy = not any("estimator_feature_time_sec" in run for run in seed_runs)
    feature_times = [run.get("feature_time_sec", 0.0) for run in seed_runs]
    forward_times = [run.get("forward_capture_time_sec", run.get("feature_time_sec", 0.0)) for run in seed_runs]
    label_times = [run.get("label_time_sec", 0.0) for run in seed_runs]
    wall_label_times = [run.get("wall_label_time_sec", run.get("label_time_sec", 0.0)) for run in seed_runs]
    alloc_times = [run.get("readout_alloc_time_sec", 0.0) for run in seed_runs]
    test_eval_times = [run.get("test_eval_time_sec", 0.0) for run in seed_runs]

    feat_mean = _mean(feature_times)
    fwd_mean = _mean(forward_times)
    label_mean = _mean(label_times)
    wall_label_mean = _mean(wall_label_times)
    alloc_mean = _mean(alloc_times)
    test_eval_mean = _mean(test_eval_times)

    supervised = set(SUPERVISED_ESTIMATORS)
    unsupervised = set(UNSUPERVISED_SCALARS)
    oracles = set(ORACLE_ESTIMATORS)

    costs = {}
    for est in estimators:
        est_feat = _mean_per_est("estimator_feature_time_sec", est)
        est_readout = _mean_per_est("estimator_readout_time_sec", est)
        est_ilp = _mean_per_est("estimator_ilp_time_sec", est)
        if legacy:  # results written before the per-estimator timers existed
            if est in unsupervised:
                total = full_bank = single_fold = feat_mean + alloc_mean
            elif est in oracles:
                total = full_bank = single_fold = label_mean + alloc_mean
            else:
                single_fold = feat_mean + single_fold_frac * label_mean + alloc_mean
                total = full_bank = feat_mean + 1.0 * label_mean + alloc_mean
        elif est in unsupervised:
            total = full_bank = single_fold = fwd_mean + est_feat + est_ilp
        elif est in oracles:
            total = full_bank = single_fold = label_mean + est_ilp
        elif est in supervised:
            # Whole-model allocation uses 100% of layers' labels across the union of all 4 folds.
            # Single-fold training uses single_fold_frac of the layers' labels.
            single_fold = fwd_mean + est_feat + single_fold_frac * label_mean + est_readout + est_ilp
            total = full_bank = fwd_mean + est_feat + 1.0 * label_mean + est_readout + est_ilp
        else:
            total = full_bank = single_fold = fwd_mean + est_feat + label_mean + est_readout + est_ilp

        costs[est] = {
            "feature_time_sec": feat_mean,
            "forward_capture_time_sec": fwd_mean,
            "estimator_feature_time_sec": est_feat,
            "estimator_readout_time_sec": est_readout,
            "estimator_ilp_time_sec": est_ilp,
            "label_time_sec": label_mean,
            "wall_label_time_sec": wall_label_mean,
            "readout_alloc_time_sec": alloc_mean,
            "test_eval_time_sec": test_eval_mean,
            "total_calib_time_sec": float(total),
            "full_bank_calib_time_sec": float(full_bank),
            "single_fold_calib_time_sec": float(single_fold),
        }

    return {
        "n_seeds": n_seeds,
        "n_folds": n_folds,
        "ranking_metrics": ranking_agg,
        "ppl_metrics": ppl_agg,
        "uniform_baselines": uniform_agg,
        "allocation_stability": stability_agg,
        "calibration_costs": costs,
    }
