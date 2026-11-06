"""Collects HDS/scalar features and ground-truth marginal-loss labels from a Llama-style model."""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PROJ_FAMILIES as _CONFIG_PROJ_FAMILIES
from .config import ExperimentConfig
from .estimators import compute_estimators, compute_selected_estimator
from .hds_encoder import HDSEncoder
from .quantizer import quantize_weight

PROJ_FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
assert PROJ_FAMILIES == tuple(n.split(".")[-1] for n in _CONFIG_PROJ_FAMILIES)

_PROJ_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:self_attn|mlp)\.(" + "|".join(PROJ_FAMILIES) + r")$")

# Projections that consume the exact same input tensor share one captured copy.
_SHARED_INPUT_SOURCE = {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}


def discover_target_projections(model: nn.Module) -> List[dict]:
    """Return metadata for every q/k/v/o/gate/up/down ``nn.Linear`` in the transformer layers.

    ``embed_tokens`` and ``lm_head`` never match. Sorted by ``(layer_idx, family_idx)``.
    """
    found = []
    for name, module in model.named_modules():
        m = _PROJ_RE.search(name)
        if m is None or not isinstance(module, nn.Linear):
            continue
        family = m.group(2)
        found.append(
            {
                "name": name,
                "layer_idx": int(m.group(1)),
                "family": family,
                "family_idx": PROJ_FAMILIES.index(family),
                "num_params": module.weight.numel(),
                "d_out": module.out_features,
                "d_in": module.in_features,
                "module": module,
            }
        )
    found.sort(key=lambda p: (p["layer_idx"], p["family_idx"]))
    return found


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _select_token_positions(calib_input_ids: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    """Sorted flat positions (into ``[n_seqs * seq_len]``), chosen with ``config.seeds[0]``."""
    n = calib_input_ids.numel()
    T = min(config.max_encoded_tokens, n)
    g = torch.Generator().manual_seed(config.seeds[0])
    return torch.randperm(n, generator=g)[:T].sort().values


@torch.no_grad()
def collect_features_single_pass(
    model: nn.Module,
    calib_input_ids: torch.Tensor,
    config: ExperimentConfig,
    hds_encoder: HDSEncoder,
    batch_size: int = 8,
    feature_estimator: Optional[str] = None,
) -> dict:
    """One calibration forward pass over the (unmodified) model, then per-layer feature extraction.

    Pass 1: pre-forward hooks capture projection inputs ``X_p`` only at the ``T`` selected token positions
    (stored on CPU, so device memory never holds more than one batch of activations).
    Pass 2: layers are processed one at a time; only that layer's ``X_p`` live on the device.

    Returns ``{"features": {(proj_name, b): estimator_dict}, "meta": [proj dicts without "module"],
    "feature_time_sec": float, "forward_capture_time_sec": float, "estimator_feature_time_sec": {est: float}}``.

    ``forward_capture_time_sec`` is the shared cost of the single hooked forward pass plus building every
    residual ``R_pb`` (weight quantization + linear layers); ``estimator_feature_time_sec[est]`` is the standalone
    feature-extraction time of each estimator summed over all ``(proj, b)`` pairs (see ``compute_estimators``).
    """
    if calib_input_ids.ndim != 2:
        raise ValueError("calib_input_ids must be [n_seqs, seq_len]")
    if hds_encoder.hd_dim != config.hd_dim or hds_encoder.num_levels != config.num_levels:
        raise ValueError("hds_encoder dimensions do not match config.hd_dim / config.num_levels")

    device = next(model.parameters()).device
    _sync(device)
    t0 = time.perf_counter()

    proj_list = discover_target_projections(model)
    n_seqs, seq_len = calib_input_ids.shape
    selected = _select_token_positions(calib_input_ids, config)

    # ---- pass 1: capture inputs at the selected positions
    captured: Dict[Tuple[int, str], List[torch.Tensor]] = {}
    state = {"local_idx": None}

    def make_hook(key):
        def hook(_module, args):
            x = args[0]
            rows = x.reshape(-1, x.shape[-1])[state["local_idx"]]
            captured[key].append(rows.detach().cpu())

        return hook

    handles = []
    for p in proj_list:
        if p["family"] in _SHARED_INPUT_SOURCE:
            continue
        key = (p["layer_idx"], p["family"])
        captured[key] = []
        handles.append(p["module"].register_forward_pre_hook(make_hook(key)))

    was_training = model.training
    model.eval()
    base = getattr(model, "model", model)  # skip the lm_head matmul
    try:
        for start in range(0, n_seqs, batch_size):
            batch = calib_input_ids[start : start + batch_size]
            lo, hi = start * seq_len, (start + batch.shape[0]) * seq_len
            local = selected[(selected >= lo) & (selected < hi)] - lo
            if local.numel() == 0:
                continue
            state["local_idx"] = local.to(device)
            base(input_ids=batch.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    _sync(device)
    pass1_time = time.perf_counter() - t0

    # ---- pass 2: per-layer features
    features: Dict[Tuple[str, int], dict] = {}
    estimator_feature_time: Dict[str, float] = {}
    estimator_wall = 0.0  # wall time spent inside compute_estimators (subtracted from the residual/forward cost)
    t_pass2 = time.perf_counter()
    anchor_int4 = config.residual_anchor == "int4_aligned"
    layers = sorted({p["layer_idx"] for p in proj_list})
    for layer_idx in layers:
        layer_X = {
            key[1]: torch.cat(chunks, dim=0)
            for key, chunks in captured.items()
            if key[0] == layer_idx
        }
        for key in [k for k in captured if k[0] == layer_idx]:
            captured[key] = []  # release chunk lists
        for p in [q for q in proj_list if q["layer_idx"] == layer_idx]:
            X_p = layer_X[_SHARED_INPUT_SOURCE.get(p["family"], p["family"])].to(device)
            W_fp16 = p["module"].weight.data
            Y_fp16 = F.linear(X_p, W_fp16)
            fp16_col_norms = torch.linalg.vector_norm(Y_fp16.float(), ord=2, dim=0)
            if anchor_int4:
                W_ref = quantize_weight(
                    W_fp16,
                    bits=config.reference_bit,
                    group_size=config.quant_group_size,
                    clip_ratio=config.clip_ratio,
                    quantizer=config.quantizer,
                    hqq_axis=config.hqq_axis,
                    hqq_zero_policy=config.hqq_zero_policy,
                )
                Y_ref = F.linear(X_p, W_ref)
                del W_ref
            else:
                Y_ref = Y_fp16
            for b in config.marginal_bits:
                W_b = quantize_weight(
                    W_fp16,
                    bits=b,
                    group_size=config.quant_group_size,
                    clip_ratio=config.clip_ratio,
                    quantizer=config.quantizer,
                    hqq_axis=config.hqq_axis,
                    hqq_zero_policy=config.hqq_zero_policy,
                )
                Y_b = F.linear(X_p, W_b)
                R_pb = (Y_ref - Y_b).abs().float()
                _sync(device)
                t_est = time.perf_counter()
                if feature_estimator is not None:
                    features[(p["name"], b)] = compute_selected_estimator(
                        R_pb, fp16_col_norms, p["family_idx"], hds_encoder,
                        feature_estimator, coord_binding=config.coord_binding,
                    )
                else:
                    features[(p["name"], b)] = compute_estimators(
                        R_pb,
                        fp16_col_norms,
                        p["family_idx"],
                        hds_encoder,
                        coord_binding=config.coord_binding,
                        hist_bins=config.hist_bins,
                        hist_max=config.hist_max,
                        trim_frac=config.trim_frac,
                        winsor_quantile=config.winsor_quantile,
                        timings=estimator_feature_time,
                    )
                _sync(device)
                estimator_wall += time.perf_counter() - t_est
                del W_b, Y_b, R_pb
            del X_p, Y_fp16, Y_ref
        del layer_X

    _sync(device)
    pass2_time = time.perf_counter() - t_pass2
    meta = [{k: v for k, v in p.items() if k != "module"} for p in proj_list]
    return {
        "features": features,
        "meta": meta,
        "feature_time_sec": time.perf_counter() - t0,
        "forward_capture_time_sec": float(pass1_time + max(pass2_time - estimator_wall, 0.0)),
        "estimator_feature_time_sec": {k: float(v) for k, v in estimator_feature_time.items()},
    }


@torch.no_grad()
def _mean_ce_loss(model: nn.Module, input_ids: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """Exact token-weighted mean next-token cross-entropy over all of ``input_ids``."""
    total = 0.0
    count = 0
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        logits = model(input_ids=ids, use_cache=False).logits
        shift_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
        shift_labels = ids[:, 1:].reshape(-1)
        total += float(F.cross_entropy(shift_logits, shift_labels, reduction="sum").double())
        count += shift_labels.numel()
    return total / count


@torch.no_grad()
def collect_marginal_loss_labels(
    model: nn.Module,
    calib_input_ids: torch.Tensor,
    config: ExperimentConfig,
    batch_size: int = 8,
    target_layers: Optional[Set[int]] = None,
) -> dict:
    """Ground-truth ``delta_loss[(proj_name, b)] = L(p at b bits, rest at ref bits) - L(all at ref bits)``.

    All target projections are quantized in place to ``config.reference_bit`` (their reference weights are cached
    on-device so restoring after each swap is a tensor copy). Each probed projection is swapped to ``b`` bits,
    evaluated, and immediately restored. Original weights are restored bit-for-bit at the end (also on error).
    """
    if calib_input_ids.ndim != 2:
        raise ValueError("calib_input_ids must be [n_seqs, seq_len]")
    device = next(model.parameters()).device
    _sync(device)
    t0 = time.perf_counter()

    proj_list = discover_target_projections(model)
    orig_weights_cpu = {p["name"]: p["module"].weight.data.detach().cpu().clone() for p in proj_list}
    was_training = model.training
    model.eval()
    delta_loss: Dict[Tuple[str, int], float] = {}
    try:
        ref_weights = {}
        for p in proj_list:
            w = p["module"].weight.data
            ref_weights[p["name"]] = quantize_weight(
                orig_weights_cpu[p["name"]].to(device),
                bits=config.reference_bit,
                group_size=config.quant_group_size,
                clip_ratio=config.clip_ratio,
                quantizer=config.quantizer,
                hqq_axis=config.hqq_axis,
                hqq_zero_policy=config.hqq_zero_policy,
            ).to(device=w.device, dtype=w.dtype)
            w.copy_(ref_weights[p["name"]])

        base_loss_4bit = _mean_ce_loss(model, calib_input_ids, batch_size, device)
        _sync(device)
        base_overhead_time_sec = float(time.perf_counter() - t0)

        t_probe = time.perf_counter()
        for p in proj_list:
            if target_layers is not None and p["layer_idx"] not in target_layers:
                continue
            w = p["module"].weight.data
            for b in config.marginal_bits:
                w.copy_(
                    quantize_weight(
                        orig_weights_cpu[p["name"]].to(device),
                        bits=b,
                        group_size=config.quant_group_size,
                        clip_ratio=config.clip_ratio,
                        quantizer=config.quantizer,
                        hqq_axis=config.hqq_axis,
                        hqq_zero_policy=config.hqq_zero_policy,
                    )
                )
                try:
                    loss_pb = _mean_ce_loss(model, calib_input_ids, batch_size, device)
                finally:
                    w.copy_(ref_weights[p["name"]])
                delta_loss[(p["name"], b)] = float(loss_pb - base_loss_4bit)
        _sync(device)
        probe_time_sec = float(time.perf_counter() - t_probe)
    finally:
        t_rest = time.perf_counter()
        for p in proj_list:
            w = p["module"].weight.data
            w.copy_(orig_weights_cpu[p["name"]].to(device=w.device, dtype=w.dtype))
        model.train(was_training)
        _sync(device)
        restore_time_sec = float(time.perf_counter() - t_rest)

    _sync(device)
    total_time_sec = float(time.perf_counter() - t0)
    return {
        "delta_loss": delta_loss,
        "base_loss_4bit": float(base_loss_4bit),
        "base_overhead_time_sec": base_overhead_time_sec,
        "probe_time_sec": probe_time_sec,
        "restore_time_sec": restore_time_sec,
        "label_time_sec": total_time_sec,
    }
