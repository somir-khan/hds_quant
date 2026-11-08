"""Paper-defined LIM ranking with native whole-block 2/4 allocation, adapted to HQQ.

Dumitru et al., Findings of ACL 2025, sections 3.1--3.3:
https://aclanthology.org/2025.findings-acl.29/
This is an independent implementation; the authors' repository currently contains
only a README. Token-mean cosine aggregation and the matched calibration corpus
are explicit choices in this adaptation, not claims of exact paper reproduction.
"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from .config import PROJ_FAMILIES
from .evaluator import evaluate_perplexity
from .label_collector import discover_target_projections
from .run_limited_labels import memory_start, memory_finish, sync

METHOD = "LIM-Block-24"
SOURCE = "https://aclanthology.org/2025.findings-acl.29/"


@torch.no_grad()
def collect_lim_scores(model, calib_ids, batch_size=1):
    """Negative mean tokenwise cosine(input, output) for each FP decoder block.

    Inputs are fixed-length unpadded sequences, as in the existing data pipeline.
    Each token gets equal weight, including in a final partial batch. Only scalar
    sums/counts are retained; no hidden-state bank or sensitivity labels are built.
    """
    if calib_ids.ndim != 2 or calib_ids.numel() == 0 or batch_size < 1:
        raise ValueError("LIM requires nonempty fixed-length calibration sequences and positive batch size")
    backbone = getattr(model, "model", None)
    blocks = getattr(backbone, "layers", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("LIM currently supports Llama-style model.model.layers only")
    device = next(model.parameters()).device
    sums = torch.zeros(len(blocks), dtype=torch.float64, device=device)
    counts = [0] * len(blocks)
    handles = []
    was_training = model.training

    def make_hook(index):
        def hook(module, args, kwargs, output):
            hidden_in = args[0] if args else kwargs["hidden_states"]
            hidden_out = output[0] if isinstance(output, (tuple, list)) else output
            if hidden_in.shape != hidden_out.shape or hidden_in.ndim != 3:
                raise ValueError("LIM block input/output must have matching [batch, tokens, hidden] shapes")
            cosine = F.cosine_similarity(hidden_in.float(), hidden_out.float(), dim=-1, eps=1e-8)
            sums[index].add_(-cosine.sum(dtype=torch.float64))
            counts[index] += cosine.numel()
        return hook

    try:
        model.eval()
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(make_hook(index), with_kwargs=True))
        for start in range(0, len(calib_ids), batch_size):
            # The language-model head is unnecessary for hidden-state scoring.
            output = backbone(input_ids=calib_ids[start:start + batch_size].to(device),
                              use_cache=False, return_dict=True)
            del output
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if any(n != calib_ids.numel() for n in counts):
        raise RuntimeError("LIM did not observe every calibration token exactly once in every block")
    scores = (sums / torch.tensor(counts, device=device)).cpu().tolist()
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("Nonfinite LIM scores")
    return {i: score for i, score in enumerate(scores)}, counts


def allocate_lim_blocks(scores, meta, target_avg_bits):
    """Protect highest-LIM blocks at 4 bits, all others at 2; exact target payload.

    Equal-sized complete blocks are required for the native rank/count rule. Do
    not silently change it to a projection allocator or underfill the budget.
    """
    by_block = defaultdict(list)
    for projection in meta:
        by_block[projection["layer_idx"]].append(projection)
    if not by_block or set(by_block) != set(scores):
        raise ValueError("LIM scores must cover exactly the quantized blocks")
    expected_families = {name.split(".")[-1] for name in PROJ_FAMILIES}
    for projections in by_block.values():
        if len(projections) != len(expected_families) or {p["family"] for p in projections} != expected_families:
            raise ValueError("LIM requires all seven target projections in each block")
    if len({p["name"] for p in meta}) != len(meta) or any(p["num_params"] <= 0 for p in meta):
        raise ValueError("Invalid target projection metadata")
    sizes = [sum(p["num_params"] for p in ps) for ps in by_block.values()]
    if len(set(sizes)) != 1:
        raise ValueError("Native LIM allocation requires equal target-weight counts per block")
    if not math.isfinite(target_avg_bits) or not 2 <= target_avg_bits <= 4:
        raise ValueError("LIM budget must be between 2 and 4")
    if not all(math.isfinite(s) for s in scores.values()):
        raise ValueError("Nonfinite LIM scores")
    high_count = (target_avg_bits - 2) * len(scores) / 2
    if not math.isclose(high_count, round(high_count), abs_tol=1e-9, rel_tol=0):
        raise ValueError("LIM cannot exactly meet this budget with whole 2/4-bit blocks")
    ranking = sorted(scores, key=lambda i: (-scores[i], i))
    high = set(ranking[:round(high_count)])
    assignment = {p["name"]: 4 if p["layer_idx"] in high else 2 for p in meta}
    total = sum(p["num_params"] for p in meta)
    achieved = sum(p["num_params"] * assignment[p["name"]] for p in meta) / total
    if not math.isclose(achieved, target_avg_bits, abs_tol=1e-10, rel_tol=0):
        raise RuntimeError("LIM payload does not match target")
    return {"bit_assignment": assignment, "achieved_avg_bits": achieved,
            "bit_counts": dict(Counter(assignment.values())), "success": True,
            "payload_lower_bound": achieved, "total_predicted_cost": None,
            "block_ranking": ranking, "high_precision_blocks": sorted(high),
            "block_bit_counts": {2: len(scores) - len(high), 4: len(high)}}


def run_lim_trial(model, calib_ids, eval_ids, cfg, budgets, seed, batch_size):
    """Independent directly timed LIM calibration and shared-HQQ evaluation."""
    print(f"Starting seed={seed} method={METHOD}; no sensitivity labels", flush=True)
    device = next(model.parameters()).device
    meta = discover_target_projections(model)
    # Reject unsupported geometry/budgets before spending a calibration pass.
    for budget in budgets:
        allocate_lim_blocks({i: 0.0 for i in range(cfg.num_layers)}, meta, budget)
    baseline = memory_start(device)
    sync(device)
    start = time.perf_counter()
    scores, counts = collect_lim_scores(model, calib_ids, batch_size)
    sync(device)
    shared = time.perf_counter() - start
    allocations = {}
    for budget in budgets:
        t0 = time.perf_counter()
        allocation = allocate_lim_blocks(scores, meta, budget)
        sync(device)
        elapsed = time.perf_counter() - t0
        allocations[f"{budget:.2f}"] = dict(allocation, allocation_sec=elapsed,
                                            calibration_to_budget_sec=shared + elapsed)
    sync(device)
    result = {"seed": seed, "method": METHOD, "trial_kind": "external_baseline",
              "k_layers": 0, "acquired_layers": [], "probe_label_count": 0,
              "source": SOURCE, "implementation": "independent paper-defined HQQ adaptation",
              "score_definition": "negative mean over all calibration tokens of FP block input/output cosine",
              "score_forward_scope": "decoder backbone; no LM head", "scores": scores,
              "tokens_per_block": counts, "allocation_granularity": "block",
              "allocation_candidate_bits": [2, 4], "label_acquisition_sec": 0.0, "fit_sec": 0.0,
              "feature_extraction_sec": shared, "calibration_shared_sec": shared,
              "calibration_wall_sec": time.perf_counter() - start,
              "feature_payload_bytes": 8 * len(scores), "encoder_codebook_bytes": 0,
              "allocations": allocations, **memory_finish(device, baseline)}
    for allocation in allocations.values():
        allocation["evaluation"] = evaluate_perplexity(model, eval_ids, meta, cfg,
            bit_assignment=allocation["bit_assignment"], batch_size=batch_size)
    print(f"Done: {METHOD} calibration={result['calibration_wall_sec']:.2f}s, labels=0", flush=True)
    return result
