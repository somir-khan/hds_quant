"""Allocation-only intervention using an immutable completed limited-label run.

No sensitivity probes, feature extraction, regression, or calibration are run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
import csv
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .allocator import _FEASIBLE_PAYLOAD_CACHE, solve_bit_allocation_ilp
from .config import ExperimentConfig
from .data_utils import get_eval_tokens
from .evaluator import evaluate_perplexity
from .label_collector import discover_target_projections
from .quantizer import get_effective_hqq_config
from .readout_cv import _build_alloc_costs
from .run_experiment import _build_smoke_model

SCHEMA = "allocation_only_no1bit_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def decode_pairs(values):
    result = {}
    for key, value in values.items():
        name, bit = key.rsplit("|", 1)
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite source prediction: {key}")
        result[(name, int(bit))] = float(value)
    return result


def reconstruct_costs(trial, meta, cfg):
    """Exactly reconstruct the original mixed measured/predicted cost bank."""
    measured = decode_pairs(trial["measured_labels"])
    predicted = decode_pairs(trial["heldout_predictions"])
    if measured.keys() & predicted.keys():
        raise ValueError("Source measured labels overlap held-out predictions")
    expected = {(m["name"], b) for m in meta for b in cfg.marginal_bits}
    if set(measured) | set(predicted) != expected:
        raise ValueError("Source predictions/labels do not cover the original projection/bit pairs")
    acquired = set(trial["acquired_layers"])
    expected_measured = {(m["name"], b) for m in meta if m["layer_idx"] in acquired for b in cfg.marginal_bits}
    if set(measured) != expected_measured:
        raise ValueError("Source measured labels do not match acquired blocks")
    raw = {**predicted, **measured}
    return _build_alloc_costs({trial["method"]: raw}, meta, (trial["method"],),
        marginal_bits=cfg.marginal_bits, reference_bit=cfg.reference_bit)[trial["method"]]


def payload(assignment, meta):
    if set(assignment) != {m["name"] for m in meta}:
        raise ValueError("Assignment does not cover exactly the model's target projections")
    return sum(int(assignment[m["name"]]) * int(m["num_params"]) for m in meta)


def plan_intervention(trial, budget_key, meta, cfg):
    costs = reconstruct_costs(trial, meta, cfg)
    source = trial["allocations"][budget_key]
    original = source["bit_assignment"]
    if not source.get("success") or any(b not in cfg.candidate_bits for b in original.values()):
        raise ValueError("Invalid original allocation")
    cost = sum(costs[(name, b)] for name, b in original.items())
    if not math.isclose(cost, source["total_predicted_cost"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Reconstructed objective differs from the saved source objective")
    source_payload = payload(original, meta)
    total_params = sum(m["num_params"] for m in meta)
    if not math.isclose(source_payload / total_params, source["achieved_avg_bits"], abs_tol=1e-10):
        raise ValueError("Source payload metadata is inconsistent")
    restricted_bits = tuple(b for b in cfg.candidate_bits if b != 1)
    if not min(restricted_bits) <= float(budget_key) <= max(restricted_bits):
        raise ValueError("Source budget is infeasible without 1-bit choices")
    one_bit = sorted(n for n, b in original.items() if b == 1)
    if one_bit:
        # Costs are constructed using the ORIGINAL marginal bits. Dropping bit 1
        # changes only the feasible allocation set, not fitting or objective values.
        restricted_costs = {key: value for key, value in costs.items() if key[1] in restricted_bits}
        _FEASIBLE_PAYLOAD_CACHE.clear()
        start = time.perf_counter()
        after = solve_bit_allocation_ilp(restricted_costs, meta, float(budget_key), candidate_bits=restricted_bits)
        solve_sec = time.perf_counter() - start
        if not after["success"]:
            raise RuntimeError("Restricted allocation failed")
    else:
        # Preserve a valid original solution rather than introduce an ILP tie-breaking
        # difference in a control where forbidding 1 bit has no effect.
        after = {key: source[key] for key in ("bit_assignment", "achieved_avg_bits", "total_predicted_cost", "bit_counts", "success", "payload_lower_bound")}
        solve_sec = 0.0
    if payload(after["bit_assignment"], meta) != source_payload:
        raise ValueError("Restricted allocation does not exactly match the original bit payload")
    if any(b == 1 for b in after["bit_assignment"].values()):
        raise RuntimeError("Restricted allocation still contains 1-bit projections")
    changes = [{"projection": n, "before": original[n], "after": after["bit_assignment"][n]}
               for n in sorted(original) if original[n] != after["bit_assignment"][n]]
    return {"original_one_bit_projections": one_bit, "changed_projections": changes,
            "restricted_allocation": after, "restricted_allocation_sec": solve_sec,
            "matched_integer_payload": source_payload,
            "prediction_bank_sha256": digest({"measured": trial["measured_labels"], "predicted": trial["heldout_predictions"]}),
            "original_cost_bank_sha256": digest({f"{n}|{b}": v for (n, b), v in costs.items()})}


def check_evaluation(actual, expected, atol):
    for value in (actual, expected):
        if (not math.isfinite(value["nll"]) or not math.isfinite(value["ppl"]) or value["ppl"] <= 0 or
            len(value["seq_nlls"]) != len(value["seq_token_counts"])):
            raise ValueError("Invalid source/replayed evaluation data")
    if actual["seq_token_counts"] != expected["seq_token_counts"]:
        raise ValueError("Evaluation token/chunk counts differ from source")
    gap = abs(actual["nll"] - expected["nll"])
    max_chunk_gap = max(abs(a-b) for a, b in zip(actual["seq_nlls"], expected["seq_nlls"]))
    if not math.isfinite(gap) or gap > atol or max_chunk_gap > atol:
        raise ValueError(f"Source evaluation replay failed: NLL gap={gap}, maximum chunk gap={max_chunk_gap}")
    return {"nll_abs_difference": gap, "max_chunk_nll_abs_difference": max_chunk_gap, "passed": True}


def summarize(report, output):
    rows = []
    for r in report["comparisons"]:
        before, after = r["source_evaluation"]["ppl"], r["restricted_evaluation"]["ppl"]
        rows.append({"seed": r["seed"], "method": r["method"], "k_layers": r["k_layers"], "budget": r["budget"],
            "source_ppl": before, "no1bit_ppl": after, "relative_change_percent": 100*(after/before-1),
            "source_1bit_count": len(r["original_one_bit_projections"]), "changed_projection_count": len(r["changed_projections"]),
            "original_replayed": r["original_replay"] is not None, "restricted_evaluation_origin": r["restricted_evaluation_origin"],
            "restricted_allocation_sec": r["restricted_allocation_sec"]})
    with (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["k_layers"], row["budget"])].append(row)
    lines = ["Allocation-only diagnostic: forbid 1-bit choices", "",
        "Negative change means lower perplexity after restricting candidates to {2,3,4}.",
        "Predictions and acquired labels are fixed. Exact integer weight payload is matched.",
        "The intervention includes all resulting reallocations; it does not isolate a single weight matrix.",
        "No new calibration-speed claim: only reallocation and evaluation are executed.", "",
        "Seed | Method | K | Bits | Original PPL | No-1-bit PPL | Change | Original 1-bit count | Changed projections"]
    for r in rows:
        lines.append(f"{r['seed']} | {r['method']} | {r['k_layers']} | {r['budget']} | {r['source_ppl']:.4f} | {r['no1bit_ppl']:.4f} | {r['relative_change_percent']:+.2f}% | {r['source_1bit_count']} | {r['changed_projection_count']}")
    lines += ["", "Across-seed means (standard deviations are not confidence intervals):"]
    for (method, k, budget), vals in sorted(groups.items()):
        bef, aft = [v["source_ppl"] for v in vals], [v["no1bit_ppl"] for v in vals]
        lines.append(f"{method}, K={k}, bits={budget}, n={len(vals)}: {statistics.mean(bef):.4f} +/- {statistics.pstdev(bef):.4f} -> {statistics.mean(aft):.4f} +/- {statistics.pstdev(aft):.4f}")
    (output / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(args):
    source_path = Path(args.source).resolve()
    source_bytes = source_path.read_bytes()
    data = json.loads(source_bytes)
    if data.get("schema") != "true_limited_labels_v1" or not data.get("complete"):
        raise ValueError("Source must be a completed true_limited_labels_v1 run")
    cfg = ExperimentConfig(**data["config"])
    cfg.candidate_bits = tuple(cfg.candidate_bits)
    if cfg.quantizer != "hqq" or tuple(sorted(cfg.candidate_bits)) != (1, 2, 3, 4):
        raise ValueError("Expected original HQQ candidate set {1,2,3,4}")
    if args.nll_atol < 0:
        raise ValueError("NLL tolerance must be nonnegative")
    selected = [t for t in data["trials"] if
                (args.seeds is None or t["seed"] in args.seeds) and
                (args.k_layers is None or t["k_layers"] in args.k_layers) and
                (args.methods is None or t["method"] in args.methods)]
    if not selected:
        raise ValueError("No source trials match the requested filters")
    for field, requested in (("seed", args.seeds), ("k_layers", args.k_layers), ("method", args.methods)):
        if requested and set(requested) - {t[field] for t in selected}:
            raise ValueError(f"Requested {field} values are absent from source")
    if args.budgets and any(f"{b:.2f}" not in t["allocations"] for t in selected for b in args.budgets):
        raise ValueError("Requested budget absent from a selected source trial")
    smoke = data["args"]["smoke_test"]
    if not smoke and not torch.cuda.is_available():
        raise RuntimeError("CUDA required to replay the CUDA source experiment")
    output = Path(args.output_dir).resolve()
    # Dedicated output only: never replace an earlier diagnostic or source run.
    output.mkdir(parents=True, exist_ok=False)
    if smoke:
        model, ids = _build_smoke_model()
    else:
        model_id = data["args"]["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda").eval()
        ids = get_eval_tokens(tokenizer, data["args"]["eval_dataset"], data["args"]["eval_seq_len"])
    batch_size = data["args"]["batch_size"]
    meta = discover_target_projections(model)
    current_hqq = get_effective_hqq_config(cfg.quant_group_size, next(model.parameters()).device,
        next(model.parameters()).dtype, axis=cfg.hqq_axis, zero_policy=cfg.hqq_zero_policy)
    if current_hqq != data["hqq_config"]:
        raise ValueError("Current effective HQQ configuration differs from source")
    report = {"schema": SCHEMA, "complete": False,
        "source": {"path": str(source_path), "sha256": hashlib.sha256(source_bytes).hexdigest(), "schema": data["schema"]},
        "args": vars(args), "source_config": data["config"], "hqq_config": current_hqq,
        "restricted_candidate_bits": [2, 3, 4], "comparisons": [], "environment": {}, "verification": {}}
    for package in ("torch", "transformers", "hqq", "scipy"):
        report["environment"][package] = importlib.metadata.version(package)
    report["evaluation_token_sha256"] = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    report["model_resolved_revision"] = getattr(model.config, "_commit_hash", None)
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), Path(__file__).with_name("allocator.py"), Path(__file__).with_name("readout_cv.py"),
         Path(__file__).with_name("evaluator.py"), Path(__file__).with_name("quantizer.py")]}

    def save():
        temp = output / "diagnostic.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(output / "diagnostic.json")

    save()
    # Validate model/data replay before any intervention. This also warms inference.
    for name, bit in (("FP16", None), ("Uniform-3bit", 3)):
        actual = evaluate_perplexity(model, ids, meta, cfg, uniform_bit=bit, batch_size=batch_size)
        report["verification"][name] = dict(check_evaluation(actual, data["uniform_baselines"][name], args.nll_atol), evaluation=actual)
        save()
        print(f"Source baseline verified: {name}", flush=True)
    save()
    for trial in selected:
        budgets = [f"{b:.2f}" for b in args.budgets] if args.budgets else list(trial["allocations"])
        for budget in budgets:
            print(f"Diagnostic seed={trial['seed']} K={trial['k_layers']} method={trial['method']} budget={budget}", flush=True)
            plan = plan_intervention(trial, budget, meta, cfg)
            original = trial["allocations"][budget]
            replay = None
            if plan["original_one_bit_projections"] or args.replay_all:
                evaluation = evaluate_perplexity(model, ids, meta, cfg,
                    bit_assignment=original["bit_assignment"], batch_size=batch_size)
                replay = dict(check_evaluation(evaluation, original["evaluation"], args.nll_atol), evaluation=evaluation)
            if plan["changed_projections"]:
                restricted = evaluate_perplexity(model, ids, meta, cfg,
                    bit_assignment=plan["restricted_allocation"]["bit_assignment"], batch_size=batch_size)
                origin = "new_evaluation"
            else:
                restricted = replay["evaluation"] if replay else original["evaluation"]
                origin = "replayed_unchanged_assignment" if replay else "unchanged_source_assignment"
            report["comparisons"].append({"seed": trial["seed"], "method": trial["method"], "k_layers": trial["k_layers"],
                "budget": budget, "source_evaluation": original["evaluation"], "source_assignment": original["bit_assignment"],
                "original_replay": replay, "restricted_evaluation": restricted, "restricted_evaluation_origin": origin, **plan})
            save()
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != report["source"]["sha256"]:
        raise RuntimeError("Source file changed while diagnostic was running")
    report["complete"] = True
    save()
    summarize(report, output)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--k-layers", type=int, nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--budgets", type=float, nargs="+")
    p.add_argument("--nll-atol", type=float, default=1e-5)
    p.add_argument("--replay-all", action="store_true", help="Also replay unchanged control allocations instead of reusing their saved evaluation")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
