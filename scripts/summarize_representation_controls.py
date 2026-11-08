"""Validate the focused protocol and emit paired seed differences with explicit scopes."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .summarize_limited_labels import summarize as summarize_shared
except ImportError:  # Direct script invocation from the launcher.
    from summarize_limited_labels import summarize as summarize_shared

LEARNED = ("HDS-Hybrid", "Coord-Rank-Hist", "Coord-Rank-Exact")
LIM = "LIM-Block-24"
PAIRS = (("HDS-Hybrid", "Coord-Rank-Exact"), ("HDS-Hybrid", "Coord-Rank-Hist"),
         ("HDS-Hybrid", LIM), ("Coord-Rank-Exact", "Coord-Rank-Hist"), ("Coord-Rank-Exact", LIM))


def validate_result(data):
    if not data.get("complete") or data.get("comparison_protocol", {}).get("schema") != "representation_controls_v1":
        raise ValueError("Expected a complete representation-controls run")
    args = data["args"]
    expected = {(seed, method) for seed in args["seeds"] for method in (*LEARNED, LIM)}
    lookup = {(t["seed"], t["method"]): t for t in data["trials"]}
    if len(lookup) != len(data["trials"]) or set(lookup) != expected or data["references"]:
        raise ValueError("Missing, duplicate or unexpected trials/references")
    budget_keys = {f"{b:.2f}" for b in args["budgets"]}
    eval_counts = data["uniform_baselines"]["FP16"]["seq_token_counts"]
    for seed in args["seeds"]:
        control = lookup[(seed, "HDS-Hybrid")]
        for method in (*LEARNED, LIM):
            trial = lookup[(seed, method)]
            if method in LEARNED:
                if (trial["k_layers"] != args["k_layers"][0] or
                        trial["acquired_layers"] != control["acquired_layers"] or
                        trial["probe_label_count"] != 7 * trial["k_layers"] * 3 or
                        trial["measured_labels"].keys() != control["measured_labels"].keys()):
                    raise ValueError("Representation controls did not acquire the same K-block labels")
                if any(not math.isclose(v, control["measured_labels"][key], abs_tol=1e-6, rel_tol=1e-5)
                       for key, v in trial["measured_labels"].items()):
                    raise ValueError("Repeated measured labels differ; inspect raw trials before comparing")
            elif trial["probe_label_count"] != 0 or trial["k_layers"] != 0:
                raise ValueError("LIM must use zero sensitivity labels")
            if set(trial["allocations"]) != budget_keys:
                raise ValueError("Incomplete budget allocations")
            for budget, allocation in trial["allocations"].items():
                allowed = {2, 4} if method == LIM else {2, 3, 4}
                if (not allocation["success"] or not allocation["bit_assignment"] or
                        not set(allocation["bit_assignment"].values()) <= allowed or
                        not math.isclose(allocation["achieved_avg_bits"], float(budget), rel_tol=0, abs_tol=1e-8)):
                    raise ValueError("Invalid allocation or unmatched weight-bit budget")
                evaluation = allocation["evaluation"]
                if evaluation["seq_token_counts"] != eval_counts or len(evaluation["seq_nlls"]) != len(eval_counts):
                    raise ValueError("Mismatched evaluation token counts")
                if not all(math.isfinite(evaluation[key]) for key in ("ppl", "nll")):
                    raise ValueError("Nonfinite evaluation")
    return lookup


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, make_plots=True):
    root = Path(root)
    data = json.loads((root / "limited_labels.json").read_text(encoding="utf-8"))
    lookup = validate_result(data)
    summary = summarize_shared(root, make_plots=make_plots)
    pairs = []
    for seed in data["args"]["seeds"]:
        for left, right in PAIRS:
            a, b = lookup[(seed, left)], lookup[(seed, right)]
            for budget in a["allocations"]:
                aa, bb = a["allocations"][budget], b["allocations"][budget]
                pairs.append({"seed": seed, "method_a": left, "method_b": right, "budget": float(budget),
                    "ppl_a_minus_b": aa["evaluation"]["ppl"] - bb["evaluation"]["ppl"],
                    "nll_a_minus_b": aa["evaluation"]["nll"] - bb["evaluation"]["nll"],
                    "calibration_sec_a_minus_b": aa["calibration_to_budget_sec"] - bb["calibration_to_budget_sec"],
                    "feature_bytes_a_minus_b": a["feature_payload_bytes"] - b["feature_payload_bytes"]})
    grouped = defaultdict(list)
    for row in pairs:
        grouped[(row["method_a"], row["method_b"], row["budget"])].append(row)
    aggregate = []
    for (a, b, budget), rows in sorted(grouped.items()):
        row = {"method_a": a, "method_b": b, "budget": budget, "n_seeds": len(rows),
               "seeds_a_lower_ppl": sum(r["ppl_a_minus_b"] < 0 for r in rows)}
        for metric in ("ppl_a_minus_b", "nll_a_minus_b", "calibration_sec_a_minus_b", "feature_bytes_a_minus_b"):
            values = [r[metric] for r in rows]
            row[metric + "_mean"] = statistics.mean(values)
            row[metric + "_std"] = statistics.pstdev(values)
        aggregate.append(row)
    write_csv(root / "paired_trials.csv", pairs)
    write_csv(root / "paired_summary.csv", aggregate)
    protocols = [{"method": method, "allocation_bits": [2, 4] if method == LIM else [2, 3, 4],
                  "granularity": "block" if method == LIM else "projection",
                  "k_label_blocks": 0 if method == LIM else data["args"]["k_layers"][0]}
                 for method in (*LEARNED, LIM)]
    (root / "method_protocols.json").write_text(json.dumps(protocols, indent=2), encoding="utf-8")
    notes = ["FOCUSED REPRESENTATION CONTROLS + LIM/HQQ", "",
             *[data["comparison_protocol"][key] for key in
               ("representation_comparison", "external_comparison", "budget_scope", "label_scope")],
             "No full-bank reference rerun. Prior results are not pooled into these tables.",
             "Paired files: difference = method A minus method B; negative PPL/NLL/time favors A.",
             "Three calibration seeds (default) describe variability; their standard deviation is not a confidence interval."]
    summary_path = root / "SUMMARY.txt"
    summary_path.write_text("\n".join(notes) + "\n\n" + summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    print("\n".join(notes))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summarize(args.input_dir, make_plots=not args.no_plots)
