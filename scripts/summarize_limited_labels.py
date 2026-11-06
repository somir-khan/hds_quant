"""Summarize directly measured cost versus PPL; optionally plot with matplotlib."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(root, make_plots=True):
    data = json.loads((root / "limited_labels.json").read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise ValueError("Run is incomplete; raw checkpoints remain available for diagnosis")
    rows = []
    for trial in data["trials"] + data["references"]:
        for budget, a in trial["allocations"].items():
            rows.append({"seed": trial["seed"], "method": trial["method"], "k_layers": trial["k_layers"],
                "budget": float(budget), "achieved_bits": a["achieved_avg_bits"], "ppl": a["evaluation"]["ppl"],
                "nll": a["evaluation"]["nll"], "calibration_sec": a["calibration_to_budget_sec"],
                "both_budgets_calibration_wall_sec": trial["calibration_wall_sec"],
                "evaluation_sec": a["evaluation"]["eval_time_sec"],
                "feature_payload_bytes": trial["feature_payload_bytes"],
                "encoder_codebook_bytes": trial["encoder_codebook_bytes"],
                "peak_cuda_allocated_bytes": trial["peak_cuda_allocated_bytes"],
                "peak_cuda_increment_bytes": trial["peak_cuda_increment_bytes"]})
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["method"], r["k_layers"], r["budget"])].append(r)
    summary = []
    for (method, k, budget), values in sorted(grouped.items()):
        result = {"method": method, "k_layers": k, "budget": budget, "n_seeds": len(values)}
        for field in ("ppl", "calibration_sec", "feature_payload_bytes", "encoder_codebook_bytes",
                      "peak_cuda_allocated_bytes", "peak_cuda_increment_bytes"):
            vals = [r[field] for r in values if r[field] is not None]
            result[field + "_mean"] = statistics.mean(vals) if vals else None
            result[field + "_std"] = statistics.pstdev(vals) if vals else None
        summary.append(result)
    for name, values in (("trials.csv", rows), ("summary.csv", summary)):
        with (root / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]))
            w.writeheader()
            w.writerows(values)
    info = data["args"]
    identity = [f"Experiment: {info.get('experiment_id', 'historical_limited_labels')}",
                f"Purpose: {info.get('experiment_purpose', 'benchmark')}",
                f"Model: {info['model_id']}", f"Dataset: {info['eval_dataset']}",
                f"Probe candidates: {data['config']['candidate_bits']}",
                f"Allocation candidates: {data.get('allocation_candidate_bits', data['config']['candidate_bits'])}"]
    if info.get("experiment_purpose") == "preflight":
        identity.append("PREFLIGHT ONLY: short evaluation, one seed, no full-label reference; not a full benchmark.")
    lines = ["True limited-label calibration", *identity, "", data["timing_scope"], data["memory_scope"], "",
        "Times below are measured shared calibration + the allocation solve for that budget.",
        "Standard deviations are across calibration seeds, not confidence intervals.", "",
        "Method | K blocks | Bits | PPL mean +/- std | Calibration seconds mean +/- std"]
    for r in summary:
        lines.append(f"{r['method']} | {r['k_layers']} | {r['budget']:.2f} | {r['ppl_mean']:.3f} +/- {r['ppl_std']:.3f} | {r['calibration_sec_mean']:.2f} +/- {r['calibration_sec_std']:.2f}")
    lines += ["", "Uniform baselines (their actual payloads are in their names):"]
    lines += [f"{name}: {v['ppl']:.4f} PPL" for name, v in data["uniform_baselines"].items()]
    (root / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if make_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Tables saved. Install matplotlib and rerun this summarizer to produce the plots.")
        else:
            for budget in sorted({r["budget"] for r in summary}):
                fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
                for method in sorted({r["method"] for r in summary}):
                    points = sorted((r for r in summary if r["method"] == method and r["budget"] == budget), key=lambda r:r["k_layers"])
                    ax.errorbar([p["calibration_sec_mean"] for p in points], [p["ppl_mean"] for p in points],
                                xerr=[p["calibration_sec_std"] for p in points], yerr=[p["ppl_std"] for p in points],
                                fmt="o-", capsize=3, label=method)
                    for p in points:
                        ax.annotate(f"K={p['k_layers']}", (p["calibration_sec_mean"], p["ppl_mean"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
                if budget == 3.0:
                    ax.axhline(data["uniform_baselines"]["Uniform-3bit"]["ppl"], linestyle="--", color="gray", label="Uniform 3-bit")
                ax.set(xlabel="Measured calibration time (seconds)", ylabel="WikiText-2 perplexity (lower is better)" if data["args"]["eval_dataset"] == "wikitext2" else "Perplexity (lower is better)",
                       title=f"Calibration cost versus quality at {budget:.2f} average bits")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
                fig.savefig(root / f"cost_quality_{budget:.2f}b.png", dpi=180)
                fig.savefig(root / f"cost_quality_{budget:.2f}b.pdf")
                plt.close(fig)
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summarize(args.input_dir, make_plots=not args.no_plots)
