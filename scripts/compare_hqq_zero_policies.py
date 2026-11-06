"""Summarize paired zero-point policies without selecting a winner automatically."""
import argparse
import csv
import json
import math
from pathlib import Path


def compare(root):
    runs = {p: json.loads((root / p / f"run_{p}.json").read_text(encoding="utf-8"))
            for p in ("floating", "round4")}
    a, b = (runs[p] for p in ("floating", "round4"))
    for key in ("model_id", "eval_dataset", "hqq_axis", "quant_group_size", "seeds",
                "candidate_bits", "budgets", "residual_anchor", "coord_binding", "readout_grouping",
                "target_transform", "n_calib_seqs", "calib_seq_len", "max_encoded_tokens",
                "eval_seq_len", "batch_size", "cv_folds", "estimators"):
        if a["config"].get(key) != b["config"].get(key):
            raise ValueError(f"Unmatched experiment setting: {key}")
    for policy, d in runs.items():
        if d["config"].get("hqq_zero_policy") != policy:
            raise ValueError(f"Wrong zero policy in {policy}")
    for name in ("FP16", "Uniform-1bit", "Uniform-2bit", "Uniform-3bit"):
        if not math.isclose(a["uniform_baselines"][name]["nll"], b["uniform_baselines"][name]["nll"], abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Unexpected change to unchanged baseline: {name}")
    rows = []

    def add(name, budget, av, bv):
        if not all(math.isfinite(v) and v > 0 for v in (av, bv)):
            raise ValueError(f"Invalid PPL for {name}")
        rows.append(dict(method=name, average_bits=budget, floating_ppl=av, round4_ppl=bv,
                         round4_minus_floating=bv-av, relative_change_percent=100*(bv/av-1)))

    for name, v in a["uniform_baselines"].items():
        add(name, "reference", v["ppl"], b["uniform_baselines"][name]["ppl"])
    for est, budgets in a["summary"]["ppl_metrics"].items():
        for budget, v in budgets.items():
            other = b["summary"]["ppl_metrics"][est][budget]
            if not math.isclose(v["achieved_avg_bits_mean"], other["achieved_avg_bits_mean"], abs_tol=1e-9):
                raise ValueError(f"Payload mismatch for {est} at {budget}")
            add(est, budget, v["ppl_mean"], other["ppl_mean"])
    with (root / "zero_policy_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["HQQ zero-point policy comparison", "", "Negative percentage means round4 has lower perplexity.",
             "One calibration seed (42): exploratory comparison, not a statistical significance test.",
             "Both mixed-precision runs use the full held-out evaluation stream.", "",
             "Method | Budget | Floating PPL | Round4 PPL | Change"]
    for r in rows:
        lines.append(f"{r['method']} | {r['average_bits']} | {r['floating_ppl']:.4f} | {r['round4_ppl']:.4f} | {r['relative_change_percent']:+.2f}%")
    text = "\n".join(lines) + "\n"
    (root / "zero_policy_comparison.txt").write_text(text, encoding="utf-8")
    print(text)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("results_hqq_zero_ablation"))
    compare(p.parse_args().input_dir)
