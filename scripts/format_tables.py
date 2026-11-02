"""Formats results from run_*.json into Markdown reports, LaTeX booktabs, and flat CSVs."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

PM = "\u00b1"  # plus-minus
RHO = "\u03c1"
TAU = "\u03c4"

ORDERED_ESTIMATORS = [
    "Scalar-MSE",
    "Scalar-L1",
    "Scalar-TrimmedMean",
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

# Mojibake variants (UTF-8 bytes of plus-minus / rho / tau mis-decoded as cp1252 or latin-1) come FIRST, so no stray
# lead character is left behind once the clean Unicode forms are mapped.
_MOJIBAKE_TO_LATEX = (
    ("\u00c2\u00b1", r"$\pm$"),  # plus-minus decoded as cp1252
    ("\u00cf\u0081", r"$\rho$"),  # rho followed by a C1 control char
    ("\u00cf\u00a0", r"$\rho$"),  # rho followed by NBSP
    ("\u00cf ", r"$\rho$"),  # rho followed by a plain space
    ("\u00cf\u201e", r"$\tau$"),  # tau decoded as cp1252
)
_UNICODE_TO_LATEX = (
    (PM, r"$\pm$"),
    (RHO, r"$\rho$"),
    (TAU, r"$\tau$"),
)


def _latex_escape(text: str) -> str:
    """Escape LaTeX specials (% & _ #) and map plus-minus / rho / tau (and mojibake forms) to math macros."""
    s = str(text)
    for bad, good in _MOJIBAKE_TO_LATEX:
        s = s.replace(bad, good)
    for ch, esc in ("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"):
        s = s.replace(ch, esc)
    for uni, good in _UNICODE_TO_LATEX:
        s = s.replace(uni, good)
    return s


def parse_args():
    parser = argparse.ArgumentParser(description="Format HDS-Quant experiment results.")
    parser.add_argument("--input-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def _is_missing(val) -> bool:
    return val is None or (isinstance(val, (int, float, np.floating)) and np.isnan(val))


def _fmt_ms(mean, std=0.0, precision: int = 4) -> str:
    if _is_missing(mean):
        return "-"
    if _is_missing(std) or float(std) <= 0:
        return f"{float(mean):.{precision}f}"
    return f"{float(mean):.{precision}f} {PM} {float(std):.{precision}f}"


def build_table1(summary: dict) -> Tuple[str, pd.DataFrame]:
    """Table 1: Sensitivity Ranking & Prediction Accuracy on Held-Out Layers."""
    ranking = summary.get("ranking_metrics", {})
    stability = summary.get("allocation_stability", {})
    est_list = [e for e in ORDERED_ESTIMATORS if e in ranking] or list(ranking.keys())

    rows = []
    for est in est_list:
        rm = ranking[est]
        # Cross-seed stability: average across budgets
        stab_vals = [float(v) for v in stability.get(est, {}).values() if not _is_missing(v)]
        stab_txt = f"{float(np.mean(stab_vals)):.3f}" if stab_vals else "-"

        rows.append({
            "Estimator": est,
            f"Spearman {RHO} (Global)": _fmt_ms(rm.get("spearman_global_mean"), rm.get("spearman_global_std")),
            f"Kendall {TAU}": _fmt_ms(rm.get("kendall_global_mean"), rm.get("kendall_global_std")),
            f"Within-Family {RHO}": _fmt_ms(rm.get("spearman_within_family_mean"), rm.get("spearman_within_family_std")),
            "Top-20% Recall": _fmt_ms(rm.get("top20_recall_mean"), rm.get("top20_recall_std")),
            "Cross-Seed Stability": stab_txt,
        })
    df = pd.DataFrame(rows)
    md = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    return md, df


def _ppl_bits_cell(ppl_mean, ppl_std, bits) -> str:
    """``PPL (achieved bits)``: PPL is only comparable between methods at matched achieved payloads."""
    bits_txt = f"{float(bits):.3f}b" if not _is_missing(bits) else "-"
    return f"{_fmt_ms(ppl_mean, ppl_std, precision=2)} ({bits_txt})"


def build_table2(summary: dict) -> Tuple[str, pd.DataFrame]:
    """Table 2: Held-Out Perplexity at Matched Weight-Payload Budgets (PPL with achieved average bits)."""
    ppl_data = summary.get("ppl_metrics", {})
    stability = summary.get("allocation_stability", {})
    calib_costs = summary.get("calibration_costs", {})
    uniform = summary.get("uniform_baselines", {})

    n_folds = summary.get("n_folds")
    train_col_header = (
        f"{int(n_folds) - 1}-of-{int(n_folds)} Fold Train Time (s)"
        if (isinstance(n_folds, int) and n_folds >= 2)
        else "Single-Fold Train Time (s)"
    )

    est_list = [e for e in ORDERED_ESTIMATORS if e in ppl_data] or list(ppl_data.keys())
    budgets = sorted(next(iter(ppl_data.values())).keys()) if ppl_data else []
    uniform_bits = {"FP16": 16.0, "Uniform-4bit": 4.0, "Uniform-3bit": 3.0, "Uniform-2bit": 2.0}

    rows = []
    # Add Uniform baseline rows first
    for u_name in ("FP16", "Uniform-4bit", "Uniform-3bit", "Uniform-2bit"):
        if u_name in uniform:
            ppl_val = uniform[u_name]["ppl_mean"]
            ppl_std = uniform[u_name]["ppl_std"]
            u_row = {"Estimator": f"Baseline: {u_name}"}
            for b in budgets:
                u_row[f"PPL @ {b}b (bits)"] = _ppl_bits_cell(ppl_val, ppl_std, uniform_bits[u_name])
            u_row["Max Payload Gap (b)"] = "-"
            u_row["Seed Agreement"] = "1.000"
            u_row["Total Calib Time (s)"] = "0.0s"
            u_row[train_col_header] = "0.0s"
            rows.append(u_row)

    for est in est_list:
        row = {"Estimator": est}
        gaps = []
        for b in budgets:
            b_info = ppl_data[est].get(b, {})
            achieved = b_info.get("achieved_avg_bits_mean")
            row[f"PPL @ {b}b (bits)"] = _ppl_bits_cell(
                b_info.get("ppl_mean"), b_info.get("ppl_std"), achieved
            )
            if not _is_missing(achieved):
                gaps.append(abs(float(b) - float(achieved)))
        row["Max Payload Gap (b)"] = f"{max(gaps):.3f}" if gaps else "-"

        stab_vals = [float(v) for v in stability.get(est, {}).values() if not _is_missing(v)]
        stab_txt = f"{float(np.mean(stab_vals)):.3f}" if stab_vals else "-"
        row["Seed Agreement"] = stab_txt

        c = calib_costs.get(est, {}).get("total_calib_time_sec", 0.0)
        c_str = f"{float(c):.1f}s" if not _is_missing(c) else "-"
        row["Total Calib Time (s)"] = c_str
        c_single = calib_costs.get(est, {}).get("single_fold_calib_time_sec", c)
        c_single_str = f"{float(c_single):.1f}s" if not _is_missing(c_single) else "-"
        row[train_col_header] = c_single_str
        rows.append(row)

    df = pd.DataFrame(rows)
    md = tabulate(df, headers="keys", tablefmt="github", showindex=False)
    return md, df


def build_table3_ablation(run_files: List[Path]) -> str:
    """Table 3: Ablation Comparison Across Runs."""
    if len(run_files) < 2:
        return "*Ablation table requires at least 2 run JSON files (e.g., strict_baseline and salience_int4_tweak).*"

    rows = []
    for p in run_files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        sm = data.get("summary", {})
        tag = cfg.get("run_tag", p.stem.replace("run_", ""))
        rm = sm.get("ranking_metrics", {})
        ppl = sm.get("ppl_metrics", {})

        # Extract HDS-Hybrid metrics as anchor
        hybrid_rm = rm.get("HDS-Hybrid", {})
        hybrid_ppl3 = ppl.get("HDS-Hybrid", {}).get("3.00", {}).get("ppl_mean")
        mse_ppl3 = ppl.get("Scalar-MSE", {}).get("3.00", {}).get("ppl_mean")

        rows.append({
            "Run Tag": tag,
            "Anchor": cfg.get("residual_anchor", "-"),
            "Binding": cfg.get("coord_binding", "-"),
            "Grouping": cfg.get("readout_grouping", "-"),
            "Transform": cfg.get("target_transform", "-"),
            f"HDS-Hybrid Spearman {RHO}": f"{float(hybrid_rm.get('spearman_global_mean', 0.0)):.4f}" if not _is_missing(hybrid_rm.get('spearman_global_mean')) else "-",
            "HDS-Hybrid Top-20%": f"{float(hybrid_rm.get('top20_recall_mean', 0.0)):.4f}" if not _is_missing(hybrid_rm.get('top20_recall_mean')) else "-",
            "PPL @ 3.0b (HDS-Hybrid)": f"{float(hybrid_ppl3):.2f}" if not _is_missing(hybrid_ppl3) else "-",
            "PPL @ 3.0b (Scalar-MSE)": f"{float(mse_ppl3):.2f}" if not _is_missing(mse_ppl3) else "-",
        })
    df = pd.DataFrame(rows)
    return tabulate(df, headers="keys", tablefmt="github", showindex=False)


def build_table4_low_shot(data: dict) -> str:
    """Table 4: Fixed-K Layer-Acquisition Curve (K labeled layers in total; ranking on the unlabeled layers)."""
    curve = data.get("low_shot_curve", {})
    if not curve:
        return "*No low-shot curve data available in run.*"

    rows = []
    k_layers = sorted(curve.keys(), key=int)
    for k in k_layers:
        for est, stats in curve[k].items():
            calib_t = stats.get("acquisition_cost_sec_mean")
            calib_str = f"{float(calib_t):.1f}s" if not _is_missing(calib_t) else "-"
            rows.append({
                "Labeled Layers (K)": k,
                "Estimator": est,
                f"Spearman {RHO} (Held-Out Layers)": _fmt_ms(stats.get("spearman_global_mean"), stats.get("spearman_global_std")),
                "Top-20% Recall": _fmt_ms(stats.get("top20_recall_mean"), stats.get("top20_recall_std")),
                "PPL @ 3.00b (bits)": _ppl_bits_cell(
                    stats.get("ppl_3bit_mean"),
                    stats.get("ppl_3bit_std"),
                    stats.get("achieved_avg_bits_mean"),
                ),
                "Total K-Layer Calib Time (s)": calib_str,
            })
    df = pd.DataFrame(rows)
    return tabulate(df, headers="keys", tablefmt="github", showindex=False)


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert a dataframe into clean LaTeX booktabs format (headers, caption and cells are LaTeX-escaped)."""
    cols = "l" + "c" * (len(df.columns) - 1)
    header = " & ".join(_latex_escape(c) for c in df.columns) + r" \\"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df.iterrows():
        escaped_row = [_latex_escape(x) for x in row]
        lines.append(" & ".join(escaped_row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def build_summary_csv(run_files: List[Path], out_csv: Path):
    """Flatten all runs and metrics into a CSV for plotting."""
    records = []
    for p in run_files:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tag = data.get("config", {}).get("run_tag", p.stem)
        sm = data.get("summary", {})
        rm = sm.get("ranking_metrics", {})
        ppl = sm.get("ppl_metrics", {})
        costs = sm.get("calibration_costs", {})

        for est, rm_stats in rm.items():
            rec = {
                "run_tag": tag,
                "estimator": est,
                "spearman_global_mean": rm_stats.get("spearman_global_mean"),
                "spearman_global_std": rm_stats.get("spearman_global_std"),
                "kendall_global_mean": rm_stats.get("kendall_global_mean"),
                "spearman_within_family_mean": rm_stats.get("spearman_within_family_mean"),
                "top20_recall_mean": rm_stats.get("top20_recall_mean"),
                "total_calib_time_sec": costs.get(est, {}).get("total_calib_time_sec"),
            }
            if est in ppl:
                for b, b_stats in ppl[est].items():
                    rec[f"ppl_{b}b_mean"] = b_stats.get("ppl_mean")
                    rec[f"ppl_{b}b_std"] = b_stats.get("ppl_std")
                    rec[f"bits_{b}b_mean"] = b_stats.get("achieved_avg_bits_mean")
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved flat summary CSV to {out_csv}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("run_*.json"))
    if not json_files:
        print(f"No run_*.json files found in {input_dir}.")
        return

    # Use the primary/latest run file for Table 1 and Table 2
    primary_file = json_files[-1]
    for p in json_files:
        if "tweak" in p.name or "canonical" in p.name:
            primary_file = p
            break

    print(f"Generating reports using primary file: {primary_file.name}")
    with open(primary_file, "r", encoding="utf-8") as f:
        primary_data = json.load(f)

    sm = primary_data.get("summary", {})
    t1_md, t1_df = build_table1(sm)
    t2_md, t2_df = build_table2(sm)
    t3_md = build_table3_ablation(json_files)
    t4_md = build_table4_low_shot(primary_data)

    # 1. SUPERVISOR_REPORT.md
    report_content = f"""# HDS-Quant Experiment Evaluation & Supervisor Report

**Primary Run Artifact**: `{primary_file.name}`  
**Model**: `{primary_data.get('config', {}).get('model_id', 'N/A')}`  
**Anchor / Binding**: `{primary_data.get('config', {}).get('residual_anchor', 'N/A')}` / `{primary_data.get('config', {}).get('coord_binding', 'N/A')}`  
**Readout Grouping / Transform**: `{primary_data.get('config', {}).get('readout_grouping', 'N/A')}` / `{primary_data.get('config', {}).get('target_transform', 'N/A')}`  
**Seeds Evaluated**: `{primary_data.get('config', {}).get('seeds', [])}`  

---

## Table 1: Sensitivity Ranking & Prediction Accuracy on Held-Out Layers

{t1_md}

---

## Table 2: Held-Out Perplexity at Matched Weight-Payload Budgets

{t2_md}

---

## Table 3: Ablation Comparison Across Runs

{t3_md}

---

## Table 4: Low-Shot Layer Calibration Amortization Curve

{t4_md}
"""
    report_file = output_dir / "SUPERVISOR_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Generated Supervisor Report at {report_file}")

    # 2. LaTeX Tables
    t1_tex = df_to_latex(t1_df, "Sensitivity Ranking & Prediction Accuracy on Held-Out Layers", "tab:ranking_metrics")
    t2_tex = df_to_latex(t2_df, "Held-Out Perplexity at Matched Weight-Payload Budgets", "tab:ppl_budgets")
    tex_file = output_dir / "tables.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX booktabs tables for HDS-Quant\n\n" + t1_tex + "\n\n" + t2_tex + "\n")
    print(f"Generated LaTeX tables at {tex_file}")

    # 3. Flat CSV
    build_summary_csv(json_files, output_dir / "summary_metrics.csv")


if __name__ == "__main__":
    main()
