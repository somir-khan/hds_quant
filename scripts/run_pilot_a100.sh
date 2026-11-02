#!/bin/bash
# ==============================================================================
# SLURM Submission Script for LONI (1x NVIDIA A100 80GB GPU)
# Project: HDS-Quant End-to-End Pilot Experiment
# ==============================================================================
#SBATCH --job-name=hds_quant_pilot
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results/pilot_%j.log
#SBATCH --error=results/pilot_%j.err

set -euo pipefail

echo "=== Starting HDS-Quant Pilot Experiment on $(hostname) ==="
date

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

mkdir -p results/cache

# ------------------------------------------------------------------------------
# Pre-flight: CUDA Kernel Verification & HDS Latency Budget (<5 ms)
# ------------------------------------------------------------------------------
echo ">>> Running CUDA verification and latency tests..."
python -m pytest tests/test_stage1_math.py -k "cuda or encode_speed_budget" -v

# ------------------------------------------------------------------------------
# Run 1: Strict Proposal Baseline
# Residual anchor: fp16_strict, Coord binding: raw_index_strict, Readout: pooled
# Computes marginal loss labels for all seeds and caches them.
# ------------------------------------------------------------------------------
echo ">>> [1/3] Executing Run 1: Strict Proposal Baseline..."
python -m hds_quant.run_experiment \
    --model-id "meta-llama/Llama-3.2-1B" \
    --eval-dataset "wikitext2" \
    --residual-anchor "fp16_strict" \
    --coord-binding "raw_index_strict" \
    --readout-grouping "pooled" \
    --target-transform "signed_log1p" \
    --seeds 42 100 2026 \
    --budgets 2.50 2.75 3.00 3.25 3.50 \
    --cache-dir "results/cache" \
    --reuse-cache \
    --output-dir "results" \
    --run-tag "strict_baseline"

# ------------------------------------------------------------------------------
# Run 2: Tweaked Canonical Model
# Residual anchor: int4_aligned, Coord binding: salience_bucket, Readout: per_bit
# Reuses the exact same cached labels (0.0s label time) and evaluates low-shot curve.
# ------------------------------------------------------------------------------
echo ">>> [2/3] Executing Run 2: Tweaked Canonical Model with Low-Shot Curve..."
python -m hds_quant.run_experiment \
    --model-id "meta-llama/Llama-3.2-1B" \
    --eval-dataset "wikitext2" \
    --residual-anchor "int4_aligned" \
    --coord-binding "salience_bucket" \
    --readout-grouping "per_bit" \
    --target-transform "signed_log1p" \
    --seeds 42 100 2026 \
    --budgets 2.50 2.75 3.00 3.25 3.50 \
    --run-low-shot-curve \
    --cache-dir "results/cache" \
    --reuse-cache \
    --output-dir "results" \
    --run-tag "salience_int4_tweak"

# ------------------------------------------------------------------------------
# Step 3: Format tables, Markdown report, LaTeX booktabs, and summary CSV
# ------------------------------------------------------------------------------
echo ">>> [3/3] Generating Supervisor Report and Conference Tables..."
python scripts/format_tables.py --input-dir "results" --output-dir "results"

echo "=== All runs completed successfully ==="
date
