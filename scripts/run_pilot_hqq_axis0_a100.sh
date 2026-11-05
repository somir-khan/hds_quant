#!/bin/bash
# ==============================================================================
# SLURM Submission Script for LONI (1x NVIDIA A100 80GB GPU)
# Project: Validate HQQ, then run axis-0 HQQ + Instruct pilot
# ==============================================================================
#SBATCH --job-name=hds_quant_hqq_axis0
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=results_hqq_axis0/pilot_hqq_%j.log
#SBATCH --error=results_hqq_axis0/pilot_hqq_%j.err

set -euo pipefail

# Submit from the project root; SLURM must find results_hqq_axis0/ before this script starts.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "hds_quant/run_experiment.py" ]; then
    echo "Submit this job from the project directory containing hds_quant/." >&2
    exit 1
fi

# Inherit the cache used for pre-downloading. Export HF_HOME before sbatch
# when using a non-default location; do not replace it with a job-local cache.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "Hugging Face cache root: $HF_HOME"

echo "=== Starting HDS-Quant Pilot Experiment (HQQ + Instruct) on $(hostname) ==="
date

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

mkdir -p results_hqq_axis0/cache

# ------------------------------------------------------------------------------
# Pre-flight: require CUDA, then run correctness and configured latency checks.
# ------------------------------------------------------------------------------
echo ">>> Running CUDA verification and latency tests..."
python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. This GPU pilot will not fall back to CPU.")
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
python -m pytest tests/test_stage1_math.py -k "cuda or encode_speed_budget" -v

# ------------------------------------------------------------------------------
# First validate both axes on the same full WikiText-2 test stream against packed
# HQQLinear. An NLL parity failure stops the pilot. These are accuracy checks,
# not measurements of packed inference speed or memory savings.
python -m pytest tests/test_hqq_axes.py -q
python -m hds_quant.validate_hqq \
    --model-id "meta-llama/Llama-3.2-1B-Instruct" \
    --device cuda --axes 0 1 --group-sizes 128 --bits 2 3 4 \
    --output results_hqq_axis0/hqq_validation.json

# ------------------------------------------------------------------------------
# Run 1: Strict Proposal Baseline
# Residual anchor: fp16_strict, Coord binding: raw_index_strict, Readout: pooled
# Computes marginal loss labels for all seeds and caches them.
# ------------------------------------------------------------------------------
echo ">>> [1/3] Executing Run 1: Strict Proposal Baseline (HQQ)..."
python -m hds_quant.run_experiment \
    --model-id "meta-llama/Llama-3.2-1B-Instruct" \
    --eval-dataset "wikitext2" \
    --quantizer "hqq" \
    --hqq-axis 0 \
    --quant-group-size 128 \
    --residual-anchor "fp16_strict" \
    --coord-binding "raw_index_strict" \
    --readout-grouping "pooled" \
    --target-transform "signed_log1p" \
    --seeds 42 100 2026 \
    --candidate-bits 1 2 3 4 \
    --budgets 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 \
    --cache-dir "results_hqq_axis0/cache" \
    --reuse-cache \
    --output-dir "results_hqq_axis0" \
    --run-tag "strict_baseline_hqq_axis0"

# ------------------------------------------------------------------------------
# Run 2: Tweaked Canonical Model
# Residual anchor: int4_aligned, Coord binding: salience_bucket, Readout: per_bit
# Reuses the exact same cached labels (0.0s label time) and evaluates low-shot curve.
# ------------------------------------------------------------------------------
echo ">>> [2/3] Executing Run 2: Tweaked Canonical Model with Low-Shot Curve (HQQ)..."
python -m hds_quant.run_experiment \
    --model-id "meta-llama/Llama-3.2-1B-Instruct" \
    --eval-dataset "wikitext2" \
    --quantizer "hqq" \
    --hqq-axis 0 \
    --quant-group-size 128 \
    --residual-anchor "int4_aligned" \
    --coord-binding "salience_bucket" \
    --readout-grouping "per_bit" \
    --target-transform "signed_log1p" \
    --seeds 42 100 2026 \
    --candidate-bits 1 2 3 4 \
    --budgets 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 \
    --run-low-shot-curve \
    --cache-dir "results_hqq_axis0/cache" \
    --reuse-cache \
    --output-dir "results_hqq_axis0" \
    --run-tag "salience_int4_tweak_hqq_axis0"

# ------------------------------------------------------------------------------
# Step 3: Format tables, Markdown report, LaTeX booktabs, and summary CSV
# ------------------------------------------------------------------------------
echo ">>> [3/3] Generating Supervisor Report and Conference Tables..."
python scripts/format_tables.py --input-dir "results_hqq_axis0" --output-dir "results_hqq_axis0"

echo "=== All runs completed successfully ==="
date

