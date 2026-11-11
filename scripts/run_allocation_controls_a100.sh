#!/bin/bash
#SBATCH --job-name=hqq_alloc_controls
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results_hqq_allocation_controls/controls_%j.log
#SBATCH --error=results_hqq_allocation_controls/controls_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python environment not found: $PYTHON_BIN. Set HQQ_PYTHON in your existing .env." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
extra=()
if [ -n "${1:-}" ]; then extra+=(--source-dir "$1"); fi
"$PYTHON_BIN" -m hds_quant.run_allocation_controls \
  --config configs/allocation_controls/llama31_8b_wikitext_seed42_v1.json \
  --output-dir "results_hqq_allocation_controls/job_${SLURM_JOB_ID:-manual}_llama31_8b_wikitext_allocation_controls_seed42_v1" \
  "${extra[@]}"
