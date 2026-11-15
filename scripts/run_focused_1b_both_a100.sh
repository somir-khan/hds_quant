#!/bin/bash
#SBATCH --job-name=hqq_1b_both
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results_hqq_1b_focused/run_%j.log
#SBATCH --error=results_hqq_1b_focused/run_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then echo "Set HQQ_PYTHON to your existing environment." >&2; exit 1; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$PYTHON_BIN" -m hds_quant.run_focused_1b \
  --config configs/coverage_generalization/llama32_1b_both_focused_v1.json \
  --output-root "results_hqq_1b_focused/job_${SLURM_JOB_ID:-manual}_both_v1" "$@"
